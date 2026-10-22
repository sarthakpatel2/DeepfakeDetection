from pathlib import Path
import argparse
import sys
import os
import signal

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add backend root to path so we can import app modules when run as a script
backend_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, backend_dir)

from app.config import settings
from app.models.detector import ResNetDeepfakeDetector
from app.models.losses import BCELossWithLogits

# Import data loader (works both as module and as script)
try:  # When run as a module: python -m train.train
	from .data_loader import make_loader  # type: ignore
except Exception:  # When run as a script: python train/train.py
	from data_loader import make_loader  # type: ignore


def _eval_accuracy(model, loader, device):
	"""Compute accuracy on a dataset. Model returns logits."""
	model.eval()
	correct, total = 0, 0
	tau_low = float(getattr(model, "tau_low", settings.UNCERTAIN_LOW))
	tau_high = float(getattr(model, "tau_high", settings.UNCERTAIN_HIGH))
	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			logits = model(x)
			prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
			yt = y.detach().cpu().numpy().reshape(-1).astype(int)
			preds = np.where(prob < tau_low, 0, np.where(prob > tau_high, 1, -1))
			mask = preds != -1
			if mask.sum() == 0:
				continue
			correct += int(np.sum(preds[mask] == yt[mask]))
			total += int(mask.sum())
	model.train()
	return correct / total if total else 0.0


def train(
	real_dir: str,
	fake_dir: str,
	epochs: int = 1,
	lr: float = 1e-3,
	batch_size: int = 16,
	save_path: str = "./data/detector.pt",
	weights: str | None = None,
	val_real_dir: str | None = None,
	val_fake_dir: str | None = None,
	num_workers: int = 0,
) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	model = ResNetDeepfakeDetector(pretrained=True).to(device)

	ckpt_path = Path(weights or save_path)
	if ckpt_path.exists():
		state_dict = torch.load(ckpt_path, map_location=device)
		# Unwrap if saved as {"state_dict": ...}
		state_dict = state_dict.get("state_dict", state_dict)
		# Skip if old SimpleConvDiscriminator checkpoint (incompatible with ResNet)
		if "conv1.weight" in state_dict:
			print(f"Checkpoint at {ckpt_path} is from old model (SimpleConvDiscriminator). Starting fresh with ResNet.")
		else:
			# Load only matching keys (ignore head if shapes differ)
			model_dict = model.state_dict()
			loaded = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
			if loaded:
				model.load_state_dict(loaded, strict=False)
				print(f"Loaded {len(loaded)} layers from {ckpt_path}")
			else:
				print(f"No compatible weights in {ckpt_path}, using pretrained ResNet")

	criterion = BCELossWithLogits()
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
	print("Building training dataset (scanning image dirs)...", flush=True)
	loader: DataLoader = make_loader(
		real_dir,
		fake_dir,
		batch_size=batch_size,
		num_workers=num_workers,
		use_face_detection=settings.USE_FACE_DETECTION,
		min_face_size=settings.MIN_FACE_SIZE,
		face_confidence=settings.FACE_DETECTION_CONFIDENCE,
		face_padding=settings.FACE_PADDING,
		use_lighting_normalization=settings.USE_LIGHTING_NORMALIZATION,
		use_augment=True,
	)

	if len(loader.dataset) == 0:
		print("No training samples found. Check your --real-dir / --fake-dir paths.")
		return

	print(f"Training set: {len(loader.dataset)} samples ({len(loader)} batches/epoch)")
	progress_every = min(100, max(1, len(loader) // 10))  # every 10% or 100, whichever is smaller
	print(f"Starting epoch 1... (progress every {progress_every} batches)")

	val_loader = None
	if val_real_dir and val_fake_dir:
		if Path(val_real_dir).exists() and Path(val_fake_dir).exists():
			val_loader = make_loader(
				val_real_dir, val_fake_dir,
				batch_size=batch_size, shuffle=False, num_workers=num_workers,
				use_face_detection=settings.USE_FACE_DETECTION,
				min_face_size=settings.MIN_FACE_SIZE,
				face_confidence=settings.FACE_DETECTION_CONFIDENCE,
				face_padding=settings.FACE_PADDING,
				use_lighting_normalization=settings.USE_LIGHTING_NORMALIZATION,
				use_augment=False,
			)
			if len(val_loader.dataset) > 0:
				print(f"Validation set: {len(val_loader.dataset)} samples")
			else:
				val_loader = None

		best_val_acc = 0.0
	model.train()
	for epoch in range(epochs):
		total_loss = 0.0
		num_batches = 0

		for batch_idx, (x, y) in enumerate(loader):
			if batch_idx == 0:
				print(f"  Epoch {epoch + 1} batch 1 - loading OK", flush=True)
			x = x.to(device)
			y = y.to(device)
			optimizer.zero_grad()
			logits = model(x)
			loss = criterion(logits, y)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
			num_batches += 1
			# Progress every N batches (10% of epoch or 100, whichever is smaller)
			if (batch_idx + 1) % progress_every == 0:
				print(f"  Epoch {epoch + 1} batch {batch_idx + 1}/{len(loader)} | loss: {loss.item():.4f}")

		avg_loss = total_loss / num_batches if num_batches else 0.0
		msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} (batches: {num_batches})"

		if val_loader is not None:
			print(f"  Running validation ({len(val_loader.dataset)} samples)...", flush=True)
			val_acc = _eval_accuracy(model, val_loader, device)
			scheduler.step(val_acc)
			msg += f" | Val Acc: {val_acc:.4f}"
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				save_file = Path(save_path)
				save_file.parent.mkdir(parents=True, exist_ok=True)
				torch.save(model.state_dict(), save_file)
				msg += " [best, saved]"
		else:
			save_file = Path(save_path)
			save_file.parent.mkdir(parents=True, exist_ok=True)
			torch.save(model.state_dict(), save_file)

		print(msg)


def _force_exit(signum, frame):
	"""Force immediate exit on Ctrl+C (handles TensorFlow/PyTorch hangs)."""
	print("\nInterrupted. Exiting...", flush=True)
	os._exit(1)


if __name__ == "__main__":
	signal.signal(signal.SIGINT, _force_exit)
	try:
		signal.signal(signal.SIGTERM, _force_exit)
	except (AttributeError, ValueError):
		pass  # SIGTERM not available on Windows
	parser = argparse.ArgumentParser(description="Train deepfake classifier on images/frames.")
	parser.add_argument("--real-dir", type=str, default="./data/train/real", help="Directory with real images/frames.")
	parser.add_argument("--fake-dir", type=str, default="./data/train/fake", help="Directory with fake images/frames.")
	parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
	parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (lower for fine-tuning).")
	parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
	parser.add_argument("--save-path", type=str, default="./data/detector.pt", help="Where to save model weights.")
	parser.add_argument("--weights", type=str, default=None, help="Optional path to existing weights for warm start.")
	parser.add_argument("--val-real-dir", type=str, default="./data/val/real", help="Validation real images.")
	parser.add_argument("--val-fake-dir", type=str, default="./data/val/fake", help="Validation fake images.")
	parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0=main thread; try 2 or 4 for faster loading).")

	args = parser.parse_args()
	print("Starting training with configuration:")
	for k, v in vars(args).items():
		print(f"  {k}: {v}")

	train(
		real_dir=args.real_dir,
		fake_dir=args.fake_dir,
		epochs=args.epochs,
		lr=args.lr,
		batch_size=args.batch_size,
		save_path=args.save_path,
		weights=args.weights,
		val_real_dir=args.val_real_dir,
		val_fake_dir=args.val_fake_dir,
		num_workers=args.num_workers,
	)

