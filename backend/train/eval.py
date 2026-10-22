from pathlib import Path
import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add backend root to path so we can import app modules when run as a script
backend_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, backend_dir)

from app.config import settings
from app.models.detector import ResNetDeepfakeDetector
try:  # When run as module
	from .data_loader import make_loader  # type: ignore
except Exception:  # When run as script
	from data_loader import make_loader  # type: ignore
import torch.nn.functional as F


def evaluate(real_dir: str, fake_dir: str, batch_size: int = 16, ckpt_path: str = "./data/detector.pt"):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ResNetDeepfakeDetector(pretrained=True).to(device)
	if Path(ckpt_path).exists():
		state = torch.load(ckpt_path, map_location=device)
		if isinstance(state, dict) and "model_state_dict" in state:
			model.load_state_dict(state["model_state_dict"], strict=False)
		else:
			model.load_state_dict(state, strict=False)
	model.eval()
	loader: DataLoader = make_loader(
		real_dir,
		fake_dir,
		batch_size=batch_size,
		shuffle=False,
		use_face_detection=settings.USE_FACE_DETECTION,
		min_face_size=settings.MIN_FACE_SIZE,
		face_confidence=settings.FACE_DETECTION_CONFIDENCE,
		face_padding=settings.FACE_PADDING,
		use_lighting_normalization=settings.USE_LIGHTING_NORMALIZATION,
		use_augment=False,
	)
	correct = 0
	total = 0
	tau_low = float(getattr(model, "tau_low", settings.UNCERTAIN_LOW))
	tau_high = float(getattr(model, "tau_high", settings.UNCERTAIN_HIGH))
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			logits = model(x)
			prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
			yt = y.detach().cpu().numpy().reshape(-1).astype(int)
			preds = np.where(prob < tau_low, 0, np.where(prob > tau_high, 1, -1))
			mask = preds != -1
			if mask.sum() == 0:
				continue
			correct += int(np.sum(preds[mask] == yt[mask]))
			total += int(mask.sum())
	acc = correct / total if total else 0.0
	print(f"Accuracy: {acc:.4f}")
	return acc


if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
	parser.add_argument("--real-dir", type=str, default="./data/val/real",
		help="Directory containing real images")
	parser.add_argument("--fake-dir", type=str, default="./data/val/fake",
		help="Directory containing fake images")
	parser.add_argument("--batch-size", type=int, default=16,
		help="Batch size")
	parser.add_argument("--ckpt-path", type=str, default="./data/detector.pt",
		help="Path to model checkpoint")
	
	args = parser.parse_args()
	
	print(f"Evaluating model...")
	print(f"  Real images: {args.real_dir}")
	print(f"  Fake images: {args.fake_dir}")
	print(f"  Model: {args.ckpt_path}")
	print()
	
	accuracy = evaluate(
		real_dir=args.real_dir,
		fake_dir=args.fake_dir,
		batch_size=args.batch_size,
		ckpt_path=args.ckpt_path
	)
	
	print(f"\nFinal Accuracy: {accuracy:.2%}")



