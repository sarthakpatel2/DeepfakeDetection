from typing import Tuple
import torch
import torch.nn as nn

from ..config import settings


class ResNetDeepfakeDetector(nn.Module):
	"""
	Pretrained ResNet18 backbone for deepfake detection. Better generalization
	than a small CNN for diverse lighting, backgrounds, and image types.
	"""

	def __init__(self, pretrained: bool = True, dropout: float = 0.5):
		super().__init__()
		from torchvision.models import resnet18
		try:
			from torchvision.models import ResNet18_Weights
			weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
		except ImportError:
			weights = "IMAGENET1K_V1" if pretrained else None
		backbone = resnet18(weights=weights)
		# Remove original classifier
		self.features = nn.Sequential(*list(backbone.children())[:-1])
		self.dropout = nn.Dropout(p=dropout)
		self.head = nn.Linear(512, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Returns logits (no sigmoid) for BCEWithLogitsLoss
		h = self.features(x)
		h = torch.flatten(h, 1)
		h = self.dropout(h)
		return self.head(h)


class SimpleConvDiscriminator(nn.Module):
	"""Legacy small CNN - kept for reference. Use ResNetDeepfakeDetector instead."""

	def __init__(self, dropout: float = 0.3):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
		self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
		self.dropout = nn.Dropout(p=dropout)
		self.head = nn.Linear(64 * 16 * 16, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = nn.functional.relu(self.conv1(x))
		x = nn.functional.relu(self.conv2(x))
		x = nn.functional.relu(self.conv3(x))
		x = torch.flatten(x, 1)
		x = self.dropout(x)
		return self.head(x)


def _get_detector_model() -> nn.Module:
	"""Factory for the active detector model (ResNet-based)."""
	return ResNetDeepfakeDetector(pretrained=True, dropout=0.5)


def load_detector(device: torch.device | None = None, checkpoint_path: str = "./data/detector.pt") -> nn.Module:
	"""Load the detector model, optionally from a checkpoint.

	Uses ResNet18 pretrained on ImageNet. Old SimpleConvDiscriminator checkpoints
	will not load; retrain to get ResNet weights.
	"""
	from pathlib import Path
	device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
	model = _get_detector_model().to(device)

	ckpt_path = Path(checkpoint_path)
	if ckpt_path.exists():
		try:
			state = torch.load(ckpt_path, map_location=device)
			# Handle both raw state_dict and wrapped format
			if isinstance(state, dict) and "model_state_dict" in state:
				state_dict = state["model_state_dict"]
				model.tau_low = float(state.get("tau_low", settings.UNCERTAIN_LOW))
				model.tau_high = float(state.get("tau_high", settings.UNCERTAIN_HIGH))
			else:
				state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
				model.tau_low = float(settings.UNCERTAIN_LOW)
				model.tau_high = float(settings.UNCERTAIN_HIGH)
			# If this is our packaged checkpoint, load strictly to avoid silent partial loads.
			# (No architecture change; this only ensures weights really match the model.)
			strict = bool(isinstance(state, dict) and "model_state_dict" in state)
			model.load_state_dict(state_dict, strict=strict)
			print(f"Loaded model weights from {ckpt_path}")
		except Exception as e:
			print(f"Warning: Could not load checkpoint {ckpt_path}: {e}")
			print("Using pretrained ResNet + random classifier head (retrain recommended)")
			model.tau_low = float(settings.UNCERTAIN_LOW)
			model.tau_high = float(settings.UNCERTAIN_HIGH)
	else:
		print(f"No checkpoint at {ckpt_path}, using pretrained ResNet (retrain recommended)")
		model.tau_low = float(settings.UNCERTAIN_LOW)
		model.tau_high = float(settings.UNCERTAIN_HIGH)

	model.eval()
	return model


def score_to_label(score: float, uncertain_low: float = 0.4, uncertain_high: float = 0.6) -> str:
	"""Convert raw score to label: REAL, UNCERTAIN, or FAKE."""
	if score < uncertain_low:
		return "REAL"
	if score >= uncertain_high:
		return "FAKE"
	return "UNCERTAIN"


@torch.no_grad()
def predict_image(model: nn.Module, image_tensor: torch.Tensor,
				  uncertain_low: float = 0.4, uncertain_high: float = 0.6) -> Tuple[float, str]:
	"""Predict deepfake score. image_tensor: 3x224x224, ImageNet-normalized."""
	image_tensor = image_tensor.unsqueeze(0)
	logits = model(image_tensor)
	prob = torch.sigmoid(logits).item()
	label = score_to_label(prob, uncertain_low, uncertain_high)
	return prob, label
