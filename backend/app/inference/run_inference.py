from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import torch
import logging
from PIL import Image

from ..models.detector import load_detector
from ..config import settings
from .heatmap import save_dummy_heatmap
from ..models.utils import preprocess_image_for_model, load_image_bgr

logger = logging.getLogger(__name__)

def _preprocess(image_path: Path, device: torch.device) -> torch.Tensor:
	"""
	Paper pipeline preprocessing:
	- CLAHE (clipLimit=2.0, grid=8x8)
	- MTCNN detect (min_face_size=20, thresholds=[0.6,0.7,0.7], select largest, padding=0.2)
	- Resize with OpenCV INTER_AREA
	- Eq. 6 + Eq. 7 normalization (ImageNet)
	- Inference-only fallback: CenterCrop if face detection fails
	"""
	img_bgr = load_image_bgr(str(image_path))
	tensor = preprocess_image_for_model(
		img_bgr,
		size=(224, 224),
		use_face_detection=True,
		face_padding=0.2,
		use_lighting_normalization=True,
		use_imagenet_normalization=True,
		inference_fallback_center_crop=True,
		device=device,
	)
	return tensor.unsqueeze(0)


def run_image_inference(image_path: Path) -> Dict[str, Any]:
	"""Run inference on a single image.

	Uses PIL + torchvision transforms matching Kaggle training.
	"""
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = load_detector(device=device)

	tensor = _preprocess(image_path, device=device).to(device)
	model.eval()
	torch.set_grad_enabled(False)
	with torch.no_grad():
		logits = model(tensor)
		prob = torch.sigmoid(logits).item()
	torch.set_grad_enabled(True)

	tau_low = float(getattr(model, "tau_low", settings.UNCERTAIN_LOW))
	tau_high = float(getattr(model, "tau_high", settings.UNCERTAIN_HIGH))
	if prob >= tau_high:
		label = "FAKE"
	elif prob < tau_low:
		label = "REAL"
	else:
		label = "UNCERTAIN"

	heatmap_path = save_dummy_heatmap(image_path, prob)

	return {
		"score": float(prob),
		"label": label,
		"heatmap_path": str(heatmap_path),
		"face_detected": False,
	}


def extract_video_frames(video_path: Path, out_dir: Path, max_frames: int = 64) -> List[Path]:
	out_dir.mkdir(parents=True, exist_ok=True)
	cap = cv2.VideoCapture(str(video_path))
	count = 0
	frame_paths: List[Path] = []
	success, frame = cap.read()
	while success and count < max_frames:
		filename = out_dir / f"frame_{count:04d}.png"
		cv2.imwrite(str(filename), frame)
		frame_paths.append(filename)
		count += 1
		success, frame = cap.read()
	cap.release()
	return frame_paths


def run_video_inference(frame_paths: List[Path]) -> Tuple[float, str, List[float]]:
	"""Run inference on video frames.

	Uses PIL + torchvision transforms matching Kaggle training.
	"""
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model = load_detector(device=device)
	per_frame_scores: List[float] = []

	for frame_path in frame_paths:
		try:
			tensor = _preprocess(frame_path, device=device).to(device)
			model.eval()
			torch.set_grad_enabled(False)
			with torch.no_grad():
				logits = model(tensor)
				prob = torch.sigmoid(logits).item()
			torch.set_grad_enabled(True)
			per_frame_scores.append(float(prob))
		except Exception as e:
			logger.warning(f"Failed to process frame {frame_path}: {e}")
			continue

	if len(per_frame_scores) == 0:
		return 0.0, "REAL", per_frame_scores

	avg_score = float(np.mean(per_frame_scores))
	tau_low = float(getattr(model, "tau_low", settings.UNCERTAIN_LOW))
	tau_high = float(getattr(model, "tau_high", settings.UNCERTAIN_HIGH))
	if avg_score < tau_low:
		label = "REAL"
	elif avg_score >= tau_high:
		label = "FAKE"
	else:
		label = "UNCERTAIN"

	logger.info(f"Processed {len(per_frame_scores)}/{len(frame_paths)} frames")
	return avg_score, label, per_frame_scores
















