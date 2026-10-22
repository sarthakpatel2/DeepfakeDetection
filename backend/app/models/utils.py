from typing import Tuple, Optional, List
import cv2
import numpy as np
import torch
import logging

from ..config import settings, IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)

# Single shared MTCNN (paper config)
_mtcnn = None


def normalize_lighting(img_bgr: np.ndarray) -> np.ndarray:
	"""
	Normalize lighting using CLAHE on luminance to handle dramatic highlights/shadows,
	indoor lighting, and warm tones. Makes face detection and model input more consistent.
	
	Returns:
		BGR image with normalized lighting
	"""
	lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
	L, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	L_norm = clahe.apply(L)
	lab_norm = cv2.merge([L_norm, a, b])
	return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)


def _get_mtcnn(device: torch.device):
	"""Lazy init MTCNN (must match paper Tools table everywhere)."""
	global _mtcnn
	if _mtcnn is None:
		try:
			from facenet_pytorch import MTCNN
			_mtcnn = MTCNN(
				image_size=224,
				margin=int(224 * 0.2),   # padding=0.2 as in paper
				min_face_size=20,
				thresholds=[0.6, 0.7, 0.7],
				keep_all=False,
				select_largest=True,
				device=device,
			)
			logger.info("MTCNN initialized (facenet_pytorch)")
		except Exception as e:
			logger.warning(f"Failed to initialize MTCNN: {e}")
			_mtcnn = False  # Mark as unavailable
	return _mtcnn


def load_image_bgr(path: str) -> np.ndarray:
	"""Load image in BGR format using OpenCV."""
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	if img is None:
		raise ValueError(f"Failed to read image: {path}")
	return img


def detect_face_box_xyxy(img_rgb: np.ndarray, device: torch.device) -> Optional[np.ndarray]:
	"""Detect the largest face and return the box [x1, y1, x2, y2]."""
	mtcnn = _get_mtcnn(device)
	if mtcnn is False or mtcnn is None:
		return None
	try:
		from PIL import Image
		img_pil = Image.fromarray(img_rgb)
		boxes, _ = mtcnn.detect(img_pil)
		if boxes is None or len(boxes) == 0:
			return None
		return boxes[0]
	except Exception as e:
		logger.warning(f"Face detection failed: {e}")
		return None


def extract_face_region(img_rgb: np.ndarray, box_xyxy: np.ndarray, padding: float = 0.2) -> np.ndarray:
	"""
	Extract and crop face region from image with padding.
	
	Args:
		img_rgb: RGB image as numpy array
		box_xyxy: Face box [x1, y1, x2, y2]
		padding: Padding ratio around face (0.2 = 20% padding)
	
	Returns:
		Cropped face region as RGB image
	"""
	x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
	x = int(np.floor(x1))
	y = int(np.floor(y1))
	w = int(np.floor(x2 - x1))
	h = int(np.floor(y2 - y1))
	img_h, img_w = img_rgb.shape[:2]
	padding_x = int(np.floor(w * padding))
	padding_y = int(np.floor(h * padding))
	xx1 = max(0, x - padding_x)
	yy1 = max(0, y - padding_y)
	xx2 = min(img_w, x + w + padding_x)
	yy2 = min(img_h, y + h + padding_y)
	return img_rgb[yy1:yy2, xx1:xx2]


def detect_and_extract_largest_face(img_rgb: np.ndarray, device: torch.device, padding: float = 0.2) -> Optional[np.ndarray]:
	"""
	Detect faces and extract the largest face region. Retries with lower confidence
	if no face found (helps with dramatic lighting, indoor, warm tones).
	
	Args:
		img_rgb: RGB image as numpy array
		min_face_size: Minimum face size in pixels
		confidence: Minimum confidence threshold
		padding: Padding ratio around face
		retry_lower_confidence: If True, retry with 0.75 then 0.6 when no face found
	
	Returns:
		Cropped face region as RGB image, or None if no face detected
	"""
	box = detect_face_box_xyxy(img_rgb, device=device)
	if box is None:
		return None
	return extract_face_region(img_rgb, box, padding)


def preprocess_image_for_model(img_bgr: np.ndarray, size: Tuple[int, int] | None = None,
							   use_face_detection: bool = True,
							   face_padding: float = 0.2,
							   use_lighting_normalization: bool = True,
							   use_imagenet_normalization: bool = True,
							   inference_fallback_center_crop: bool = False,
							   device: torch.device | None = None) -> torch.Tensor:
	"""
	Preprocess image for model input (ResNet expects 224x224, ImageNet-normalized).
	
	Args:
		img_bgr: BGR image as numpy array
		size: Target size (default 224x224 for ResNet)
		use_face_detection: Whether to detect and extract face first
		min_face_size: Minimum face size for detection
		face_confidence: Face detection confidence threshold
		face_padding: Padding ratio around detected face
		use_lighting_normalization: CLAHE before face detection for challenging lighting
		use_imagenet_normalization: Apply ImageNet mean/std (required for ResNet)
	
	Returns:
		Preprocessed tensor (3, H, W), ImageNet-normalized if use_imagenet_normalization
	"""
	if size is None:
		s = settings.MODEL_INPUT_SIZE
		size = (s, s)
	device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
	# Lighting normalization first - helps face detection in dramatic/indoor/warm lighting
	if use_lighting_normalization:
		img_bgr = normalize_lighting(img_bgr)
	
	# Convert BGR to RGB
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	
	# Face detection and extraction
	if use_face_detection:
		face_crop = detect_and_extract_largest_face(img_rgb, device=device, padding=face_padding)
		if face_crop is not None:
			img_rgb = face_crop
			logger.debug("Face detected and extracted")
		else:
			if inference_fallback_center_crop:
				from PIL import Image
				import torchvision.transforms as T
				fallback = T.Compose([
					T.Resize(256),
					T.CenterCrop(224),
					T.ToTensor(),
					T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
				])
				return fallback(Image.fromarray(img_rgb))
			logger.warning("No face detected, using full image")
	
	# Resize to target size
	img_resized = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
	
	# Normalize to [0, 1]
	img_normalized = img_resized.astype(np.float32) / 255.0
	
	# Convert HWC to CHW
	img_tensor = np.transpose(img_normalized, (2, 0, 1))
	tensor = torch.from_numpy(img_tensor).float()

	# ImageNet normalization for pretrained ResNet
	if use_imagenet_normalization:
		mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
		std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
		tensor = (tensor - mean) / std

	return tensor
















