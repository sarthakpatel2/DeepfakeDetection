import argparse
import sys
from pathlib import Path

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from app.models.detector import load_detector, predict_image
from app.models.utils import load_image_bgr, preprocess_image_for_model
from app.config import settings


def main():
    parser = argparse.ArgumentParser(description="Test trained detector on a single image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--ckpt", default="./data/detector.pt", help="Path to model checkpoint")
    parser.add_argument("--no-face-detection", action="store_true", 
                       help="Disable face detection preprocessing")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_detector(checkpoint_path=args.ckpt)
    device = next(model.parameters()).device

    img = load_image_bgr(str(image_path))
    
    # Use face detection unless explicitly disabled
    use_face_detection = settings.USE_FACE_DETECTION and not args.no_face_detection
    
    tensor = preprocess_image_for_model(
        img,
        use_face_detection=use_face_detection,
        face_padding=0.2,
        use_lighting_normalization=True,
        inference_fallback_center_crop=True,
        device=device,
    )
    tensor = tensor.to(device)

    tau_low = float(getattr(model, "tau_low", settings.UNCERTAIN_LOW))
    tau_high = float(getattr(model, "tau_high", settings.UNCERTAIN_HIGH))
    score, label = predict_image(model, tensor, uncertain_low=tau_low, uncertain_high=tau_high)
    print(f"Image: {image_path}")
    print(f"Face detection: {'Enabled' if use_face_detection else 'Disabled'}")
    print(f"Score: {score:.4f}")
    print(f"Label: {label}")


if __name__ == "__main__":
    main()
