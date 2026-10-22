"""
diagnose_eval.py
================
Quick diagnostic: MTCNN face-detection rate on your REAL/FAKE folders.

Usage (from backend/):
  python scripts/diagnose_eval.py --real_dir data/val/real --fake_dir data/val/fake
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image


EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def diagnose(real_dir: str, fake_dir: str, max_images: int = 200) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(
        image_size=224,
        margin=int(224 * 0.2),  # paper padding=0.2
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        keep_all=False,
        select_largest=True,
        device=device,
    )

    for label, folder in [("REAL", real_dir), ("FAKE", fake_dir)]:
        paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS][:max_images]
        face_detected = 0
        sizes = []

        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                sizes.append(img.size)
                boxes, _ = mtcnn.detect(img)
                if boxes is not None and len(boxes) > 0:
                    face_detected += 1
            except Exception:
                pass

        avg_w = float(np.mean([s[0] for s in sizes])) if sizes else 0.0
        avg_h = float(np.mean([s[1] for s in sizes])) if sizes else 0.0
        n = len(paths)
        rate = (face_detected / max(1, n)) * 100.0

        print(f"\n[{label}] {n} images sampled")
        print(f"  Face detection rate : {face_detected}/{n} = {rate:.1f}%")
        print(f"  Avg image size      : {avg_w:.0f} x {avg_h:.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", required=True)
    p.add_argument("--fake_dir", required=True)
    p.add_argument("--max_images", type=int, default=200)
    args = p.parse_args()

    diagnose(args.real_dir, args.fake_dir, max_images=args.max_images)

