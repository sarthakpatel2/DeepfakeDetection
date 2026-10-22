# ============================================================
# evaluate_model.py  —  aligned exactly with training script
# Usage:
#   python scripts/evaluate_model.py \
#       --model_path backend/data/detector.pt \
#       --real_dir  data/val/real \
#       --fake_dir  data/val/fake
# ============================================================

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T
from facenet_pytorch import MTCNN
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────
# SEED — must match training
# ─────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
FACE_PADDING_P = 0.2
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ─────────────────────────────────────────────────────────────
# PREPROCESSING — identical to training script
# ─────────────────────────────────────────────────────────────
def apply_clahe_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE on L channel (paper Tools table: clipLimit=2.0, grid=8x8)."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    out_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def crop_with_paper_padding(img_rgb, box_xyxy, p=FACE_PADDING_P):
    """Paper Eq. 1-5 padding crop."""
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    x, y = int(np.floor(x1)), int(np.floor(y1))
    w, h = int(np.floor(x2 - x1)), int(np.floor(y2 - y1))
    px, py = int(np.floor(w * p)), int(np.floor(h * p))
    xx1, yy1 = max(0, x - px), max(0, y - py)
    xx2, yy2 = min(W, x + w + px), min(H, y + h + py)
    return img_rgb[yy1:yy2, xx1:xx2]


def build_mtcnn(device):
    """Identical params to training script."""
    return MTCNN(
        image_size=224,
        margin=int(224 * 0.2),
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        keep_all=False,
        select_largest=True,
        device=device,
    )


# Val/Test transform — NO augmentation (same as VAL_TF in training)
EVAL_TF = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Fallback if MTCNN finds no face
FALLBACK_TF = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TTA_TFS = [
    T.Compose([T.Resize((224,224)), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.Resize((256,256)), T.CenterCrop(224), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.Resize((232,232)), T.CenterCrop(224), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
]


def preprocess_image(path: Path, mtcnn: MTCNN, model: nn.Module, device: torch.device) -> tuple[float, bool]:
    """
    Full pipeline: CLAHE → MTCNN face detect → paper padding crop → normalize.
    Falls back to CenterCrop if no face detected.
    Matches training script __getitem__ exactly.
    """
    try:
        img_pil = Image.open(path).convert("RGB")
        img_rgb = np.array(img_pil)
        img_rgb = apply_clahe_rgb(img_rgb)          # CLAHE first

        boxes, _ = mtcnn.detect(Image.fromarray(img_rgb))
        if boxes is not None and len(boxes) > 0:
            face_rgb = crop_with_paper_padding(img_rgb, boxes[0])
            img = Image.fromarray(face_rgb)
            probs = []
            for tf in TTA_TFS:
                x = tf(img).unsqueeze(0).to(device)
                logits = model(x)
                probs.append(torch.sigmoid(logits).item())
            return float(np.mean(probs)), True
        else:
            img = Image.fromarray(img_rgb)
            probs = []
            for tf in TTA_TFS:
                x = tf(img).unsqueeze(0).to(device)
                logits = model(x)
                probs.append(torch.sigmoid(logits).item())
            return float(np.mean(probs)), False
    except Exception as e:
        print(f"  [WARN] Failed to load {path.name}: {e}")
        return 0.5, False


# ─────────────────────────────────────────────────────────────
# MODEL — identical architecture to training script
# ─────────────────────────────────────────────────────────────
class ResNet18Detector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout  = nn.Dropout(p=0.5)
        self.head     = nn.Linear(512, 1)

    def forward(self, x):
        h = self.features(x)
        h = torch.flatten(h, 1)
        h = self.dropout(h)
        return self.head(h)


def load_model(model_path: str, device: torch.device):
    """
    Load checkpoint and extract tau_low, tau_high saved during training.
    Falls back to 0.4 / 0.6 if not found (never uses raw 0.5).
    """
    ckpt = torch.load(model_path, map_location=device)

    model = ResNet18Detector().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()  # CRITICAL: eval mode before any inference

    tau_low  = ckpt.get('tau_low',  0.4)
    tau_high = ckpt.get('tau_high', 0.6)
    val_acc  = ckpt.get('val_accuracy', None)

    print(f"Checkpoint loaded from: {model_path}")
    print(f"  Saved val accuracy : {val_acc*100:.2f}%" if val_acc else "  (no val_accuracy in checkpoint)")
    print(f"  tau_low  = {tau_low:.2f}")
    print(f"  tau_high = {tau_high:.2f}")
    return model, tau_low, tau_high


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(model, mtcnn, image_paths, labels, tau_low, tau_high, device, batch_size=32):
    """
    Run inference with:
      - model.eval() enforced
      - torch.no_grad() enforced
      - Same preprocessing as training
      - Paper three-class threshold (tau_low / tau_high / UNCERTAIN)
    """
    model.eval()

    all_probs  = []
    all_preds  = []   # three-class: 0=REAL, 1=FAKE, -1=UNCERTAIN
    all_labels = []
    face_found_total = 0
    face_not_found_total = 0
    face_found_real = 0
    face_not_found_real = 0
    face_found_fake = 0
    face_not_found_fake = 0

    print(f"\nRunning eval on {len(image_paths)} images...")
    print(f"Using tau_low={tau_low:.2f}, tau_high={tau_high:.2f}")

    with torch.no_grad():
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            prob, used_face = preprocess_image(path, mtcnn, model, device)
            if used_face:
                face_found_total += 1
                if label == 0:
                    face_found_real += 1
                else:
                    face_found_fake += 1
            else:
                face_not_found_total += 1
                if label == 0:
                    face_not_found_real += 1
                else:
                    face_not_found_fake += 1

            if prob < tau_low:
                pred = 0       # REAL
            elif prob > tau_high:
                pred = 1       # FAKE
            else:
                pred = -1      # UNCERTAIN

            all_probs.append(prob)
            all_preds.append(pred)
            all_labels.append(label)

            if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
                print(f"  [{i+1}/{len(image_paths)}] processed", flush=True)

    n_total = len(image_paths)
    print(f"\nFace detection summary:")
    print(f"  Total images processed : {n_total}")
    print(f"  Face detected (crop)   : {face_found_total}")
    print(f"  No face (fallback)     : {face_not_found_total}")
    real_total = int(np.sum(np.array(labels) == 0))
    fake_total = int(np.sum(np.array(labels) == 1))
    real_rate = (face_found_real / max(1, real_total)) * 100.0
    fake_rate = (face_found_fake / max(1, fake_total)) * 100.0
    print(f"  Face detection rate REAL: {face_found_real}/{real_total} = {real_rate:.1f}%")
    print(f"  Face detection rate FAKE: {face_found_fake}/{fake_total} = {fake_rate:.1f}%")
    if fake_rate < 60.0:
        print("WARNING: Low face detection on FAKE images. Eval set may not match")
        print("training distribution. Use eval_set.zip exported from Kaggle instead.")

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def print_report(probs, preds, labels, tau_low, tau_high):
    uncertain_mask = preds == -1
    decided_mask   = preds != -1

    n_total     = len(labels)
    n_uncertain = int(uncertain_mask.sum())
    n_decided   = int(decided_mask.sum())

    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Total images   : {n_total}")
    print(f"UNCERTAIN      : {n_uncertain} ({n_uncertain/n_total*100:.1f}%) — skipped from metrics")
    print(f"Decided        : {n_decided} ({n_decided/n_total*100:.1f}%)")

    if n_decided == 0:
        print("No decided predictions — check threshold values.")
        return

    d_preds  = preds[decided_mask]
    d_labels = labels[decided_mask]

    correct = int((d_preds == d_labels).sum())
    acc     = correct / n_decided

    TP = int(((d_preds == 1) & (d_labels == 1)).sum())
    TN = int(((d_preds == 0) & (d_labels == 0)).sum())
    FP = int(((d_preds == 1) & (d_labels == 0)).sum())
    FN = int(((d_preds == 0) & (d_labels == 1)).sum())

    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    print(f"\nAccuracy   : {acc*100:.2f}%")
    print(f"Precision  : {prec*100:.2f}%  (of predicted FAKE, how many were FAKE)")
    print(f"Recall     : {rec*100:.2f}%   (of actual FAKE, how many did we catch)")
    print(f"F1 Score   : {f1*100:.2f}%")
    print(f"\nConfusion Matrix (decided only):")
    print(f"               Predicted REAL   Predicted FAKE")
    print(f"Actual REAL    TN={TN:<6}          FP={FP}")
    print(f"Actual FAKE    FN={FN:<6}          TP={TP}")

    # Also report at binary 0.5 for comparison
    binary_preds = (probs > 0.5).astype(int)
    b_correct = int((binary_preds == labels).sum())
    b_acc = b_correct / n_total
    print(f"\n--- For reference: binary 0.5 threshold on ALL images ---")
    print(f"Accuracy: {b_acc*100:.2f}%")

    # Score distribution
    fake_probs = probs[labels == 1]
    real_probs = probs[labels == 0]
    print(f"\nScore distribution:")
    print(f"  REAL images — mean prob: {real_probs.mean():.3f}  std: {real_probs.std():.3f}")
    print(f"  FAKE images — mean prob: {fake_probs.mean():.3f}  std: {fake_probs.std():.3f}")
    print(f"  (If FAKE mean < 0.5, the model is biased toward REAL — retrain with better balance)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to detector.pt")
    parser.add_argument("--real_dir",   required=True, help="Directory of real images")
    parser.add_argument("--fake_dir",   required=True, help="Directory of fake images")
    parser.add_argument("--max_images", type=int, default=None, help="Cap per class (optional)")
    args = parser.parse_args()

    real_paths = sorted([p for p in Path(args.real_dir).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS])
    fake_paths = sorted([p for p in Path(args.fake_dir).rglob("*") if p.suffix.lower() in SUPPORTED_EXTS])

    if args.max_images:
        real_paths = real_paths[:args.max_images]
        fake_paths = fake_paths[:args.max_images]

    print(f"Real images: {len(real_paths)}")
    print(f"Fake images: {len(fake_paths)}")

    image_paths = real_paths + fake_paths
    labels      = [0] * len(real_paths) + [1] * len(fake_paths)

    model, tau_low, tau_high = load_model(args.model_path, DEVICE)
    mtcnn = build_mtcnn(DEVICE)

    probs, preds, label_arr = evaluate(
        model, mtcnn, image_paths, labels,
        tau_low, tau_high, DEVICE
    )

    print_report(probs, preds, label_arr, tau_low, tau_high)


if __name__ == "__main__":
    main()