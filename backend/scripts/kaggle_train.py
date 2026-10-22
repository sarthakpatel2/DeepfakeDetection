# ============================================================
# Deepfake Detection — Kaggle Multi-Dataset Training Script
# MODEL: ResNet-18 (matches local backend/app/models/detector.py)
# Pipeline: MTCNN face detect + paper padding + CLAHE + ImageNet normalization
# Handles: Images + Videos (auto-extracts frames from .mp4/.avi)
# ============================================================
# Datasets added:
#   - Celeb DF (v2)
#   - FaceForensics++ Dataset (C23)
#   - 140k Real and Fake Faces
# ============================================================

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SEED (must be called before anything else)
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)  # Call this before anything else

import os
import random
import cv2
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
from facenet_pytorch import MTCNN

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_PATH      = Path("/kaggle/working/detector.pt")
FRAMES_DIR       = Path("/kaggle/working/extracted_frames")  # where to save video frames
IMAGE_SIZE       = 224
BATCH_SIZE       = 32      # reduce to 16 if OOM error
EPOCHS           = 6
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
NUM_WORKERS      = 2
VAL_SPLIT        = 0.15
SEED             = 42
FRAMES_PER_VIDEO  = 64      # paper: extract up to max_frames=64 (Tools table)
MAX_VIDEOS_PER_DIR = 500   # cap per folder to avoid FF++ taking forever (500×10 = 5,000 frames)
EXTRACT_WORKERS   = 8      # parallel threads for frame extraction
MAX_SAMPLES_PER_LABEL = 15000  # 15k real + 15k fake per dataset = 90k total max

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO FRAME EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_frames_from_video(video_path: Path, out_dir: Path, n_frames: int = 64) -> List[Path]:
    """Extract n evenly-spaced frames from a video file. Returns list of saved frame paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    saved = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = out_dir / f"f{idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved.append(frame_path)
    cap.release()
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# PAPER-CONFIG FACE PIPELINE (MTCNN + CLAHE + padding equations + Eq. 6/7 norm)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
FACE_PADDING_P = 0.2  # paper p=0.2 (Eq. 1-5)


def apply_clahe_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE on L channel (paper Tools table). img_rgb: uint8 RGB."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    out_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def crop_with_paper_padding(img_rgb: np.ndarray, box_xyxy: np.ndarray, p: float = FACE_PADDING_P) -> np.ndarray:
    """Crop using paper Eq. 1-5 with padding p. box_xyxy = [x1, y1, x2, y2]."""
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    x = int(np.floor(x1))
    y = int(np.floor(y1))
    w = int(np.floor(x2 - x1))
    h = int(np.floor(y2 - y1))
    padding_x = int(np.floor(w * p))
    padding_y = int(np.floor(h * p))
    xx1 = max(0, x - padding_x)
    yy1 = max(0, y - padding_y)
    xx2 = min(W, x + w + padding_x)
    yy2 = min(H, y + h + padding_y)
    return img_rgb[yy1:yy2, xx1:xx2]


def build_mtcnn(device: torch.device) -> MTCNN:
    # Must match paper Tools table and be identical across train/val/test/infer.
    return MTCNN(
        image_size=224,
        margin=int(224 * 0.2),   # padding=0.2 as in paper
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        keep_all=False,
        select_largest=True,
        device=device,
    )


_MTCNN = build_mtcnn(torch.device("cpu"))


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD MAPS FOR FOLDER AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────
REAL_KEYWORDS = {
    "real", "original", "authentic", "genuine", "0",
    "original_sequences",                # FaceForensics++
    "celeb-real", "celeb_real",          # Celeb-DF v2
    "youtube-real", "youtube_real",      # Celeb-DF v2
    "actors",                            # FF++ sub-folder
    # 140k Real and Fake Faces dataset keywords
    "real_vs_fake", "real-vs-fake",
    "train/real", "valid/real", "test/real",
}

FAKE_KEYWORDS = {
    "fake", "manipulated", "deepfake", "deepfakes", "forged",
    "generated", "synthetic", "ai_generated", "1",
    "manipulated_sequences",             # FaceForensics++
    "face2face", "faceswap",             # FF++ sub-methods
    "neuraltextures", "faceshifter",     # FF++ sub-methods
    "deepfakedetection",                 # FF++ sub-method
    "celeb-synthesis", "celeb_synthesis",# Celeb-DF v2
    # 140k Real and Fake Faces dataset keywords
    "fake_vs_real", "fake-vs-real",
    "train/fake", "valid/fake", "test/fake",
}


def collect_from_dir(dirpath: Path, label: int, frames_base: Path) -> List[Tuple[Path, int]]:
    """
    Collect images from a directory. If only videos found, extract frames in parallel.
    Caps at MAX_VIDEOS_PER_DIR videos to prevent runaway extraction on large datasets.
    """
    samples = []

    # Try images first
    imgs = [p for p in dirpath.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    if imgs:
        samples = [(p, label) for p in imgs]
        print(f"      → {len(imgs)} images")
        return samples

    # No images found — try videos and extract frames in parallel
    videos = [p for p in dirpath.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"      → [WARNING] No images or videos found in {dirpath.name}")
        return samples

    # Cap number of videos to avoid runaway extraction
    if len(videos) > MAX_VIDEOS_PER_DIR:
        random.shuffle(videos)
        videos = videos[:MAX_VIDEOS_PER_DIR]
        print(f"      → {len(videos)} videos (capped at {MAX_VIDEOS_PER_DIR}), extracting {FRAMES_PER_VIDEO} frames each in parallel...")
    else:
        print(f"      → {len(videos)} videos found, extracting {FRAMES_PER_VIDEO} frames each in parallel...")

    tag = "real" if label == 0 else "fake"
    subdir_name = dirpath.name.lower().replace(" ", "_")
    frames_out = frames_base / tag / subdir_name

    all_frame_paths: List[Path] = []
    lock_counter = [0]  # mutable counter for progress reporting

    def _extract_one(args):
        i, vid = args
        vid_out = frames_out / f"v{i:05d}"
        return extract_frames_from_video(vid, vid_out, n_frames=FRAMES_PER_VIDEO)

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        futures = {pool.submit(_extract_one, (i, vid)): i for i, vid in enumerate(videos)}
        for fut in as_completed(futures):
            result = fut.result()
            all_frame_paths.extend(result)
            lock_counter[0] += 1
            if lock_counter[0] % 50 == 0 or lock_counter[0] == len(videos):
                print(f"        [{lock_counter[0]}/{len(videos)}] videos processed, {len(all_frame_paths)} frames so far...", flush=True)

    samples = [(f, label) for f in all_frame_paths]
    print(f"      → Extracted {len(samples)} frames total")
    return samples


def find_all_samples(root: Path, frames_base: Path) -> List[Tuple[Path, int]]:
    """
    Scan a dataset root, identify real/fake subdirs by keyword,
    and collect image+video samples (extracting frames from videos).
    """
    samples = []
    seen: set = set()

    for dirpath in sorted(root.rglob("*")):
        if not dirpath.is_dir() or dirpath in seen:
            continue
        name = dirpath.name.lower()
        try:
            rel = dirpath.relative_to(root)
            rel_norm = str(rel).lower().replace("\\", "/")
        except Exception:
            rel_norm = ""

        label = None
        if name in REAL_KEYWORDS or rel_norm in REAL_KEYWORDS:
            label = 0
        elif name in FAKE_KEYWORDS or rel_norm in FAKE_KEYWORDS:
            label = 1

        if label is not None:
            tag = "REAL" if label == 0 else "FAKE"
            print(f"    [{tag}]  {rel if 'rel' in locals() else dirpath.name}")
            found = collect_from_dir(dirpath, label, frames_base)
            samples.extend(found)
            # Mark this dir and all parents as seen
            seen.add(dirpath)
            for p in dirpath.parents:
                seen.add(p)

    by_label = defaultdict(list)
    for path, label in samples:
        by_label[label].append((path, label))

    capped = []
    for label, items in by_label.items():
        random.shuffle(items)
        capped.extend(items[:MAX_SAMPLES_PER_LABEL])
    samples = capped

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# SCAN ALL DATASETS in /kaggle/input/
# ─────────────────────────────────────────────────────────────────────────────
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
kaggle_input  = Path("/kaggle/input")
all_samples: List[Tuple[Path, int]] = []

print(f"\n{'='*60}")
print("Scanning /kaggle/input/ for deepfake datasets...")
print(f"{'='*60}")

available = sorted([d for d in kaggle_input.iterdir() if d.is_dir()])
if not available:
    raise RuntimeError("No datasets found in /kaggle/input/. Please add a dataset first.")

for ds_dir in available:
    print(f"\nDataset: {ds_dir.name}")
    found = find_all_samples(ds_dir, FRAMES_DIR / ds_dir.name)
    all_samples.extend(found)
    real_c = sum(1 for _, l in found if l == 0)
    fake_c = sum(1 for _, l in found if l == 1)
    print(f"  ▶ {real_c} real  +  {fake_c} fake  =  {len(found)} total")

if not all_samples:
    raise RuntimeError(
        "No images/videos found. Check that your datasets have "
        "folders named 'real'/'fake'/'manipulated'/'Celeb-real' etc."
    )

total_real = sum(1 for _, l in all_samples if l == 0)
total_fake = sum(1 for _, l in all_samples if l == 1)
print(f"\n{'='*60}")
print(f"GRAND TOTAL: {total_real} real  +  {total_fake} fake  =  {len(all_samples)}")
print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
random.shuffle(all_samples)
split   = int(len(all_samples) * (1 - VAL_SPLIT))
train_s = all_samples[:split]
val_s   = all_samples[split:]
print(f"\nTrain: {len(train_s)}  |  Val: {len(val_s)}")

import shutil, zipfile
EVAL_EXPORT = Path("/kaggle/working/eval_export")
for lbl, tag in [(0,"real"),(1,"fake")]:
    out = EVAL_EXPORT / tag
    out.mkdir(parents=True, exist_ok=True)
    subset = [(p,l) for p,l in val_s if l==lbl][:300]
    for i,(p,_) in enumerate(subset):
        shutil.copy2(p, out / f"{tag}_{i:05d}{p.suffix}")
with zipfile.ZipFile("/kaggle/working/eval_set.zip","w",
                     zipfile.ZIP_DEFLATED) as zf:
    for f in EVAL_EXPORT.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(EVAL_EXPORT))
print("Eval set exported to /kaggle/working/eval_set.zip")


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_TF = T.Compose([
    T.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomRotation(8),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
    T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1)], p=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None, is_train: bool = False):
        self.samples   = samples
        self.transform = transform
        self.is_train  = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # Open via PIL, then operate in numpy RGB for CLAHE + MTCNN detect/crop.
            img_pil = Image.open(path).convert("RGB")
            img_rgb = np.array(img_pil)  # uint8 RGB
            img_rgb = apply_clahe_rgb(img_rgb)

            # MTCNN detect with paper parameters on the same (CLAHE-processed) image.
            boxes, _ = _MTCNN.detect(Image.fromarray(img_rgb))
            if boxes is not None and len(boxes) > 0:
                face_rgb = crop_with_paper_padding(img_rgb, boxes[0], p=FACE_PADDING_P)
                img = Image.fromarray(face_rgb)
            else:
                img = Image.fromarray(img_rgb)
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), 0)

        def random_jpeg_compress(img_pil, quality_range=(50, 95)):
            import io
            if random.random() < 0.3:
                buf = io.BytesIO()
                img_pil.save(buf, format='JPEG', quality=random.randint(*quality_range))
                buf.seek(0)
                return Image.open(buf).convert('RGB')
            return img_pil

        if self.is_train:
            img = random_jpeg_compress(img)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(float(label))


train_loader = DataLoader(DeepfakeDataset(train_s, TRAIN_TF, is_train=True),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True)
val_loader   = DataLoader(DeepfakeDataset(val_s, VAL_TF, is_train=False),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — ResNet-18 (matches local backend/app/models/detector.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding ResNet-18 model...")
backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

features = nn.Sequential(*list(backbone.children())[:-1])
dropout  = nn.Dropout(p=0.5)
head     = nn.Linear(512, 1)


class ResNet18Detector(nn.Module):
    def __init__(self, features, dropout, head):
        super().__init__()
        self.features = features
        self.dropout  = dropout
        self.head     = head

    def forward(self, x):
        h = self.features(x)
        h = torch.flatten(h, 1)
        h = self.dropout(h)
        return self.head(h)


model = ResNet18Detector(features, dropout, head).to(DEVICE)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Params: {total:,} total  |  {trainable:,} trainable")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SETUP
# ─────────────────────────────────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

def evaluate(model, loader, tau_low: float, tau_high: float, return_probs: bool = False):
    model.eval()
    torch.set_grad_enabled(False)
    correct = total_n = 0
    tp = tn = fp = fn = 0
    val_loss = 0.0
    all_probs: List[float] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            logits = model(x)
            val_loss += criterion(logits, y).item()
            prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            yt = y.detach().cpu().numpy().reshape(-1).astype(int)

            all_probs.extend(prob.tolist())
            all_labels.extend(yt.tolist())

            preds = np.where(prob < tau_low, 0, np.where(prob > tau_high, 1, -1))
            mask = preds != -1
            if mask.sum() == 0:
                continue
            correct += int(np.sum(preds[mask] == yt[mask]))
            total_n += int(mask.sum())
            tp += int(np.sum((preds[mask] == 1) & (yt[mask] == 1)))
            tn += int(np.sum((preds[mask] == 0) & (yt[mask] == 0)))
            fp += int(np.sum((preds[mask] == 1) & (yt[mask] == 0)))
            fn += int(np.sum((preds[mask] == 0) & (yt[mask] == 1)))

    model.train()
    torch.set_grad_enabled(True)
    acc  = correct / total_n if total_n else 0
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    if return_probs:
        return acc, val_loss / max(1, len(loader)), prec, rec, f1, np.array(all_probs), np.array(all_labels)
    return acc, val_loss / max(1, len(loader)), prec, rec, f1


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training: {EPOCHS} epochs | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
print(f"{'='*60}\n")

best_val_acc = 0.0
history      = []
best_state_dict = None
tau_low, tau_high = 0.4, 0.6

for epoch in range(EPOCHS):
    model.train()
    total_loss  = 0.0
    num_batches = 0
    prog_every  = max(1, len(train_loader) // 5)

    for bi, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss  += loss.item()
        num_batches += 1
        if (bi + 1) % prog_every == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {bi+1}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / num_batches
    val_acc, val_loss, prec, rec, f1 = evaluate(model, val_loader, tau_low=tau_low, tau_high=tau_high)
    scheduler.step(val_acc)

    status = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        status = "  ← BEST saved ✓"

    history.append(dict(epoch=epoch+1, train_loss=avg_loss, val_loss=val_loss,
                        val_acc=val_acc, prec=prec, rec=rec, f1=f1))

    print(f"Epoch {epoch+1:>2}/{EPOCHS} | "
          f"TrLoss={avg_loss:.4f} | ValLoss={val_loss:.4f} | "
          f"Acc={val_acc*100:.1f}% | P={prec*100:.1f}% R={rec*100:.1f}% F1={f1*100:.1f}%"
          f"{status}")

if best_state_dict is not None:
    model.load_state_dict(best_state_dict, strict=True)

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD SEARCH (paper Section IV-C) on validation set
# ─────────────────────────────────────────────────────────────────────────────
_, _, _, _, _, probs, val_labels = evaluate(model, val_loader, tau_low=tau_low, tau_high=tau_high, return_probs=True)
best_f1, best_low, best_high = 0.0, 0.4, 0.6
for low in np.arange(0.2, 0.5, 0.05):
    for high in np.arange(0.5, 0.8, 0.05):
        preds = np.where(probs < low, 0, np.where(probs > high, 1, -1))
        mask = preds != -1
        if mask.sum() == 0:
            continue
        f1v = f1_score(val_labels[mask], preds[mask])
        if f1v > best_f1:
            best_f1, best_low, best_high = float(f1v), float(low), float(high)

print(f"\nBest thresholds on val: tau_low={best_low:.2f} tau_high={best_high:.2f} (F1={best_f1:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT (save thresholds alongside weights)
# ─────────────────────────────────────────────────────────────────────────────
torch.save({
    'model_state_dict': model.state_dict(),
    'tau_low': best_low,
    'tau_high': best_high,
    'val_accuracy': best_val_acc,
    'seed': 42,
}, OUTPUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE  |  Best Val Accuracy: {best_val_acc*100:.2f}%")
print(f"  Model saved: {OUTPUT_PATH}")
print(f"{'='*60}")
print(f"\n{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | {'Acc%':>7} | {'Prec%':>7} | {'Rec%':>7} | {'F1%':>7}")
print("-" * 65)
for h in history:
    print(f"  {h['epoch']:>2} | {h['train_loss']:>8.4f} | {h['val_loss']:>8.4f} | "
          f"{h['val_acc']*100:>6.1f}% | {h['prec']*100:>6.1f}% | "
          f"{h['rec']*100:>6.1f}% | {h['f1']*100:>6.1f}%")

print("""
╔══════════════════════════════════════════════════════════╗
║  NEXT STEPS after training:                              ║
║  1. Output tab → Download detector.pt                   ║
║  2. Put at: backend/data/detector.pt                    ║
║  3. Run: python scripts/evaluate_model.py               ║
║      --real_dir data/val/real                           ║
║      --fake_dir data/val/fake                           ║
╚══════════════════════════════════════════════════════════╝
""")
