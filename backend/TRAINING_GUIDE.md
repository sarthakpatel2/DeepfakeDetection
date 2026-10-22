# Training Guide for DeepFake Detection Model

This guide explains how to train the deepfake detection model for this project.

## Overview

The model uses a **pretrained ResNet18** backbone for better generalization across diverse lighting, backgrounds, and image types. It classifies images as REAL (0) or FAKE (1) using binary cross-entropy loss. Input is 224×224 with ImageNet normalization.

## Prerequisites

1. **Dataset**: You need a dataset with real and fake images
2. **GPU** (optional but recommended): Training is much faster on GPU
3. **PyTorch**: Already included in requirements.txt

## Step 1: Prepare Your Dataset

Organize your dataset in the following structure:

```
data/
├── train/
│   ├── real/          # Real images (jpg, png, jpeg)
│   │   ├── img1.jpg
│   │   ├── img2.png
│   │   └── ...
│   └── fake/          # Fake/deepfake images
│       ├── fake1.jpg
│       ├── fake2.png
│       └── ...
└── val/               # Optional: validation set
    ├── real/
    └── fake/
```

**Dataset Requirements:**
- Images can be in subdirectories (recursive search)
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Images will be resized to 224×224 (ImageNet standard) during training
- Balanced dataset recommended (similar number of real/fake images)

**Popular Datasets:**
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **DFDC (Deepfake Detection Challenge)**: https://www.kaggle.com/c/deepfake-detection-challenge
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics

## Step 2: Run Training

### Basic Training

From the `backend` directory:

```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate     # Linux/Mac

# Run training with default settings (uses same preprocessing as inference)
python train/train.py
```

**Important**: Training now uses the same face detection and preprocessing as inference (from `USE_FACE_DETECTION` in config). This prevents train/inference mismatch and improves performance on new images. Data augmentation (flip, brightness, gamma) is enabled by default.

**Validation**: If `data/val/real` and `data/val/fake` exist, training will evaluate on them each epoch and save the best model (by validation accuracy) to `data/detector.pt`.

### Custom Training Parameters

Edit `train/train.py` or create a custom training script:

```python
from train.train import train

train(
    real_dir="./data/train/real",
    fake_dir="./data/train/fake",
    epochs=10,              # Number of training epochs
    lr=1e-3,                # Learning rate
    batch_size=16,          # Batch size 
    save_path="./data/detector.pt"  # Where to save the model
)
```

### Training with GPU

The script automatically uses GPU if available. To check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Tips:**
- Increase `batch_size` if you have more GPU memory
- Use `batch_size=32` or `64` for larger GPUs
- Monitor GPU usage: `nvidia-smi` (if using NVIDIA GPU)

## Step 3: Evaluate the Model

After training, evaluate on validation/test set:

```bash
python train/eval.py
```

Or customize:

```python
from train.eval import evaluate

accuracy = evaluate(
    real_dir="./data/val/real",
    fake_dir="./data/val/fake",
    batch_size=16,
    ckpt_path="./data/detector.pt"
)
```

## Step 4: Use the Trained Model

The trained model is saved to `./data/detector.pt` by default. The inference code automatically loads it:

1. **Check if model exists**: The `load_detector()` function in `app/models/detector.py` loads the model
2. **Update model path** (if needed): Modify `app/models/detector.py` to load from a different path
3. **Test inference**: Upload images/videos through the API to test

### Loading Custom Model

To load a trained model in your code:

```python
from app.models.detector import SimpleConvDiscriminator
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleConvDiscriminator().to(device)
model.load_state_dict(torch.load("./data/detector.pt", map_location=device))
model.eval()
```

## Training Tips

### 1. Data Augmentation
Consider adding data augmentation to improve generalization:
- Random flips
- Random crops
- Color jitter
- Rotation

### 2. Learning Rate Scheduling
- Start with `lr=1e-3`
- Reduce by 0.1 every 5-10 epochs if loss plateaus
- Use learning rate finder to find optimal LR

### 3. Early Stopping
- Monitor validation loss
- Stop if validation loss doesn't improve for N epochs
- Save best model based on validation accuracy

### 4. Model Architecture
The current model is a simple scaffold. For better performance, consider:
- Deeper networks (ResNet, EfficientNet)
- Transfer learning from pre-trained models
- Attention mechanisms
- Temporal models for video

### 5. Training Monitoring
Add logging/tensorboard to track:
- Training loss per epoch
- Validation accuracy
- Learning rate
- GPU utilization

## Example Training Workflow

### Quick Setup

```bash
cd backend

# 1. Create directory structure
python train/quick_start.py

# 2. Add your images
# Copy real images to: data/train/real/
# Copy fake images to: data/train/fake/
# (Optional) Copy validation images to: data/val/real/ and data/val/fake/

# 3. Train model
python train/train.py --epochs 10 --batch-size 16

# 4. Evaluate
python train/eval.py

# 5. Test via API
# Start API server - it will automatically load the trained model
uvicorn app.main:app --reload
```

### Command Line Examples

**Basic training:**
```bash
python train/train.py
```

**Custom training:**
```bash
python train/train.py \
  --real-dir ./data/train/real \
  --fake-dir ./data/train/fake \
  --epochs 20 \
  --lr 0.001 \
  --batch-size 32 \
  --save-path ./data/my_detector.pt
```

**Evaluation:**
```bash
python train/eval.py \
  --real-dir ./data/val/real \
  --fake-dir ./data/val/fake \
  --ckpt-path ./data/detector.pt
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` (try 8, 4, or even 2)
- Use gradient accumulation
- Process smaller images

### Poor Accuracy
- Check dataset quality and balance
- Increase training epochs
- Try different learning rates
- Add data augmentation
- Use a more sophisticated model architecture

### Slow Training
- Use GPU if available
- Increase `batch_size` (if memory allows)
- Use `num_workers > 0` in DataLoader (if on Linux/Mac)
- Consider mixed precision training

## Next Steps

1. **Improve Model**: Replace `SimpleConvDiscriminator` with a state-of-the-art architecture
2. **Add Metrics**: Implement precision, recall, F1-score, ROC-AUC
3. **Hyperparameter Tuning**: Use grid search or optuna
4. **Ensemble Models**: Train multiple models and combine predictions
5. **Video-Level Training**: Train on video sequences instead of individual frames

