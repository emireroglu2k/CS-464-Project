"""Shared utilities for Random Forest experiments."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torchvision import models, transforms


# Constants
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_image_rgb(path: Path | str) -> Image.Image:
    """Load an image and convert to RGB.
    
    Args:
        path: Path to image file
    
    Returns:
        PIL Image in RGB format
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def default_image_transform(image_size: int = 224) -> transforms.Compose:
    """Return default image transform for ResNet50.
    
    Args:
        image_size: Target image size (default: 224)
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def next_versioned_file(dir_path: Path, stem: str, suffix: str) -> Path:
    """Return a path like {stem}_v001{suffix} without overwriting.
    
    Args:
        dir_path: Directory where the file will be created
        stem: File name prefix (e.g., 'rf_model', 'outcome_train')
        suffix: File extension (e.g., '.joblib', '.txt')
    
    Returns:
        Path with incremented version number
    """
    ensure_dir(dir_path)

    pattern = re.compile(rf"^{re.escape(stem)}_v(\\d{{3}}){re.escape(suffix)}$")
    max_ver = 0
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            max_ver = max(max_ver, int(m.group(1)))

    return dir_path / f"{stem}_v{max_ver + 1:03d}{suffix}"


def write_outcome(path: Path, lines: Sequence[str]) -> None:
    """Write versioned outcome file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def build_resnet50_feature_extractor(device: str = "cpu") -> nn.Module:
    """Build pretrained ResNet50 feature extractor (frozen, fc replaced with Identity).
    
    Args:
        device: 'cpu' or 'cuda'
    
    Returns:
        ResNet50 model ready for feature extraction (outputs 2048-D embeddings)
    """
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()  # Remove classification head
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    images: torch.Tensor,
    device: str,
) -> np.ndarray:
    """Extract features from images using the model.
    
    Args:
        model: Feature extractor (e.g., ResNet50)
        images: Batch of images (B, C, H, W)
        device: Device to run on
    
    Returns:
        Numpy array of shape (B, 2048)
    """
    images = images.to(device, non_blocking=True)
    feats = model(images)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    feats = feats.detach().cpu().numpy()
    return feats


def train_eval_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str]
) -> Tuple[str, np.ndarray]:
    """Generate classification report and confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        Tuple of (classification_report_str, confusion_matrix_array)
    """
    report = classification_report(y_true, y_pred, target_names=list(class_names), digits=4)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm


def format_confusion_analysis(cm: np.ndarray, class_names: Sequence[str], top_n: int = 3) -> str:
    """Format confusion matrix into readable table showing which breeds are confused.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        top_n: Number of top confusions to show per class
    
    Returns:
        Formatted string showing confusion patterns
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CONFUSION ANALYSIS (Top confusions per breed)")
    lines.append("=" * 80)
    lines.append("")
    
    for true_idx, true_label in enumerate(class_names):
        # Get predictions for this true class
        row = cm[true_idx]
        total_samples = row.sum()
        
        if total_samples == 0:
            continue
        
        # Get top confused classes (excluding correct predictions)
        confused_with = []
        for pred_idx, count in enumerate(row):
            if pred_idx != true_idx and count > 0:
                confused_with.append((class_names[pred_idx], count, count / total_samples * 100))
        
        # Sort by count descending
        confused_with.sort(key=lambda x: x[1], reverse=True)
        
        # Format output
        correct_count = row[true_idx]
        accuracy = correct_count / total_samples * 100
        
        lines.append(f"{true_label}:")
        lines.append(f"  Correct: {correct_count}/{total_samples} ({accuracy:.2f}%)")
        
        if confused_with:
            lines.append(f"  Most confused with:")
            for pred_label, count, pct in confused_with[:top_n]:
                lines.append(f"    → {pred_label}: {count} times ({pct:.2f}%)")
        else:
            lines.append(f"  No confusions (100% accurate)")
        
        lines.append("")
    
    return "\n".join(lines)


def save_joblib(path: Path, obj: object) -> None:
    """Save object using joblib."""
    ensure_dir(path.parent)
    joblib.dump(obj, str(path))


def load_joblib(path: Path) -> object:
    """Load object using joblib."""
    return joblib.load(str(path))


def save_class_names(path: Path, class_names: Sequence[str]) -> None:
    """Save class names to JSON."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(class_names), f, indent=2)


def load_class_names(path: Path) -> List[str]:
    """Load class names from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return list(json.load(f))


# For preprocess.py compatibility
def dataset_imagefolder_splits(root: Path):
    """Helper for preprocess.py - returns train/validation/test paths."""
    from dataclasses import dataclass
    
    @dataclass(frozen=True)
    class SplitPaths:
        train: Path
        valid: Path
        test: Path
    
    return SplitPaths(
        train=root / "train",
        valid=root / "validation",
        test=root / "test"
    )
