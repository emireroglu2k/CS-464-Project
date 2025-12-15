"""Data loader for Model 1: Baseline using ml/dataset as-is."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm


class Model1DataLoader:
    """Loads data from ml/dataset (ImageFolder structure: train/validation/test/<class>/)."""
    
    def __init__(
        self,
        dataset_root: Path,
        image_size: int = 224,
        batch_size: int = 32,
    ):
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def load_splits(self) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder, List[str]]:
        """Load train/validation/test splits using ImageFolder.
        
        Uses preprocessed_train for training (existing preprocessed images from ml/preprocess.py).
        """
        train_dir = self.dataset_root / "preprocessed_train"
        valid_dir = self.dataset_root / "validation"
        test_dir = self.dataset_root / "test"
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Train folder not found: {train_dir}")
        if not valid_dir.exists():
            raise FileNotFoundError(f"Validation folder not found: {valid_dir}")
        if not test_dir.exists():
            raise FileNotFoundError(f"Test folder not found: {test_dir}")
        
        train_ds = datasets.ImageFolder(str(train_dir), transform=self.transform)
        valid_ds = datasets.ImageFolder(str(valid_dir), transform=self.transform)
        test_ds = datasets.ImageFolder(str(test_dir), transform=self.transform)
        
        class_names = list(train_ds.classes)
        
        return train_ds, valid_ds, test_ds, class_names
    
    def extract_features(
        self,
        dataset: datasets.ImageFolder,
        extractor: nn.Module,
        device: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ResNet50 features from ImageFolder dataset."""
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device != "cpu"),
        )
        
        x_batches: List[np.ndarray] = []
        y_batches: List[np.ndarray] = []
        
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=f"Extracting features from {len(dataset)} images"):
                images = images.to(device, non_blocking=True)
                feats = extractor(images)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
                feats = feats.detach().cpu().numpy()
                
                x_batches.append(feats)
                y_batches.append(targets.numpy())
        
        x = np.concatenate(x_batches, axis=0) if x_batches else np.zeros((0, 2048), dtype=np.float32)
        y = np.concatenate(y_batches, axis=0) if y_batches else np.zeros((0,), dtype=np.int64)
        
        return x, y
