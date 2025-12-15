"""Data loader for Model 2: Custom preprocessed train, original validation/test."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm


class FilteredImageFolder(datasets.ImageFolder):
    """ImageFolder that filters images based on augmentation strategy."""
    
    def __init__(self, root, transform=None, augmentation_filter: Optional[List[str]] = None):
        super().__init__(root, transform=transform)
        
        if augmentation_filter is not None:
            # Filter samples based on filename patterns
            filtered_samples = []
            for path, class_idx in self.samples:
                filename = Path(path).stem
                # Check if filename matches any of the allowed patterns
                if any(pattern in filename for pattern in augmentation_filter):
                    filtered_samples.append((path, class_idx))
            
            self.samples = filtered_samples
            self.targets = [s[1] for s in self.samples]
            self.imgs = self.samples


class Model2DataLoader:
    """
    Loads data with custom preprocessing:
    - Train: from model2's preprocessed_train folder (augmented)
    - Validation/Test: from original dataset folder (no augmentation)
    """
    
    def __init__(
        self,
        preprocessed_train_root: Path,
        dataset_root: Path,
        image_size: int = 224,
        batch_size: int = 32,
    ):
        self.preprocessed_train_root = preprocessed_train_root
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Transform for ResNet50 feature extraction
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def verify_paths_exist(self) -> None:
        """Check if required directories exist."""
        if not self.preprocessed_train_root.exists():
            raise FileNotFoundError(
                f"Preprocessed train folder not found: {self.preprocessed_train_root}\n"
                f"Run preprocess.py first to generate preprocessed training images."
            )
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
    
    def load_splits(self, augmentation_strategy: str = "all") -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder, List[str]]:
        """
        Load splits with different augmentation strategies:
        - Train: from preprocessed_train (filtered by strategy)
        - Validation/Test: from original dataset folder
        
        Augmentation strategies:
        - "none": Only v1 (original, no augmentation)
        - "flip": v1 + v2 (original + horizontal flip)
        - "brightness": v1 + v3 + v4 (original + bright + dark)
        - "all": v1 + v2 + v3 + v4 (all augmentations)
        """
        self.verify_paths_exist()
        
        train_dir = self.preprocessed_train_root
        valid_dir = self.dataset_root / "validation"
        test_dir = self.dataset_root / "test"
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Preprocessed train folder not found: {train_dir}")
        if not valid_dir.exists():
            raise FileNotFoundError(f"Validation folder not found: {valid_dir}")
        if not test_dir.exists():
            raise FileNotFoundError(f"Test folder not found: {test_dir}")
        
        # Define augmentation filters
        aug_filters = {
            "none": ["_v1"],
            "flip": ["_v1", "_v2_flip"],
            "brightness": ["_v1", "_v3_bright", "_v4_dark"],
            "all": ["_v1", "_v2_flip", "_v3_bright", "_v4_dark"]
        }
        
        if augmentation_strategy not in aug_filters:
            raise ValueError(f"Unknown augmentation strategy: {augmentation_strategy}. Use: {list(aug_filters.keys())}")
        
        filter_patterns = aug_filters[augmentation_strategy]
        train_ds = FilteredImageFolder(str(train_dir), transform=self.transform, augmentation_filter=filter_patterns)
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
        """Extract ResNet50 features from preprocessed ImageFolder dataset."""
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
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
