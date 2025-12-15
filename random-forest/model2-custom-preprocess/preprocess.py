"""
preprocess.py
Preprocesses the train dataset with custom augmentation and saves to model2's preprocessed_train folder.
Validation and test are loaded directly from original dataset folder.
"""
import os
import sys
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------
# Source: Original dataset train folder
base_dir = THIS_DIR.parents[1] / "dataset"
# Output: Model2's own preprocessed_train folder
out_dir = THIS_DIR / "preprocessed_train"

# ------------------------------------------------------
# 2. PROCESSING FUNCTION
# ------------------------------------------------------

def preprocess_and_save():
    """
    Preprocess ONLY the train split from original dataset.
    Applies custom augmentation (horizontal flip) to increase training data.
    Validation and test are loaded directly from original dataset folder.
    """
    input_train_dir = base_dir / "train"
    
    if not input_train_dir.exists():
        print(f"ERROR: Train folder not found: {input_train_dir}")
        sys.exit(1)
    
    print(f"Input:  {input_train_dir}")
    print(f"Output: {out_dir}")
    print("=" * 80)
    
    # Standard ResNet preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    resize = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    print("Loading train dataset...")
    dataset = datasets.ImageFolder(str(input_train_dir))
    print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")
    print("")
    
    processed_count = 0
    augmented_count = 0
    
    for idx, (img_path, label) in enumerate(tqdm(dataset.samples, desc="Preprocessing")):
        class_name = dataset.classes[label]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Create output folder for this class
        save_folder = out_dir / class_name
        save_folder.mkdir(parents=True, exist_ok=True)
        
        # Load and convert to RGB
        pil_img = Image.open(img_path).convert("RGB")
        
        # --- VERSION 1: ORIGINAL (resized and normalized) ---
        img_v1 = resize(pil_img)
        tensor_v1 = normalize(to_tensor(img_v1))
        
        # Denormalize for saving as PNG
        save_v1 = tensor_v1 * std + mean
        save_image(save_v1, save_folder / f"{base_name}_v1.png")
        processed_count += 1
        
        # --- VERSION 2: HORIZONTAL FLIP (augmentation) ---
        img_v2 = ImageOps.mirror(pil_img)
        img_v2 = resize(img_v2)
        tensor_v2 = normalize(to_tensor(img_v2))
        
        save_v2 = tensor_v2 * std + mean
        save_image(save_v2, save_folder / f"{base_name}_v2_flip.png")
        augmented_count += 1
        
        # --- VERSION 3: BRIGHT (1.2x) ---
        enhancer = ImageEnhance.Brightness(pil_img)
        img_v3 = enhancer.enhance(1.2)  # 20% brighter
        img_v3 = resize(img_v3)
        tensor_v3 = normalize(to_tensor(img_v3))
        
        save_v3 = tensor_v3 * std + mean
        save_image(save_v3, save_folder / f"{base_name}_v3_bright.png")
        augmented_count += 1
        
        # --- VERSION 4: DARK (0.8x) ---
        img_v4 = enhancer.enhance(0.8)  # 20% darker
        img_v4 = resize(img_v4)
        tensor_v4 = normalize(to_tensor(img_v4))
        
        save_v4 = tensor_v4 * std + mean
        save_image(save_v4, save_folder / f"{base_name}_v4_dark.png")
        augmented_count += 1
    
    print("")
    print("=" * 80)
    print(f"DONE! Preprocessed {processed_count} images + {augmented_count} augmented = {processed_count + augmented_count} total")
    print(f"Saved to: {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    preprocess_and_save()