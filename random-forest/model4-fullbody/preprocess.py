"""
preprocess.py - Model 4: Full Body Images (No Face Cropping)
Processes dataset-faces WITHOUT cropping faces - uses full images resized to 224x224.
Preserves original train/valid/test splits from dataset-faces.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent

# --- CONFIGURATION ---
# 1. Source: dataset-faces folder
SOURCE_DIR = THIS_DIR.parents[1] / "dataset-faces"

# 2. Output: processed-fullbody-dataset in model4 folder
DEST_DIR = THIS_DIR / "processed-fullbody-dataset"

# 3. Settings
IMG_SIZE = (224, 224)

def process_dataset():
    """
    Process dataset-faces (train/valid/test folders) WITHOUT face cropping.
    Uses full images, resized to 224x224, preserving original splits.
    """
    source_path = Path(SOURCE_DIR)
    dest_path = Path(DEST_DIR)
    
    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_path}")
        sys.exit(1)
    
    print(f"Source: {source_path}")
    print(f"Output: {dest_path}")
    print("=" * 80)
    print("FULL BODY PROCESSING (No Face Cropping)")
    print("=" * 80)
    
    # Map source folders to destination folders (preserve splits)
    split_mapping = {
        "train": "preprocessed_train",  # Read from train, save to preprocessed_train
        "valid": "valid",
        "test": "test"
    }
    total_stats = {"preprocessed_train": 0, "valid": 0, "test": 0}
    
    for source_split, dest_split in split_mapping.items():
        split_dir = source_path / source_split
        if not split_dir.exists():
            print(f"Warning: {source_split} folder not found, skipping...")
            continue
        
        print(f"\nProcessing {source_split} split...")
        
        # Find all XML files in this split
        xml_files = list(split_dir.glob("*.xml"))
        if len(xml_files) == 0:
            print(f"  No XML files found in {source_split}")
            continue
        
        print(f"  Found {len(xml_files)} XML annotations")
        
        # Process each XML file
        processed = 0
        for xml_file in tqdm(xml_files, desc=f"  {source_split}"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Find the image filename
                filename_node = root.find('filename')
                if filename_node is None:
                    continue
                
                img_filename = filename_node.text
                img_path = split_dir / img_filename
                
                # Fallback: try to find image with same stem
                if not img_path.exists():
                    possible_imgs = list(split_dir.glob(f"{xml_file.stem}.*"))
                    # Filter out XML files
                    possible_imgs = [p for p in possible_imgs if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                    if possible_imgs:
                        img_path = possible_imgs[0]
                    else:
                        continue
                
                # Get breed name from first object (we don't use bounding box)
                obj = root.find('object')
                if obj is None:
                    continue
                
                class_name = obj.find('name').text
                
                # Create output directory for this class (using dest_split)
                save_dir = dest_path / dest_split / class_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Load, resize (NO CROP), and save
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    
                    # FULL IMAGE - just resize to 224x224 (no face cropping)
                    img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                    
                    # Save original resized image
                    base_name = f"{img_path.stem}_fullbody"
                    img_resized.save(save_dir / f"{base_name}_v1.jpg")
                    processed += 1
                    total_stats[dest_split] += 1
                    
                    # Augmentation: multiple variations (train -> preprocessed_train only)
                    if source_split == "train":
                        # V2: Flip (Mirror)
                        img_flipped = ImageOps.mirror(img_resized)
                        img_flipped.save(save_dir / f"{base_name}_v2_flip.jpg")
                        processed += 1
                        total_stats[dest_split] += 1
                        
                        # V3: Bright (1.2x)
                        enhancer = ImageEnhance.Brightness(img_resized)
                        img_bright = enhancer.enhance(1.2)  # 20% brighter
                        img_bright.save(save_dir / f"{base_name}_v3_bright.jpg")
                        processed += 1
                        total_stats[dest_split] += 1
                        
                        # V4: Dark (0.8x)
                        img_dark = enhancer.enhance(0.8)  # 20% darker
                        img_dark.save(save_dir / f"{base_name}_v4_dark.jpg")
                        processed += 1
                        total_stats[dest_split] += 1
                        
            except Exception as e:
                print(f"\n  Warning: Error processing {xml_file.name}: {e}")
        
        print(f"  Processed {processed} images for {source_split}")
    
    # Final report
    print("\n" + "=" * 80)
    print("FULL BODY PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total Images Generated:")
    for dest_split in ["preprocessed_train", "valid", "test"]:
        if dest_split in total_stats:
            aug_note = " (includes augmentation)" if dest_split == "preprocessed_train" else ""
            print(f"  {dest_split.capitalize():16}: {total_stats[dest_split]}{aug_note}")
    print("-" * 80)
    print(f"Output: {dest_path}")
    print("=" * 80)

if __name__ == "__main__":
    process_dataset()
