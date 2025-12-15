# Model 3: Face-Cropped Dataset

This model uses the **dataset-faces** which contains cat face bounding boxes from XML annotations. Images are cropped to show only the cat's face, providing a more focused dataset for classification.


## Workflow

### Step 1: Preprocess Face Dataset

Run preprocessing to crop faces from XML annotations:

```bash
python .\random-forest\model3-faces\preprocess.py
```

This will:
- Read XML annotations from `ml/dataset-faces/train/`, `valid/`, `test/`
- Crop cat faces using bounding box coordinates
- Resize to 224x224
- Create 4 augmented versions per face:
  - `_v1`: Original face crop
  - `_v2_flip`: Horizontal flip
  - `_v3_bright`: Brightness +20%
  - `_v4_dark`: Brightness -20%
- Save to `processed-dataset-faces/` in ImageFolder structure
- Original: ~2000 images → Augmented: ~8000 face crops (4x)

### Step 2: Run All Experiments

**Option A: Train with ALL augmentations (default)**
```bash
python .\random-forest\model3-faces\train_experiments.py
```

**Option B: Train with specific augmentation strategy**
```bash
# No augmentation (baseline)
python .\random-forest\model3-faces\train_experiments.py --augmentation-strategy none

# Only horizontal flip
python .\random-forest\model3-faces\train_experiments.py --augmentation-strategy flip

# Only brightness variations
python .\random-forest\model3-faces\train_experiments.py --augmentation-strategy brightness

# All augmentations (default)
python .\random-forest\model3-faces\train_experiments.py --augmentation-strategy all
```

**Option C: Compare baseline vs full augmentation (RECOMMENDED)**
```bash
python .\random-forest\model3-faces\train_all_augmentations.py
```
This runs 2 strategies (none and all) to compare baseline vs full augmentation impact, saves results in `artifacts/aug_none/` and `artifacts/aug_all/`, and creates a **combined comparison summary** at `augmentation_comparison_summary.txt`.

This will:
- Load processed face-cropped images with selected augmentation filter
- Extract ResNet50 features (cached per strategy)
- Train 11 experiments with different hyperparameters
- Select best parameters and train champion model
- Save all models and outcomes in `artifacts/aug_{strategy}/`
- Generate comprehensive summary

### Step 3: Make Predictions

```bash
python .\random-forest\model3-faces\predict.py path/to/image.jpg
```

Optional arguments:
- `--model rf_baseline.joblib` - Use specific model (default: rf_champion.joblib)
- `--device cuda` - Use GPU (default: auto-detect)

## Key Features

| Aspect | Description |
|--------|-------------|
| Dataset Source | dataset-faces with XML annotations |
| Image Content | Cropped cat faces only (face-focused) |
| Augmentation | Horizontal flip + Brightness (±20%) for training |
| Train Size | ~8000 face crops (4x with all augmentations) |
| Structure | ImageFolder (train/valid/test with breed subfolders) |
| Annotation Format | Pascal VOC XML with bounding boxes |

## Expected Performance

Model 3 should potentially show:
- Better accuracy on face features (focused on face only)
- More robust to background variations
- May struggle with profile/partial faces
- Smaller effective image size (cropped region)

Compare results with Model 1 and Model 2 to evaluate the impact of face-cropping vs full-body images.
