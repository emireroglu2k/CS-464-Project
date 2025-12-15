# Model 4: Full Body Images (No Face Cropping)

This model uses the **dataset-faces** but processes **full body images** without cropping to cat faces. Images are resized to 224x224 while preserving the entire image content. This allows direct comparison with Model 3 (face-cropped) to evaluate the impact of face-focused vs full-body classification.

## Workflow

### Step 1: Preprocess Full Body Dataset

Run preprocessing to resize images without face cropping:

```bash
python .\random-forest\model4-fullbody\preprocess.py
```

This will:
- Read images from `ml/dataset-faces/train/`, `valid/`, `test/`
- Resize full images to 224x224 (NO face cropping)
- Create 4 augmented versions per image:
  - `_v1`: Original full body
  - `_v2_flip`: Horizontal flip
  - `_v3_bright`: Brightness +20%
  - `_v4_dark`: Brightness -20%
- Preserve original train/valid/test splits
- Save to `processed-fullbody-dataset/` in ImageFolder structure
- Original: ~2000 images → Augmented: ~8000 images (4x)

### Step 2: Run All Experiments

**Option A: Train with ALL augmentations (default)**
```bash
python .\random-forest\model4-fullbody\train_experiments.py
```

**Option B: Train with specific augmentation strategy**
```bash
# No augmentation (baseline)
python .\random-forest\model4-fullbody\train_experiments.py --augmentation-strategy none

# Only horizontal flip
python .\random-forest\model4-fullbody\train_experiments.py --augmentation-strategy flip

# Only brightness variations
python .\random-forest\model4-fullbody\train_experiments.py --augmentation-strategy brightness

# All augmentations (default)
python .\random-forest\model4-fullbody\train_experiments.py --augmentation-strategy all
```

**Option C: Compare baseline vs full augmentation (RECOMMENDED)**
```bash
python .\random-forest\model4-fullbody\train_all_augmentations.py
```
This runs 2 strategies (none and all) to compare baseline vs full augmentation impact, saves results in `artifacts/aug_none/` and `artifacts/aug_all/`, and creates a **combined comparison summary** at `augmentation_comparison_summary.txt`.

This will:
- Load processed full body images with selected augmentation filter
- Extract ResNet50 features (cached per strategy)
- Train 11 experiments with different hyperparameters
- Select best parameters and train champion model
- Save all models and outcomes in `artifacts/aug_{strategy}/`
- Generate comprehensive summary

### Step 3: Make Predictions

```bash
python .\random-forest\model4-fullbody\predict.py path/to/image.jpg
```

Optional arguments:
- `--model rf_baseline.joblib` - Use specific model (default: rf_champion.joblib)
- `--device cuda` - Use GPU (default: auto-detect)

## Key Features

| Aspect | Description |
|--------|-------------|
| Dataset Source | dataset-faces (same as Model 3) |
| Image Content | Full body images (entire image, no cropping) |
| Image Processing | Resize to 224x224 (maintains full context) |
| Augmentation | Horizontal flip + Brightness (±20%) for training |
| Train Size | ~8000 images (4x with all augmentations) |
| Structure | ImageFolder (preprocessed_train/valid/test with breed subfolders) |
| Split Preservation | Original train/valid/test splits maintained |

## Comparison with Model 3

**Model 3 (Face-Cropped)**:
- Focuses on facial features only
- Smaller effective image region
- May lose body/context information

**Model 4 (Full Body)**:
- Includes entire cat (face + body + background)
- More contextual information
- May be affected by background noise

Compare `artifacts/experiments_summary_full.txt` from both models to evaluate which approach works better for cat breed classification.

## Expected Behavior

Model 4 should potentially show:
- Better use of body features (coat patterns, size)
- More robust to face orientation/occlusion
- May be affected by background variations
- Direct comparison possible with Model 3 (same source data, different preprocessing)
