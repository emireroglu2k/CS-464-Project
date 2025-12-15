# Model 2: Custom Preprocessed Train with Augmentation

This model uses a **custom preprocessing pipeline** for the training set with data augmentation (horizontal flip), while using the original validation and test sets.


## Workflow

### Step 1: Preprocess Training Data

Run preprocessing to create augmented training set:

```bash
python .\random-forest\model2-custom-preprocess\preprocess.py
```

This will:
- Load images from `ml/dataset/train/`
- Apply custom preprocessing (resize + normalize)
- Create 4 augmented versions per image:
  - `_v1`: Original image
  - `_v2_flip`: Horizontal flip
  - `_v3_bright`: Brightness +20%
  - `_v4_dark`: Brightness -20%
- Save to `preprocessed_train/` folder
- Original: ~2000 images → Augmented: ~8000 images (4x)

### Step 2: Run All Experiments

**Option A: Train with ALL augmentations (default)**
```bash
python .\random-forest\model2-custom-preprocess\train_experiments.py
```

**Option B: Train with specific augmentation strategy**
```bash
# No augmentation (baseline)
python .\random-forest\model2-custom-preprocess\train_experiments.py --augmentation-strategy none

# Only horizontal flip
python .\random-forest\model2-custom-preprocess\train_experiments.py --augmentation-strategy flip

# Only brightness variations
python .\random-forest\model2-custom-preprocess\train_experiments.py --augmentation-strategy brightness

# All augmentations (default)
python .\random-forest\model2-custom-preprocess\train_experiments.py --augmentation-strategy all
```

**Option C: Compare baseline vs full augmentation (RECOMMENDED)**
```bash
python .\random-forest\model2-custom-preprocess\train_all_augmentations.py
```
This runs 2 strategies (none and all) to compare baseline vs full augmentation impact, saves results in `artifacts/aug_none/` and `artifacts/aug_all/`, and creates a **combined comparison summary** at `augmentation_comparison_summary.txt`.

This will:
- Load preprocessed train images with selected augmentation filter
- Extract ResNet50 features (cached per strategy)
- Train 11 experiments with different hyperparameters
- Select best parameters and train champion model
- Save all models and outcomes in `artifacts/aug_{strategy}/`
- Generate comprehensive summary

### Step 3: Make Predictions

```bash
python .\random-forest\model2-custom-preprocess\predict.py path/to/image.jpg
```

Optional arguments:
- `--model rf_baseline.joblib` - Use specific model (default: rf_champion.joblib)
- `--device cuda` - Use GPU (default: auto-detect)

## Key Differences from Model 1

| Aspect | Model 1 | Model 2 |
|--------|---------|---------|
| Training Data | Preprocessed images (existing) | Custom preprocessed with augmentation |
| Augmentation | None | Horizontal flip + Brightness (±20%) |
| Train Size | ~2000 images | ~8000 images (4x with all augmentations) |
| Validation/Test | From `dataset/` | From `dataset/` (same) |

## Expected Performance

Model 2 should potentially show:
- Better generalization due to augmentation
- Higher training time (more samples)
- Improved robustness to horizontal variations

Compare results with Model 1's `experiments_summary_full.txt` to evaluate the impact of augmentation.
