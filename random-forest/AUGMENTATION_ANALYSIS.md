# Augmentation Strategy Analysis

## Overview

Models 2, 3, and 4 now support **4 different augmentation strategies** to analyze the impact of data augmentation on model performance.

## Augmentation Strategies

| Strategy | Description | Training Data Composition |
|----------|-------------|---------------------------|
| **none** | No augmentation | Only v1 (original images) |
| **flip** | Horizontal flip only | v1 + v2_flip |
| **brightness** | Brightness variations only | v1 + v3_bright + v4_dark |
| **all** | All augmentations | v1 + v2_flip + v3_bright + v4_dark |

### Augmentation Details

- **v1**: Original preprocessed images
- **v2_flip**: Horizontal flip (mirror)
- **v3_bright**: Brightness +20% (factor 1.2)
- **v4_dark**: Brightness -20% (factor 0.8)

## How to Use

### Single Strategy Training

Train with a specific augmentation strategy:

```bash
# Model 2
python .\random-forest\model2-custom-preprocess\train_experiments.py --augmentation-strategy none

# Model 3
python .\random-forest\model3-faces\train_experiments.py --augmentation-strategy flip

# Model 4
python .\random-forest\model4-fullbody\train_experiments.py --augmentation-strategy brightness
```

### Compare All Strategies

Run all 4 strategies automatically:

```bash
# Model 2
python .\random-forest\model2-custom-preprocess\train_all_augmentations.py

# Model 3
python .\random-forest\model3-faces\train_all_augmentations.py

# Model 4
python .\random-forest\model4-fullbody\train_all_augmentations.py
```

## Results Structure

Each strategy saves results in a separate folder:

```
artifacts/
├── aug_none/
│   ├── models/
│   ├── outcomes/
│   ├── feature_cache/
│   └── experiments_summary_full.txt
├── aug_flip/
│   └── ...
├── aug_brightness/
│   └── ...
└── aug_all/
    └── ...
```

## Analysis Questions

### 1. Training Data Size Impact
- **none**: Baseline (smallest dataset)
- **flip**: 2x original size
- **brightness**: 3x original size
- **all**: 4x original size

**Question**: Does more data always lead to better performance?

### 2. Augmentation Type Impact
- **flip** vs **brightness**: Which type of augmentation is more effective?
- **all**: Does combining augmentations improve or harm performance?

### 3. Model-Specific Effects
- **Model 2** (custom preprocessing): How does augmentation affect already preprocessed data?
- **Model 3** (face-cropped): Are face features robust to brightness changes?
- **Model 4** (full body): Does background variation from brightness affect full-body classification?

## Expected Outcomes

### Potential Benefits of Augmentation:
- ✅ Better generalization
- ✅ Reduced overfitting
- ✅ More robust to variations
- ✅ Higher accuracy on test set

### Potential Drawbacks:
- ❌ Longer training time
- ❌ More disk space for features
- ❌ May introduce unrealistic variations
- ❌ Diminishing returns with too much augmentation

## Comparison Process

1. **Run all strategies** using `train_all_augmentations.py`
2. **Compare summaries** in each `artifacts/aug_*/experiments_summary_full.txt`
3. **Look for**:
   - Test accuracy differences
   - F1 score improvements
   - Training time trade-offs
   - Champion model performance
4. **Analyze**:
   - Which strategy gives best accuracy?
   - Is the improvement worth the data size increase?
   - Do certain breeds benefit more from specific augmentations?

## Example Comparison Table

After running all strategies, you can create a summary like:

| Strategy | Train Size | Test Acc | Test F1 | Champion Acc | Training Time |
|----------|-----------|----------|---------|--------------|---------------|
| none | 1000 | 0.75 | 0.74 | 0.76 | 5min |
| flip | 2000 | 0.78 | 0.77 | 0.79 | 8min |
| brightness | 3000 | 0.77 | 0.76 | 0.78 | 12min |
| all | 4000 | 0.80 | 0.79 | 0.81 | 15min |

*(Example values - actual results will vary)*

## Recommendations

1. **Start with `none`** to establish baseline performance
2. **Run `all`** to see maximum augmentation impact
3. **Compare with `flip` and `brightness`** to identify which augmentation type is more effective
4. **Choose strategy** based on:
   - Accuracy improvement
   - Training time budget
   - Available disk space
   - Deployment requirements
