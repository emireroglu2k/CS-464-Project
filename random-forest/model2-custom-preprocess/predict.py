"""
predict.py - Model 2: Custom Preprocessed Train
Predict a single image using trained Random Forest model.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent))

import rf_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict cat breed from image using Model 2")
    p.add_argument("image_path", type=str, help="Path to image file")
    p.add_argument("--model", type=str, default="rf_champion.joblib", help="Model filename in artifacts/models/")
    p.add_argument("--device", type=str, default="cuda" if rf_utils.torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    # Setup paths
    artifacts_dir = THIS_DIR / "artifacts"
    models_dir = artifacts_dir / "models"
    
    model_path = models_dir / args.model
    class_names_path = models_dir / "class_names.json"
    image_path = Path(args.image_path)
    
    # Verify files exist
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1
    if not class_names_path.exists():
        print(f"ERROR: Class names not found: {class_names_path}")
        return 1
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return 1
    
    print("=" * 80)
    print(f"MODEL 2 PREDICTION: {args.model}")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Device: {args.device}")
    print("")
    
    # Load model and class names
    print("Loading model...")
    rf_model = rf_utils.load_joblib(model_path)
    class_names = rf_utils.load_class_names(class_names_path)
    
    # Load and preprocess image
    print("Loading image...")
    img_rgb = rf_utils.load_image_rgb(image_path)
    img_tensor = rf_utils.default_image_transform()(img_rgb).unsqueeze(0)
    
    # Extract features
    print("Extracting features...")
    extractor = rf_utils.build_resnet50_feature_extractor(device=args.device)
    extractor.eval()
    
    img_tensor = img_tensor.to(args.device)
    with rf_utils.torch.no_grad():
        features = extractor(img_tensor)
        if isinstance(features, (tuple, list)):
            features = features[0]
        features_np = features.cpu().numpy()
    
    # Predict
    print("Predicting...")
    prediction = rf_model.predict(features_np)[0]
    probabilities = rf_model.predict_proba(features_np)[0]
    
    predicted_class = class_names[prediction]
    
    # Get top 3 predictions
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    
    print("")
    print("=" * 80)
    print(f"PREDICTION: {predicted_class}")
    print("=" * 80)
    print("")
    print("Top 3 Predictions:")
    for idx in top3_indices:
        print(f"  {class_names[idx]:<25} {probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
