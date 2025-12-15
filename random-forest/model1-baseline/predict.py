from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent))  # allow `import rf_utils` from ../rf_utils.py

import rf_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict a single image using the latest (or provided) RandomForest trained on ResNet50 embeddings."
    )
    p.add_argument("--image", type=str, required=True, help="Path to an image file.")
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default=str(THIS_DIR / "artifacts"),
        help="Where to read models and write outcomes.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a .joblib model. If omitted, uses latest rf_model_v###.joblib in artifacts/models.",
    )
    p.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Path to class_names_v###.json. If omitted, inferred from model version.",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if rf_utils.torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return p.parse_args()


def _latest_versioned_model(models_dir: Path) -> Path:
    candidates = sorted(models_dir.glob("rf_model_v*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No models found in {models_dir} (expected rf_model_v###.joblib)")

    def ver(p: Path) -> int:
        s = p.stem
        try:
            return int(s.split("_v")[-1])
        except Exception:
            return -1

    candidates.sort(key=ver)
    return candidates[-1]


def main() -> int:
    args = parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    models_dir = artifacts_dir / "models"
    outcomes_dir = artifacts_dir / "outcomes"

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.model:
        model_path = Path(args.model)
    else:
        model_path = _latest_versioned_model(models_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.class_names:
        class_names_path = Path(args.class_names)
    else:
        ver = model_path.stem.split("_v")[-1]
        class_names_path = models_dir / f"class_names_v{ver}.json"

    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    rf = rf_utils.load_joblib(model_path)
    class_names = rf_utils.load_class_names(class_names_path)

    extractor = rf_utils.build_resnet50_feature_extractor(device=args.device)
    transform = rf_utils.default_image_transform(image_size=args.image_size)

    img = transform(rf_utils.load_image_rgb(image_path)).unsqueeze(0)
    feats = rf_utils.extract_embeddings(extractor, img, device=args.device)

    # sklearn returns int class index
    pred_idx = int(rf.predict(feats)[0])
    proba = getattr(rf, "predict_proba", None)

    top3 = []
    if callable(proba):
        probs = proba(feats)[0]
        top = np.argsort(-probs)[:3]
        top3 = [(int(i), float(probs[i])) for i in top]

    pred_label = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)

    # Print results to terminal only
    print(f"\nPrediction: {pred_label}")
    
    if top3:
        print("\nTop 3 predictions:")
        for i, p in top3:
            label = class_names[i] if 0 <= i < len(class_names) else str(i)
            print(f"  {label}: {p:.4f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
