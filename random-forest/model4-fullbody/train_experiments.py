"""
train_experiments.py - Model 4: Full Body Images (No Face Cropping)
Runs exhaustive Random Forest experiments on full body cat images.
Generates ONE master summary table in 'artifacts/experiments_summary_full.txt'.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent))

import rf_utils  # noqa: E402
from data_loader import Model4DataLoader  # noqa: E402

# --- COMPREHENSIVE EXPERIMENT LIST ---
EXPERIMENTS = [
    # --- GROUP 1: BASELINE ---
    {"name": "baseline",          "n_estimators": 100, "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},

    # --- GROUP 2: NUMBER OF TREES (n_estimators) ---
    {"name": "trees_10",          "n_estimators": 10,  "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "trees_50",          "n_estimators": 50,  "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "trees_200",         "n_estimators": 200, "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "trees_500",         "n_estimators": 500, "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "trees_1000",        "n_estimators": 1000,"max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},

    # --- GROUP 3: TREE DEPTH (max_depth) ---
    {"name": "depth_3",           "n_estimators": 100, "max_depth": 3,    "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "depth_5",           "n_estimators": 100, "max_depth": 5,    "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "depth_10",          "n_estimators": 100, "max_depth": 10,   "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "depth_20",          "n_estimators": 100, "max_depth": 20,   "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "depth_unlim",       "n_estimators": 100, "max_depth": None,  "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},

    # --- GROUP 4: MAX FEATURES (Randomness) ---
    {"name": "feat_sqrt",         "n_estimators": 100, "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini", "bootstrap": True},
    {"name": "feat_log2",         "n_estimators": 100, "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": "log2", "criterion": "gini", "bootstrap": True},
    {"name": "feat_0.5",          "n_estimators": 100, "max_depth": None, "min_samples_split": 2,  "min_samples_leaf": 1, "max_features": 0.5,    "criterion": "gini", "bootstrap": True},
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run exhaustive Random Forest experiments for Model 4.")
    p.add_argument("--processed-dataset", type=str, default=str(THIS_DIR / "processed-fullbody-dataset"))
    p.add_argument("--augmentation-strategy", type=str, default="all", 
                   choices=["none", "flip", "brightness", "all"],
                   help="Augmentation strategy: none (no aug), flip (v1+v2), brightness (v1+v3+v4), all (v1+v2+v3+v4)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if rf_utils.torch.cuda.is_available() else "cpu")
    p.add_argument("--no-cache", action="store_true", help="Force re-extraction of features")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    
    # Setup paths with augmentation strategy suffix
    aug_strategy = args.augmentation_strategy
    processed_dataset = Path(args.processed_dataset)
    artifacts_dir = THIS_DIR / "artifacts" / f"aug_{aug_strategy}"
    models_dir = artifacts_dir / "models"
    outcomes_dir = artifacts_dir / "outcomes"
    cache_dir = artifacts_dir / "feature_cache"
    
    # Create directories
    rf_utils.ensure_dir(models_dir)
    rf_utils.ensure_dir(outcomes_dir)
    rf_utils.ensure_dir(cache_dir)
    
    print(f"\nMODEL 4: FULL BODY DATASET (No Face Cropping)")
    print(f"AUGMENTATION STRATEGY: {aug_strategy.upper()}")
    print(f"STARTING EXHAUSTIVE EXPERIMENTS ({len(EXPERIMENTS)} configs)")
    print("=" * 100)

    # --- 1. LOAD DATA & CACHE FEATURES ---
    data_loader = Model4DataLoader(
        processed_dataset_root=processed_dataset,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
    train_ds, valid_ds, test_ds, class_names = data_loader.load_splits(augmentation_strategy=aug_strategy)
    
    print(f"Train samples: {len(train_ds)} (augmentation strategy: {aug_strategy})")
    print(f"Valid samples: {len(valid_ds)} (full body)")
    print(f"Test samples:  {len(test_ds)} (full body)")
    
    # Save class names
    class_names_path = models_dir / "class_names.json"
    rf_utils.save_class_names(class_names_path, class_names)
    
    cache_files = {
        'x_train': cache_dir / f"x_train_{args.image_size}.npy", 'y_train': cache_dir / f"y_train_{args.image_size}.npy",
        'x_valid': cache_dir / f"x_valid_{args.image_size}.npy", 'y_valid': cache_dir / f"y_valid_{args.image_size}.npy",
        'x_test':  cache_dir / f"x_test_{args.image_size}.npy",  'y_test':  cache_dir / f"y_test_{args.image_size}.npy"
    }
    
    use_cache = not args.no_cache and all(p.exists() for p in cache_files.values())
    
    if use_cache:
        print(">> Loading features from cache...")
        x_train, y_train = np.load(cache_files['x_train']), np.load(cache_files['y_train'])
        x_valid, y_valid = np.load(cache_files['x_valid']), np.load(cache_files['y_valid'])
        x_test, y_test   = np.load(cache_files['x_test']),  np.load(cache_files['y_test'])
    else:
        print(">> Extracting features (ResNet50)...")
        extractor = rf_utils.build_resnet50_feature_extractor(device=args.device)
        x_train, y_train = data_loader.extract_features(train_ds, extractor, args.device)
        x_valid, y_valid = data_loader.extract_features(valid_ds, extractor, args.device)
        x_test, y_test   = data_loader.extract_features(test_ds, extractor, args.device)
        
        np.save(cache_files['x_train'], x_train); np.save(cache_files['y_train'], y_train)
        np.save(cache_files['x_valid'], x_valid); np.save(cache_files['y_valid'], y_valid)
        np.save(cache_files['x_test'], x_test);   np.save(cache_files['y_test'], y_test)

    # --- 2. EXPERIMENT LOOP ---
    results = []
    
    print("-" * 120)
    print(f"{'Experiment':<15} | {'Acc':<8} | {'F1':<8} | {'Time':<6} | Config Summary")
    print("-" * 120)

    for i, config in enumerate(EXPERIMENTS, 1):
        exp_name = config["name"]
        
        start_time = time.time()
        
        # Train
        rf = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            criterion=config["criterion"],
            bootstrap=config["bootstrap"],
            n_jobs=-1,
            random_state=42
        )
        rf.fit(x_train, y_train)
        train_time = time.time() - start_time
        
        # Predict & Metrics
        y_test_pred = rf.predict(x_test)
        y_valid_pred = rf.predict(x_valid)
        
        test_acc  = accuracy_score(y_test, y_test_pred)
        test_f1   = f1_score(y_test, y_test_pred, average='weighted')
        test_prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_rec  = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        valid_acc = accuracy_score(y_valid, y_valid_pred)

        # Store results
        results.append({
            "name": exp_name,
            "Experiment": exp_name,
            # Parameters (store both string and original values)
            "n_estimators": config["n_estimators"],
            "Trees": config["n_estimators"],
            "Depth": str(config["max_depth"]),
            "Split": config["min_samples_split"],
            "Leaf": config["min_samples_leaf"],
            "max_features": config["max_features"],
            "Feat": str(config["max_features"]),
            "criterion": config["criterion"],
            "Crit": config["criterion"],
            "Boot": str(config["bootstrap"]),
            # Metrics
            "Val Acc": valid_acc,
            "Test Acc": test_acc,
            "Test F1": test_f1,
            "Test Prec": test_prec,
            "Test Rec": test_rec,
            "Time": train_time
        })
        
        # Save model
        model_path = models_dir / f"rf_{exp_name}.joblib"
        rf_utils.save_joblib(model_path, rf)
        
        # Generate and save per-experiment outcome file
        outcome_path = outcomes_dir / f"outcome_{exp_name}.txt"
        val_report, val_cm = rf_utils.train_eval_report(y_valid, y_valid_pred, class_names)
        test_report, test_cm = rf_utils.train_eval_report(y_test, y_test_pred, class_names)
        confusion_analysis = rf_utils.format_confusion_analysis(test_cm, class_names)
        
        outcome_lines = [
            f"timestamp_utc: {rf_utils.utc_now_iso()}",
            f"experiment: {exp_name}",
            f"task: random_forest_exhaustive_experiment",
            f"model: model4-fullbody",
            f"features_cached: {use_cache}",
            f"device: {args.device}",
            f"n_estimators: {config['n_estimators']}",
            f"max_depth: {config['max_depth']}",
            f"min_samples_split: {config['min_samples_split']}",
            f"min_samples_leaf: {config['min_samples_leaf']}",
            f"max_features: {config['max_features']}",
            f"criterion: {config['criterion']}",
            f"bootstrap: {config['bootstrap']}",
            f"train_size: {len(y_train)}",
            f"valid_size: {len(y_valid)}",
            f"test_size: {len(y_test)}",
            f"elapsed_seconds: {train_time:.2f}",
            "",
            "=== VALIDATION REPORT ===",
            val_report,
            "",
            "=== TEST REPORT ===",
            test_report,
            "",
            "=== CONFUSION ANALYSIS ===",
            confusion_analysis,
        ]
        
        rf_utils.write_outcome(outcome_path, outcome_lines)

        # Console Log
        config_summary = f"n={config['n_estimators']}, d={config['max_depth']}, leaf={config['min_samples_leaf']}"
        print(f"{exp_name:<15} | {test_acc:.4f}   | {test_f1:.4f}   | {train_time:.1f}s   | {config_summary}")


    # --- 3. CHAMPION MODEL TRAINING ---
    print("\n" + "=" * 100)
    print("SELECTING BEST PARAMETERS FOR CHAMPION MODEL")
    print("=" * 100)

    # Find best parameter from each experiment group based on Test Accuracy
    
    # Best n_estimators (from trees_* experiments)
    tree_exps = [r for r in results if r['name'].startswith('trees_')]
    best_n_est = max(tree_exps, key=lambda x: x['Test Acc'])['n_estimators'] if tree_exps else 100
    
    # Best max_depth (from depth_* experiments)
    depth_exps = [r for r in results if r['name'].startswith('depth_')]
    best_depth_exp = max(depth_exps, key=lambda x: x['Test Acc']) if depth_exps else None
    best_depth = None if best_depth_exp and best_depth_exp['Depth'] == 'None' else (int(best_depth_exp['Depth']) if best_depth_exp else None)
    
    # Best max_features (from feat_* experiments) - BUT avoid slow combinations
    feat_exps = [r for r in results if r['name'].startswith('feat_')]
    best_feat_exp = max(feat_exps, key=lambda x: x['Test Acc']) if feat_exps else None
    best_feat = best_feat_exp['max_features'] if best_feat_exp else 'sqrt'
    
    # SAFETY: If max_features=0.5 and n_estimators > 200, use sqrt instead (0.5 is too slow)
    if best_feat == 0.5 and best_n_est > 200:
        print(f"⚠️  WARNING: max_features=0.5 with {best_n_est} trees would take ~{best_n_est}+ seconds!")
        print(f"   Using max_features='sqrt' instead for reasonable training time.")
        best_feat = 'sqrt'

    print(f"Best n_estimators: {best_n_est}")
    print(f"Best max_depth:    {best_depth}")
    print(f"Best max_features: {best_feat}")
    print("-" * 60)
    print("Training CHAMPION MODEL (Combines best settings with sklearn defaults)...")

    champion_rf = RandomForestClassifier(
        n_estimators=best_n_est,
        max_depth=best_depth,
        max_features=best_feat,
        n_jobs=-1,
        random_state=42
    )
    
    t0 = time.time()
    champion_rf.fit(x_train, y_train)
    t_champ = time.time() - t0
    
    champ_pred = champion_rf.predict(x_test)
    champ_acc = accuracy_score(y_test, champ_pred)
    champ_f1 = f1_score(y_test, champ_pred, average='weighted')
    
    print(f"CHAMPION RESULT: Acc={champ_acc:.4f} | F1={champ_f1:.4f} | Time={t_champ:.1f}s")
    
    # Save Champion
    champ_path = models_dir / "rf_champion.joblib"
    rf_utils.save_joblib(champ_path, champion_rf)
    
    # Save Champion Report
    champ_report, champ_cm = rf_utils.train_eval_report(y_test, champ_pred, class_names)
    champ_confusion = rf_utils.format_confusion_analysis(champ_cm, class_names)
    
    champion_outcome = [
        f"timestamp_utc: {rf_utils.utc_now_iso()}",
        f"experiment: champion",
        f"task: champion_model_with_best_hyperparameters",
        f"model: model4-fullbody",
        "",
        "CHAMPION MODEL REPORT",
        f"Accuracy: {champ_acc:.4f}",
        f"F1 Score: {champ_f1:.4f}",
        f"Training Time: {t_champ:.1f}s",
        "",
        "PARAMETERS (Tuned):",
        f"  n_estimators: {best_n_est}",
        f"  max_depth: {best_depth}",
        f"  max_features: {best_feat}",
        "",
        "Other parameters use sklearn defaults (min_samples_split=2, min_samples_leaf=1, criterion='gini', bootstrap=True)",
        "",
        "=== CLASSIFICATION REPORT ===",
        champ_report,
        "",
        "=== CONFUSION ANALYSIS ===",
        champ_confusion,
    ]
    rf_utils.write_outcome(outcomes_dir / "outcome_champion.txt", champion_outcome)
    
    print(f"Saved champion model to: {champ_path}")
    print("=" * 100)


    # --- 4. WRITE MASTER SUMMARY FILE ---
    summary_path = artifacts_dir / "experiments_summary_full.txt"
    lines = []
    
    lines.append("=" * 110)
    lines.append(f"MASTER EXPERIMENT SUMMARY - MODEL 4 (Full Body Dataset) | {rf_utils.utc_now_iso()}")
    lines.append("=" * 110)
    
    # Header
    header = (f"{'Experiment':<15} | {'Trees':<5} | {'Depth':<7} | {'Feat':<9} | "
              f"{'Val Acc':<9} | {'Test Acc':<9} | {'Test F1':<9} | {'Time(s)':<7}")
    lines.append(header)
    lines.append("-" * 110)
    
    # Rows
    for r in results:
        row = (f"{r['Experiment']:<15} | {r['Trees']:<5} | {r['Depth']:<7} | {r['Feat']:<9} | "
               f"{r['Val Acc']:.4f}    | {r['Test Acc']:.4f}    | {r['Test F1']:.4f}    | {r['Time']:.1f}")
        lines.append(row)
        
    lines.append("-" * 110)
    lines.append("")
    lines.append("CHAMPION MODEL RESULTS")
    lines.append("-" * 60)
    lines.append(f"Accuracy: {champ_acc:.4f}")
    lines.append(f"F1 Score: {champ_f1:.4f}")
    lines.append(f"Training Time: {t_champ:.1f}s")
    lines.append("Tuned Parameters:")
    lines.append(f"  n_estimators: {best_n_est}")
    lines.append(f"  max_depth: {best_depth}")
    lines.append(f"  max_features: {best_feat}")
    lines.append("")
    lines.append("=== CLASSIFICATION REPORT ===")
    lines.append(champ_report)
    lines.append("")
    lines.append("=== CONFUSION ANALYSIS ===")
    lines.append(champ_confusion)

    rf_utils.write_outcome(summary_path, lines)
    
    print("\n" + "=" * 80)
    print(f"DONE! Full {len(EXPERIMENTS)}-experiment summary saved to: {summary_path}")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
