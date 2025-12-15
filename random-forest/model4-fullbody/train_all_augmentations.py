"""
train_all_augmentations.py - Model 4
Runs train_experiments.py with all 4 augmentation strategies sequentially.
Compares the impact of different augmentation approaches.
Creates a combined summary comparing all strategies.
"""
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

THIS_DIR = Path(__file__).resolve().parent

# Define augmentation strategies (reduced to 2 for faster comparison)
STRATEGIES = ["none", "all"]

STRATEGY_DESCRIPTIONS = {
    "none": "No augmentation (only v1 original images)",
    "all": "All augmentations (v1 + v2_flip + v3_bright + v4_dark)"
}

def extract_champion_results(summary_path):
    """Extract champion model results from summary file."""
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract champion section
    champion_section = content.split("CHAMPION MODEL RESULTS")[-1] if "CHAMPION MODEL RESULTS" in content else ""
    
    # Extract metrics using regex
    results = {}
    
    acc_match = re.search(r"Accuracy:\s+([0-9.]+)", champion_section)
    f1_match = re.search(r"F1 Score:\s+([0-9.]+)", champion_section)
    time_match = re.search(r"Training Time:\s+([0-9.]+)s", champion_section)
    n_est_match = re.search(r"n_estimators:\s+(\d+)", champion_section)
    depth_match = re.search(r"max_depth:\s+(\w+)", champion_section)
    feat_match = re.search(r"max_features:\s+(\w+)", champion_section)
    
    results['accuracy'] = float(acc_match.group(1)) if acc_match else 0.0
    results['f1_score'] = float(f1_match.group(1)) if f1_match else 0.0
    results['train_time'] = float(time_match.group(1)) if time_match else 0.0
    results['n_estimators'] = int(n_est_match.group(1)) if n_est_match else 100
    results['max_depth'] = depth_match.group(1) if depth_match else "None"
    results['max_features'] = feat_match.group(1) if feat_match else "sqrt"
    
    return results

def main():
    print("=" * 100)
    print("MODEL 4: TRAINING WITH AUGMENTATION COMPARISON (Full Body)")
    print("=" * 100)
    print("This will run 2 separate training sessions:")
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"  {i}. {strategy.upper():12} - {STRATEGY_DESCRIPTIONS[strategy]}")
    print("=" * 100)
    print("")
    
    train_script = THIS_DIR / "train_experiments.py"
    all_results = {}
    
    for i, strategy in enumerate(STRATEGIES, 1):
        print(f"\n{'#' * 100}")
        print(f"# STRATEGY {i}/2: {strategy.upper()}")
        print(f"# {STRATEGY_DESCRIPTIONS[strategy]}")
        print(f"{'#' * 100}\n")
        
        # Run train_experiments.py with this strategy
        cmd = [sys.executable, str(train_script), "--augmentation-strategy", strategy]
        
        result = subprocess.run(cmd, cwd=str(THIS_DIR))
        
        if result.returncode != 0:
            print(f"\n❌ ERROR: Training failed for strategy '{strategy}'")
            print(f"Exit code: {result.returncode}")
            all_results[strategy] = None
        else:
            print(f"\n✅ COMPLETED: Strategy '{strategy}' finished successfully")
            # Extract results
            summary_path = THIS_DIR / "artifacts" / f"aug_{strategy}" / "experiments_summary_full.txt"
            all_results[strategy] = extract_champion_results(summary_path)
    
    # Create combined summary
    combined_summary_path = THIS_DIR / "augmentation_comparison_summary.txt"
    
    lines = []
    lines.append("=" * 120)
    lines.append(f"AUGMENTATION STRATEGY COMPARISON - MODEL 4 (Full Body) | {datetime.utcnow().isoformat()}Z")
    lines.append("=" * 120)
    lines.append("")
    lines.append("CHAMPION MODEL RESULTS COMPARISON")
    lines.append("-" * 120)
    lines.append(f"{'Strategy':<12} | {'Description':<40} | {'Acc':>8} | {'F1':>8} | {'Time(s)':>8} | {'Trees':>6} | {'Depth':>7} | {'Feat':>7}")
    lines.append("-" * 120)
    
    for strategy in STRATEGIES:
        results = all_results.get(strategy)
        desc = STRATEGY_DESCRIPTIONS[strategy]
        
        if results:
            line = (f"{strategy:<12} | {desc:<40} | {results['accuracy']:.4f}   | {results['f1_score']:.4f}   | "
                   f"{results['train_time']:>8.1f} | {results['n_estimators']:>6} | {results['max_depth']:>7} | {results['max_features']:>7}")
        else:
            line = f"{strategy:<12} | {desc:<40} | FAILED"
        
        lines.append(line)
    
    lines.append("-" * 120)
    lines.append("")
    lines.append("ANALYSIS")
    lines.append("-" * 60)
    
    # Find best strategy
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        best_acc_strategy = max(valid_results.keys(), key=lambda k: valid_results[k]['accuracy'])
        best_f1_strategy = max(valid_results.keys(), key=lambda k: valid_results[k]['f1_score'])
        fastest_strategy = min(valid_results.keys(), key=lambda k: valid_results[k]['train_time'])
        
        lines.append(f"Best Accuracy:     {best_acc_strategy.upper()} ({valid_results[best_acc_strategy]['accuracy']:.4f})")
        lines.append(f"Best F1 Score:     {best_f1_strategy.upper()} ({valid_results[best_f1_strategy]['f1_score']:.4f})")
        lines.append(f"Fastest Training:  {fastest_strategy.upper()} ({valid_results[fastest_strategy]['train_time']:.1f}s)")
        lines.append("")
        
        # Calculate improvements
        if 'none' in valid_results and 'all' in valid_results:
            baseline_acc = valid_results['none']['accuracy']
            all_acc = valid_results['all']['accuracy']
            improvement = (all_acc - baseline_acc) * 100
            lines.append(f"Augmentation Impact: {improvement:+.2f}% accuracy improvement")
            lines.append(f"  Baseline (none): {baseline_acc:.4f}")
            lines.append(f"  All augmentations: {all_acc:.4f}")
    
    lines.append("")
    lines.append("=" * 120)
    lines.append("")
    lines.append("DETAILED RESULTS")
    lines.append("-" * 60)
    lines.append("Individual strategy results saved in:")
    for strategy in STRATEGIES:
        artifacts_path = THIS_DIR / "artifacts" / f"aug_{strategy}"
        lines.append(f"  - {artifacts_path / 'experiments_summary_full.txt'}")
    lines.append("=" * 120)
    
    # Write combined summary
    with open(combined_summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print("\n" + "=" * 100)
    print("ALL AUGMENTATION STRATEGIES COMPLETED!")
    print("=" * 100)
    print(f"\n📊 COMBINED SUMMARY: {combined_summary_path}")
    print("\n" + "\n".join(lines[4:20]))  # Print comparison table
    print("=" * 100)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
