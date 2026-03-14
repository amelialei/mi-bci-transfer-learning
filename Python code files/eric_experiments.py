"""
Eric's Complete Experimental Suite (CORRECTED)
- Layer Freezing Ablation (6 subjects × 2 reps × 4 strategies = 48 experiments)
- Sample Efficiency (6 subjects × 2 reps × 5 fractions × 2 conditions = 120 experiments)
TOTAL: 168 experiments (~11 hours runtime)

Prerequisites:
- phase1_infrastructure.py (from Hridyanshu)
- Dataset files in correct location
- PyTorch, NumPy, Matplotlib installed

Usage:
    python eric_experiments.py
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import from Hridyanshu's infrastructure
from phase1_infrastructure import ExperimentRunner


# ============================================================
# SETUP AND CONFIGURATION
# ============================================================

class EricExperimentRunner:
    def __init__(self, data_path):
        self.data_path = data_path
        self.results_dir = Path("results")
        self.ablation_dir = self.results_dir / "ablation_study"
        self.sample_eff_dir = self.results_dir / "sample_efficiency"

        # Create directories
        for directory in [self.ablation_dir, self.sample_eff_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.runner = ExperimentRunner(data_path)

    def save_results(self, results, filename, directory):
        """Save experiment results to JSON"""
        filepath = directory / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved: {filepath}")


# ============================================================
# EXPERIMENT 1: LAYER FREEZING ABLATION
# Runtime: ~4 hours
# ============================================================

def run_ablation_study(data_path):
    """
    Test 4 different layer freezing strategies:
    1. temporal_only: Freeze temporal layer only
    2. spatial_temporal: Freeze temporal + spatial
    3. all_except_classifier: Freeze all except final layer
    4. none: No freezing (full fine-tuning)

    6 subjects × 2 replicates × 4 strategies = 48 experiments
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: LAYER FREEZING ABLATION STUDY")
    print("=" * 70)

    runner = EricExperimentRunner(data_path)
    freeze_strategies = ['temporal_only', 'spatial_temporal',
                         'all_except_classifier', 'none']
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    replicates = [1, 2]

    all_results = []
    total_experiments = len(subjects) * len(replicates) * len(freeze_strategies)
    current_exp = 0
    start_time = time.time()

    for subject_id in subjects:
        print(f"\n{'=' * 70}")
        print(f"SUBJECT {subject_id}/9")
        print(f"{'=' * 70}")

        for replicate in replicates:
            print(f"\n  Replicate {replicate}/2")
            print(f"  {'-' * 60}")

            for strategy in freeze_strategies:
                current_exp += 1
                exp_start = time.time()

                print(f"    [{current_exp}/{total_experiments}] Strategy: {strategy}")

                # Run experiment
                try:
                    result = runner.runner.run_transfer_experiment(
                        subject_id=subject_id,
                        freeze_strategy=strategy,
                        random_seed=42 + replicate
                    )

                    # Add metadata
                    result.update({
                        'subject_id': subject_id,
                        'replicate': replicate,
                        'freeze_strategy': strategy,
                        'experiment_type': 'ablation',
                        'timestamp': datetime.now().isoformat()
                    })

                    # Rename 'accuracy' to 'target_accuracy' for consistency
                    if 'accuracy' in result:
                        result['target_accuracy'] = result.pop('accuracy')

                    # Remove model object (not JSON serializable)
                    if 'model' in result:
                        del result['model']

                    all_results.append(result)

                    exp_time = time.time() - exp_start
                    print(f"      Target: {result.get('target_accuracy', 0):.2%}")
                    print(f"      Time: {exp_time / 60:.1f} min")

                except Exception as e:
                    print(f"      ❌ ERROR: {e}")
                    continue

    # Save results
    runner.save_results(all_results, 'ablation_results.json', runner.ablation_dir)

    total_time = time.time() - start_time
    print(f"\n✅ Ablation study complete!")
    print(f"   Total time: {total_time / 3600:.2f} hours")
    print(f"   Experiments: {len(all_results)}/{total_experiments}")

    return all_results


# ============================================================
# EXPERIMENT 2: SAMPLE EFFICIENCY (LEARNING CURVES)
# Runtime: ~7 hours
# ============================================================

def run_sample_efficiency_study(data_path):
    """
    Test transfer learning with varying amounts of target data:
    - 20%, 40%, 60%, 80%, 100% of training data
    - Compare baseline vs transfer for each

    6 subjects × 2 replicates × 5 fractions × 2 conditions = 120 experiments
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SAMPLE EFFICIENCY STUDY")
    print("=" * 70)

    runner = EricExperimentRunner(data_path)
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    replicates = [1, 2]
    data_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    all_results = []
    total_experiments = len(subjects) * len(replicates) * len(data_fractions) * 2
    current_exp = 0
    start_time = time.time()

    for subject_id in subjects:
        print(f"\n{'=' * 70}")
        print(f"SUBJECT {subject_id}/9")
        print(f"{'=' * 70}")

        for replicate in replicates:
            print(f"\n  Replicate {replicate}/2")
            print(f"  {'-' * 60}")

            for fraction in data_fractions:
                # Baseline
                current_exp += 1
                exp_start = time.time()
                print(f"    [{current_exp}/{total_experiments}] Baseline @ {fraction * 100:.0f}% data")

                try:
                    baseline_result = runner.runner.run_sample_efficiency_experiment(
                        subject_id=subject_id,
                        data_fraction=fraction,
                        use_transfer=False,
                        random_seed=42 + replicate
                    )

                    baseline_result.update({
                        'subject_id': subject_id,
                        'replicate': replicate,
                        'data_fraction': fraction,
                        'condition': 'baseline',
                        'experiment_type': 'sample_efficiency',
                        'timestamp': datetime.now().isoformat()
                    })

                    # Remove model object if present (not JSON serializable)
                    if 'model' in baseline_result:
                        del baseline_result['model']

                    all_results.append(baseline_result)
                    exp_time = time.time() - exp_start
                    print(f"      Accuracy: {baseline_result['accuracy']:.2%} | Time: {exp_time / 60:.1f} min")

                except Exception as e:
                    print(f"      ❌ ERROR: {e}")

                # Transfer
                current_exp += 1
                exp_start = time.time()
                print(f"    [{current_exp}/{total_experiments}] Transfer @ {fraction * 100:.0f}% data")

                try:
                    transfer_result = runner.runner.run_sample_efficiency_experiment(
                        subject_id=subject_id,
                        data_fraction=fraction,
                        use_transfer=True,
                        random_seed=42 + replicate
                    )

                    transfer_result.update({
                        'subject_id': subject_id,
                        'replicate': replicate,
                        'data_fraction': fraction,
                        'condition': 'transfer',
                        'experiment_type': 'sample_efficiency',
                        'timestamp': datetime.now().isoformat()
                    })

                    # Remove model object if present (not JSON serializable)
                    if 'model' in transfer_result:
                        del transfer_result['model']

                    all_results.append(transfer_result)
                    exp_time = time.time() - exp_start

                    # Calculate benefit
                    benefit = transfer_result['accuracy'] - baseline_result['accuracy']
                    print(
                        f"      Accuracy: {transfer_result['accuracy']:.2%} | Benefit: {benefit:+.2%} | Time: {exp_time / 60:.1f} min")

                except Exception as e:
                    print(f"      ❌ ERROR: {e}")

    # Save results
    runner.save_results(all_results, 'sample_efficiency_results.json', runner.sample_eff_dir)

    total_time = time.time() - start_time
    print(f"\n✅ Sample efficiency study complete!")
    print(f"   Total time: {total_time / 3600:.2f} hours")
    print(f"   Experiments: {len(all_results)}/{total_experiments}")

    return all_results


# ============================================================
# ANALYSIS AND VISUALIZATION
# ============================================================

def analyze_ablation_results(results):
    """Analyze layer freezing ablation results"""
    print("\n" + "=" * 70)
    print("ABLATION ANALYSIS")
    print("=" * 70)

    if not results:
        print("⚠️ No results to analyze!")
        return {}

    # Group by freeze strategy
    by_strategy = defaultdict(list)
    for r in results:
        if 'target_accuracy' in r:
            by_strategy[r['freeze_strategy']].append(r['target_accuracy'])

    # Calculate statistics
    print("\nMean accuracy by freeze strategy:")
    print(f"{'Strategy':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 65)

    stats = {}
    for strategy in ['temporal_only', 'spatial_temporal', 'all_except_classifier', 'none']:
        accs = by_strategy[strategy]
        if len(accs) == 0:
            print(f"{strategy:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)

        print(f"{strategy:<25} {mean_acc:<10.2%} {std_acc:<10.2%} {min_acc:<10.2%} {max_acc:<10.2%}")
        stats[strategy] = {
            'mean': mean_acc,
            'std': std_acc,
            'min': min_acc,
            'max': max_acc,
            'n': len(accs)
        }

    return stats


def analyze_sample_efficiency_results(results):
    """Analyze sample efficiency (learning curves) results"""
    print("\n" + "=" * 70)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("=" * 70)

    if not results:
        print("⚠️ No results to analyze!")
        return {}

    # Group by fraction and condition
    by_fraction = defaultdict(lambda: {'baseline': [], 'transfer': []})
    for r in results:
        if 'accuracy' in r and 'data_fraction' in r and 'condition' in r:
            fraction = r['data_fraction']
            condition = r['condition']
            by_fraction[fraction][condition].append(r['accuracy'])

    # Calculate statistics
    print("\nAccuracy by data amount:")
    print(f"{'Data %':<10} {'Baseline':<15} {'Transfer':<15} {'Benefit':<15}")
    print("-" * 55)

    stats = {}
    for fraction in sorted(by_fraction.keys()):
        baseline_accs = by_fraction[fraction]['baseline']
        transfer_accs = by_fraction[fraction]['transfer']

        if len(baseline_accs) == 0 or len(transfer_accs) == 0:
            print(f"{fraction * 100:<10.0f} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
            continue

        baseline_mean = np.mean(baseline_accs)
        transfer_mean = np.mean(transfer_accs)
        benefit = transfer_mean - baseline_mean

        print(f"{fraction * 100:<10.0f} {baseline_mean:<15.2%} {transfer_mean:<15.2%} {benefit:<+15.2%}")

        stats[fraction] = {
            'baseline_mean': baseline_mean,
            'baseline_std': np.std(baseline_accs),
            'transfer_mean': transfer_mean,
            'transfer_std': np.std(transfer_accs),
            'benefit': benefit
        }

    return stats


def create_ablation_figure(stats, save_path):
    """Create bar chart comparing freeze strategies"""
    if not stats:
        print("⚠️ No stats to plot!")
        return

    strategies = ['temporal_only', 'spatial_temporal', 'all_except_classifier', 'none']
    labels = ['Temporal Only', 'Spatial +\nTemporal', 'All Except\nClassifier', 'No Freezing']
    means = [stats.get(s, {}).get('mean', 0) * 100 for s in strategies]
    stds = [stats.get(s, {}).get('std', 0) * 100 for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(strategies))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                  color=['#3498db', '#2ecc71', '#e67e22', '#e74c3c'])

    ax.set_xlabel('Freeze Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Layer Freezing Strategy Comparison\n(6 subjects × 2 replicates per strategy)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        if mean > 0:  # Only add label if we have data
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + std + 1,
                    f'{mean:.1f}%',
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {save_path}")
    plt.close()


def create_learning_curves_figure(stats, save_path):
    """Create learning curves plot"""
    if not stats:
        print("⚠️ No stats to plot!")
        return

    fractions = sorted(stats.keys())
    baseline_means = [stats[f]['baseline_mean'] * 100 for f in fractions]
    baseline_stds = [stats[f]['baseline_std'] * 100 for f in fractions]
    transfer_means = [stats[f]['transfer_mean'] * 100 for f in fractions]
    transfer_stds = [stats[f]['transfer_std'] * 100 for f in fractions]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = [f * 100 for f in fractions]

    ax.errorbar(x, baseline_means, yerr=baseline_stds,
                marker='o', linewidth=2, capsize=5, label='Baseline (No Transfer)',
                color='#3498db')
    ax.errorbar(x, transfer_means, yerr=transfer_stds,
                marker='s', linewidth=2, capsize=5, label='Transfer Learning',
                color='#e67e22')

    ax.set_xlabel('Training Data Amount (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves: Transfer vs Baseline\n(6 subjects × 2 replicates per data point)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim([40, 80])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {save_path}")
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print("ERIC'S COMPREHENSIVE EXPERIMENTAL SUITE")
    print("=" * 70)
    print("\nThis will run:")
    print("  1. Layer Freezing Ablation: 72 experiments (~6 hours)")
    print("  2. Sample Efficiency: 180 experiments (~10.5 hours)")
    print("\nTotal: 252 experiments, ~16.5 hours estimated runtime")
    print("=" * 70)

    # Get data path
    data_path = input("\nEnter path to BCI Competition IV data directory: ").strip().strip('"')
    if not data_path:
        data_path = r"D:\Documents 1 Jan 2026\BCICIV"
        print(f"Using default: {data_path}")

    # Confirm
    print("\nReady to start experiments?")
    print("Note: This will take ~16.5 hours. You can interrupt and resume later.")
    response = input("Continue? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Cancelled.")
        return

    overall_start = time.time()

    # Run experiments
    try:
        # Experiment 1: Ablation
        ablation_results = run_ablation_study(data_path)
        ablation_stats = analyze_ablation_results(ablation_results)
        create_ablation_figure(ablation_stats,
                               Path("results/ablation_study/ablation_comparison.png"))

        # Experiment 2: Sample Efficiency
        sample_eff_results = run_sample_efficiency_study(data_path)
        sample_eff_stats = analyze_sample_efficiency_results(sample_eff_results)
        create_learning_curves_figure(sample_eff_stats,
                                      Path("results/sample_efficiency/learning_curves.png"))

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Partial results have been saved.")
        return
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return

    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"\nTotal runtime: {total_time / 3600:.2f} hours")
    print(f"\nResults saved to:")
    print(f"  - results/ablation_study/")
    print(f"  - results/sample_efficiency/")
    print(f"\nFigures generated:")
    print(f"  - ablation_comparison.png")
    print(f"  - learning_curves.png")
    print("\n✅ Eric's work is complete!")


if __name__ == "__main__":
    main()