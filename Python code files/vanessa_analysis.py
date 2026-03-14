"""
Vanessa's Complete Analysis Suite (CORRECTED)
- LOSO Cross-Validation (baseline + transfer models)
- Variance Decomposition
- Subject Similarity Clustering
- Correlation Analysis
- Error Pattern Analysis

Prerequisites:
- All 135 saved models from Hridyanshu's work
- results/9x5_study/ directory with subject_XX_intermediate.json files
- phase1_infrastructure.py
- PyTorch, NumPy, scikit-learn, Matplotlib, SciPy installed

Usage:
    python vanessa_analysis.py
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

# Import from Hridyanshu's infrastructure
from phase1_infrastructure import (
    GrazDataLoader,
    EEGPreprocessor,
    EEGNet,
    create_train_val_test_splits
)


# ============================================================
# SETUP AND CONFIGURATION
# ============================================================

class VanessaAnalysisRunner:
    def __init__(self, results_path, models_path, data_path):
        self.results_path = Path(results_path)
        self.models_path = Path(models_path)
        self.data_path = data_path

        # Output directory
        self.output_dir = Path("results/vanessa_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load experiment data from subject intermediate files
        print("Loading experiment data from subject files...")
        self.main_results = {'experiments': []}

        for subject_id in range(1, 10):
            subject_file = self.results_path / f"subject_{subject_id:02d}_intermediate.json"
            if subject_file.exists():
                with open(subject_file, 'r') as f:
                    subject_replicates = json.load(f)  # This is a list

                # Convert to expected format
                for rep_data in subject_replicates:
                    self.main_results['experiments'].append({
                        'subject_id': rep_data['subject_id'],
                        'replicate': rep_data['replicate'],
                        'baseline_accuracy': rep_data['baseline'] / 100.0,  # Convert % to decimal
                        'misaligned_accuracy': rep_data['misaligned'] / 100.0,
                        'aligned_accuracy': rep_data['aligned'] / 100.0,
                        'random_seed': rep_data.get('random_seed', 42)
                    })
            else:
                print(f"⚠️ Warning: {subject_file} not found")

        if not self.main_results['experiments']:
            raise FileNotFoundError("No experiment data found. Make sure subject_XX_intermediate.json files exist.")

        print(
            f"✅ Loaded {len(self.main_results['experiments'])} experiments from {len(set(e['subject_id'] for e in self.main_results['experiments']))} subjects")

    def save_results(self, results, filename):
        """Save analysis results to JSON"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Saved: {filepath}")


# ============================================================
# ANALYSIS 1: LOSO CROSS-VALIDATION (BASELINE + TRANSFER)
# ============================================================

def run_loso_validation(runner):
    """
    Leave-One-Subject-Out cross-validation
    Test both baseline and transfer models for generalization
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: LOSO CROSS-VALIDATION")
    print("=" * 70)

    loader = GrazDataLoader(runner.data_path)
    preprocessor = EEGPreprocessor()
    loso_results = []

    for held_out_subject in range(1, 10):
        print(f"\n{'=' * 70}")
        print(f"HELD-OUT SUBJECT: {held_out_subject}")
        print(f"{'=' * 70}")

        # Load held-out subject's test data
        try:
            X_raw, y_raw = loader.load_dataset_2b(held_out_subject)

            # CRITICAL FIX: Preprocess the data (bandpass + normalize)
            X_preprocessed = preprocessor.preprocess(X_raw)

            # CRITICAL FIX: Use same split strategy as training
            splits = create_train_val_test_splits(
                X_preprocessed, y_raw,
                val_size=0.2, test_size=0.2,
                random_state=42
            )

            X_test, y_test = splits['test']

            # Convert to tensors
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)

            print(f"  Loaded test data: {X_test.shape[0]} trials")
            print(f"  Test labels: {np.unique(y_test)} (remapped to 0-indexed)")

        except Exception as e:
            print(f"  ❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            continue

        # LOSO for BASELINE models
        print(f"\n  Loading baseline models from other 8 subjects...")
        baseline_models = []
        for subject_id in range(1, 10):
            if subject_id == held_out_subject:
                continue

            for replicate in range(1, 6):
                try:
                    model_path = runner.models_path / f"subject_{subject_id:02d}_rep_{replicate}_baseline.pth"
                    checkpoint = torch.load(model_path, weights_only=False)

                    model = EEGNet(n_channels=3, n_classes=2)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    baseline_models.append(model)

                except Exception as e:
                    print(f"    Warning: Could not load {model_path}: {e}")
                    continue

        print(f"  Loaded {len(baseline_models)} baseline models")

        # LOSO for TRANSFER models (misaligned)
        print(f"\n  Loading transfer models from other 8 subjects...")
        transfer_models = []
        for subject_id in range(1, 10):
            if subject_id == held_out_subject:
                continue

            for replicate in range(1, 6):
                try:
                    model_path = runner.models_path / f"subject_{subject_id:02d}_rep_{replicate}_misaligned.pth"
                    checkpoint = torch.load(model_path, weights_only=False)

                    model = EEGNet(n_channels=3, n_classes=2)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    transfer_models.append(model)

                except Exception as e:
                    print(f"    Warning: Could not load {model_path}: {e}")
                    continue

        print(f"  Loaded {len(transfer_models)} transfer models")

        # Make predictions with ensemble
        # Baseline ensemble
        with torch.no_grad():
            baseline_probs = []
            for model in baseline_models:
                outputs = model(X_test_tensor)
                probs = torch.softmax(outputs, dim=1)
                baseline_probs.append(probs.numpy())

            baseline_avg_probs = np.mean(baseline_probs, axis=0)
            baseline_preds = np.argmax(baseline_avg_probs, axis=1)
            baseline_accuracy = np.mean(baseline_preds == y_test)

        # Transfer ensemble
        with torch.no_grad():
            transfer_probs = []
            for model in transfer_models:
                outputs = model(X_test_tensor)
                probs = torch.softmax(outputs, dim=1)
                transfer_probs.append(probs.numpy())

            transfer_avg_probs = np.mean(transfer_probs, axis=0)
            transfer_preds = np.argmax(transfer_avg_probs, axis=1)
            transfer_accuracy = np.mean(transfer_preds == y_test)

        print(f"\n  Results:")
        print(f"    Baseline LOSO accuracy: {baseline_accuracy:.2%}")
        print(f"    Transfer LOSO accuracy: {transfer_accuracy:.2%}")

        # Get within-subject accuracy for comparison
        within_subject_results = [r for r in runner.main_results['experiments']
                                  if r['subject_id'] == held_out_subject]
        within_baseline = np.mean([r['baseline_accuracy'] for r in within_subject_results])

        print(f"    Within-subject baseline: {within_baseline:.2%}")
        print(f"    Generalization gap: {within_baseline - baseline_accuracy:+.2%}")

        loso_results.append({
            'held_out_subject': held_out_subject,
            'baseline_loso_accuracy': baseline_accuracy,
            'transfer_loso_accuracy': transfer_accuracy,
            'within_subject_baseline': within_baseline,
            'generalization_gap': within_baseline - baseline_accuracy,
            'n_baseline_models': len(baseline_models),
            'n_transfer_models': len(transfer_models)
        })

    # Save results
    runner.save_results(loso_results, 'loso_validation_results.json')

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("LOSO SUMMARY")
    print(f"{'=' * 70}")

    baseline_loso_accs = [r['baseline_loso_accuracy'] for r in loso_results]
    transfer_loso_accs = [r['transfer_loso_accuracy'] for r in loso_results]
    within_accs = [r['within_subject_baseline'] for r in loso_results]
    gaps = [r['generalization_gap'] for r in loso_results]

    print(f"\nBaseline LOSO accuracy:  {np.mean(baseline_loso_accs):.2%} ± {np.std(baseline_loso_accs):.2%}")
    print(f"Transfer LOSO accuracy:  {np.mean(transfer_loso_accs):.2%} ± {np.std(transfer_loso_accs):.2%}")
    print(f"Within-subject baseline: {np.mean(within_accs):.2%} ± {np.std(within_accs):.2%}")
    print(f"Mean generalization gap: {np.mean(gaps):.2%} ± {np.std(gaps):.2%}")

    # Statistical test
    t_stat, p_value = stats.ttest_rel(within_accs, baseline_loso_accs)
    print(f"\nPaired t-test (within vs LOSO):")
    print(f"  t(8) = {t_stat:.3f}, p = {p_value:.4f}")

    return loso_results


# ============================================================
# ANALYSIS 2: VARIANCE DECOMPOSITION
# ============================================================

def run_variance_analysis(runner):
    """
    Decompose variance by condition and subject
    Test if transfer increases variance
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: VARIANCE DECOMPOSITION")
    print("=" * 70)

    experiments = runner.main_results['experiments']

    # Group by subject and condition
    by_subject_condition = defaultdict(lambda: defaultdict(list))
    for exp in experiments:
        subject_id = exp['subject_id']
        by_subject_condition[subject_id]['baseline'].append(exp['baseline_accuracy'])
        by_subject_condition[subject_id]['misaligned'].append(exp['misaligned_accuracy'])
        by_subject_condition[subject_id]['aligned'].append(exp['aligned_accuracy'])

    # Calculate within-subject variance for each condition
    variance_results = []

    print(f"\n{'Subject':<10} {'Baseline σ':<12} {'Misaligned σ':<15} {'Aligned σ':<12} {'Highest Var':<15}")
    print("-" * 70)

    for subject_id in range(1, 10):
        baseline_std = np.std(by_subject_condition[subject_id]['baseline'])
        misaligned_std = np.std(by_subject_condition[subject_id]['misaligned'])
        aligned_std = np.std(by_subject_condition[subject_id]['aligned'])

        variances = {
            'baseline': baseline_std,
            'misaligned': misaligned_std,
            'aligned': aligned_std
        }
        highest_var = max(variances, key=variances.get)

        print(f"{subject_id:<10} {baseline_std:<12.2%} {misaligned_std:<15.2%} {aligned_std:<12.2%} {highest_var:<15}")

        variance_results.append({
            'subject_id': subject_id,
            'baseline_std': baseline_std,
            'misaligned_std': misaligned_std,
            'aligned_std': aligned_std,
            'highest_variance_condition': highest_var
        })

    # Levene's test for equality of variances
    all_baseline = [v for subj in by_subject_condition.values() for v in subj['baseline']]
    all_misaligned = [v for subj in by_subject_condition.values() for v in subj['misaligned']]
    all_aligned = [v for subj in by_subject_condition.values() for v in subj['aligned']]

    levene_stat, levene_p = stats.levene(all_baseline, all_misaligned, all_aligned)

    print(f"\nLevene's test for equality of variances:")
    print(f"  F(2, 132) = {levene_stat:.3f}, p = {levene_p:.4f}")
    if levene_p < 0.05:
        print(f"  ✅ Variances differ significantly across conditions")
    else:
        print(f"  ❌ Variances do not differ significantly")

    # Save results
    runner.save_results(variance_results, 'variance_decomposition.json')

    return variance_results


# ============================================================
# ANALYSIS 3: SUBJECT SIMILARITY CLUSTERING
# ============================================================

def run_subject_clustering(runner):
    """
    Cluster subjects based on baseline performance patterns
    Identify which subjects are similar
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: SUBJECT SIMILARITY CLUSTERING")
    print("=" * 70)

    experiments = runner.main_results['experiments']

    # Create performance matrix: subjects × replicates
    performance_matrix = []
    subject_ids = []

    for subject_id in range(1, 10):
        subject_exps = [e for e in experiments if e['subject_id'] == subject_id]
        baseline_accs = [e['baseline_accuracy'] for e in subject_exps]
        performance_matrix.append(baseline_accs)
        subject_ids.append(subject_id)

    performance_matrix = np.array(performance_matrix)

    print(f"\nPerformance matrix shape: {performance_matrix.shape}")
    print(f"(9 subjects × 5 replicates)")

    # Compute pairwise distances (correlation distance)
    distances = pdist(performance_matrix, metric='correlation')
    distance_matrix = squareform(distances)

    # Hierarchical clustering
    linkage_matrix = linkage(distances, method='ward')

    # Print distance matrix
    print(f"\nPairwise correlation distances:")
    print(f"{'':>5}", end='')
    for i in range(1, 10):
        print(f"{i:>8}", end='')
    print()
    print("-" * 80)

    for i, row in enumerate(distance_matrix):
        print(f"S{i + 1:>3}", end='')
        for val in row:
            if val == 0:
                print(f"{'--':>8}", end='')
            else:
                print(f"{val:>8.3f}", end='')
        print()

    # Find most similar pairs
    print(f"\nMost similar subject pairs:")
    pairs = []
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            pairs.append((i + 1, j + 1, distance_matrix[i, j]))

    pairs.sort(key=lambda x: x[2])
    for s1, s2, dist in pairs[:5]:
        print(f"  Subject {s1} ↔ Subject {s2}: distance = {dist:.3f}")

    # Save results
    clustering_results = {
        'distance_matrix': distance_matrix.tolist(),
        'linkage_matrix': linkage_matrix.tolist(),
        'most_similar_pairs': [
            {'subject_1': int(s1), 'subject_2': int(s2), 'distance': float(dist)}
            for s1, s2, dist in pairs[:10]
        ]
    }
    runner.save_results(clustering_results, 'subject_clustering.json')

    return clustering_results, linkage_matrix


# ============================================================
# ANALYSIS 4: CORRELATION ANALYSIS
# ============================================================

def run_correlation_analysis(runner):
    """
    Analyze correlations between baseline performance and transfer benefit
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: CORRELATION ANALYSIS")
    print("=" * 70)

    experiments = runner.main_results['experiments']

    # Group by subject
    by_subject = defaultdict(lambda: {
        'baseline': [],
        'misaligned_benefit': [],
        'aligned_benefit': []
    })

    for exp in experiments:
        subject_id = exp['subject_id']
        by_subject[subject_id]['baseline'].append(exp['baseline_accuracy'])
        by_subject[subject_id]['misaligned_benefit'].append(
            exp['misaligned_accuracy'] - exp['baseline_accuracy']
        )
        by_subject[subject_id]['aligned_benefit'].append(
            exp['aligned_accuracy'] - exp['baseline_accuracy']
        )

    # Calculate mean values per subject
    subject_means = []
    for subject_id in range(1, 10):
        subject_means.append({
            'subject_id': subject_id,
            'baseline_mean': np.mean(by_subject[subject_id]['baseline']),
            'baseline_std': np.std(by_subject[subject_id]['baseline']),
            'misaligned_benefit_mean': np.mean(by_subject[subject_id]['misaligned_benefit']),
            'aligned_benefit_mean': np.mean(by_subject[subject_id]['aligned_benefit'])
        })

    # Correlation: baseline vs misaligned benefit
    baseline_accs = [s['baseline_mean'] for s in subject_means]
    misaligned_benefits = [s['misaligned_benefit_mean'] for s in subject_means]
    aligned_benefits = [s['aligned_benefit_mean'] for s in subject_means]
    baseline_stds = [s['baseline_std'] for s in subject_means]

    r_misaligned, p_misaligned = stats.pearsonr(baseline_accs, misaligned_benefits)
    r_aligned, p_aligned = stats.pearsonr(baseline_accs, aligned_benefits)
    r_variance, p_variance = stats.pearsonr(baseline_stds, misaligned_benefits)

    print(f"\nCorrelation: Baseline accuracy vs Transfer benefit")
    print(f"  Misaligned: r = {r_misaligned:+.3f}, p = {p_misaligned:.4f}")
    print(f"  Aligned:    r = {r_aligned:+.3f}, p = {p_aligned:.4f}")

    print(f"\nCorrelation: Baseline variance vs Transfer benefit")
    print(f"  Misaligned: r = {r_variance:+.3f}, p = {p_variance:.4f}")

    # Bootstrap confidence intervals
    print(f"\nBootstrap 95% confidence intervals (1000 iterations):")

    def bootstrap_correlation(x, y, n_iterations=1000):
        correlations = []
        for _ in range(n_iterations):
            indices = np.random.choice(len(x), len(x), replace=True)
            x_boot = [x[i] for i in indices]
            y_boot = [y[i] for i in indices]
            r, _ = stats.pearsonr(x_boot, y_boot)
            correlations.append(r)
        return np.percentile(correlations, [2.5, 97.5])

    ci_misaligned = bootstrap_correlation(baseline_accs, misaligned_benefits)
    ci_aligned = bootstrap_correlation(baseline_accs, aligned_benefits)

    print(f"  Misaligned: [{ci_misaligned[0]:+.3f}, {ci_misaligned[1]:+.3f}]")
    print(f"  Aligned:    [{ci_aligned[0]:+.3f}, {ci_aligned[1]:+.3f}]")

    # Save results
    correlation_results = {
        'baseline_vs_misaligned_benefit': {
            'r': r_misaligned,
            'p': p_misaligned,
            'ci_95': ci_misaligned.tolist()
        },
        'baseline_vs_aligned_benefit': {
            'r': r_aligned,
            'p': p_aligned,
            'ci_95': ci_aligned.tolist()
        },
        'variance_vs_benefit': {
            'r': r_variance,
            'p': p_variance
        },
        'subject_data': subject_means
    }
    runner.save_results(correlation_results, 'correlation_analysis.json')

    return correlation_results


# ============================================================
# ANALYSIS 5: ERROR PATTERN ANALYSIS
# ============================================================

def run_error_analysis(runner):
    """
    Analyze error patterns for subjects with negative transfer
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 5: ERROR PATTERN ANALYSIS")
    print("=" * 70)

    # Identify subjects with negative transfer
    experiments = runner.main_results['experiments']

    by_subject = defaultdict(lambda: {'baseline': [], 'misaligned': [], 'aligned': []})
    for exp in experiments:
        subject_id = exp['subject_id']
        by_subject[subject_id]['baseline'].append(exp['baseline_accuracy'])
        by_subject[subject_id]['misaligned'].append(exp['misaligned_accuracy'])
        by_subject[subject_id]['aligned'].append(exp['aligned_accuracy'])

    negative_transfer_subjects = []
    for subject_id in range(1, 10):
        baseline_mean = np.mean(by_subject[subject_id]['baseline'])
        misaligned_mean = np.mean(by_subject[subject_id]['misaligned'])

        if misaligned_mean < baseline_mean:
            negative_transfer_subjects.append({
                'subject_id': subject_id,
                'baseline_mean': baseline_mean,
                'misaligned_mean': misaligned_mean,
                'degradation': baseline_mean - misaligned_mean
            })

    print(f"\nSubjects with negative transfer:")
    print(f"{'Subject':<10} {'Baseline':<12} {'Misaligned':<15} {'Degradation':<15}")
    print("-" * 55)

    for s in sorted(negative_transfer_subjects, key=lambda x: x['degradation'], reverse=True):
        print(
            f"{s['subject_id']:<10} {s['baseline_mean']:<12.2%} {s['misaligned_mean']:<15.2%} {s['degradation']:<15.2%}")

    print(f"\nTotal: {len(negative_transfer_subjects)}/9 subjects experienced negative transfer")

    # Save results
    runner.save_results({
        'negative_transfer_subjects': negative_transfer_subjects,
        'count': len(negative_transfer_subjects),
        'percentage': len(negative_transfer_subjects) / 9
    }, 'error_pattern_analysis.json')

    return negative_transfer_subjects


# ============================================================
# VISUALIZATION
# ============================================================

def create_loso_figure(loso_results, save_path):
    """Create LOSO comparison figure"""
    subjects = [r['held_out_subject'] for r in loso_results]
    baseline_loso = [r['baseline_loso_accuracy'] * 100 for r in loso_results]
    transfer_loso = [r['transfer_loso_accuracy'] * 100 for r in loso_results]
    within_subject = [r['within_subject_baseline'] * 100 for r in loso_results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(subjects))
    width = 0.25

    bars1 = ax.bar(x - width, within_subject, width, label='Within-Subject',
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, baseline_loso, width, label='LOSO (Baseline Models)',
                   color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, transfer_loso, width, label='LOSO (Transfer Models)',
                   color='#e67e22', alpha=0.8)

    ax.set_xlabel('Held-Out Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Leave-One-Subject-Out Cross-Validation\n(Ensemble of 40 models per subject)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in subjects])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {save_path}")
    plt.close()


def create_variance_figure(variance_results, save_path):
    """Create variance comparison figure"""
    subjects = [r['subject_id'] for r in variance_results]
    baseline_std = [r['baseline_std'] * 100 for r in variance_results]
    misaligned_std = [r['misaligned_std'] * 100 for r in variance_results]
    aligned_std = [r['aligned_std'] * 100 for r in variance_results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(subjects))
    width = 0.25

    ax.bar(x - width, baseline_std, width, label='Baseline',
           color='#3498db', alpha=0.8)
    ax.bar(x, misaligned_std, width, label='Misaligned Transfer',
           color='#e67e22', alpha=0.8)
    ax.bar(x + width, aligned_std, width, label='Task-Aligned Transfer',
           color='#9b59b6', alpha=0.8)

    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standard Deviation (% points)', fontsize=12, fontweight='bold')
    ax.set_title('Within-Subject Variance by Condition\n(Across 5 replicates)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in subjects])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {save_path}")
    plt.close()


def create_dendrogram_figure(linkage_matrix, save_path):
    """Create hierarchical clustering dendrogram"""
    fig, ax = plt.subplots(figsize=(10, 6))

    dendrogram(linkage_matrix,
               labels=[f'S{i}' for i in range(1, 10)],
               ax=ax,
               color_threshold=0.7)

    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance (Correlation)', fontsize=12, fontweight='bold')
    ax.set_title('Subject Similarity Clustering\n(Based on baseline performance patterns)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {save_path}")
    plt.close()


def create_correlation_figure(correlation_results, save_path):
    """Create correlation scatter plots"""
    subject_data = correlation_results['subject_data']
    baseline_accs = [s['baseline_mean'] * 100 for s in subject_data]
    misaligned_benefits = [s['misaligned_benefit_mean'] * 100 for s in subject_data]
    aligned_benefits = [s['aligned_benefit_mean'] * 100 for s in subject_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Baseline vs Misaligned Benefit
    ax1.scatter(baseline_accs, misaligned_benefits, s=100, alpha=0.7, color='#e67e22')
    for i, s in enumerate(subject_data):
        ax1.annotate(f'S{s["subject_id"]}',
                     (baseline_accs[i], misaligned_benefits[i]),
                     fontsize=9, ha='center', va='bottom')

    # Fit line
    z = np.polyfit(baseline_accs, misaligned_benefits, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(baseline_accs), max(baseline_accs), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)

    r = correlation_results['baseline_vs_misaligned_benefit']['r']
    p_val = correlation_results['baseline_vs_misaligned_benefit']['p']
    ax1.text(0.05, 0.95, f'r = {r:+.3f}\np = {p_val:.4f}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel('Baseline Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Misaligned Transfer Benefit (% points)', fontsize=11, fontweight='bold')
    ax1.set_title('Baseline Performance vs Transfer Benefit\n(Misaligned)', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Plot 2: Baseline vs Aligned Benefit
    ax2.scatter(baseline_accs, aligned_benefits, s=100, alpha=0.7, color='#9b59b6')
    for i, s in enumerate(subject_data):
        ax2.annotate(f'S{s["subject_id"]}',
                     (baseline_accs[i], aligned_benefits[i]),
                     fontsize=9, ha='center', va='bottom')

    # Fit line
    z = np.polyfit(baseline_accs, aligned_benefits, 1)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)

    r = correlation_results['baseline_vs_aligned_benefit']['r']
    p_val = correlation_results['baseline_vs_aligned_benefit']['p']
    ax2.text(0.05, 0.95, f'r = {r:+.3f}\np = {p_val:.4f}',
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Baseline Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Task-Aligned Transfer Benefit (% points)', fontsize=11, fontweight='bold')
    ax2.set_title('Baseline Performance vs Transfer Benefit\n(Task-Aligned)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

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
    print("VANESSA'S COMPREHENSIVE ANALYSIS SUITE")
    print("=" * 70)
    print("\nThis will run:")
    print("  1. LOSO Cross-Validation (baseline + transfer)")
    print("  2. Variance Decomposition")
    print("  3. Subject Similarity Clustering")
    print("  4. Correlation Analysis")
    print("  5. Error Pattern Analysis")
    print("\nEstimated runtime: ~30 minutes")
    print("=" * 70)

    # Get paths (with quote stripping)
    results_path = input("\nEnter path to results/9x5_study/ directory: ").strip().strip('"')
    if not results_path:
        results_path = "results/9x5_study"
        print(f"Using default: {results_path}")

    models_path = input("Enter path to saved models directory: ").strip().strip('"')
    if not models_path:
        models_path = "results/9x5_study/models"
        print(f"Using default: {models_path}")

    data_path = input("Enter path to BCI Competition IV data: ").strip().strip('"')
    if not data_path:
        data_path = r"D:\Documents 1 Jan 2026\BCICIV"
        print(f"Using default: {data_path}")

    # Initialize
    runner = VanessaAnalysisRunner(results_path, models_path, data_path)

    # Confirm
    print("\nReady to start analysis?")
    response = input("Continue? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Cancelled.")
        return

    try:
        # Run analyses
        loso_results = run_loso_validation(runner)
        create_loso_figure(loso_results, runner.output_dir / "loso_comparison.png")

        variance_results = run_variance_analysis(runner)
        create_variance_figure(variance_results, runner.output_dir / "variance_by_condition.png")

        clustering_results, linkage_matrix = run_subject_clustering(runner)
        create_dendrogram_figure(linkage_matrix, runner.output_dir / "subject_clustering_dendrogram.png")

        correlation_results = run_correlation_analysis(runner)
        create_correlation_figure(correlation_results, runner.output_dir / "correlation_scatter_plots.png")

        error_results = run_error_analysis(runner)

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Partial results have been saved.")
        return
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return

    # Final summary
    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {runner.output_dir}")
    print(f"\nFigures generated:")
    print(f"  - loso_comparison.png")
    print(f"  - variance_by_condition.png")
    print(f"  - subject_clustering_dendrogram.png")
    print(f"  - correlation_scatter_plots.png")
    print("\n✅ Vanessa's work is complete!")


if __name__ == "__main__":
    main()