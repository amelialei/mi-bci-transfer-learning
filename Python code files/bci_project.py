"""
BCI Transfer Learning Project - Complete Study
5 replicates × 9 subjects = 45 total experiments
Estimated time: ~90 hours (4 days)
"""

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from scipy import stats
import json
import time
from datetime import datetime, timedelta

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

DATA_PATH = r"D:\Documents 1 Jan 2026\BCICIV"

SAMPLING_RATE = 250
LOWCUT, HIGHCUT = 8, 30
EPOCH_START, EPOCH_END = 0.5, 2.5
VALIDATION_SPLIT = 0.2

N_SUBJECTS = 9
N_REPLICATES = 5  # Number of runs per subject

# ============================================================================
# SECTION 2: DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(dataset='2a', subject_id=1, hands_only=False):
    """Load and preprocess data - returns train/val splits"""

    if dataset == '2a':
        filename = f"A{subject_id:02d}T.gdf"
        filepath = Path(DATA_PATH) / filename
        raw = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
        events, event_dict = mne.events_from_annotations(raw)
        raw.pick(raw.ch_names[:22])

        if hands_only:
            mi_event_ids = {key: val for key, val in event_dict.items()
                           if key in ['769', '770']}
        else:
            mi_event_ids = {key: val for key, val in event_dict.items()
                           if key in ['769', '770', '771', '772']}
        n_channels = 22
    else:
        raw_list, events_list = [], []
        event_dict = None
        for sess in ['01T', '02T', '03T']:
            filepath = Path(DATA_PATH) / f"B{subject_id:02d}{sess}.gdf"
            if filepath.exists():
                r = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
                e, ed = mne.events_from_annotations(r)
                if event_dict is None:
                    event_dict = ed
                r.pick(r.ch_names[:3])
                raw_list.append(r)
                events_list.append(e)
        raw = mne.concatenate_raws(raw_list)
        events = np.vstack(events_list)
        mi_event_ids = {key: val for key, val in event_dict.items()
                       if key in ['769', '770']}
        n_channels = 3

    epochs = mne.Epochs(raw, events, event_id=mi_event_ids,
                       tmin=EPOCH_START, tmax=EPOCH_END,
                       baseline=None, preload=True,
                       reject_by_annotation=True, verbose=False,
                       event_repeated='drop')

    data = epochs.get_data()
    nyquist = 0.5 * SAMPLING_RATE
    b, a = butter(4, [LOWCUT/nyquist, HIGHCUT/nyquist], btype='band')
    data = filtfilt(b, a, data, axis=-1)
    data = (data - data.mean(axis=-1, keepdims=True)) / (data.std(axis=-1, keepdims=True) + 1e-8)

    labels = epochs.events[:, -1]
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[label] for label in labels])

    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=VALIDATION_SPLIT,
        stratify=labels)

    return (torch.FloatTensor(X_train), torch.FloatTensor(X_val),
            torch.LongTensor(y_train), torch.LongTensor(y_val))

# ============================================================================
# SECTION 3: NEURAL NETWORK MODEL
# ============================================================================

class EEGNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8))

        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25))

        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))
        self.classifier = nn.Linear(16 * 4, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

# ============================================================================
# SECTION 4: TRAINING FUNCTION
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=150, lr=0.001, verbose=False):
    """Train a model and return best validation accuracy"""

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0

    batch_size = 32
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).float().mean().item()
            best_acc = max(best_acc, accuracy)

        if verbose and (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Val Acc = {accuracy:.4f}")

    return best_acc

# ============================================================================
# SECTION 5: SINGLE EXPERIMENT (ONE REPLICATE)
# ============================================================================

def run_single_experiment(subject_id, replicate, random_seed, verbose=False):
    """Run one complete experiment for a subject"""

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    results = {
        'subject_id': subject_id,
        'replicate': replicate,
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }

    if verbose:
        print(f"  Loading data...")

    # Load data
    X_train_2b, X_val_2b, y_train_2b, y_val_2b = load_data('2b', subject_id=subject_id)

    # ========================================================================
    # BASELINE
    # ========================================================================
    if verbose:
        print(f"  Training baseline...")

    baseline_model = EEGNet(n_channels=3, n_classes=2)
    baseline_acc = train_model(baseline_model, X_train_2b, y_train_2b,
                               X_val_2b, y_val_2b, epochs=150, verbose=verbose)
    results['baseline'] = baseline_acc * 100

    # ========================================================================
    # MISALIGNED TRANSFER (4-class)
    # ========================================================================
    if verbose:
        print(f"  Pre-training on 4-class...")

    X_train_2a_full, X_val_2a_full, y_train_2a_full, y_val_2a_full = load_data(
        '2a', subject_id=subject_id, hands_only=False)

    model_2a_full = EEGNet(n_channels=22, n_classes=4)
    train_model(model_2a_full, X_train_2a_full, y_train_2a_full,
                X_val_2a_full, y_val_2a_full, epochs=150, verbose=verbose)

    if verbose:
        print(f"  Fine-tuning misaligned...")

    transfer_misaligned = EEGNet(n_channels=3, n_classes=2)
    transfer_misaligned.temporal = model_2a_full.temporal

    misaligned_acc = train_model(transfer_misaligned, X_train_2b, y_train_2b,
                                 X_val_2b, y_val_2b, epochs=150, verbose=verbose)
    results['misaligned'] = misaligned_acc * 100
    results['misaligned_delta'] = (misaligned_acc - baseline_acc) * 100

    # ========================================================================
    # TASK-ALIGNED TRANSFER (2-class hands only)
    # ========================================================================
    if verbose:
        print(f"  Pre-training on hands-only...")

    X_train_2a_hands, X_val_2a_hands, y_train_2a_hands, y_val_2a_hands = load_data(
        '2a', subject_id=subject_id, hands_only=True)

    model_2a_hands = EEGNet(n_channels=22, n_classes=2)
    train_model(model_2a_hands, X_train_2a_hands, y_train_2a_hands,
                X_val_2a_hands, y_val_2a_hands, epochs=150, verbose=verbose)

    if verbose:
        print(f"  Fine-tuning task-aligned...")

    transfer_aligned = EEGNet(n_channels=3, n_classes=2)
    transfer_aligned.temporal = model_2a_hands.temporal

    aligned_acc = train_model(transfer_aligned, X_train_2b, y_train_2b,
                              X_val_2b, y_val_2b, epochs=150, verbose=verbose)
    results['aligned'] = aligned_acc * 100
    results['aligned_delta'] = (aligned_acc - baseline_acc) * 100

    return results

# ============================================================================
# SECTION 6: MULTI-REPLICATE SUBJECT
# ============================================================================

def run_subject_with_replicates(subject_id, n_replicates=5, verbose=True):
    """Run multiple replicates for one subject"""

    print(f"\n{'='*70}")
    print(f"SUBJECT {subject_id}/{N_SUBJECTS} - {n_replicates} REPLICATES")
    print(f"{'='*70}")

    start_time = time.time()

    subject_results = []

    for rep in range(n_replicates):
        print(f"\n  Replicate {rep+1}/{n_replicates}")
        print(f"  {'-'*60}")

        # Unique seed for each replicate
        random_seed = 42 + subject_id * 1000 + rep

        rep_start = time.time()
        result = run_single_experiment(subject_id, rep+1, random_seed, verbose=False)
        rep_time = time.time() - rep_start

        subject_results.append(result)

        print(f"  Baseline:      {result['baseline']:.2f}%")
        print(f"  Misaligned:    {result['misaligned']:.2f}%  ({result['misaligned_delta']:+.2f}%)")
        print(f"  Task-Aligned:  {result['aligned']:.2f}%  ({result['aligned_delta']:+.2f}%)")
        print(f"  Time: {rep_time/60:.1f} min")

        # Save intermediate results
        save_intermediate_results(subject_id, subject_results)

    # Compute subject-level statistics
    baseline_vals = [r['baseline'] for r in subject_results]
    misaligned_vals = [r['misaligned'] for r in subject_results]
    aligned_vals = [r['aligned'] for r in subject_results]
    misaligned_deltas = [r['misaligned_delta'] for r in subject_results]
    aligned_deltas = [r['aligned_delta'] for r in subject_results]

    summary = {
        'subject_id': subject_id,
        'n_replicates': n_replicates,
        'baseline_mean': np.mean(baseline_vals),
        'baseline_std': np.std(baseline_vals, ddof=1),
        'misaligned_mean': np.mean(misaligned_vals),
        'misaligned_std': np.std(misaligned_vals, ddof=1),
        'aligned_mean': np.mean(aligned_vals),
        'aligned_std': np.std(aligned_vals, ddof=1),
        'misaligned_delta_mean': np.mean(misaligned_deltas),
        'misaligned_delta_std': np.std(misaligned_deltas, ddof=1),
        'aligned_delta_mean': np.mean(aligned_deltas),
        'aligned_delta_std': np.std(aligned_deltas, ddof=1),
        'all_results': subject_results
    }

    elapsed = time.time() - start_time

    print(f"\n  {'='*60}")
    print(f"  SUBJECT {subject_id} SUMMARY ({n_replicates} replicates)")
    print(f"  {'='*60}")
    print(f"  Baseline:      {summary['baseline_mean']:.2f}% ± {summary['baseline_std']:.2f}%")
    print(f"  Misaligned:    {summary['misaligned_delta_mean']:+.2f}% ± {summary['misaligned_delta_std']:.2f}%")
    print(f"  Task-Aligned:  {summary['aligned_delta_mean']:+.2f}% ± {summary['aligned_delta_std']:.2f}%")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")

    return summary

# ============================================================================
# SECTION 7: COMPLETE STUDY
# ============================================================================

def run_complete_study(n_subjects=9, n_replicates=5):
    """Run the complete study on all subjects"""

    print("="*70)
    print("BCI TRANSFER LEARNING - COMPLETE STUDY")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Subjects: {n_subjects}")
    print(f"  Replicates per subject: {n_replicates}")
    print(f"  Total experiments: {n_subjects * n_replicates * 3} models")
    print(f"  Estimated time: ~{n_subjects * n_replicates * 2} hours ({n_subjects * n_replicates * 2 / 24:.1f} days)")

    input("\nPress Enter to start the experiment...")

    study_start = time.time()
    all_subject_summaries = []

    for subject_id in range(1, n_subjects + 1):
        try:
            summary = run_subject_with_replicates(subject_id, n_replicates)
            all_subject_summaries.append(summary)

            # Estimate remaining time
            elapsed = time.time() - study_start
            avg_time_per_subject = elapsed / len(all_subject_summaries)
            remaining_subjects = n_subjects - len(all_subject_summaries)
            eta_seconds = avg_time_per_subject * remaining_subjects
            eta = datetime.now() + timedelta(seconds=eta_seconds)

            print(f"\n  Progress: {len(all_subject_summaries)}/{n_subjects} subjects complete")
            print(f"  ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')} ({eta_seconds/3600:.1f} hours remaining)")

        except Exception as e:
            print(f"\n  ❌ ERROR with subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - study_start

    print(f"\n{'='*70}")
    print(f"ALL SUBJECTS COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/86400:.2f} days)")

    # Generate final analysis
    generate_final_analysis(all_subject_summaries)

    return all_subject_summaries

# ============================================================================
# SECTION 8: RESULTS SAVING
# ============================================================================

def save_intermediate_results(subject_id, results):
    """Save intermediate results after each replicate"""

    filename = f'subject_{subject_id:02d}_intermediate.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def generate_final_analysis(all_summaries):
    """Generate comprehensive final analysis"""

    print(f"\n{'='*70}")
    print("FINAL ANALYSIS")
    print(f"{'='*70}\n")

    n_subjects = len(all_summaries)

    # Extract data
    baseline_means = [s['baseline_mean'] for s in all_summaries]
    misaligned_deltas = [s['misaligned_delta_mean'] for s in all_summaries]
    aligned_deltas = [s['aligned_delta_mean'] for s in all_summaries]

    # ========================================================================
    # DESCRIPTIVE STATISTICS
    # ========================================================================

    print("DESCRIPTIVE STATISTICS")
    print("-"*70)
    print(f"\nBaseline Performance:")
    print(f"  Mean: {np.mean(baseline_means):.2f}%")
    print(f"  Std:  {np.std(baseline_means, ddof=1):.2f}%")
    print(f"  Min:  {np.min(baseline_means):.2f}%")
    print(f"  Max:  {np.max(baseline_means):.2f}%")

    print(f"\nTransfer Learning Improvements:")
    print(f"  Misaligned Transfer:")
    print(f"    Mean: {np.mean(misaligned_deltas):+.2f}%")
    print(f"    Std:  {np.std(misaligned_deltas, ddof=1):.2f}%")
    print(f"    Min:  {np.min(misaligned_deltas):+.2f}%")
    print(f"    Max:  {np.max(misaligned_deltas):+.2f}%")

    print(f"  Task-Aligned Transfer:")
    print(f"    Mean: {np.mean(aligned_deltas):+.2f}%")
    print(f"    Std:  {np.std(aligned_deltas, ddof=1):.2f}%")
    print(f"    Min:  {np.min(aligned_deltas):+.2f}%")
    print(f"    Max:  {np.max(aligned_deltas):+.2f}%")

    # ========================================================================
    # STATISTICAL TESTS
    # ========================================================================

    print(f"\n{'='*70}")
    print("STATISTICAL TESTS")
    print("-"*70)

    # Test 1: Misaligned vs zero
    t_mis, p_mis = stats.ttest_1samp(misaligned_deltas, 0)
    print(f"\nMisaligned Transfer vs Baseline:")
    print(f"  One-sample t-test: t({n_subjects-1}) = {t_mis:.3f}, p = {p_mis:.4f}")
    if p_mis < 0.05:
        print(f"  ✅ SIGNIFICANT (α=0.05)")
    else:
        print(f"  ❌ Not significant")

    # Test 2: Task-aligned vs zero
    t_ali, p_ali = stats.ttest_1samp(aligned_deltas, 0)
    print(f"\nTask-Aligned Transfer vs Baseline:")
    print(f"  One-sample t-test: t({n_subjects-1}) = {t_ali:.3f}, p = {p_ali:.4f}")
    if p_ali < 0.05:
        print(f"  ✅ SIGNIFICANT (α=0.05)")
    else:
        print(f"  ❌ Not significant")

    # Test 3: Misaligned vs Task-aligned
    t_comp, p_comp = stats.ttest_rel(misaligned_deltas, aligned_deltas)
    print(f"\nMisaligned vs Task-Aligned:")
    print(f"  Paired t-test: t({n_subjects-1}) = {t_comp:.3f}, p = {p_comp:.4f}")
    if p_comp < 0.05:
        print(f"  ✅ SIGNIFICANT (α=0.05)")
    else:
        print(f"  ❌ Not significant")

    # ========================================================================
    # EFFECT SIZES
    # ========================================================================

    print(f"\n{'='*70}")
    print("EFFECT SIZES (Cohen's d)")
    print("-"*70)

    def cohens_d_onesample(x, mu=0):
        return (np.mean(x) - mu) / np.std(x, ddof=1)

    d_mis = cohens_d_onesample(misaligned_deltas)
    d_ali = cohens_d_onesample(aligned_deltas)

    print(f"\nMisaligned Transfer: d = {d_mis:.3f}", end="")
    if abs(d_mis) < 0.2:
        print(" (negligible)")
    elif abs(d_mis) < 0.5:
        print(" (small)")
    elif abs(d_mis) < 0.8:
        print(" (medium)")
    else:
        print(" (large)")

    print(f"Task-Aligned Transfer: d = {d_ali:.3f}", end="")
    if abs(d_ali) < 0.2:
        print(" (negligible)")
    elif abs(d_ali) < 0.5:
        print(" (small)")
    elif abs(d_ali) < 0.8:
        print(" (medium)")
    else:
        print(" (large)")

    # ========================================================================
    # CONFIDENCE INTERVALS
    # ========================================================================

    print(f"\n{'='*70}")
    print("95% CONFIDENCE INTERVALS")
    print("-"*70)

    def ci_95(data):
        mean = np.mean(data)
        se = stats.sem(data)
        ci = se * stats.t.ppf(0.975, len(data)-1)
        return mean, mean - ci, mean + ci

    mis_mean, mis_lower, mis_upper = ci_95(misaligned_deltas)
    ali_mean, ali_lower, ali_upper = ci_95(aligned_deltas)

    print(f"\nMisaligned Transfer: {mis_mean:+.2f}% [{mis_lower:+.2f}%, {mis_upper:+.2f}%]")
    print(f"Task-Aligned Transfer: {ali_mean:+.2f}% [{ali_lower:+.2f}%, {ali_upper:+.2f}%]")

    # ========================================================================
    # SUCCESS RATES
    # ========================================================================

    print(f"\n{'='*70}")
    print("SUCCESS RATES")
    print("-"*70)

    mis_wins = sum(1 for d in misaligned_deltas if d > 0)
    ali_wins = sum(1 for d in aligned_deltas if d > 0)

    print(f"\nSubjects improved with:")
    print(f"  Misaligned Transfer:   {mis_wins}/{n_subjects} ({mis_wins/n_subjects*100:.0f}%)")
    print(f"  Task-Aligned Transfer: {ali_wins}/{n_subjects} ({ali_wins/n_subjects*100:.0f}%)")

    # ========================================================================
    # PER-SUBJECT TABLE
    # ========================================================================

    print(f"\n{'='*70}")
    print("PER-SUBJECT RESULTS")
    print("-"*70)
    print(f"\n{'Subj':<6} {'Baseline':<18} {'Misaligned Δ':<20} {'Aligned Δ':<20}")
    print("-" * 70)

    for s in all_summaries:
        print(f"{s['subject_id']:<6} "
              f"{s['baseline_mean']:>6.2f}% ± {s['baseline_std']:>4.2f}%    "
              f"{s['misaligned_delta_mean']:>+6.2f}% ± {s['misaligned_delta_std']:>4.2f}%      "
              f"{s['aligned_delta_mean']:>+6.2f}% ± {s['aligned_delta_std']:>4.2f}%")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    # Save JSON
    results_dict = {
        'metadata': {
            'n_subjects': n_subjects,
            'n_replicates': all_summaries[0]['n_replicates'],
            'total_experiments': n_subjects * all_summaries[0]['n_replicates'],
            'timestamp': datetime.now().isoformat()
        },
        'summary_statistics': {
            'baseline_mean': float(np.mean(baseline_means)),
            'baseline_std': float(np.std(baseline_means, ddof=1)),
            'misaligned_mean': float(np.mean(misaligned_deltas)),
            'misaligned_std': float(np.std(misaligned_deltas, ddof=1)),
            'aligned_mean': float(np.mean(aligned_deltas)),
            'aligned_std': float(np.std(aligned_deltas, ddof=1))
        },
        'statistical_tests': {
            'misaligned_vs_baseline': {'t': float(t_mis), 'p': float(p_mis)},
            'aligned_vs_baseline': {'t': float(t_ali), 'p': float(p_ali)},
            'misaligned_vs_aligned': {'t': float(t_comp), 'p': float(p_comp)}
        },
        'effect_sizes': {
            'misaligned': float(d_mis),
            'aligned': float(d_ali)
        },
        'confidence_intervals': {
            'misaligned': {'mean': float(mis_mean), 'lower': float(mis_lower), 'upper': float(mis_upper)},
            'aligned': {'mean': float(ali_mean), 'lower': float(ali_lower), 'upper': float(ali_upper)}
        },
        'subject_summaries': all_summaries
    }

    with open('complete_study_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✅ Complete results saved to 'complete_study_results.json'")

    # Save text report
    with open('final_report.txt', 'w') as f:
        f.write("BCI TRANSFER LEARNING STUDY - FINAL REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Study Parameters:\n")
        f.write(f"  Subjects: {n_subjects}\n")
        f.write(f"  Replicates per subject: {all_summaries[0]['n_replicates']}\n")
        f.write(f"  Total experiments: {n_subjects * all_summaries[0]['n_replicates']}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Baseline: {np.mean(baseline_means):.2f}% ± {np.std(baseline_means, ddof=1):.2f}%\n")
        f.write(f"  Misaligned Transfer: {np.mean(misaligned_deltas):+.2f}% ± {np.std(misaligned_deltas, ddof=1):.2f}% (p={p_mis:.4f})\n")
        f.write(f"  Task-Aligned Transfer: {np.mean(aligned_deltas):+.2f}% ± {np.std(aligned_deltas, ddof=1):.2f}% (p={p_ali:.4f})\n\n")
        f.write(f"Statistical Significance:\n")
        f.write(f"  Misaligned: {'YES (p<0.05)' if p_mis < 0.05 else 'NO (p≥0.05)'}\n")
        f.write(f"  Task-Aligned: {'YES (p<0.05)' if p_ali < 0.05 else 'NO (p≥0.05)'}\n\n")
        f.write(f"Effect Sizes (Cohen's d):\n")
        f.write(f"  Misaligned: {d_mis:.3f}\n")
        f.write(f"  Task-Aligned: {d_ali:.3f}\n\n")
        f.write(f"95% Confidence Intervals:\n")
        f.write(f"  Misaligned: [{mis_lower:+.2f}%, {mis_upper:+.2f}%]\n")
        f.write(f"  Task-Aligned: [{ali_lower:+.2f}%, {ali_upper:+.2f}%]\n")

    print(f"✅ Text report saved to 'final_report.txt'")

    # Create visualizations
    create_final_visualizations(all_summaries)

    print(f"\n{'='*70}")
    print("STUDY COMPLETE!")
    print(f"{'='*70}\n")

# ============================================================================
# SECTION 9: VISUALIZATIONS
# ============================================================================

def create_final_visualizations(all_summaries):
    """Create comprehensive visualizations"""

    n_subjects = len(all_summaries)

    # Extract data
    subjects = [s['subject_id'] for s in all_summaries]
    baseline_means = [s['baseline_mean'] for s in all_summaries]
    baseline_stds = [s['baseline_std'] for s in all_summaries]
    mis_means = [s['baseline_mean'] + s['misaligned_delta_mean'] for s in all_summaries]
    mis_stds = [s['misaligned_delta_std'] for s in all_summaries]
    ali_means = [s['baseline_mean'] + s['aligned_delta_mean'] for s in all_summaries]
    ali_stds = [s['aligned_delta_std'] for s in all_summaries]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # ========================================================================
    # Plot 1: Bar chart with error bars
    # ========================================================================
    ax1 = plt.subplot(2, 2, 1)

    x = np.arange(n_subjects)
    width = 0.25

    ax1.bar(x - width, baseline_means, width, yerr=baseline_stds,
            label='Baseline', color='gray', alpha=0.7, capsize=3)
    ax1.bar(x, mis_means, width, yerr=mis_stds,
            label='Misaligned', color='orange', alpha=0.7, capsize=3)
    ax1.bar(x + width, ali_means, width, yerr=ali_stds,
            label='Task-Aligned', color='green', alpha=0.7, capsize=3)

    ax1.set_xlabel('Subject', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Per-Subject Results with Error Bars', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='Chance')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # ========================================================================
    # Plot 2: Box plot
    # ========================================================================
    ax2 = plt.subplot(2, 2, 2)

    data = [baseline_means, mis_means, ali_means]
    bp = ax2.boxplot(data, labels=['Baseline', 'Misaligned', 'Task-Aligned'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('gray')
    bp['boxes'][1].set_facecolor('orange')
    bp['boxes'][2].set_facecolor('green')

    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Distribution Across Subjects', fontsize=12, fontweight='bold')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)

    # ========================================================================
    # Plot 3: Improvement deltas
    # ========================================================================
    ax3 = plt.subplot(2, 2, 3)

    mis_deltas = [s['misaligned_delta_mean'] for s in all_summaries]
    ali_deltas = [s['aligned_delta_mean'] for s in all_summaries]

    ax3.bar(x - width/2, mis_deltas, width,
            label='Misaligned', color='orange', alpha=0.7)
    ax3.bar(x + width/2, ali_deltas, width,
            label='Task-Aligned', color='green', alpha=0.7)

    ax3.set_xlabel('Subject', fontsize=11)
    ax3.set_ylabel('Improvement over Baseline (%)', fontsize=11)
    ax3.set_title('Transfer Learning Improvements', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(subjects)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # ========================================================================
    # Plot 4: Individual trajectories
    # ========================================================================
    ax4 = plt.subplot(2, 2, 4)

    for i in range(n_subjects):
        ax4.plot([1, 2, 3],
                [baseline_means[i], mis_means[i], ali_means[i]],
                'o-', alpha=0.6, label=f'S{subjects[i]}')

    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['Baseline', 'Misaligned', 'Task-Aligned'])
    ax4.set_ylabel('Accuracy (%)', fontsize=11)
    ax4.set_title('Individual Subject Trajectories', fontsize=12, fontweight='bold')
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    ax4.grid(alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('final_results.png', dpi=300, bbox_inches='tight')
    print(f"✅ Visualizations saved to 'final_results.png'")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BCI TRANSFER LEARNING - GOLD STANDARD STUDY")
    print("="*70)
    print("\nThis will run:")
    print(f"  • {N_SUBJECTS} subjects")
    print(f"  • {N_REPLICATES} replicates per subject")
    print(f"  • {N_SUBJECTS * N_REPLICATES} total experiments")
    print(f"  • ~{N_SUBJECTS * N_REPLICATES * 2} hours (~{N_SUBJECTS * N_REPLICATES * 2 / 24:.1f} days)")
    print("\nResults will be saved continuously.")
    print("You can stop and resume by commenting out completed subjects.\n")

    # Run the complete study
    results = run_complete_study(n_subjects=N_SUBJECTS, n_replicates=N_REPLICATES)

    print("\n🎉 Complete study finished!")
    print("Check these files:")
    print("  • complete_study_results.json - Full data")
    print("  • final_report.txt - Summary report")
    print("  • final_results.png - Visualizations")
    print("  • subject_XX_intermediate.json - Per-subject data")