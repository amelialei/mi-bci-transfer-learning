"""
CROSS-SESSION TRANSFER LEARNING - ADAPTED FROM PHASE 1 INFRASTRUCTURE
======================================================================

Extends phase1_infrastructure.py to support within-subject cross-session transfer.

Tests: "Can I train on yesterday's session and transfer to today's session?"

Authors: Based on Hetvi Gandhi + Hridyanshu's Phase 1 infrastructure
Adapted for: Cross-session experiments

Usage:
    from cross_session_infrastructure import CrossSessionExperimentRunner

    runner = CrossSessionExperimentRunner("D:\\Documents 1 Jan 2026\\BCICIV")
    result = runner.run_cross_session_sample_efficiency(
        subject_id=1,
        data_fraction=0.2,
        use_transfer=True
    )
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict
import mne

# Import from existing infrastructure
from phase1_infrastructure import (
    EEGPreprocessor,
    EEGNet,
    train_model,
    create_train_val_test_splits
)


# ============================================================================
# CROSS-SESSION DATA LOADER (Extension of GrazDataLoader)
# ============================================================================

class CrossSessionDataLoader:
    """
    Extension of GrazDataLoader to support session-specific loading.

    BCI Competition IV 2a structure:
    - A01T.gdf = Subject 1, Session 1 (Training day)
    - A01E.gdf = Subject 1, Session 2 (Evaluation day)
    - Sessions recorded on different days
    """

    def __init__(self, data_path: str):
        """
        Initialize cross-session data loader.

        Args:
            data_path: Path to directory containing GDF files
        """
        self.data_path = Path(data_path)
        self.sampling_rate = 250  # Hz
        self.preprocessor = EEGPreprocessor()

    def load_session(self,
                     subject_id: int,
                     session: str = 'T',
                     hands_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load specific session for a subject.

        Args:
            subject_id: Subject number (1-9)
            session: 'T' for Session 1 (Training), 'E' for Session 2 (Evaluation)
            hands_only: If True, only load left/right hand trials

        Returns:
            data: EEG data (n_trials, n_channels, n_timepoints)
            labels: Trial labels (n_trials,)
        """
        # Construct filename
        filename = f"A{subject_id:02d}{session}.gdf"
        filepath = self.data_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Session file not found: {filepath}")

        # Load with MNE
        raw = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
        events, event_dict = mne.events_from_annotations(raw, verbose=False)

        # Select 22 EEG channels
        raw.pick(raw.ch_names[:22])

        # Motor imagery event IDs: 769=left, 770=right, 771=feet, 772=tongue
        if hands_only:
            mi_event_ids = {key: val for key, val in event_dict.items()
                            if key in ['769', '770']}
        else:
            mi_event_ids = {key: val for key, val in event_dict.items()
                            if key in ['769', '770', '771', '772']}

        # Create epochs
        epochs = mne.Epochs(raw, events, event_id=mi_event_ids,
                            tmin=0.5, tmax=2.5, baseline=None,
                            preload=True, verbose=False,
                            reject_by_annotation=True,
                            event_repeated='drop')

        data = epochs.get_data()  # (n_trials, n_channels, n_timepoints)
        labels = epochs.events[:, -1]

        return data, labels


# ============================================================================
# CROSS-SESSION EXPERIMENT RUNNER
# ============================================================================

class CrossSessionExperimentRunner:
    """
    Run within-subject cross-session transfer learning experiments.

    Experiments:
    1. Train on Session 1 (Day 1)
    2. Transfer to Session 2 (Different Day) with varying amounts of data
    3. Compare to baseline (train from scratch on Session 2)

    This tests: "Can I reuse yesterday's calibration?"
    """

    def __init__(self, data_path: str):
        """
        Initialize experiment runner.

        Args:
            data_path: Path to BCI Competition IV 2a data
        """
        self.data_path = data_path
        self.data_loader = CrossSessionDataLoader(data_path)
        self.preprocessor = EEGPreprocessor()

    def run_cross_session_sample_efficiency(self,
                                            subject_id: int,
                                            data_fraction: float = 1.0,
                                            use_transfer: bool = False,
                                            random_seed: int = 42,
                                            hands_only: bool = False) -> Dict:
        """
        Run within-subject cross-session transfer with limited Session 2 data.

        Workflow:
        1. Load Session 1 (Training day) - full data
        2. Load Session 2 (Evaluation day) - subsample by data_fraction
        3. If use_transfer: Train on Session 1, fine-tune on Session 2
        4. If baseline: Train from scratch on Session 2
        5. Evaluate on held-out Session 2 data

        Args:
            subject_id: Subject number (1-9)
            data_fraction: Fraction of Session 2 data to use (0.2 = 20%, 1.0 = 100%)
            use_transfer: If True, use cross-session transfer learning
            random_seed: Random seed for reproducibility
            hands_only: If True, only use left/right hand (2 classes)

        Returns:
            results: Dictionary with accuracy and metadata
        """
        # Set seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        n_classes = 2 if hands_only else 4

        # Load Session 1 (source session)
        data_session1, labels_session1 = self.data_loader.load_session(
            subject_id, session='T', hands_only=hands_only
        )
        data_session1 = self.preprocessor.preprocess(data_session1)

        # Load Session 2 (target session)
        data_session2, labels_session2 = self.data_loader.load_session(
            subject_id, session='E', hands_only=hands_only
        )
        data_session2 = self.preprocessor.preprocess(data_session2)

        # Split Session 2 into train/test
        # We'll use data_fraction of Session 2 for training, rest for testing
        splits_session2 = create_train_val_test_splits(
            data_session2, labels_session2,
            val_size=0.15,  # Small validation set
            test_size=0.5,  # Large test set (to ensure we have held-out data)
            random_state=random_seed
        )

        # Subsample Session 2 training data by data_fraction
        X_train_full, y_train_full = splits_session2['train']
        if data_fraction < 1.0:
            n_samples = max(1, int(len(X_train_full) * data_fraction))
            indices = np.random.choice(len(X_train_full), n_samples, replace=False)
            X_train_session2 = X_train_full[indices]
            y_train_session2 = y_train_full[indices]
        else:
            X_train_session2 = X_train_full
            y_train_session2 = y_train_full

        X_val_session2, y_val_session2 = splits_session2['val']
        X_test_session2, y_test_session2 = splits_session2['test']

        if use_transfer:
            # CROSS-SESSION TRANSFER LEARNING

            # Step 1: Train model on Session 1 (full data)
            splits_session1 = create_train_val_test_splits(
                data_session1, labels_session1, random_state=random_seed
            )

            source_model = EEGNet(n_channels=22, n_classes=n_classes)
            train_model(
                source_model,
                splits_session1['train'][0], splits_session1['train'][1],
                splits_session1['val'][0], splits_session1['val'][1],
                epochs=150,
                verbose=False
            )

            # Step 2: Create target model (same architecture, copy weights)
            target_model = EEGNet(n_channels=22, n_classes=n_classes)
            target_model.load_state_dict(source_model.state_dict())

            # Step 3: Fine-tune on Session 2 data (if any)
            if data_fraction > 0:
                train_results = train_model(
                    target_model,
                    X_train_session2, y_train_session2,
                    X_val_session2, y_val_session2,
                    epochs=150,
                    verbose=False
                )
            else:
                # Pure transfer (no Session 2 training data)
                # Evaluate directly on validation set
                target_model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_session2)
                    outputs = target_model(X_val_tensor)
                    _, predicted = torch.max(outputs, 1)
                    y_val_tensor = torch.LongTensor(y_val_session2)
                    val_acc = (predicted == y_val_tensor).float().mean().item()
                train_results = {'best_val_acc': val_acc}

        else:
            # BASELINE: Train from scratch on Session 2

            if data_fraction > 0:
                model = EEGNet(n_channels=22, n_classes=n_classes)
                train_results = train_model(
                    model,
                    X_train_session2, y_train_session2,
                    X_val_session2, y_val_session2,
                    epochs=150,
                    verbose=False
                )
                target_model = model
            else:
                # 0% data baseline = random model
                target_model = EEGNet(n_channels=22, n_classes=n_classes)
                target_model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_session2)
                    outputs = target_model(X_val_tensor)
                    _, predicted = torch.max(outputs, 1)
                    y_val_tensor = torch.LongTensor(y_val_session2)
                    val_acc = (predicted == y_val_tensor).float().mean().item()
                train_results = {'best_val_acc': val_acc}

        # Final evaluation on test set
        target_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_session2)
            outputs = target_model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            y_test_tensor = torch.LongTensor(y_test_session2)
            test_acc = (predicted == y_test_tensor).float().mean().item()

        return {
            'subject_id': subject_id,
            'data_fraction': data_fraction,
            'use_transfer': use_transfer,
            'random_seed': random_seed,
            'hands_only': hands_only,
            'accuracy': train_results['best_val_acc'],  # Validation accuracy for consistency with Eric's experiments
            'test_accuracy': test_acc,  # Also report test accuracy
            'n_training_samples': len(X_train_session2),
            'n_test_samples': len(X_test_session2),
            'condition': 'transfer' if use_transfer else 'baseline',
            'experiment_type': 'cross_session'
        }


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_cross_session_infrastructure():
    """
    Quick test to verify cross-session infrastructure works.
    """
    print("=" * 70)
    print("TESTING CROSS-SESSION INFRASTRUCTURE")
    print("=" * 70)

    data_path = "D:\\Documents 1 Jan 2026\\BCICIV"

    print("\n1. Testing data loading...")
    try:
        loader = CrossSessionDataLoader(data_path)

        # Load both sessions
        data_s1, labels_s1 = loader.load_session(1, session='T')
        print(f"   ✅ Session 1: {data_s1.shape} - {len(np.unique(labels_s1))} classes")

        data_s2, labels_s2 = loader.load_session(1, session='E')
        print(f"   ✅ Session 2: {data_s2.shape} - {len(np.unique(labels_s2))} classes")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    print("\n2. Testing cross-session experiment...")
    try:
        runner = CrossSessionExperimentRunner(data_path)

        # Test baseline
        result = runner.run_cross_session_sample_efficiency(
            subject_id=1,
            data_fraction=0.5,
            use_transfer=False,
            random_seed=42
        )
        print(f"   ✅ Baseline (50% data): {result['accuracy'] * 100:.2f}% validation")
        print(f"                           {result['test_accuracy'] * 100:.2f}% test")

        # Test transfer
        result = runner.run_cross_session_sample_efficiency(
            subject_id=1,
            data_fraction=0.5,
            use_transfer=True,
            random_seed=42
        )
        print(f"   ✅ Transfer (50% data): {result['accuracy'] * 100:.2f}% validation")
        print(f"                           {result['test_accuracy'] * 100:.2f}% test")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - READY FOR FULL EXPERIMENTS")
    print("=" * 70)
    print("\nYou can now run:")
    print("  python run_cross_session_experiments.py")

    return True


if __name__ == '__main__':
    test_cross_session_infrastructure()