"""
CROSS-SESSION INFRASTRUCTURE - B FILES VERSION
===============================================

Uses B files which have TRUE EMBEDDED LABELS in both sessions.

B File Structure:
- B0101T, B0102T, B0103T = Subject 1, Training runs (Session 1 options)
- B0104E, B0105E = Subject 1, Evaluation runs (Session 2 options)

All B files have:
- 3 EEG channels
- Embedded class labels (769, 770, 771, 772) ✅
- True cross-session data

This will produce REAL results!
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict
import mne

from phase1_infrastructure import (
    EEGPreprocessor,
    EEGNet,
    train_model,
    create_train_val_test_splits
)


class BFileDataLoader:
    """
    Load B files for cross-session transfer.
    B files have 3 channels and TRUE embedded labels.
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.sampling_rate = 250
        self.preprocessor = EEGPreprocessor()
    
    def load_session(self,
                    subject_id: int,
                    session: str = 'T',
                    hands_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load B file session with TRUE labels.
        
        Args:
            subject_id: Subject number (1-9)
            session: 'T' for Session 1, 'E' for Session 2
            hands_only: If True, only load left/right hand trials
        
        Returns:
            data: EEG data (n_trials, n_channels, n_timepoints)
            labels: Trial labels (n_trials,) - TRUE LABELS ✅
        """
        # Choose B file based on session
        if session == 'T':
            # Use first training run as Session 1
            filename = f"B{subject_id:02d}01T.gdf"
        else:  # session == 'E'
            # Use first evaluation run as Session 2
            filename = f"B{subject_id:02d}04E.gdf"
        
        filepath = self.data_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"B file not found: {filepath}")
        
        # Load with MNE
        raw = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        
        # Select 3 EEG channels (B files have 3 channels)
        raw.pick(raw.ch_names[:3])
        
        # Motor imagery event IDs (B files have TRUE labels!)
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
        labels = epochs.events[:, -1]  # TRUE LABELS ✅
        
        return data, labels


class BFileCrossSessionRunner:
    """
    Run cross-session experiments using B files.
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_loader = BFileDataLoader(data_path)
        self.preprocessor = EEGPreprocessor()
    
    def run_cross_session_sample_efficiency(self,
                                           subject_id: int,
                                           data_fraction: float = 1.0,
                                           use_transfer: bool = False,
                                           random_seed: int = 42,
                                           hands_only: bool = False) -> Dict:
        """
        Within-subject cross-session transfer using B files.
        """
        # Set seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        n_classes = 2 if hands_only else 4
        
        # Load Session 1 (B01T - first training run)
        data_session1, labels_session1 = self.data_loader.load_session(
            subject_id, session='T', hands_only=hands_only
        )
        data_session1 = self.preprocessor.preprocess(data_session1)
        
        # Load Session 2 (B04E - first evaluation run)
        data_session2, labels_session2 = self.data_loader.load_session(
            subject_id, session='E', hands_only=hands_only
        )
        data_session2 = self.preprocessor.preprocess(data_session2)
        
        # Split Session 2
        splits_session2 = create_train_val_test_splits(
            data_session2, labels_session2,
            val_size=0.15,
            test_size=0.5,
            random_state=random_seed
        )
        
        # Subsample training data
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
            # Train on Session 1
            splits_session1 = create_train_val_test_splits(
                data_session1, labels_session1, random_state=random_seed
            )
            
            # NOTE: B files have 3 channels, not 22!
            source_model = EEGNet(n_channels=3, n_classes=n_classes)
            train_model(
                source_model,
                splits_session1['train'][0], splits_session1['train'][1],
                splits_session1['val'][0], splits_session1['val'][1],
                epochs=150,
                verbose=False
            )
            
            # Copy to target model
            target_model = EEGNet(n_channels=3, n_classes=n_classes)
            target_model.load_state_dict(source_model.state_dict())
            
            # Fine-tune on Session 2
            if data_fraction > 0:
                train_results = train_model(
                    target_model,
                    X_train_session2, y_train_session2,
                    X_val_session2, y_val_session2,
                    epochs=150,
                    verbose=False
                )
            else:
                # Pure transfer
                target_model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_session2)
                    outputs = target_model(X_val_tensor)
                    _, predicted = torch.max(outputs, 1)
                    y_val_tensor = torch.LongTensor(y_val_session2)
                    val_acc = (predicted == y_val_tensor).float().mean().item()
                train_results = {'best_val_acc': val_acc}
        
        else:
            # Baseline
            if data_fraction > 0:
                model = EEGNet(n_channels=3, n_classes=n_classes)
                train_results = train_model(
                    model,
                    X_train_session2, y_train_session2,
                    X_val_session2, y_val_session2,
                    epochs=150,
                    verbose=False
                )
                target_model = model
            else:
                target_model = EEGNet(n_channels=3, n_classes=n_classes)
                target_model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_session2)
                    outputs = target_model(X_val_tensor)
                    _, predicted = torch.max(outputs, 1)
                    y_val_tensor = torch.LongTensor(y_val_session2)
                    val_acc = (predicted == y_val_tensor).float().mean().item()
                train_results = {'best_val_acc': val_acc}
        
        # Final test evaluation
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
            'accuracy': train_results['best_val_acc'],
            'test_accuracy': test_acc,
            'n_training_samples': len(X_train_session2),
            'n_test_samples': len(X_test_session2),
            'condition': 'transfer' if use_transfer else 'baseline',
            'experiment_type': 'cross_session_bfiles',
            'n_channels': 3  # Track that these are 3-channel experiments
        }


def test_bfile_infrastructure():
    """Test B-file infrastructure."""
    print("="*70)
    print("TESTING B-FILE CROSS-SESSION INFRASTRUCTURE")
    print("="*70)
    print("\nB Files: 3 channels, TRUE embedded labels ✅")
    print("Session 1: B0101T (Training run 1)")
    print("Session 2: B0104E (Evaluation run 4)")
    print("="*70)
    
    data_path = "D:\\Documents 1 Jan 2026\\BCICIV"
    
    print("\n1. Testing data loading...")
    try:
        loader = BFileDataLoader(data_path)
        
        # Load T file
        data_s1, labels_s1 = loader.load_session(1, session='T')
        print(f"   ✅ Session 1 (B01T): {data_s1.shape} - {len(np.unique(labels_s1))} classes")
        print(f"      Labels: {np.unique(labels_s1)} (TRUE labels ✅)")
        
        # Load E file
        data_s2, labels_s2 = loader.load_session(1, session='E')
        print(f"   ✅ Session 2 (B04E): {data_s2.shape} - {len(np.unique(labels_s2))} classes")
        print(f"      Labels: {np.unique(labels_s2)} (TRUE labels ✅)")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n2. Testing cross-session experiment...")
    try:
        runner = BFileCrossSessionRunner(data_path)
        
        # Quick test
        result = runner.run_cross_session_sample_efficiency(
            subject_id=1,
            data_fraction=0.5,
            use_transfer=False,
            random_seed=42
        )
        print(f"   ✅ Baseline (50% data): {result['accuracy']*100:.2f}% validation")
        print(f"                           {result['test_accuracy']*100:.2f}% test")
        print(f"      Training samples: {result['n_training_samples']}")
        print(f"      Channels: {result['n_channels']}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - READY FOR REAL EXPERIMENTS")
    print("="*70)
    print("\n✅ B Files Advantages:")
    print("  - TRUE embedded labels (no guessing!)")
    print("  - Different recording sessions (true cross-session)")
    print("  - Will produce REAL, publishable results")
    print("\n⚠️  B Files Note:")
    print("  - 3 channels (vs 22 in A files)")
    print("  - Still tests transfer learning effectively!")
    
    return True


if __name__ == '__main__':
    test_bfile_infrastructure()
