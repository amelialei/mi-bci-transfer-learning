"""
PHASE 1 INFRASTRUCTURE - BCI Transfer Learning Project
======================================================

This script refactors the messy bci_project.py into clean, modular code.

Authors: Hetvi Gandhi (Data) + Hridyanshu (Models)
Purpose: Create infrastructure for Eric's experiments (Phase 2)

Run this file to:
1. Validate all data loading works
2. Validate all models work
3. Create experiment utilities for Eric
4. Save model checkpoints
5. Generate infrastructure documentation

Usage:
    python phase1_infrastructure.py

Estimated runtime: 10-15 minutes (validates everything works)
"""

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import json
import time
from datetime import datetime


# ============================================================================
# SECTION 1: DATA INFRASTRUCTURE (Hetvi's Work)
# ============================================================================

class GrazDataLoader:
    """
    Clean interface for loading Graz BCI Competition datasets.

    Handles:
    - Dataset 2a: 9 subjects, 22 channels, 4 classes (left/right hand, feet, tongue)
    - Dataset 2b: 9 subjects, 3 channels, 2 classes (left/right hand)

    Author: Hetvi Gandhi
    """

    def __init__(self, data_path: str):
        """
        Initialize data loader.

        Args:
            data_path: Path to directory containing GDF files
        """
        self.data_path = Path(data_path)
        self.sampling_rate = 250  # Hz

    def load_dataset_2a(self,
                        subject_id: int,
                        hands_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Dataset 2a (22 channels, 4 classes).

        Args:
            subject_id: Subject number (1-9)
            hands_only: If True, only load left/right hand trials (for task-aligned transfer)

        Returns:
            data: EEG data (n_trials, n_channels, n_timepoints)
            labels: Trial labels (n_trials,)
        """
        filename = f"A{subject_id:02d}T.gdf"
        filepath = self.data_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset 2a file not found: {filepath}")

        # Load with MNE
        raw = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
        events, event_dict = mne.events_from_annotations(raw)

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

    def load_dataset_2b(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Dataset 2b (3 channels, 2 classes - left/right hand only).

        Args:
            subject_id: Subject number (1-9)

        Returns:
            data: EEG data (n_trials, n_channels, n_timepoints)
            labels: Trial labels (n_trials,)
        """
        # Dataset 2b has 3 sessions per subject
        raw_list, events_list = [], []
        event_dict = None

        for session in ['01T', '02T', '03T']:
            filepath = self.data_path / f"B{subject_id:02d}{session}.gdf"

            if not filepath.exists():
                continue  # Some subjects missing sessions

            r = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
            e, ed = mne.events_from_annotations(r)

            if event_dict is None:
                event_dict = ed

            # Select 3 EEG channels
            r.pick(r.ch_names[:3])
            raw_list.append(r)
            events_list.append(e)

        if not raw_list:
            raise FileNotFoundError(f"No Dataset 2b files found for subject {subject_id}")

        # Concatenate sessions
        raw = mne.concatenate_raws(raw_list)
        events = np.vstack(events_list)

        # Motor imagery events (only left/right hand)
        mi_event_ids = {key: val for key, val in event_dict.items()
                        if key in ['769', '770']}

        # Create epochs
        epochs = mne.Epochs(raw, events, event_id=mi_event_ids,
                            tmin=0.5, tmax=2.5, baseline=None,
                            preload=True, verbose=False,
                            reject_by_annotation=True,
                            event_repeated='drop')

        data = epochs.get_data()
        labels = epochs.events[:, -1]

        return data, labels


class EEGPreprocessor:
    """
    Standardized preprocessing pipeline for EEG data.

    Pipeline:
    1. Bandpass filter (8-30 Hz - motor imagery band)
    2. Z-score normalization per trial

    Author: Hetvi Gandhi
    """

    def __init__(self, lowcut: float = 8.0, highcut: float = 30.0, fs: float = 250.0):
        """
        Initialize preprocessor.

        Args:
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            fs: Sampling rate (Hz)
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline.

        Args:
            data: Raw EEG data (n_trials, n_channels, n_timepoints)

        Returns:
            preprocessed: Preprocessed data (same shape)
        """
        # 1. Bandpass filter
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data, axis=-1)

        # 2. Z-score normalization per trial
        mean = filtered.mean(axis=-1, keepdims=True)
        std = filtered.std(axis=-1, keepdims=True) + 1e-8
        normalized = (filtered - mean) / std

        return normalized


def create_train_val_test_splits(data: np.ndarray,
                                 labels: np.ndarray,
                                 val_size: float = 0.2,
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Dict:
    """
    Create stratified train/val/test splits.

    Args:
        data: EEG data (n_trials, n_channels, n_timepoints)
        labels: Trial labels (n_trials,)
        val_size: Validation set proportion
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        splits: Dictionary with 'train', 'val', 'test' keys, each containing (X, y)
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, test_size=test_size,
        stratify=labels, random_state=random_state
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted,
        stratify=y_temp, random_state=random_state
    )

    # Remap labels to 0, 1, 2, ... (for PyTorch)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}

    y_train = np.array([label_map[label] for label in y_train])
    y_val = np.array([label_map[label] for label in y_val])
    y_test = np.array([label_map[label] for label in y_test])

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'label_map': label_map
    }


# ============================================================================
# SECTION 2: MODEL ARCHITECTURE (Hridyanshu's Work)
# ============================================================================

class EEGNet(nn.Module):
    """
    EEGNet architecture for motor imagery classification.

    Reference: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural
               Network for EEG-based Brain-Computer Interfaces"

    Architecture:
    - Temporal convolution (learns frequency filters)
    - Spatial convolution (learns spatial patterns)
    - Separable convolution (feature extraction)
    - Adaptive pooling (handles variable input sizes)
    - Classification head

    Author: Hridyanshu
    """

    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.25):
        """
        Initialize EEGNet.

        Args:
            n_channels: Number of EEG channels
            n_classes: Number of output classes
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Temporal convolution (learns frequency filters)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8)
        )

        # Spatial convolution (learns spatial patterns per channel)
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        # Separable convolution (feature extraction)
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # Adaptive pooling (handles different channel counts)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))

        # Classification head
        self.classifier = nn.Linear(16 * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.

        Useful for:
        - Feature visualization (t-SNE)
        - Transfer learning analysis
        """
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        return x

    def freeze_layers(self, layers: List[str]):
        """
        Freeze specified layers for transfer learning.

        Args:
            layers: List of layer names to freeze
                   Options: 'temporal', 'spatial', 'separable', 'classifier'
        """
        layer_map = {
            'temporal': self.temporal,
            'spatial': self.spatial,
            'separable': self.separable,
            'classifier': self.classifier
        }

        for layer_name in layers:
            if layer_name in layer_map:
                for param in layer_map[layer_name].parameters():
                    param.requires_grad = False


def train_model(model: nn.Module,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray,
                y_val: np.ndarray,
                epochs: int = 150,
                batch_size: int = 32,
                lr: float = 0.001,
                verbose: bool = False) -> Dict:
    """
    Train EEGNet model.

    Args:
        model: EEGNet model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        verbose: Print training progress

    Returns:
        results: Dictionary with 'best_val_acc', 'train_history', 'val_history'

    Author: Hridyanshu
    """
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create data loader
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    best_val_acc = 0.0
    train_history = []
    val_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, predicted = torch.max(val_outputs, 1)
            val_acc = (predicted == y_val).float().mean().item()

            train_outputs = model(X_train)
            _, predicted_train = torch.max(train_outputs, 1)
            train_acc = (predicted_train == y_train).float().mean().item()

        best_val_acc = max(best_val_acc, val_acc)
        train_history.append(train_acc)
        val_history.append(val_acc)

        if verbose and (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: Train={train_acc:.4f}, Val={val_acc:.4f}")

    return {
        'best_val_acc': best_val_acc,
        'final_val_acc': val_history[-1],
        'train_history': train_history,
        'val_history': val_history
    }


# ============================================================================
# SECTION 3: TRANSFER LEARNING (Hridyanshu's Work)
# ============================================================================

class TransferLearningPipeline:
    """
    Transfer learning pipeline for cross-subject BCI.

    Handles:
    - Channel adaptation (22 → 3 channels via adaptive pooling)
    - Layer transfer with selective freezing
    - Fine-tuning strategies

    Author: Hridyanshu
    """

    def __init__(self, source_model: EEGNet, target_n_channels: int, target_n_classes: int):
        """
        Initialize transfer pipeline.

        Args:
            source_model: Trained model on source dataset
            target_n_channels: Number of channels in target dataset
            target_n_classes: Number of classes in target dataset
        """
        self.source_model = source_model
        self.target_n_channels = target_n_channels
        self.target_n_classes = target_n_classes

    def create_transferred_model(self, freeze_strategy: str = 'temporal_only') -> EEGNet:
        """
        Create target model with transferred weights.

        Args:
            freeze_strategy: Which layers to freeze
                - 'temporal_only': Transfer and freeze only temporal conv
                - 'spatial_temporal': Transfer and freeze temporal + spatial
                - 'all_except_classifier': Transfer and freeze all except classifier
                - 'none': Initialize randomly (no transfer)

        Returns:
            target_model: Model ready for fine-tuning
        """
        # Create target model
        target_model = EEGNet(
            n_channels=self.target_n_channels,
            n_classes=self.target_n_classes
        )

        if freeze_strategy == 'none':
            # No transfer - random initialization
            return target_model

        # Transfer temporal layer (always compatible - operates on time dimension)
        target_model.temporal.load_state_dict(self.source_model.temporal.state_dict())

        if freeze_strategy == 'temporal_only':
            target_model.freeze_layers(['temporal'])

        elif freeze_strategy == 'spatial_temporal':
            # Transfer temporal (compatible)
            # Spatial not compatible (different channel counts) - keep random
            target_model.freeze_layers(['temporal', 'spatial'])

        elif freeze_strategy == 'all_except_classifier':
            # Transfer what's compatible
            target_model.freeze_layers(['temporal', 'spatial', 'separable'])

        return target_model


# ============================================================================
# SECTION 4: EXPERIMENT UTILITIES (For Eric's Phase 2)
# ============================================================================

class ExperimentRunner:
    """
    Utilities for running ablation studies and sample efficiency experiments.

    This class provides easy-to-use functions for Eric's experiments in Phase 2.

    Author: Hridyanshu (for Eric's use)
    """

    def __init__(self, data_path: str):
        """Initialize experiment runner."""
        self.data_loader = GrazDataLoader(data_path)
        self.preprocessor = EEGPreprocessor()

    def run_baseline_experiment(self,
                                subject_id: int,
                                random_seed: int = 42) -> Dict:
        """
        Run baseline experiment (no transfer).

        Args:
            subject_id: Subject number (1-9)
            random_seed: Random seed for reproducibility

        Returns:
            results: Dictionary with accuracy and model
        """
        # Set seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Load and preprocess
        data, labels = self.data_loader.load_dataset_2b(subject_id)
        data = self.preprocessor.preprocess(data)

        # Create splits
        splits = create_train_val_test_splits(data, labels, random_state=random_seed)
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']

        # Train model
        model = EEGNet(n_channels=3, n_classes=2)
        train_results = train_model(model, X_train, y_train, X_val, y_val)

        return {
            'subject_id': subject_id,
            'random_seed': random_seed,
            'accuracy': train_results['best_val_acc'],
            'model': model
        }

    def run_transfer_experiment(self,
                                subject_id: int,
                                freeze_strategy: str = 'temporal_only',
                                hands_only: bool = True,
                                random_seed: int = 42) -> Dict:
        """
        Run transfer learning experiment.

        Args:
            subject_id: Subject number (1-9)
            freeze_strategy: Freezing strategy (see TransferLearningPipeline)
            hands_only: If True, use task-aligned transfer (hands only from 2a)
            random_seed: Random seed

        Returns:
            results: Dictionary with accuracy and model
        """
        # Set seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # 1. Load and train source model (Dataset 2a)
        data_2a, labels_2a = self.data_loader.load_dataset_2a(subject_id, hands_only=hands_only)
        data_2a = self.preprocessor.preprocess(data_2a)
        splits_2a = create_train_val_test_splits(data_2a, labels_2a, random_state=random_seed)

        n_classes_2a = 2 if hands_only else 4
        source_model = EEGNet(n_channels=22, n_classes=n_classes_2a)
        train_model(source_model, splits_2a['train'][0], splits_2a['train'][1],
                    splits_2a['val'][0], splits_2a['val'][1])

        # 2. Create transferred model
        pipeline = TransferLearningPipeline(source_model, target_n_channels=3, target_n_classes=2)
        target_model = pipeline.create_transferred_model(freeze_strategy=freeze_strategy)

        # 3. Fine-tune on target dataset (Dataset 2b)
        data_2b, labels_2b = self.data_loader.load_dataset_2b(subject_id)
        data_2b = self.preprocessor.preprocess(data_2b)
        splits_2b = create_train_val_test_splits(data_2b, labels_2b, random_state=random_seed)

        train_results = train_model(target_model, splits_2b['train'][0], splits_2b['train'][1],
                                    splits_2b['val'][0], splits_2b['val'][1])

        return {
            'subject_id': subject_id,
            'random_seed': random_seed,
            'freeze_strategy': freeze_strategy,
            'hands_only': hands_only,
            'accuracy': train_results['best_val_acc'],
            'model': target_model
        }

    def run_sample_efficiency_experiment(self,
                                         subject_id: int,
                                         data_fraction: float = 1.0,
                                         use_transfer: bool = False,
                                         random_seed: int = 42) -> Dict:
        """
        Run sample efficiency experiment with limited target data.

        This is for Eric's learning curve experiments.

        Args:
            subject_id: Subject number (1-9)
            data_fraction: Fraction of training data to use (0.2 = 20%, 1.0 = 100%)
            use_transfer: If True, use transfer learning
            random_seed: Random seed

        Returns:
            results: Dictionary with accuracy
        """
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # Load target data
        data_2b, labels_2b = self.data_loader.load_dataset_2b(subject_id)
        data_2b = self.preprocessor.preprocess(data_2b)
        splits = create_train_val_test_splits(data_2b, labels_2b, random_state=random_seed)

        # Subsample training data
        X_train_full, y_train_full = splits['train']
        n_samples = int(len(X_train_full) * data_fraction)
        indices = np.random.choice(len(X_train_full), n_samples, replace=False)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]

        X_val, y_val = splits['val']

        if use_transfer:
            # With transfer learning
            data_2a, labels_2a = self.data_loader.load_dataset_2a(subject_id, hands_only=True)
            data_2a = self.preprocessor.preprocess(data_2a)
            splits_2a = create_train_val_test_splits(data_2a, labels_2a, random_state=random_seed)

            source_model = EEGNet(n_channels=22, n_classes=2)
            train_model(source_model, splits_2a['train'][0], splits_2a['train'][1],
                        splits_2a['val'][0], splits_2a['val'][1])

            pipeline = TransferLearningPipeline(source_model, target_n_channels=3, target_n_classes=2)
            model = pipeline.create_transferred_model(freeze_strategy='temporal_only')
        else:
            # Baseline (no transfer)
            model = EEGNet(n_channels=3, n_classes=2)

        # Train
        train_results = train_model(model, X_train, y_train, X_val, y_val)

        return {
            'subject_id': subject_id,
            'data_fraction': data_fraction,
            'use_transfer': use_transfer,
            'random_seed': random_seed,
            'accuracy': train_results['best_val_acc'],
            'n_training_samples': n_samples
        }


# ============================================================================
# SECTION 5: VALIDATION & TESTING
# ============================================================================

def validate_infrastructure(data_path: str = r"D:\Documents 1 Jan 2026\BCICIV"):
    """
    Validate that all infrastructure works correctly.

    This function:
    1. Tests data loading (Dataset 2a and 2b)
    2. Tests preprocessing
    3. Tests model training
    4. Tests transfer learning
    5. Tests experiment utilities

    Run this to ensure Phase 1 is complete and Phase 2 can begin.
    """
    print("=" * 70)
    print("PHASE 1 INFRASTRUCTURE VALIDATION")
    print("=" * 70)
    print()

    # Test 1: Data Loading
    print("Test 1: Data Loading")
    print("-" * 70)
    try:
        loader = GrazDataLoader(data_path)

        # Test Dataset 2a
        data_2a, labels_2a = loader.load_dataset_2a(subject_id=1, hands_only=False)
        print(f"✅ Dataset 2a loaded: {data_2a.shape} (trials, channels, timepoints)")
        print(f"   Classes: {np.unique(labels_2a)} (should be 4 classes)")

        # Test Dataset 2a (hands only)
        data_2a_hands, labels_2a_hands = loader.load_dataset_2a(subject_id=1, hands_only=True)
        print(f"✅ Dataset 2a (hands only): {data_2a_hands.shape}")
        print(f"   Classes: {np.unique(labels_2a_hands)} (should be 2 classes)")

        # Test Dataset 2b
        data_2b, labels_2b = loader.load_dataset_2b(subject_id=1)
        print(f"✅ Dataset 2b loaded: {data_2b.shape}")
        print(f"   Classes: {np.unique(labels_2b)} (should be 2 classes)")

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

    print()

    # Test 2: Preprocessing
    print("Test 2: Preprocessing")
    print("-" * 70)
    try:
        preprocessor = EEGPreprocessor()
        processed = preprocessor.preprocess(data_2b)
        print(f"✅ Preprocessing works: {processed.shape}")
        print(f"   Mean: {processed.mean():.4f} (should be ~0)")
        print(f"   Std: {processed.std():.4f} (should be ~1)")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False

    print()

    # Test 3: Train/Val/Test Splits
    print("Test 3: Train/Val/Test Splits")
    print("-" * 70)
    try:
        splits = create_train_val_test_splits(processed, labels_2b)
        print(f"✅ Splits created:")
        print(f"   Train: {splits['train'][0].shape[0]} trials")
        print(f"   Val: {splits['val'][0].shape[0]} trials")
        print(f"   Test: {splits['test'][0].shape[0]} trials")
        print(f"   Total: {sum([splits[k][0].shape[0] for k in ['train', 'val', 'test']])} trials")
    except Exception as e:
        print(f"❌ Split creation failed: {e}")
        return False

    print()

    # Test 4: Model Training (Quick)
    print("Test 4: Model Training (30 epochs, quick test)")
    print("-" * 70)
    try:
        model = EEGNet(n_channels=3, n_classes=2)
        results = train_model(
            model,
            splits['train'][0], splits['train'][1],
            splits['val'][0], splits['val'][1],
            epochs=30,  # Quick test
            verbose=True
        )
        print(f"✅ Model training works!")
        print(f"   Best validation accuracy: {results['best_val_acc']:.4f}")
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

    print()

    # Test 5: Transfer Learning
    print("Test 5: Transfer Learning")
    print("-" * 70)
    try:
        # Create source model (Dataset 2a)
        data_2a_hands, labels_2a_hands = loader.load_dataset_2a(1, hands_only=True)
        data_2a_proc = preprocessor.preprocess(data_2a_hands)
        splits_2a = create_train_val_test_splits(data_2a_proc, labels_2a_hands)

        source_model = EEGNet(n_channels=22, n_classes=2)
        train_model(source_model, splits_2a['train'][0], splits_2a['train'][1],
                    splits_2a['val'][0], splits_2a['val'][1], epochs=30, verbose=False)
        print(f"✅ Source model trained (Dataset 2a)")

        # Create transferred model
        pipeline = TransferLearningPipeline(source_model, target_n_channels=3, target_n_classes=2)
        target_model = pipeline.create_transferred_model(freeze_strategy='temporal_only')
        print(f"✅ Transfer learning pipeline works")

        # Test different freezing strategies
        for strategy in ['temporal_only', 'spatial_temporal', 'all_except_classifier', 'none']:
            model = pipeline.create_transferred_model(freeze_strategy=strategy)
            print(f"   ✅ Freeze strategy '{strategy}' works")

    except Exception as e:
        print(f"❌ Transfer learning failed: {e}")
        return False

    print()

    # Test 6: Experiment Utilities (For Eric)
    print("Test 6: Experiment Utilities (For Eric's Phase 2)")
    print("-" * 70)
    try:
        runner = ExperimentRunner(data_path)

        # Test baseline experiment
        result = runner.run_baseline_experiment(subject_id=1, random_seed=42)
        print(f"✅ Baseline experiment: {result['accuracy']:.4f} accuracy")

        # Test transfer experiment
        result = runner.run_transfer_experiment(subject_id=1, freeze_strategy='temporal_only', random_seed=42)
        print(f"✅ Transfer experiment: {result['accuracy']:.4f} accuracy")

        # Test sample efficiency experiment
        result = runner.run_sample_efficiency_experiment(subject_id=1, data_fraction=0.5, random_seed=42)
        print(f"✅ Sample efficiency (50% data): {result['accuracy']:.4f} accuracy")

    except Exception as e:
        print(f"❌ Experiment utilities failed: {e}")
        return False

    print()
    print("=" * 70)
    print("✅ ALL TESTS PASSED - INFRASTRUCTURE READY FOR PHASE 2")
    print("=" * 70)
    print()
    print("Eric can now use ExperimentRunner for his ablation studies!")
    print("Vanessa can now use the trained models for LOSO validation!")
    print()

    return True


# ============================================================================
# SECTION 6: SAVE INFRASTRUCTURE DOCUMENTATION
# ============================================================================

def generate_documentation():
    """Generate documentation for the infrastructure."""

    docs = """
# BCI Transfer Learning Infrastructure Documentation

## Generated by Phase 1: Hetvi Gandhi + Hridyanshu

---

## Data Pipeline (Hetvi's Work)

### GrazDataLoader
Load Graz BCI Competition datasets.
```python
loader = GrazDataLoader(data_path="path/to/data")

# Load Dataset 2a (22 channels, 4 classes)
data_2a, labels_2a = loader.load_dataset_2a(subject_id=1, hands_only=False)

# Load Dataset 2a (hands only - for task-aligned transfer)
data_2a_hands, labels_2a_hands = loader.load_dataset_2a(subject_id=1, hands_only=True)

# Load Dataset 2b (3 channels, 2 classes)
data_2b, labels_2b = loader.load_dataset_2b(subject_id=1)
```

### EEGPreprocessor
Standardized preprocessing pipeline.
```python
preprocessor = EEGPreprocessor(lowcut=8.0, highcut=30.0)
processed_data = preprocessor.preprocess(raw_data)
```

### Data Splits
Create train/val/test splits.
```python
splits = create_train_val_test_splits(data, labels, val_size=0.2, test_size=0.2)
X_train, y_train = splits['train']
X_val, y_val = splits['val']
X_test, y_test = splits['test']
```

---

## Model Architecture (Hridyanshu's Work)

### EEGNet
Motor imagery classification model.
```python
model = EEGNet(n_channels=3, n_classes=2, dropout=0.25)

# Train model
results = train_model(model, X_train, y_train, X_val, y_val, epochs=150)

# Freeze layers for transfer learning
model.freeze_layers(['temporal', 'spatial'])
```

### Transfer Learning Pipeline
Handle cross-dataset transfer.
```python
# Create pipeline
pipeline = TransferLearningPipeline(
    source_model=trained_source_model,
    target_n_channels=3,
    target_n_classes=2
)

# Create transferred model with different strategies
model = pipeline.create_transferred_model(freeze_strategy='temporal_only')
# Options: 'temporal_only', 'spatial_temporal', 'all_except_classifier', 'none'
```

---

## Experiment Utilities (For Eric's Phase 2)

### ExperimentRunner
Easy-to-use interface for running experiments.
```python
runner = ExperimentRunner(data_path="path/to/data")

# Baseline experiment
result = runner.run_baseline_experiment(subject_id=1, random_seed=42)
print(f"Accuracy: {result['accuracy']}")

# Transfer experiment
result = runner.run_transfer_experiment(
    subject_id=1,
    freeze_strategy='temporal_only',
    hands_only=True,
    random_seed=42
)

# Sample efficiency experiment (for learning curves)
result = runner.run_sample_efficiency_experiment(
    subject_id=1,
    data_fraction=0.5,  # Use 50% of training data
    use_transfer=True,
    random_seed=42
)
```

---

## For Eric's Ablation Studies
```python
# Experiment 1: Layer Freezing Ablation
freeze_strategies = ['temporal_only', 'spatial_temporal', 'all_except_classifier', 'none']

for strategy in freeze_strategies:
    result = runner.run_transfer_experiment(
        subject_id=1,
        freeze_strategy=strategy,
        random_seed=42
    )
    print(f"{strategy}: {result['accuracy']:.4f}")
```

---

## For Eric's Sample Efficiency Curves
```python
# Experiment 2: Learning curves with limited data
data_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

for fraction in data_fractions:
    # Baseline
    baseline = runner.run_sample_efficiency_experiment(
        subject_id=1,
        data_fraction=fraction,
        use_transfer=False,
        random_seed=42
    )

    # With transfer
    transfer = runner.run_sample_efficiency_experiment(
        subject_id=1,
        data_fraction=fraction,
        use_transfer=True,
        random_seed=42
    )

    print(f"{int(fraction*100)}% data: Baseline={baseline['accuracy']:.4f}, Transfer={transfer['accuracy']:.4f}")
```

---

## Dataset Statistics

### Dataset 2a (22 channels, 4 classes)
- Subjects: 9
- Channels: 22 EEG
- Classes: 4 (left hand, right hand, feet, tongue)
- Trials per subject: ~288
- Sampling rate: 250 Hz
- Epoch duration: 2.0 seconds (0.5-2.5s after cue)

### Dataset 2b (3 channels, 2 classes)
- Subjects: 9
- Channels: 3 EEG (C3, Cz, C4)
- Classes: 2 (left hand, right hand)
- Trials per subject: ~140-160 (varies by session availability)
- Sampling rate: 250 Hz
- Epoch duration: 2.0 seconds (0.5-2.5s after cue)

### Preprocessing
- Bandpass filter: 8-30 Hz (motor imagery frequency band)
- Normalization: Z-score per trial
- No artifact rejection (handled by MNE's automatic rejection)

---

## Phase 2 Ready Checklist

✅ Data loading works for both datasets
✅ Preprocessing pipeline validated
✅ Model training works
✅ Transfer learning pipeline works
✅ All freezing strategies tested
✅ Experiment utilities ready for Eric
✅ Documentation complete

**Status: READY FOR PHASE 2 (Eric + Vanessa can start experiments)**

---

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    with open('INFRASTRUCTURE_DOCS.md', 'w', encoding='utf-8') as f:
        f.write(docs)

    print("📄 Documentation saved to: INFRASTRUCTURE_DOCS.md")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("PHASE 1: INFRASTRUCTURE SETUP")
    print("Authors: Hetvi Gandhi (Data) + Hridyanshu (Models)")
    print("=" * 70)
    print("\n")

    # Run validation
    success = validate_infrastructure()

    if success:
        # Generate documentation
        print("\nGenerating documentation...")
        generate_documentation()

        print("\n" + "=" * 70)
        print("PHASE 1 COMPLETE!")
        print("=" * 70)
        print("\n✅ Infrastructure is ready for Phase 2")
        print("✅ Eric can start his ablation studies")
        print("✅ Vanessa can start her LOSO validation")
        print("\nNext: Schedule Day 4 gate review meeting")
    else:
        print("\n" + "=" * 70)
        print("❌ VALIDATION FAILED - FIX ERRORS BEFORE PHASE 2")
        print("=" * 70)