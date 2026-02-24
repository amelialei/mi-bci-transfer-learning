import mne
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from data_infra.config import *
from data_infra.data_loader import load_dataset_2a, load_dataset_2b

def bandpass_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=SAMPLING_RATE):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')

    return filtfilt(b, a, data, axis=-1)


def normalize_data(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1
    return (data - mean) / std


def preprocess_epochs(epochs):
    data = epochs.get_data()
    filtered = bandpass_filter(data)
    normalized = normalize_data(filtered)
    
    return normalized

def get_subject_data(dataset='2a', subject_id=1, session='T', return_split=True):
    print(f"Loading {dataset} subject {subject_id}...")

    if dataset == '2a':
        raw, events, event_dict = load_dataset_2a(subject_id, session)
    else:
        raw, events, event_dict = load_dataset_2b(subject_id, session)

    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_dict,
        tmin=EPOCH_START,
        tmax=EPOCH_END,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )

    data = preprocess_epochs(epochs)
    labels = epochs.events[:, -1] 

    print(f"  Loaded {len(labels)} trials")
    print(f"  Data shape: {data.shape}")
    
    if return_split:
        X_train, X_val, y_train, y_val = train_test_split(
            data, 
            labels,
            test_size=VALIDATION_SPLIT,
            random_state=RANDOM_SEED,
            stratify=labels
        )
        print(f"  Train: {X_train.shape}, Validation: {X_val.shape}")
        return X_train, X_val, y_train, y_val
    else:
        return data, labels

