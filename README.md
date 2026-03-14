# mi-bci-transfer-learning

Motor imagery (MI) BCI transfer learning experiments on **Graz / BCI Competition IV** datasets using an **EEGNet**-style CNN and multiple transfer setups (cross-dataset and cross-session/run).

## What’s in this repo

- **Phase 1 “infrastructure”**: `Python code files/phase1_infrastructure.py`
  - Data loading for Graz datasets (2a + 2b)
  - Standard preprocessing (8–30 Hz bandpass + per-trial z-score)
  - EEGNet model
  - Transfer-learning utilities (layer transfer + freezing strategies)
  - Experiment utilities (`ExperimentRunner`)
- **Phase 1 complete study**: `Python code files/bci_project.py`
- **B-file “true-label” studies**:
  - Multi-run transfer: `Python code files/run_bfile_multirun_experiments.py`
  - Cross-session transfer: `Python code files/run_bfile_cross_session_experiments.py`
- **Results/artifacts**: `results/` (JSON summaries, checkpoints, logs, figures, saved models)
  - A full handoff doc with file locations and Phase 1 findings lives at `results/9x5_study/HANDOFF_TO_PHASE2.md`

## Datasets 

- **Dataset 2a** (A-files): `A01T.gdf` … `A09T.gdf`
  - 22 channels, 4 MI classes (left, right, feet, tongue)
  - Some experiments also use a **hands-only** (2-class) subset (left/right) for “task-aligned” transfer.
- **Dataset 2b** (B-files, multiple sessions/runs per subject): examples include
  - `B01T.gdf`, `B02T.gdf`, `B03T.gdf` (training runs)
  - Some cross-session scripts also reference evaluation files like `B04E.gdf`


## Environment setup

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy scikit-learn matplotlib mne torch
```

Notes:
- **PyTorch install** can be platform-specific; if `pip install torch` fails, install from the official PyTorch instructions for your OS.
- These scripts are research-code style and can be compute-heavy.
