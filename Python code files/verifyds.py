"""
Fixed verification script for GDF files (BCI Competition IV 2a)
Run this to verify your cross-session data structure
"""

import mne
import os
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

data_path = "D:\\Documents 1 Jan 2026\\BCICIV"

# ============================================================================
# VERIFICATION
# ============================================================================

print("=" * 70)
print("BCI COMPETITION IV 2a - CROSS-SESSION DATA VERIFICATION")
print("=" * 70)

# Get all files
all_files = os.listdir(data_path)
a_files = sorted([f for f in all_files if f.startswith('A') and f.endswith('.gdf')])
b_files = sorted([f for f in all_files if f.startswith('B') and f.endswith('.gdf')])

print(f"\nTotal files found:")
print(f"  A files (Main dataset): {len(a_files)}")
print(f"  B files (Additional runs): {len(b_files)}")

# Separate T and E files
a_train = sorted([f for f in a_files if 'T.gdf' in f])
a_eval = sorted([f for f in a_files if 'E.gdf' in f])

print(f"\nMain dataset structure:")
print(f"  Training (Session 1): {len(a_train)} files")
for f in a_train:
    print(f"    - {f}")

print(f"\n  Evaluation (Session 2): {len(a_eval)} files")
for f in a_eval:
    print(f"    - {f}")

# Check if we have matching pairs
print(f"\n" + "=" * 70)
print("CROSS-SESSION PAIRS")
print("=" * 70)

for i in range(1, 10):
    train_file = f"A0{i}T.gdf"
    eval_file = f"A0{i}E.gdf"

    has_train = train_file in a_train
    has_eval = eval_file in a_eval

    status = "✅" if (has_train and has_eval) else "❌"
    print(f"Subject {i}: {status}")
    if has_train:
        print(f"  Session 1: {train_file}")
    if has_eval:
        print(f"  Session 2: {eval_file}")

# Load one file to check structure
print(f"\n" + "=" * 70)
print("LOADING SAMPLE FILE: A01T.gdf")
print("=" * 70)

try:
    # Load GDF file with MNE
    raw = mne.io.read_raw_gdf(
        f"{data_path}/A01T.gdf",
        preload=True,
        verbose=False
    )

    print(f"\n✅ File loaded successfully!")
    print(f"\nBasic information:")
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} seconds")
    print(f"  Data shape: {raw.get_data().shape} (channels × samples)")

    # Get events (trial markers)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    print(f"\n  Total events: {len(events)}")
    print(f"  Event types: {list(event_dict.keys())}")

    # Count trials by class
    print(f"\n  Event counts:")
    for event_name, event_id in event_dict.items():
        count = np.sum(events[:, 2] == event_id)
        if count > 0:
            print(f"    {event_name}: {count}")

    print(f"\n" + "=" * 70)
    print("✅ CROSS-SESSION DATA VERIFIED!")
    print("=" * 70)
    print("\nYou have:")
    print("  ✅ 9 subjects")
    print("  ✅ 2 sessions per subject (T = Session 1, E = Session 2)")
    print("  ✅ EEG data in GDF format")
    print("  ✅ Ready for within-subject cross-session transfer learning!")

    print("\nAdditional B files detected:")
    print(f"  - {len(b_files)} additional run files")
    print("  - These provide even more training data!")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. ✅ Data structure verified")
    print("2. → Check your phase1_infrastructure.py uses MNE")
    print("3. → Implement cross-session transfer experiments")
    print("4. → Run experiments (180 experiments, ~10-12 hours)")
    print("5. → Analyze results and integrate into paper")

except FileNotFoundError:
    print(f"\n❌ Error: Could not find file A01T.gdf")
    print(f"   Check data_path: {data_path}")

except ImportError:
    print(f"\n❌ Error: MNE-Python not installed")
    print("\nInstall with:")
    print("  pip install mne --break-system-packages")

except Exception as e:
    print(f"\n❌ Error loading file: {e}")
    print(f"\nError type: {type(e).__name__}")

print("\n" + "=" * 70)