"""
Diagnostic script to understand E file structure
"""

import mne
import numpy as np
from pathlib import Path

data_path = Path("D:\\Documents 1 Jan 2026\\BCICIV")

print("=" * 70)
print("ANALYZING T vs E FILE STRUCTURE")
print("=" * 70)

for session in ['T', 'E']:
    filename = f"A01{session}.gdf"
    filepath = data_path / filename

    print(f"\n{session} FILE: {filename}")
    print("-" * 70)

    raw = mne.io.read_raw_gdf(str(filepath), preload=True, verbose=False)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    print(f"Total events: {len(events)}")
    print(f"Event types: {list(event_dict.keys())}")
    print(f"\nEvent counts:")
    for event_name, event_id in sorted(event_dict.items(), key=lambda x: x[1]):
        count = np.sum(events[:, 2] == event_id)
        print(f"  {event_name:10s} (ID {event_id:4d}): {count:3d} occurrences")

    # Show first 10 events
    print(f"\nFirst 10 events:")
    print(f"{'Time':<10s} {'Sample':<10s} {'Event ID':<10s} {'Event Name':<15s}")
    for i in range(min(10, len(events))):
        event_sample = events[i, 0]
        event_time = event_sample / raw.info['sfreq']
        event_id = events[i, 2]
        event_name = [k for k, v in event_dict.items() if v == event_id][
            0] if event_id in event_dict.values() else "Unknown"
        print(f"{event_time:<10.2f} {event_sample:<10d} {event_id:<10d} {event_name:<15s}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)