import mne
import numpy as np
from pathlib import Path
from data_infra.config import *


def load_dataset_2a(subject_id, session='T'):
    fname = f"A{subject_id:02d}{session}.gdf"
    fpath = Path(DATASET_2A_PATH) / fname
    
    if not fpath.exists():
        raise FileNotFoundError(f"Cannot find file: {fpath}")
    
    raw = mne.io.read_raw_gdf(str(fpath), preload=True, verbose=False)
    event, eventDict = mne.events_from_annotations(raw)
    
    raw.pick_channels(raw.ch_names[:DATASET_2A_CHANNELS])
    
    return raw, event, eventDict

def load_dataset_2b(subject_id, sess='T'):
    if sess == 'T':
        sessIds = ['01T', '02T', '03T']
    else:
        sessIds = ['04E', '05E']
    
    rawList = []
    eventList = []
    
    for sess_id in sessIds:
        fname = f"B{subject_id:02d}{sess_id}.gdf"
        fpath = Path(DATASET_2B_PATH) / fname
        
        if not fpath.exists():
            print(f"Warning: Skipping {fpath}")
            continue

        raw = mne.io.read_raw_gdf(str(fpath), preload=True, verbose=False)
        event, eventDict = mne.events_from_annotations(raw)

        raw.pick_channels(raw.ch_names[:DATASET_2B_CHANNELS])
        
        rawList.append(raw)
        eventList.append(event)
    if len(rawList) > 1:
        raw = mne.concatenate_raws(rawList)
        event = np.vstack(eventList)
    else:
        raw = rawList[0]
        event = eventList[0]
    
    return raw, event, eventDict