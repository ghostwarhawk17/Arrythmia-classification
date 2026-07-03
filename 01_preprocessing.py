"""
STAGE 1: ECG PREPROCESSING MODULE
==================================
Based on Serhani et al. (2025) - "Enhancing arrhythmia prediction through an
adaptive deep reinforcement learning framework for ECG signal analysis"

What this does (per paper Section 5.2, point 1):
 - Loads a raw MIT-BIH record (.dat/.hea/.atr) using the WFDB package
 - Removes powerline noise interference, baseline wander, and muscle
   artifact (the paper uses BioSPPy's ecg module for this)
 - Saves the cleaned signal + annotations for the next stage (segmentation)

Input expected: a folder containing <record_id>.dat, .hea, .atr
   e.g. 100.dat, 100.hea, 100.atr  (place all three in the same folder)

Usage:
   python 01_preprocessing.py --record_path /path/to/100 --out_dir ./processed
"""

import os
import argparse
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, iirnotch


# ---------------------------------------------------------------------------
# 1. LOAD RAW RECORD (signal + beat annotations)
# ---------------------------------------------------------------------------
def load_record(record_path):
    """
    record_path: path WITHOUT extension, e.g. 'data/100'
    Returns:
        signal   -> 1D numpy array (channel 0, e.g. MLII lead)
        fs       -> sampling frequency (360 Hz for MIT-BIH)
        ann_sym  -> list of beat annotation symbols ('N','L','V','R','A',...)
        ann_pos  -> list of sample indices where each beat's R-peak is located
    """
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]     # first channel (e.g. MLII)
    fs = record.fs

    ann_sym = np.array(annotation.symbol)
    ann_pos = np.array(annotation.sample)

    print(f"Loaded record: {record_path}")
    print(f"  Signal length : {len(signal)} samples")
    print(f"  Sampling rate : {fs} Hz")
    print(f"  Total beats   : {len(ann_pos)}")
    print(f"  Unique labels : {sorted(set(ann_sym))}")

    return signal, fs, ann_sym, ann_pos


# ---------------------------------------------------------------------------
# 2. DENOISING FILTERS
#    Paper: BioSPPy filters out muscle activity, 50 Hz powerline noise,
#    and baseline wandering [Section 5.2 / Fig. 4,5]
# ---------------------------------------------------------------------------
def bandpass_filter(signal, fs, lowcut=0.5, highcut=45.0, order=3):
    """Butterworth bandpass: removes baseline wander (<0.5Hz) and
    high-frequency muscle noise (>45Hz), matching BioSPPy's default
    ECG filtering band."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, freq=50.0, quality=30.0):
    """Removes powerline interference (50 Hz, or use 60 Hz for US mains)."""
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal)


def denoise_ecg(signal, fs):
    """Full denoising pipeline matching the paper's approach:
       1) Bandpass filter to remove baseline wander + high-freq noise
       2) Notch filter to remove powerline interference
    """
    filtered = bandpass_filter(signal, fs, lowcut=0.5, highcut=45.0)
    filtered = notch_filter(filtered, fs, freq=50.0)
    return filtered


# Optional alternative: use BioSPPy directly (exactly what the paper used)
def denoise_ecg_biosppy(signal, fs):
    from biosppy.signals import ecg
    out = ecg.ecg(signal=signal, sampling_rate=fs, show=False)
    return out['filtered']   # cleaned signal, same length as input


# ---------------------------------------------------------------------------
# 3. Z-SCORE NORMALIZATION
#    Paper mentions Z-score normalization to reduce offset effects and
#    equalize amplitude variations across recordings (Section 2.1)
# ---------------------------------------------------------------------------
def z_normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(record_path, out_dir, use_biosppy=False):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load
    signal, fs, ann_sym, ann_pos = load_record(record_path)

    # 2. Denoise
    if use_biosppy:
        filtered = denoise_ecg_biosppy(signal, fs)
    else:
        filtered = denoise_ecg(signal, fs)

    # 3. Normalize
    filtered_norm = z_normalize(filtered)

    # 4. Save processed outputs for next stage (segmentation)
    record_id = os.path.basename(record_path)
    np.savez(
        os.path.join(out_dir, f"{record_id}_processed.npz"),
        signal_raw=signal,
        signal_filtered=filtered_norm,
        fs=fs,
        ann_sym=ann_sym,
        ann_pos=ann_pos,
    )
    print(f"\nSaved processed data -> {out_dir}/{record_id}_processed.npz")
    return filtered_norm, fs, ann_sym, ann_pos


if __name__ == "__main__":
    import glob

    DEFAULT_DB_DIR = os.path.join(
        "mit-bih-arrhythmia-database-1.0.0",
        "mit-bih-arrhythmia-database-1.0.0",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", default=None,
                         help="Path to a single record without extension, "
                              "e.g. data/100. If omitted, all records in "
                              "--db_dir are processed.")
    parser.add_argument("--db_dir", default=DEFAULT_DB_DIR,
                         help="Directory containing MIT-BIH .hea/.dat/.atr "
                              "files (default: %(default)s)")
    parser.add_argument("--out_dir", default="./processed")
    parser.add_argument("--use_biosppy", action="store_true",
                         help="Use BioSPPy filtering (matches paper exactly) "
                              "instead of the manual scipy filter pipeline")
    args = parser.parse_args()

    if args.record_path:
        # Process a single record
        main(args.record_path, args.out_dir, args.use_biosppy)
    else:
        # Auto-discover and process all records in the database directory
        hea_files = sorted(glob.glob(os.path.join(args.db_dir, "*.hea")))
        if not hea_files:
            print(f"ERROR: No .hea files found in {args.db_dir}")
            exit(1)
        print(f"Found {len(hea_files)} records in {args.db_dir}\n")
        for hea in hea_files:
            record_path = hea.replace(".hea", "")  # strip extension
            try:
                main(record_path, args.out_dir, args.use_biosppy)
            except Exception as e:
                print(f"  !! Skipping {record_path}: {e}\n")
        print("\nAll records processed.")
