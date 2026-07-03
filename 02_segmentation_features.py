"""
STAGE 2: ECG PROCESSING MODULE
================================
Follows directly from 01_preprocessing.py output.

Implements (per paper Section 5.2, points 2-3 and Section 4):
 A) R-peak-based heartbeat segmentation
      - 260 samples per beat: 99 samples BEFORE the R-peak,
        160 samples AFTER the R-peak (method of Yildirim et al.,
        replicated in this paper)
 B) Temporal feature extraction per beat:
      - Pre-RR interval
      - Post-RR interval
      - Local average RR interval (avg of last 10 RR intervals, ~10s window)
      - Global average RR interval (avg of last 10 RR intervals, ~5min window)
 C) Class filtering + rebalancing:
      - Keep only 6 classes: N, L, V, R, U, A  (paper drops F, Q, S -
        too few samples, see Fig. 6)
      - Upsample minority classes to match the majority class (Fig. 7)

Usage:
   python 02_segmentation_features.py --npz_path ./processed/100_processed.npz --out_dir ./features
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import resample

# Beat window sizes used in the paper (Yildirim et al. method)
PRE_SAMPLES = 99
POST_SAMPLES = 160
BEAT_LEN = PRE_SAMPLES + POST_SAMPLES + 1  # +1 for the R-peak sample itself = 260

# The 6 classes retained by the paper (see Fig. 6 / Fig. 7)
KEEP_CLASSES = ['N', 'L', 'V', 'R', 'U', 'A']


# ---------------------------------------------------------------------------
# A. HEARTBEAT SEGMENTATION
# ---------------------------------------------------------------------------
def segment_beats(signal, ann_sym, ann_pos, pre=PRE_SAMPLES, post=POST_SAMPLES):
    """
    Extract a fixed-length window centered on each annotated R-peak.

    Returns:
        beats  -> (num_beats, BEAT_LEN) array of segmented waveforms
        labels -> (num_beats,) array of beat symbols
        rpeaks -> (num_beats,) array of R-peak sample indices (kept beats only)
    """
    beats, labels, rpeaks = [], [], []

    for sym, pos in zip(ann_sym, ann_pos):
        start, end = pos - pre, pos + post + 1
        # Skip beats too close to the start/end of the recording
        if start < 0 or end > len(signal):
            continue
        beats.append(signal[start:end])
        labels.append(sym)
        rpeaks.append(pos)

    beats = np.array(beats)
    labels = np.array(labels)
    rpeaks = np.array(rpeaks)

    print(f"Segmented {len(beats)} beats of length {BEAT_LEN} samples each")
    return beats, labels, rpeaks


# ---------------------------------------------------------------------------
# B. TEMPORAL FEATURE EXTRACTION
#    (Pre-RR, Post-RR, Local avg RR, Global avg RR — Section 5.2)
# ---------------------------------------------------------------------------
def extract_temporal_features(rpeaks, fs, local_window_sec=10, global_window_sec=300):
    """
    For each beat (by R-peak position), compute:
        pre_rr    : RR interval to the previous beat
        post_rr   : RR interval to the next beat
        local_rr  : mean of RR intervals within the past `local_window_sec`
        global_rr : mean of RR intervals within the past `global_window_sec`

    Paper assumes ~1 beat/second for defining the sliding windows, so we
    convert the requested time windows into an approximate beat count
    (10 RR-intervals for local, 10 RR-intervals for global window per text).
    """
    n = len(rpeaks)
    rr_intervals = np.diff(rpeaks) / fs  # in seconds

    pre_rr = np.zeros(n)
    post_rr = np.zeros(n)
    local_rr = np.zeros(n)
    global_rr = np.zeros(n)

    # Pre-RR / Post-RR
    pre_rr[1:] = rr_intervals
    pre_rr[0] = rr_intervals[0] if n > 1 else 0
    post_rr[:-1] = rr_intervals
    post_rr[-1] = rr_intervals[-1] if n > 1 else 0

    # Local / global average RR (paper: avg of last 10 RR-intervals,
    # assuming ~1 beat/sec -> 10s local window, 5min -> effectively
    # capped to available history similarly)
    for i in range(n):
        lo_start = max(0, i - 10)
        local_rr[i] = np.mean(pre_rr[lo_start:i + 1])

        gl_start = max(0, i - 10)  # per paper text, also averaged over
        global_rr[i] = np.mean(pre_rr[gl_start:i + 1])  # last 10 RR-intervals

    return pd.DataFrame({
        'pre_rr': pre_rr,
        'post_rr': post_rr,
        'local_rr': local_rr,
        'global_rr': global_rr,
    })


# ---------------------------------------------------------------------------
# C. CLASS FILTERING + REBALANCING (Fig. 6 -> Fig. 7 in the paper)
# ---------------------------------------------------------------------------
def filter_and_rebalance(beats, labels, features_df, keep_classes=KEEP_CLASSES):
    mask = np.isin(labels, keep_classes)
    beats, labels = beats[mask], labels[mask]
    features_df = features_df[mask].reset_index(drop=True)

    print("\nClass distribution BEFORE rebalancing:")
    print(pd.Series(labels).value_counts())

    # Combine everything into one dataframe for easy resampling
    df = pd.DataFrame({'label': labels})
    df = pd.concat([df, features_df], axis=1)
    df['beat_idx'] = np.arange(len(df))  # index back into `beats` array

    max_count = df['label'].value_counts().max()
    resampled_frames = []
    for cls in keep_classes:
        cls_df = df[df['label'] == cls]
        if len(cls_df) == 0:
            continue
        cls_resampled = resample(
            cls_df, replace=True, n_samples=max_count, random_state=42
        )
        resampled_frames.append(cls_resampled)

    balanced_df = pd.concat(resampled_frames).reset_index(drop=True)
    balanced_beats = beats[balanced_df['beat_idx'].values]
    balanced_labels = balanced_df['label'].values
    balanced_features = balanced_df.drop(columns=['label', 'beat_idx']).reset_index(drop=True)

    print("\nClass distribution AFTER rebalancing:")
    print(pd.Series(balanced_labels).value_counts())

    return balanced_beats, balanced_labels, balanced_features


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(npz_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    signal = data['signal_filtered']
    fs = float(data['fs'])
    ann_sym = data['ann_sym']
    ann_pos = data['ann_pos']

    # A. Segment
    beats, labels, rpeaks = segment_beats(signal, ann_sym, ann_pos)

    # B. Temporal features
    features_df = extract_temporal_features(rpeaks, fs)

    # C. Filter to 6 classes + rebalance
    beats_bal, labels_bal, features_bal = filter_and_rebalance(beats, labels, features_df)

    # Save for the CNN/RL training stage
    record_id = os.path.splitext(os.path.basename(npz_path))[0]
    out_path = os.path.join(out_dir, f"{record_id}_features.npz")
    np.savez(
        out_path,
        beats=beats_bal,                       # shape: (N, 260) -> CNN input
        labels=labels_bal,                      # class labels
        temporal_features=features_bal.values,  # shape: (N, 4)
        feature_names=features_bal.columns.tolist(),
    )
    print(f"\nSaved segmented + balanced dataset -> {out_path}")
    print(f"Final shape: beats={beats_bal.shape}, features={features_bal.shape}")


if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", default=None,
                         help="Path to a single *_processed.npz from stage 1. "
                              "If omitted, all files in --processed_dir are used.")
    parser.add_argument("--processed_dir", default="./processed",
                         help="Directory with *_processed.npz files (default: %(default)s)")
    parser.add_argument("--out_dir", default="./features")
    args = parser.parse_args()

    if args.npz_path:
        main(args.npz_path, args.out_dir)
    else:
        npz_files = sorted(glob.glob(os.path.join(args.processed_dir, "*_processed.npz")))
        if not npz_files:
            print(f"ERROR: No *_processed.npz files found in {args.processed_dir}")
            exit(1)
        print(f"Found {len(npz_files)} preprocessed records in {args.processed_dir}\n")
        for npz in npz_files:
            try:
                main(npz, args.out_dir)
            except Exception as e:
                print(f"  !! Skipping {npz}: {e}\n")
        print("\nAll records segmented + feature-extracted.")

