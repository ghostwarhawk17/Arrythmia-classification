import os
import wfdb
import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
from imblearn.over_sampling import SMOTE

# Constants according to the paper
SAMPLING_RATE = 360  # MIT-BIH dataset sampling rate
BEFORE_R = 99        # Samples before R-peak
AFTER_R = 160        # Samples after R-peak
SEGMENT_LENGTH = BEFORE_R + AFTER_R + 1 # 260 samples per beat

# MIT-BIH standard mapping to paper's classes
# The paper evaluates 6 classes: N (Normal), L (LBBB), V (PVC), R (RBBB), U (Unknown), A (APB)
CLASS_MAPPING = {
    'N': 'N', 'L': 'L', 'R': 'R', 'V': 'V', 'A': 'A',
    '/': 'U', 'f': 'U', 'Q': 'U', '?': 'U' # Mapping unclassified/paced to U
}

def extract_temporal_features(r_peaks, current_idx, current_peak):
    """
    Extract temporal features for a beat as described in the paper.
    """
    # 1. Pre-RR interval
    pre_rr = (r_peaks[current_idx] - r_peaks[current_idx-1]) / SAMPLING_RATE
    
    # 2. Post-RR interval
    post_rr = (r_peaks[current_idx+1] - r_peaks[current_idx]) / SAMPLING_RATE
    
    # 3. Local average RR interval (past 10s window)
    window_10s_start = current_peak - (10 * SAMPLING_RATE)
    rr_10s = [(r_peaks[j] - r_peaks[j-1]) / SAMPLING_RATE for j in range(1, current_idx+1) 
              if r_peaks[j] > window_10s_start and r_peaks[j] <= current_peak]
    local_avg_rr = np.mean(rr_10s) if len(rr_10s) > 0 else 0
    
    # 4. Global average RR interval (past 5 min window)
    window_5m_start = current_peak - (5 * 60 * SAMPLING_RATE)
    rr_5m = [(r_peaks[j] - r_peaks[j-1]) / SAMPLING_RATE for j in range(1, current_idx+1) 
             if r_peaks[j] > window_5m_start and r_peaks[j] <= current_peak]
    global_avg_rr = np.mean(rr_5m) if len(rr_5m) > 0 else 0
    
    return [pre_rr, post_rr, local_avg_rr, global_avg_rr]

def preprocess_record(record_name, data_path):
    """
    Preprocess a single ECG record according to the paper's methodology.
    """
    # 1. Read record and annotations (MIT-BIH Database)
    record = wfdb.rdrecord(os.path.join(data_path, record_name))
    annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')
    
    signal = record.p_signal[:, 0] # Using MLII lead generally
    
    # 2. Denoising using BioSPPy
    # BioSPPy filters interfering components (muscle activity, 50Hz noise, baseline wandering)
    # The ecg function applies a FIR bandpass filter (3 to 45 Hz)
    out = ecg.ecg(signal=signal, sampling_rate=SAMPLING_RATE, show=False)
    filtered_signal = out['filtered']
    
    # 3. R-peak detection
    # The paper relies on the annotation files for R-peaks
    r_peaks = annotation.sample
    symbols = np.array(annotation.symbol)
    
    segments = []
    labels = []
    features = []
    
    # Iterate through peaks, skipping the first and last to allow RR calculations
    for i in range(1, len(r_peaks) - 1):
        peak = r_peaks[i]
        sym = symbols[i]
        
        # Map symbol to the 6 target classes
        mapped_sym = CLASS_MAPPING.get(sym, None)
        if not mapped_sym:
            continue
            
        # 4. Heartbeat Segmentation (260 samples per beat)
        start_idx = peak - BEFORE_R
        end_idx = peak + AFTER_R + 1 # +1 to include the peak itself and get exactly 260 samples
        
        # Ensure we don't go out of bounds
        if start_idx < 0 or end_idx > len(filtered_signal):
            continue
            
        segment = filtered_signal[start_idx:end_idx]
        
        # 5. Temporal feature extraction
        temporal_features = extract_temporal_features(r_peaks, i, peak)
        
        segments.append(segment)
        features.append(temporal_features)
        labels.append(mapped_sym)
        
    return np.array(segments), np.array(features), np.array(labels)

def load_and_preprocess_all(data_path):
    """
    Loads all records, preprocesses them, and balances the dataset.
    """
    all_segments = []
    all_features = []
    all_labels = []
    
    # List all valid MIT-BIH records (e.g., 100, 101, etc.)
    records = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]
    
    for record in records:
        print(f"Processing record {record}...")
        seg, feat, lab = preprocess_record(record, data_path)
        if len(seg) > 0:
            all_segments.append(seg)
            all_features.append(feat)
            all_labels.append(lab)
        
    X_seg = np.vstack(all_segments)
    X_feat = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    print("Class distribution before resampling:")
    print(pd.Series(y).value_counts())
    
    # 6. Data Analysis (Resampling / Upscaling)
    # The paper states data resampling was performed using upscaling for the 6 classes
    # We flatten segments to apply SMOTE, then reshape them back
    X_seg_flat = X_seg.reshape(X_seg.shape[0], -1)
    X_combined = np.hstack((X_seg_flat, X_feat))
    
    print("Applying SMOTE upscaling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)
    
    print("Class distribution after resampling:")
    print(pd.Series(y_resampled).value_counts())
    
    # Separate segments and features again
    X_seg_resampled = X_resampled[:, :SEGMENT_LENGTH].reshape(-1, SEGMENT_LENGTH, 1)
    X_feat_resampled = X_resampled[:, SEGMENT_LENGTH:]
    
    return X_seg_resampled, X_feat_resampled, y_resampled

if __name__ == "__main__":
    # Example usage:
    # Set this to the directory where MIT-BIH data (.dat, .atr, .hea files) is stored.
    # MIT_BIH_DIR = "path/to/mit-bih"
    # X_segments, X_features, y_labels = load_and_preprocess_all(MIT_BIH_DIR)
    pass
