import numpy as np
import matplotlib.pyplot as plt

def visualize_processed_ecg(npz_file_path, start_sec=0, end_sec=10):
    """
    Visualizes the Raw and Filtered ECG signals from a processed .npz file.
    
    Args:
        npz_file_path (str): Path to the processed .npz file (e.g., 'processed/100_processed.npz')
        start_sec (int): Start time in seconds for the plot window
        end_sec (int): End time in seconds for the plot window
    """
    # Load the processed data
    data = np.load(npz_file_path)
    
    signal_raw = data['signal_raw']
    signal_filtered = data['signal_filtered']
    fs = int(data['fs'])
    ann_sym = data['ann_sym']
    ann_pos = data['ann_pos']
    
    # Calculate sample indices for the window
    start_idx = start_sec * fs
    end_idx = end_sec * fs
    
    # Create time array for the x-axis
    time = np.arange(start_idx, end_idx) / fs
    
    # Extract data for the chosen window
    window_raw = signal_raw[start_idx:end_idx]
    window_filtered = signal_filtered[start_idx:end_idx]
    
    # Find annotations that fall within our time window
    mask = (ann_pos >= start_idx) & (ann_pos < end_idx)
    window_ann_pos = ann_pos[mask]
    window_ann_sym = ann_sym[mask]
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 1. Plot Raw Signal
    ax1.plot(time, window_raw, color='gray', label='Raw Signal')
    ax1.set_title(f"Raw ECG Signal (Window: {start_sec}s - {end_sec}s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Plot Annotations on Raw Signal
    for pos, sym in zip(window_ann_pos, window_ann_sym):
        t_pos = pos / fs
        y_val = signal_raw[pos]
        ax1.plot(t_pos, y_val, 'ro', markersize=5)
        ax1.annotate(sym, (t_pos, y_val), textcoords="offset points", xytext=(0,10), ha='center', color='red')
    
    # 2. Plot Filtered & Normalized Signal
    ax2.plot(time, window_filtered, color='blue', label='Filtered & Z-Normalized')
    ax2.set_title("Processed ECG Signal (Bandpass + Notch + Z-Score)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.grid(True, alpha=0.3)
    
    # Plot Annotations on Filtered Signal
    for pos, sym in zip(window_ann_pos, window_ann_sym):
        t_pos = pos / fs
        y_val = signal_filtered[pos]
        ax2.plot(t_pos, y_val, 'ro', markersize=5)
        ax2.annotate(sym, (t_pos, y_val), textcoords="offset points", xytext=(0,10), ha='center', color='red')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage: Change this path to match one of your processed files
    # Make sure you have run the 01_preprocessing.py script first!
    processed_file = r"c:\MTP\processed\100_processed.npz" 
    
    try:
        visualize_processed_ecg(processed_file, start_sec=0, end_sec=5) # Plotting first 5 seconds
    except FileNotFoundError:
        print(f"File {processed_file} not found. Please ensure data is preprocessed.")
