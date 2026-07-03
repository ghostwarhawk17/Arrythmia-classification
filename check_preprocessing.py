"""
PREPROCESSING VERIFICATION SCRIPT
===================================
Checks that all 48 MIT-BIH records were preprocessed correctly by
01_preprocessing.py.  Validates:
  1. All expected .npz files exist
  2. Each file contains the required keys
  3. Signal arrays are non-empty and finite (no NaN / Inf)
  4. Filtered signal is actually z-normalized (mean ≈ 0, std ≈ 1)
  5. Annotations are consistent (same count for symbols and positions)
  6. Sampling rate is the expected 360 Hz

Usage:
    python check_preprocessing.py
    python check_preprocessing.py --processed_dir ./processed
"""

import os
import glob
import argparse
import numpy as np


# The 48 record IDs in the MIT-BIH Arrhythmia Database
EXPECTED_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222",
    "223", "228", "230", "231", "232", "233", "234",
]

REQUIRED_KEYS = {"signal_raw", "signal_filtered", "fs", "ann_sym", "ann_pos"}
EXPECTED_FS = 360  # Hz


def check_record(npz_path, record_id):
    """Validate a single processed record.  Returns (passed: bool, issues: list[str])."""
    issues = []

    # ---- 1. Load ----
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return False, [f"Cannot load file: {e}"]

    # ---- 2. Required keys ----
    present_keys = set(data.files)
    missing = REQUIRED_KEYS - present_keys
    if missing:
        issues.append(f"Missing keys: {missing}")

    # ---- 3. Signal checks ----
    for key in ("signal_raw", "signal_filtered"):
        if key not in present_keys:
            continue
        sig = data[key]
        if sig.size == 0:
            issues.append(f"{key} is empty")
        if not np.all(np.isfinite(sig)):
            nan_count = np.count_nonzero(~np.isfinite(sig))
            issues.append(f"{key} contains {nan_count} non-finite values (NaN/Inf)")

    # ---- 4. Z-normalization check on filtered signal ----
    if "signal_filtered" in present_keys:
        filt = data["signal_filtered"]
        if filt.size > 0 and np.all(np.isfinite(filt)):
            mean = np.mean(filt)
            std = np.std(filt)
            if abs(mean) > 0.01:
                issues.append(f"Filtered signal mean = {mean:.6f} (expected ≈ 0)")
            if abs(std - 1.0) > 0.01:
                issues.append(f"Filtered signal std  = {std:.6f} (expected ≈ 1)")

    # ---- 5. Sampling rate ----
    if "fs" in present_keys:
        fs = int(data["fs"])
        if fs != EXPECTED_FS:
            issues.append(f"Sampling rate = {fs} Hz (expected {EXPECTED_FS})")

    # ---- 6. Annotation consistency ----
    if "ann_sym" in present_keys and "ann_pos" in present_keys:
        n_sym = len(data["ann_sym"])
        n_pos = len(data["ann_pos"])
        if n_sym != n_pos:
            issues.append(f"Annotation mismatch: {n_sym} symbols vs {n_pos} positions")
        if n_sym == 0:
            issues.append("No annotations found")

    # ---- 7. Denoising effectiveness (filtered should differ from raw) ----
    if "signal_raw" in present_keys and "signal_filtered" in present_keys:
        raw = data["signal_raw"]
        filt = data["signal_filtered"]
        if raw.shape == filt.shape and np.allclose(raw, filt):
            issues.append("Filtered signal is identical to raw — denoising may not have worked")

    passed = len(issues) == 0
    return passed, issues


def main(processed_dir):
    print("=" * 65)
    print("  PREPROCESSING VERIFICATION REPORT")
    print("=" * 65)

    # Check which files exist
    found_files = sorted(glob.glob(os.path.join(processed_dir, "*_processed.npz")))
    found_ids = {os.path.basename(f).replace("_processed.npz", "") for f in found_files}

    missing_records = [r for r in EXPECTED_RECORDS if r not in found_ids]
    extra_records = found_ids - set(EXPECTED_RECORDS)

    print(f"\n  Directory : {os.path.abspath(processed_dir)}")
    print(f"  Expected  : {len(EXPECTED_RECORDS)} records")
    print(f"  Found     : {len(found_files)} files")
    if missing_records:
        print(f"  MISSING   : {missing_records}")
    if extra_records:
        print(f"  Extra     : {sorted(extra_records)}")

    # Validate each file
    print(f"\n{'Record':<10} {'Status':<8} {'Samples':>10} {'Beats':>8} {'Labels':>8}  Details")
    print("-" * 85)

    total_pass = 0
    total_fail = 0
    total_beats = 0

    for record_id in EXPECTED_RECORDS:
        npz_path = os.path.join(processed_dir, f"{record_id}_processed.npz")
        if not os.path.exists(npz_path):
            print(f"{record_id:<10} {'MISS':>8}")
            total_fail += 1
            continue

        passed, issues = check_record(npz_path, record_id)

        # Gather stats for display
        try:
            data = np.load(npz_path, allow_pickle=True)
            n_samples = data["signal_filtered"].shape[0] if "signal_filtered" in data.files else 0
            n_beats = len(data["ann_pos"]) if "ann_pos" in data.files else 0
            n_labels = len(set(data["ann_sym"])) if "ann_sym" in data.files else 0
            total_beats += n_beats
        except Exception:
            n_samples = n_beats = n_labels = 0

        status = "PASS" if passed else "FAIL"
        detail = "" if passed else "; ".join(issues)
        print(f"{record_id:<10} {status:<8} {n_samples:>10,} {n_beats:>8,} {n_labels:>8}  {detail}")

        if passed:
            total_pass += 1
        else:
            total_fail += 1

    # Summary
    print("-" * 85)
    print(f"\n  Results: {total_pass} PASSED / {total_fail} FAILED out of {len(EXPECTED_RECORDS)}")
    print(f"  Total beats across all records: {total_beats:,}")

    if total_fail == 0 and not missing_records:
        print("\n  [OK] ALL RECORDS PREPROCESSED CORRECTLY!")
    else:
        print("\n  [!!] Some issues detected -- see details above.")

    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify preprocessed MIT-BIH data")
    parser.add_argument("--processed_dir", default="./processed",
                        help="Directory containing *_processed.npz files")
    args = parser.parse_args()
    main(args.processed_dir)
