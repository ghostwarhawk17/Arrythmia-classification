"""
STAGE 5: EVALUATION MODULE
============================
Reproduces the paper's evaluation figures/tables (Section 6):
  - Fig. 12: Confusion matrix per scenario
  - Fig. 13-15: ROC curve (per class) + AUC per scenario
  - Fig. 16: Accuracy comparison across scenarios (line/step chart)
  - Fig. 17: Bar chart of accuracy per scenario
  - Fig. 18: Bar chart of execution time per scenario
  - Fig. 19: Bar chart of cross-entropy per scenario
  - Table 4 style: class-wise accuracy comparison table

Expects one .npz result file PER SCENARIO, each containing at minimum:
    y_true          -> (N,) integer class labels
    y_pred_prob     -> (N, num_classes) predicted probabilities
    accuracy        -> float
    cross_entropy   -> float
    execution_time_min -> float
    classes         -> array of class name strings, e.g. ['A','L','N','R','U','V']

These are exactly what 03_cnn_baseline.py and 04_rl_optimizer.py save.
If you also build a Scenario 1B (manual tuning) run, save it in the same
schema (see the "manual" example call at the bottom of 03_cnn_baseline.py
docstring, or just re-run 03 with different hardcoded hyperparameters and
rename the output file).

Usage:
    python 05_evaluation.py \
        --baseline_npz ./results/baseline_results.npz \
        --manual_npz   ./results/manual_results.npz \
        --rl_npz       ./results/rl_optimized_results.npz \
        --out_dir      ./evaluation_report
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# 1. LOAD RESULTS
# ---------------------------------------------------------------------------
def load_scenario_results(npz_path, scenario_name):
    data = np.load(npz_path, allow_pickle=True)
    result = {
        "name": scenario_name,
        "accuracy": float(data["accuracy"]),
        "cross_entropy": float(data["cross_entropy"]),
        "execution_time_min": float(data["execution_time_min"]),
        "classes": [str(c) for c in data["classes"]],
    }
    # y_true/y_pred_prob may not exist for older result files
    if "y_true" in data and "y_pred_prob" in data:
        result["y_true"] = data["y_true"]
        result["y_pred_prob"] = data["y_pred_prob"]
        result["y_pred"] = np.argmax(data["y_pred_prob"], axis=1)
    elif "confusion_matrix" in data:
        result["confusion_matrix"] = data["confusion_matrix"]
    return result


# ---------------------------------------------------------------------------
# 2. CONFUSION MATRIX  (Fig. 12 style)
# ---------------------------------------------------------------------------
def plot_confusion_matrices(scenarios, out_dir):
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        if "y_true" in sc:
            cm = confusion_matrix(sc["y_true"], sc["y_pred"])
        else:
            cm = sc["confusion_matrix"]

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sc["classes"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format='d')
        ax.set_title(f"{sc['name']}\nAccuracy: {sc['accuracy']*100:.2f}%")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# 3. CLASSIFICATION REPORT (precision/recall/F1 per class per scenario)
# ---------------------------------------------------------------------------
def save_classification_reports(scenarios, out_dir):
    all_reports = {}
    for sc in scenarios:
        if "y_true" not in sc:
            print(f"Skipping classification report for {sc['name']} (no y_true/y_pred saved)")
            continue
        report = classification_report(
            sc["y_true"], sc["y_pred"], target_names=sc["classes"],
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        all_reports[sc["name"]] = report_df
        out_path = os.path.join(out_dir, f"classification_report_{sc['name']}.csv")
        report_df.to_csv(out_path)
        print(f"Saved -> {out_path}")
    return all_reports


# ---------------------------------------------------------------------------
# 4. ROC CURVES + AUC (Fig. 13-15 style, one-vs-rest per class)
# ---------------------------------------------------------------------------
def plot_roc_curves(scenarios, out_dir):
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        if "y_true" not in sc:
            print(f"Skipping ROC for {sc['name']} (no y_pred_prob saved)")
            continue

        classes = sc["classes"]
        n_classes = len(classes)
        y_true_bin = label_binarize(sc["y_true"], classes=list(range(n_classes)))

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], sc["y_pred_prob"][:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {sc['name']}")
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# 5. SCENARIO COMPARISON BAR CHARTS (Fig. 16-19 style)
# ---------------------------------------------------------------------------
def plot_scenario_comparison(scenarios, out_dir):
    names = [sc["name"] for sc in scenarios]
    acc = [sc["accuracy"] * 100 for sc in scenarios]
    ce = [sc["cross_entropy"] for sc in scenarios]
    t = [sc["execution_time_min"] for sc in scenarios]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(names, acc, color="#4C72B0")
    axes[0].set_title("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(acc):
        axes[0].text(i, v + 1, f"{v:.1f}", ha='center')

    axes[1].bar(names, ce, color="#DD8452")
    axes[1].set_title("Cross-Entropy Loss")
    for i, v in enumerate(ce):
        axes[1].text(i, v, f"{v:.4f}", ha='center', va='bottom')

    axes[2].bar(names, t, color="#55A868")
    axes[2].set_title("Execution Time (min)")
    for i, v in enumerate(t):
        axes[2].text(i, v, f"{v:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "scenario_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")

    # Also save the raw numbers as a CSV (mirrors paper's summary values)
    summary_df = pd.DataFrame({
        "scenario": names,
        "accuracy_%": acc,
        "cross_entropy": ce,
        "execution_time_min": t,
    })
    csv_path = os.path.join(out_dir, "scenario_comparison.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved -> {csv_path}")
    return summary_df


# ---------------------------------------------------------------------------
# 6. CLASS-WISE ACCURACY COMPARISON  (Table 4 style)
# ---------------------------------------------------------------------------
def class_wise_accuracy_table(scenarios, out_dir):
    rows = []
    for sc in scenarios:
        if "y_true" not in sc:
            continue
        classes = sc["classes"]
        cm = confusion_matrix(sc["y_true"], sc["y_pred"])
        for i, cls in enumerate(classes):
            sample_size = cm[i].sum()
            true_pred = cm[i, i]
            false_pred = sample_size - true_pred
            acc_pct = (true_pred / sample_size * 100) if sample_size > 0 else 0.0
            rows.append({
                "class": cls, "scenario": sc["name"],
                "sample_size": int(sample_size), "true_prediction": int(true_pred),
                "false_prediction": int(false_pred), "accuracy_%": round(acc_pct, 2),
            })

    if not rows:
        print("No y_true/y_pred available in any scenario - skipping class-wise table")
        return None

    df = pd.DataFrame(rows).pivot(index="class", columns="scenario", values="accuracy_%")
    out_path = os.path.join(out_dir, "class_wise_accuracy.csv")
    df.to_csv(out_path)
    print(f"Saved -> {out_path}")
    print("\nClass-wise accuracy (%) comparison:\n", df)
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(baseline_npz, manual_npz, rl_npz, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    scenarios = []
    if baseline_npz:
        scenarios.append(load_scenario_results(baseline_npz, "Baseline"))
    if manual_npz:
        scenarios.append(load_scenario_results(manual_npz, "Manual"))
    if rl_npz:
        scenarios.append(load_scenario_results(rl_npz, "RL-Optimized"))

    if not scenarios:
        raise ValueError("Provide at least one of --baseline_npz / --manual_npz / --rl_npz")

    print(f"Loaded {len(scenarios)} scenario(s): {[s['name'] for s in scenarios]}")

    plot_confusion_matrices(scenarios, out_dir)
    save_classification_reports(scenarios, out_dir)
    plot_roc_curves(scenarios, out_dir)
    summary_df = plot_scenario_comparison(scenarios, out_dir)
    class_wise_accuracy_table(scenarios, out_dir)

    print("\n===== FINAL SUMMARY =====")
    print(summary_df.to_string(index=False))
    print(f"\nFull evaluation report saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_npz", default=None, help="Path to baseline_results.npz")
    parser.add_argument("--manual_npz", default=None, help="Path to manual_results.npz")
    parser.add_argument("--rl_npz", default=None, help="Path to rl_optimized_results.npz")
    parser.add_argument("--out_dir", default="./evaluation_report")
    args = parser.parse_args()

    main(args.baseline_npz, args.manual_npz, args.rl_npz, args.out_dir)
