#!/usr/bin/env python3
"""Recreate and display the saved ROC and Precision–Recall curves."""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roc-csv", default="artifacts/enhanced_bgl/ocdigcn_seed_42_roc_curve.csv"
    )
    parser.add_argument(
        "--pr-csv", default="artifacts/enhanced_bgl/ocdigcn_seed_42_precision_recall_curve.csv"
    )
    parser.add_argument(
        "--predictions", default="artifacts/enhanced_bgl/ocdigcn_seed_42_predictions.csv"
    )
    parser.add_argument(
        "--output", default="artifacts/enhanced_bgl/ocdigcn_seed_42_roc_pr_curves.png"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.roc_csv) or not os.path.isfile(args.pr_csv):
        raise FileNotFoundError(
            "Curve CSVs are missing. Run the 'Train and evaluate enhanced BGL' "
            "configuration first."
        )

    # Use Tk on Windows for an interactive window; open the saved PNG if Tk is absent.
    import matplotlib
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg")
        has_plot_window = True
    except (ImportError, RuntimeError):
        matplotlib.use("Agg")
        has_plot_window = False
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, auc

    roc = pd.read_csv(args.roc_csv)
    pr = pd.read_csv(args.pr_csv)
    roc_auc = auc(roc["false_positive_rate"], roc["true_positive_rate"])
    if os.path.isfile(args.predictions):
        predictions = pd.read_csv(args.predictions)
        prc_auc = average_precision_score(predictions["label"], predictions["anomaly_score"])
        pr_label = f"Average precision = {prc_auc:.4f}"
    else:
        prc_auc = auc(pr["recall"], pr["precision"])
        pr_label = f"PR area = {prc_auc:.4f}"

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        roc["false_positive_rate"], roc["true_positive_rate"],
        color="#1769aa", linewidth=2, label=f"ROC AUC = {roc_auc:.4f}",
    )
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(
        title="ROC curve", xlabel="False positive rate", ylabel="True positive rate",
        xlim=(0, 1), ylim=(0, 1),
    )
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        pr["recall"], pr["precision"], color="#d55e00", linewidth=2,
        label=pr_label,
    )
    axes[1].set(
        title="Precision–Recall curve", xlabel="Recall", ylabel="Precision",
        xlim=(0, 1), ylim=(0, 1.02),
    )
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.2)
    figure.suptitle("OCDiGCN BGL evaluation · seed 42")
    figure.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PRC-AUC / Average Precision: {prc_auc:.4f}")
    print(f"Saved figure: {args.output}")
    if has_plot_window:
        print("A plot window should be open. Close it to return to VS Code.")
        plt.show()
    elif os.name == "nt":
        print("Tk is unavailable; opening the saved PNG in your default image viewer.")
        os.startfile(os.path.abspath(args.output))
    else:
        print("Tk is unavailable; open the saved PNG to view the curves.")


if __name__ == "__main__":
    main()
