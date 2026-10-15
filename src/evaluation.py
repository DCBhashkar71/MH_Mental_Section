"""
Calibration and decision curve analysis utilities for Phase 10, matching
the reference manuscript's Section 2.11/3.7 methodology.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def get_calibration_data(y_true, y_proba, n_bins=10, strategy="quantile"):
    """
    Wrapper around sklearn's calibration_curve. strategy='quantile' is safer
    than 'uniform' for small test sets (avoids empty bins), but note that
    with heavily tied probability values (common for tree ensembles, and
    made more likely here by SMOTE-trained sub-fold models) the number of
    bins actually returned can be fewer than n_bins requested -- that's
    expected, not a bug.
    """
    frac_positive, mean_predicted = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy=strategy)
    return mean_predicted, frac_positive


def plot_calibration_curve(y_true, y_proba, project_root, dataset, n_bins=10, save_fig_fn=None):
    mean_predicted, frac_positive = get_calibration_data(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(mean_predicted, frac_positive, marker="o", color="#4C72B0", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(f"Calibration curve — {dataset}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_fig_fn is not None:
        save_fig_fn(fig, f"{dataset}_calibration_curve.png", project_root, dataset=dataset)

    plt.show()
    return mean_predicted, frac_positive


def decision_curve_analysis(y_true, y_proba, thresholds=None):
    """
    Standard decision curve analysis (Vickers & Elkin, 2006).
    net_benefit_model(pt) = TP/n - FP/n * (pt / (1-pt))
    net_benefit_treat_all(pt) = prevalence - (1-prevalence) * (pt / (1-pt))
    net_benefit_treat_none(pt) = 0
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    prevalence = y_true.mean()

    rows = []
    for pt in thresholds:
        pred_positive = (y_proba >= pt).astype(int)
        tp = np.sum((pred_positive == 1) & (y_true == 1))
        fp = np.sum((pred_positive == 1) & (y_true == 0))
        nb_model = (tp / n) - (fp / n) * (pt / (1 - pt))
        nb_treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({
            "threshold": pt, "net_benefit_model": nb_model,
            "net_benefit_treat_all": nb_treat_all, "net_benefit_treat_none": 0.0,
        })

    return pd.DataFrame(rows)


def plot_decision_curve(dca_df, project_root, dataset, save_fig_fn=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"], color="#4C72B0", label="Model (RF)", linewidth=2)
    ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"], color="#C44E52", linestyle="--", label="Treat all")
    ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_none"], color="gray", linestyle=":", label="Treat none")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curve analysis — {dataset}")
    ax.legend()
    y_min = min(dca_df["net_benefit_model"].min(), dca_df["net_benefit_treat_all"].min(), -0.05)
    ax.set_ylim(y_min, dca_df["net_benefit_model"].max() * 1.2 + 0.02)
    plt.tight_layout()

    if save_fig_fn is not None:
        save_fig_fn(fig, f"{dataset}_decision_curve.png", project_root, dataset=dataset)

    plt.show()
