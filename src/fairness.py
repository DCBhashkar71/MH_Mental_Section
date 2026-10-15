"""
Subgroup fairness analysis for Phase 11, matching the reference manuscript's
Section 2.12/3.8 methodology (recall/precision by subgroup, e.g. wealth
quintile there; education/financial-sufficiency here).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score


def subgroup_performance_table(y_true, y_pred, subgroup_series, min_group_size=10):
    """
    Recall, precision, and support for the positive class, computed
    separately within each level of subgroup_series. Groups smaller than
    min_group_size are flagged (recall/precision on <10 cases is too noisy
    to report reliably) rather than silently included.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred),
                        "subgroup": np.asarray(subgroup_series)})

    rows = []
    for level, group in df.groupby("subgroup"):
        n = len(group)
        n_positive_true = (group["y_true"] == 1).sum()
        if n_positive_true == 0:
            recall = np.nan
        else:
            recall = recall_score(group["y_true"], group["y_pred"], zero_division=0)
        precision = precision_score(group["y_true"], group["y_pred"], zero_division=0)
        rows.append({
            "subgroup_level": level, "n": n, "n_positive_true": n_positive_true,
            "recall": round(recall, 4) if not np.isnan(recall) else np.nan,
            "precision": round(precision, 4),
            "reliable": n >= min_group_size,
        })

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def group_level_recall_correlation(subgroup_table, group_stat_series):
    """
    Correlate subgroup-level recall against a group-level continuous
    statistic (e.g. mean income code, mean education level) -- mirrors the
    reference manuscript's state-level wealth/prevalence regression, but as
    a simple correlation since this project's subgroup counts are much
    smaller than the reference's 28-state analysis.
    """
    merged = subgroup_table.set_index("subgroup_level").join(
        group_stat_series.rename("group_stat"), how="inner")
    merged = merged.dropna(subset=["recall", "group_stat"])
    if len(merged) < 3:
        return None, merged
    corr = merged["recall"].corr(merged["group_stat"])
    return corr, merged
