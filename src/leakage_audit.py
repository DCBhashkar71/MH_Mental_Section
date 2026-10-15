"""
Reusable functions for the data-leakage audit (notebook 02) and any later
notebook that needs to re-check a dataset for item-level label circularity.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


def deterministic_sum_check(df, item_cols, score_col):
    """
    Check whether score_col is (close to) the row-wise sum of item_cols.
    Returns (match_rate, mean_abs_diff) computed only on rows where all
    item_cols and score_col are non-missing.
    """
    sub = df.dropna(subset=item_cols + [score_col])
    computed = sub[item_cols].sum(axis=1)
    match_rate = np.isclose(computed, sub[score_col]).mean()
    mean_abs_diff = (computed - sub[score_col]).abs().mean()
    return match_rate, mean_abs_diff


def label_threshold_summary(df, score_col, label_col):
    """Return min/max/count of score_col grouped by label_col — reveals the cutoff."""
    return df.groupby(label_col)[score_col].agg(["min", "max", "count"])


def naive_leaky_cv_accuracy(X, y, seed=42, n_splits=5):
    """
    Fit a plain LogisticRegression on the given (leakage) features to predict y,
    5-fold cross-validated accuracy. Used to reproduce/quantify inflated
    performance claims driven by item-level circularity.
    """
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()


def trivial_threshold_accuracy(score_series, y, threshold):
    """Accuracy of a bare `score >= threshold` rule, no model at all."""
    pred = (score_series >= threshold).astype(int)
    return accuracy_score(y, pred)


def sdoh_sniff_test_auc(X, y, seed=42, n_splits=5):
    """
    Quick (untuned) cross-validated ROC-AUC for a logistic regression using
    only legitimate (non-leakage) predictors. This is NOT the final model —
    just a sanity check that the real predictive task is realistic (i.e. well
    below 1.0), unlike the leaky reproduction above.
    """
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()


def missingness_dependency_check(df, item_cols, group_col):
    """
    Cross-tabulate whether item_cols are missing (any of them) against group_col
    (e.g. country/site). A skewed table means missingness is not random with
    respect to the group — relevant when missing rows get a default-imputed
    label, since it creates a second, subtler leakage path.
    """
    any_missing = df[item_cols].isna().any(axis=1)
    return pd.crosstab(df[group_col], any_missing, normalize="index").round(3)
