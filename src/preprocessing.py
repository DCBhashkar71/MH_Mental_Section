"""
Shared preprocessing utilities: stratified three-way splitting and
ColumnTransformer builders, used by every base-learner notebook (05a-05c
onward) so the split/encoding logic is identical and defined in one place.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def compute_group_id(df, feature_cols):
    """
    Build a group id per row = the row's full feature signature (all feature
    columns concatenated as strings, missing values treated as their own
    category). Rows with an identical signature are the "same" observation
    for splitting/CV purposes.

    WHY THIS EXISTS: with a coarse categorical/discrete feature space and a
    large enough sample, many rows end up with an *identical* recorded
    feature profile (verified on Pakistan: 6,991 unique combos out of 14,008
    rows). Standard random train/test or K-fold splitting doesn't know this
    and will happily put duplicate-profile rows on both sides, which lets a
    model "recognize" a profile it memorized during training rather than
    generalize -- inflating CV/test AUC without any real signal. Always use
    compute_group_id + the group-aware functions below instead of the plain
    three_way_split / StratifiedKFold for this project's datasets.
    """
    return df[feature_cols].fillna("MISSING").astype(str).agg("|".join, axis=1)


def three_way_split(df, label_col, train_frac, cal_frac, test_frac, seed):
    """
    DEPRECATED for this project's modeling notebooks -- kept only for cases
    where group-awareness genuinely doesn't apply. Plain stratified split,
    no protection against duplicate feature-profile rows crossing partitions.
    Use three_way_group_split instead for anything that trains a model.
    """
    assert abs(train_frac + cal_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    train_df, temp_df = train_test_split(
        df, test_size=(cal_frac + test_frac),
        stratify=df[label_col], random_state=seed
    )
    rel_test = test_frac / (cal_frac + test_frac)
    cal_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df[label_col], random_state=seed
    )
    return train_df, cal_df, test_df


def three_way_group_split(df, label_col, feature_cols, train_frac, cal_frac, test_frac, seed):
    """
    Group-aware three-way split: rows sharing an identical feature signature
    (see compute_group_id) are guaranteed to land entirely within ONE
    partition (train, calibration, or test), never split across them.

    Not exactly label-stratified (GroupShuffleSplit can't stratify by both
    group and label simultaneously), but verified in practice to stay close
    to the overall class balance since group sizes are small relative to n.
    Always sanity-check the resulting label balance after calling this.
    """
    assert abs(train_frac + cal_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    groups = compute_group_id(df, feature_cols)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(cal_frac + test_frac), random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_groups = groups.iloc[temp_idx].reset_index(drop=True)

    rel_test = test_frac / (cal_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    cal_idx, test_idx = next(gss2.split(temp_df, groups=temp_groups))
    cal_df = temp_df.iloc[cal_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, cal_df, test_df


def build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False):
    """
    Build a ColumnTransformer: median-impute (+ optional scaling) for numeric
    columns, most-frequent-impute + one-hot encode for categorical columns.

    scale_numeric=False -> use for tree-based models (XGBoost, Random Forest,
                            Extra Trees, HistGradientBoosting) which don't need it.
    scale_numeric=True  -> use for SVM, Naive Bayes (Gaussian), and the DL model.

    IMPORTANT: this transformer must only ever be fit inside a Pipeline on
    training-fold data — never fit on the full dataset before splitting.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])


def split_column_types(df, feature_cols):
    """Convenience: split a feature list into numeric vs categorical column names."""
    numeric_cols = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def save_split(train_df, cal_df, test_df, out_dir, dataset_name):
    """Save the three partitions as CSVs into processed/<dataset_name>/."""
    target_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(target_dir, exist_ok=True)
    train_df.to_csv(os.path.join(target_dir, "train.csv"), index=False)
    cal_df.to_csv(os.path.join(target_dir, "calibration.csv"), index=False)
    test_df.to_csv(os.path.join(target_dir, "test.csv"), index=False)
    print(f"Saved {dataset_name}: train={train_df.shape}, calibration={cal_df.shape}, test={test_df.shape}")
    print(f"  -> {target_dir}")
