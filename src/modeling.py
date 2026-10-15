"""
Shared modeling utilities: pipeline builders, Optuna search spaces,
threshold tuning, and bootstrap CI. Used by 05a/05b/05c (base learners),
07 (NNLS ensemble), and reused conceptually in 13 (external validation).

NOTE ON TESTING: the pipeline-construction and CV logic here was verified
against real project data using scikit-learn-only equivalents (this
environment does not have xgboost/optuna/imbalanced-learn installed).
The XGBoost, Optuna, and SMOTE-specific code paths use standard, stable
APIs but were not execution-tested in that exact form before being handed
to you -- if anything errors, send me the traceback and I'll fix it fast.
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from preprocessing import compute_group_id  # src/ is on sys.path in every notebook

# Models that benefit from scaled numeric features
SCALE_NEEDED = {"svm", "nb", "knn"}


def build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False):
    """Same logic as preprocessing.build_preprocessor, duplicated here to keep
    this module self-contained for modeling notebooks."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = SkPipeline(numeric_steps)

    categorical_transformer = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])


def get_classifier(model_name, params, seed):
    """Instantiate a classifier given a name and a hyperparameter dict."""
    if model_name == "dt":
        return DecisionTreeClassifier(random_state=seed, **params)
    elif model_name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    elif model_name == "gb":
        return GradientBoostingClassifier(random_state=seed, **params)
    elif model_name == "svm":
        return SVC(probability=True, random_state=seed, **params)
    elif model_name == "nb":
        return GaussianNB(**params)
    elif model_name == "knn":
        return KNeighborsClassifier(n_jobs=-1, **params)
    elif model_name == "xgb":
        # NOTE: use_label_encoder was removed in xgboost>=1.6 -- deliberately omitted here
        # since requirements.txt pins xgboost>=2.0. If you're on an older xgboost, add it back.
        return XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=-1, **params)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def build_model_pipeline(model_name, params, numeric_cols, categorical_cols, seed, use_smote=True):
    """
    Full pipeline: preprocessor -> (optional SMOTE) -> classifier.
    Uses imblearn's Pipeline so SMOTE is correctly fold-safe when used
    inside cross_val_score / cross_val_predict / Optuna objectives.
    """
    scale = model_name in SCALE_NEEDED
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=scale)
    clf = get_classifier(model_name, params, seed)

    steps = [("pre", pre)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", clf))

    return ImbPipeline(steps)


def suggest_params(trial, model_name):
    """Optuna search spaces, one branch per model_name."""
    if model_name == "dt":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
    elif model_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    elif model_name == "gb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    elif model_name == "svm":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True),
            "kernel": "rbf",
        }
    elif model_name == "nb":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True),
        }
    elif model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 75),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        }
    elif model_name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def make_objective(model_name, X, y, numeric_cols, categorical_cols, seed, n_splits=5,
                    subsample_n=None, groups=None):
    """
    Build an Optuna objective function for a given model. Uses 5-fold
    cross-validated ROC-AUC on the training set, with SMOTE inside each fold.

    groups: REQUIRED for correctness on this project's datasets. Pass the
    output of preprocessing.compute_group_id(X, feature_cols) so that rows
    sharing an identical feature signature never split across a fold's
    train/validation boundary (see preprocessing.py docstring for why this
    matters -- verified on Pakistan to otherwise inflate AUC from ~0.56 to
    ~0.84 with no real signal behind it). If groups is None, falls back to
    plain StratifiedKFold -- only appropriate if you've separately confirmed
    this dataset has no meaningful duplicate-row problem.

    subsample_n: if set (used for SVM per the project config), fit each trial
    on a random subsample of X/y (and groups, correspondingly) for
    computational tractability.
    """
    def objective(trial):
        params = suggest_params(trial, model_name)
        pipe = build_model_pipeline(model_name, params, numeric_cols, categorical_cols, seed)

        if subsample_n is not None and len(X) > subsample_n:
            X_sub = X.sample(n=subsample_n, random_state=seed)
            y_sub = y.loc[X_sub.index]
            groups_sub = groups.loc[X_sub.index] if groups is not None else None
        else:
            X_sub, y_sub = X, y
            groups_sub = groups

        if groups_sub is not None:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scores = cross_val_score(pipe, X_sub, y_sub, cv=cv, groups=groups_sub,
                                      scoring="roc_auc", n_jobs=1)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            scores = cross_val_score(pipe, X_sub, y_sub, cv=cv, scoring="roc_auc", n_jobs=1)

        return scores.mean()

    return objective


def tune_threshold(y_true, y_proba, n_steps=99):
    """Sweep thresholds 0.01-0.99, return the one maximizing macro-F1."""
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_thr, best_score = 0.5, -1
    y_true = np.asarray(y_true)
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        score = f1_score(y_true, pred, average="macro")
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr, best_score


def bootstrap_ci(y_true, y_proba, metric_func=roc_auc_score, n_resamples=2000, seed=42, ci=0.95):
    """Bootstrap point estimate + CI for a given metric (default: AUC-ROC)."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_func(y_true[idx], y_proba[idx]))
    scores = np.array(scores)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return scores.mean(), lower, upper


def evaluate_at_threshold(y_true, y_proba, threshold):
    """Return accuracy, macro-F1, AUC-ROC at a given decision threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "auc_roc": roc_auc_score(y_true, y_proba),
    }


def run_base_learner_suite(train, cal, test, label_col, numeric_cols, categorical_cols,
                            model_names, trial_budget, seed, n_bootstrap, models_dir,
                            tables_dir, dataset_tag, svm_subsample_n=None, n_cv_splits=5):
    """
    End-to-end base-learner suite for one dataset/label: Optuna-tune every
    model in model_names (group-aware CV), fit final pipelines, tune
    per-model decision thresholds on the calibration set, evaluate on the
    test set with bootstrap CI, save models + results + OOF predictions.

    This is exactly the logic from 05a_baseline_models_pakistan.ipynb,
    Cells 3-12, wrapped into one function so 05b/05c/05d don't repeat it
    by hand. Returns a dict with results_df, fitted_pipelines, best_params,
    oof_predictions, cal_predictions, test_predictions, groups_train.
    """
    import os
    import json
    import joblib
    from sklearn.model_selection import cross_val_predict

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    feature_cols = numeric_cols + categorical_cols
    X_train, y_train = train[feature_cols], train[label_col]
    X_cal, y_cal = cal[feature_cols], cal[label_col]
    X_test, y_test = test[feature_cols], test[label_col]

    groups_train = compute_group_id(X_train, feature_cols)
    print(f"[{dataset_tag}] Train/Cal/Test: {X_train.shape}/{X_cal.shape}/{X_test.shape} | "
          f"unique training combos: {groups_train.nunique()} of {len(X_train)}")

    # --- Optuna tuning (group-aware) ---
    best_params, best_cv_scores = {}, {}
    for model_name in model_names:
        subsample_n = svm_subsample_n if model_name == "svm" else None
        objective = make_objective(model_name, X_train, y_train, numeric_cols, categorical_cols,
                                    seed=seed, n_splits=n_cv_splits, subsample_n=subsample_n,
                                    groups=groups_train)
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=trial_budget[model_name], show_progress_bar=False)
        best_params[model_name] = study.best_params
        best_cv_scores[model_name] = study.best_value
        print(f"[{dataset_tag}] {model_name}: best CV AUC = {study.best_value:.4f}")

    with open(os.path.join(models_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # --- Fit final pipelines ---
    fitted_pipelines = {}
    for model_name in model_names:
        pipe = build_model_pipeline(model_name, best_params[model_name],
                                     numeric_cols, categorical_cols, seed=seed)
        pipe.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipe

    # --- Threshold tuning on calibration set ---
    tuned_thresholds = {}
    for model_name in model_names:
        proba_cal = fitted_pipelines[model_name].predict_proba(X_cal)[:, 1]
        thr, _ = tune_threshold(y_cal, proba_cal)
        tuned_thresholds[model_name] = thr

    # --- Evaluate on test set ---
    results = []
    for model_name in model_names:
        proba_test = fitted_pipelines[model_name].predict_proba(X_test)[:, 1]
        thr = tuned_thresholds[model_name]
        metrics = evaluate_at_threshold(y_test, proba_test, thr)
        auc_point, auc_lo, auc_hi = bootstrap_ci(y_test, proba_test, n_resamples=n_bootstrap, seed=seed)
        results.append({
            "model": model_name, "threshold": round(thr, 2),
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "auc_roc": round(metrics["auc_roc"], 4),
            "auc_ci_lower": round(auc_lo, 4), "auc_ci_upper": round(auc_hi, 4),
            "cv_auc_tuning": round(best_cv_scores[model_name], 4),
        })
    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    results_df.to_csv(os.path.join(tables_dir, f"{dataset_tag}_base_learner_results.csv"), index=False)

    for model_name, pipe in fitted_pipelines.items():
        joblib.dump(pipe, os.path.join(models_dir, f"{model_name}_pipeline.joblib"))

    # --- OOF predictions (group-aware CV) for the Phase 7 ensemble ---
    oof_predictions = pd.DataFrame({"y_true": y_train.values}, index=X_train.index)
    cv = StratifiedGroupKFold(n_splits=n_cv_splits, shuffle=True, random_state=seed)
    for model_name in model_names:
        pipe = build_model_pipeline(model_name, best_params[model_name],
                                     numeric_cols, categorical_cols, seed=seed)
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, groups=groups_train,
                                       method="predict_proba")[:, 1]
        oof_predictions[model_name] = oof_proba

    cal_predictions = pd.DataFrame({"y_true": y_cal.values})
    test_predictions = pd.DataFrame({"y_true": y_test.values})
    for model_name in model_names:
        cal_predictions[model_name] = fitted_pipelines[model_name].predict_proba(X_cal)[:, 1]
        test_predictions[model_name] = fitted_pipelines[model_name].predict_proba(X_test)[:, 1]

    print(f"[{dataset_tag}] Done. Results saved to {tables_dir}, models saved to {models_dir}.")

    return {
        "results_df": results_df, "fitted_pipelines": fitted_pipelines, "best_params": best_params,
        "oof_predictions": oof_predictions, "cal_predictions": cal_predictions,
        "test_predictions": test_predictions, "groups_train": groups_train,
    }
