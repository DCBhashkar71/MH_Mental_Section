"""
NNLS ensemble utilities for Phase 7. Mirrors the reference manuscript's
approach: fit non-negative weights via NNLS on out-of-fold base-learner
predictions (avoids the sign-flipped/unstable coefficients a logistic-
regression meta-learner produces under multicollinear tree-based learners),
normalize to sum to 1, then apply as a weighted average of probabilities.
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score


def fit_nnls_weights(oof_df, model_names, label_col="y_true"):
    """
    Fit NNLS weights on out-of-fold predictions.

    oof_df: DataFrame with one column per model_name (OOF predicted
            probabilities) plus a label_col column.
    Returns: dict {model_name: normalized_weight}, and the raw (pre-
             normalization) weights for reference.
    """
    X = oof_df[model_names].values
    y = oof_df[label_col].values

    raw_weights, residual = nnls(X, y)
    weight_sum = raw_weights.sum()

    if weight_sum == 0:
        raise ValueError("NNLS returned all-zero weights -- check that OOF predictions "
                          "carry signal (e.g. inspect individual AUCs first).")

    normalized = raw_weights / weight_sum
    weights_dict = dict(zip(model_names, normalized))
    raw_dict = dict(zip(model_names, raw_weights))
    return weights_dict, raw_dict


def apply_ensemble(pred_df, model_names, weights_dict):
    """Weighted average of probability columns using the fitted NNLS weights."""
    proba = np.zeros(len(pred_df))
    for name in model_names:
        proba += pred_df[name].values * weights_dict[name]
    return proba


def equal_weight_ensemble(model_names):
    """Simple baseline: equal weight to every base learner. Useful sanity check."""
    w = 1.0 / len(model_names)
    return {name: w for name in model_names}


def fit_auc_optimized_weights(oof_df, model_names, label_col="y_true", seed=42):
    """
    Directly optimize ensemble AUC (not squared error, unlike NNLS) over
    non-negative weights summing to 1, via constrained optimization.
    AUC is non-smooth, so this uses repeated random-restart local search
    (Nelder-Mead) rather than gradient methods.
    """
    from scipy.optimize import minimize

    X = oof_df[model_names].values
    y = oof_df[label_col].values
    n_models = len(model_names)

    def neg_auc(raw_w):
        w = np.abs(raw_w)
        w = w / (w.sum() + 1e-12)
        proba = X @ w
        return -roc_auc_score(y, proba)

    rng = np.random.RandomState(seed)
    best_result, best_score = None, np.inf
    for _ in range(10):
        x0 = rng.dirichlet(np.ones(n_models))
        res = minimize(neg_auc, x0, method="Nelder-Mead",
                        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
        if res.fun < best_score:
            best_score, best_result = res.fun, res.x

    w = np.abs(best_result)
    w = w / w.sum()
    return dict(zip(model_names, w))


def fit_logistic_stacking(oof_df, model_names, label_col="y_true", C=1.0, seed=42):
    """
    L2-regularized logistic regression meta-learner on OOF base predictions.
    Unlike an unregularized meta-learner (which the reference manuscript
    found unstable under multicollinearity), regularization keeps
    coefficients bounded even when base learners are correlated. Returns a
    fitted sklearn model -- use predict_proba directly rather than a
    weights dict, since coefficients aren't constrained to be non-negative
    or sum to 1.
    """
    from sklearn.linear_model import LogisticRegression

    X = oof_df[model_names].values
    y = oof_df[label_col].values
    meta = LogisticRegression(C=C, max_iter=1000, random_state=seed)
    meta.fit(X, y)
    return meta


def greedy_ensemble_selection(oof_df, model_names, label_col="y_true", max_iterations=50, seed=42):
    """
    Caruana et al. (2004) greedy ensemble selection with replacement:
    iteratively add whichever model (repeats allowed) most improves the
    running ensemble's AUC on OOF data. Naturally handles correlated
    learners (redundant models simply don't get picked again) without
    needing an explicit regularization parameter. Returns normalized
    weights = pick_count / total_picks per model.
    """
    X = oof_df[model_names].values
    y = oof_df[label_col].values
    n_models = len(model_names)

    picks = []
    running_sum = np.zeros(len(y))

    for i in range(max_iterations):
        best_j, best_auc = None, -np.inf
        for j in range(n_models):
            trial_sum = running_sum + X[:, j]
            trial_proba = trial_sum / (len(picks) + 1)
            auc = roc_auc_score(y, trial_proba)
            if auc > best_auc:
                best_auc, best_j = auc, j
        picks.append(best_j)
        running_sum += X[:, best_j]

    counts = np.zeros(n_models)
    for j in picks:
        counts[j] += 1
    weights = counts / counts.sum()
    return dict(zip(model_names, weights))


def _bootstrap_mean_se(y_true, proba, n_resamples=500, seed=42):
    """Bootstrap mean and standard error of AUC-ROC for one candidate's predictions."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], proba[idx]))
    scores = np.array(scores)
    return scores.mean(), scores.std()


# Complexity ranking used by the one-standard-error rule: prefer the simplest
# candidate that isn't meaningfully worse than the best one. Single base
# learners are simplest (nothing extra fit on top); equal-weight needs no
# fitting either but combines information from multiple models; the four
# fitted-weight/meta-learner strategies are most complex.
_COMPLEXITY_RANK = {
    "equal_weight": 1,
    "nnls": 2, "auc_optimized": 2, "logistic_stacking": 2, "greedy_selection": 2,
}


def _complexity_of(strategy_name):
    if strategy_name.startswith("single_"):
        return 0
    return _COMPLEXITY_RANK.get(strategy_name, 2)


def compare_ensemble_strategies(oof_df, cal_df, model_names, label_col="y_true", seed=42,
                                 greedy_iterations=50, auc_opt_restarts=10, include_single_models=True,
                                 n_bootstrap=500):
    """
    Fit NNLS, equal-weight, AUC-direct-optimized, logistic stacking, greedy
    selection, and (if include_single_models) each individual base learner
    -- all candidate weights/models fit using ONLY the OOF (training)
    predictions -- then score every candidate on the calibration set.

    SELECTION RULE: bootstrap each candidate's calibration AUC (mean + SE,
    n_bootstrap resamples) rather than trusting a single point estimate.
    With small calibration sets, a raw point-estimate winner can be pure
    noise (verified on this project's Arab5 data: a ~0.02 AUC point-estimate
    gap on n=296 turned out to reverse on the test set). Apply a
    one-standard-error rule: among all candidates within 1 SE of the best
    bootstrap mean, select the SIMPLEST one (single base learner preferred
    over equal-weight, preferred over a fitted ensemble/meta-learner). This
    is the same logic as the well-established "1-SE rule" for regularization
    parameter selection (Breiman et al.'s CART book; still standard practice,
    e.g. glmnet's lambda.1se) applied here to ensembling-strategy selection.

    Returns: (winning_strategy_name, winning_weights_or_model, comparison_df)
    comparison_df includes point-estimate AND bootstrap mean/SE columns, plus
    which candidates were within 1 SE of the best -- keep this full table in
    the manuscript's supplementary material regardless of which one wins.
    """
    strategies = {}

    w_nnls, _ = fit_nnls_weights(oof_df, model_names, label_col)
    strategies["nnls"] = ("weights", w_nnls)

    strategies["equal_weight"] = ("weights", equal_weight_ensemble(model_names))

    w_auc = fit_auc_optimized_weights(oof_df, model_names, label_col, seed=seed)
    strategies["auc_optimized"] = ("weights", w_auc)

    meta_lr = fit_logistic_stacking(oof_df, model_names, label_col, seed=seed)
    strategies["logistic_stacking"] = ("model", meta_lr)

    w_greedy = greedy_ensemble_selection(oof_df, model_names, label_col,
                                          max_iterations=greedy_iterations, seed=seed)
    strategies["greedy_selection"] = ("weights", w_greedy)

    if include_single_models:
        for name in model_names:
            single_weights = {n: (1.0 if n == name else 0.0) for n in model_names}
            strategies[f"single_{name}"] = ("weights", single_weights)

    rows = []
    for strat_name, (kind, obj) in strategies.items():
        if kind == "weights":
            proba_cal = apply_ensemble(cal_df, model_names, obj)
        else:
            proba_cal = obj.predict_proba(cal_df[model_names].values)[:, 1]
        point_auc = roc_auc_score(cal_df[label_col], proba_cal)
        boot_mean, boot_se = _bootstrap_mean_se(cal_df[label_col], proba_cal,
                                                 n_resamples=n_bootstrap, seed=seed)
        rows.append({
            "strategy": strat_name, "calibration_auc": point_auc,
            "bootstrap_mean_auc": boot_mean, "bootstrap_se": boot_se,
            "complexity_rank": _complexity_of(strat_name),
        })

    comparison_df = pd.DataFrame(rows).sort_values("bootstrap_mean_auc", ascending=False).reset_index(drop=True)

    best_mean = comparison_df["bootstrap_mean_auc"].max()
    best_se = comparison_df.loc[comparison_df["bootstrap_mean_auc"].idxmax(), "bootstrap_se"]
    threshold = best_mean - best_se
    comparison_df["within_1se_of_best"] = comparison_df["bootstrap_mean_auc"] >= threshold

    eligible = comparison_df[comparison_df["within_1se_of_best"]]
    eligible = eligible.sort_values(["complexity_rank", "bootstrap_mean_auc"], ascending=[True, False])
    winner_name = eligible.iloc[0]["strategy"]
    winner_kind, winner_obj = strategies[winner_name]

    return winner_name, winner_kind, winner_obj, comparison_df


def apply_fitted_strategy(pred_df, model_names, kind, obj):
    """Apply a strategy chosen by compare_ensemble_strategies to a new prediction DataFrame."""
    if kind == "weights":
        return apply_ensemble(pred_df, model_names, obj)
    else:
        return obj.predict_proba(pred_df[model_names].values)[:, 1]


def paired_bootstrap_delta_auc(y_true, proba_a, proba_b, n_resamples=2000, seed=42, ci=0.95):
    """
    Paired bootstrap comparison: delta = AUC(proba_a) - AUC(proba_b), computed
    on the SAME resampled indices each iteration (paired, not independent),
    matching the reference manuscript's Section 3.4 comparison methodology.
    Typical use: a = ensemble, b = best individual base learner.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    n = len(y_true)

    deltas = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_a = roc_auc_score(y_true[idx], proba_a[idx])
        auc_b = roc_auc_score(y_true[idx], proba_b[idx])
        deltas.append(auc_a - auc_b)

    deltas = np.array(deltas)
    lower = np.percentile(deltas, (1 - ci) / 2 * 100)
    upper = np.percentile(deltas, (1 + ci) / 2 * 100)
    return deltas.mean(), lower, upper
