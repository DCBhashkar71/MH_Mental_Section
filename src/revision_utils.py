"""
revision_utils.py -- shared helpers for the Study 2 revision phases (16-28).

Everything here is deliberately self-contained (numpy / pandas / scipy / sklearn only,
plus the project's own modeling, preprocessing and ensemble modules) so that every
revision notebook reads the SAME saved artefacts that the original phases (04-13)
produced, and nothing is silently re-tuned.

Conventions (identical to the original project):
  * notebooks live in notebooks/, project root = one level up
  * processed/<tag>/{train,calibration,test}.csv, base_predictions_{calibration,test}.csv,
    oof_predictions_train.csv
  * models/<models_dir>/{<model>_pipeline.joblib, best_hyperparameters.json}
  * models/ensemble_final/<tag>/strategy_comparison.csv  (written by notebook 07)
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score)
from sklearn.linear_model import LogisticRegression

MODEL_NAMES = ["dt", "rf", "gb", "svm", "nb", "knn", "xgb"]
MODEL_LABELS = {"dt": "Decision tree", "rf": "Random forest", "gb": "Gradient boosting",
                "svm": "SVM (RBF)", "nb": "Gaussian naive Bayes", "knn": "k-nearest neighbours",
                "xgb": "XGBoost"}

# ----------------------------------------------------------------------------------------
# Dataset registry (column lists copied verbatim from notebooks 05a/05b/05c/08)
# ----------------------------------------------------------------------------------------
_PK_NUM = ["Age", "Gestational Age", "Number of sons ", "Number of daughters", "Total Number of Children"]
_PK_CAT = ["Gravida", "Female Education", "Husband Education", "Working Status", "Physical Health ",
           "Previous Miscarriage", "Sufficient Money for Basic Needs", "Current Appereance Acceptance",
           "Family System", "Male Gender Preference", "Relationship with Mother in-law"]
_AR_NUM = ["WomenAge", "MariageAge", "NumberofPregnancy", "NumberofAbortions",
           "CovidDiagL1", "CovidDiagL2", "SleepingAfter", "BMI_Pregnancy"]
_AR_CAT = ["part1_country", "Locality_first", "EducationLevel", "Work", "Health_Prob",
           "StressduringCovid", "FamilyProblems", "FinancialProblem", "SocialProblem",
           "PsychologicalProb", "WorkStress", "FamilyIncome1", "part1_current_preg_first"]
_UG_NUM = ["age_numeric", "otherChildren", "numStillBorn", "numMissCarry"]
_UG_CAT = ["homeDistrict", "occupation", "relationshipStatus", "spouseOccupation", "polygamy", "education"]

DATASETS = {
    "pakistan": dict(label="y_depression", numeric=_PK_NUM, categorical=_PK_CAT, models_dir="pakistan",
                     subgroups=["Female Education", "Sufficient Money for Basic Needs"],
                     pretty="Pakistan (depression)"),
    "arab5_depression": dict(label="y_depression", numeric=_AR_NUM, categorical=_AR_CAT,
                             models_dir="arab5_depression",
                             subgroups=["EducationLevel", "FamilyIncome1", "part1_country"],
                             pretty="Arab-region (depression)"),
    "arab5_anxiety": dict(label="y_anxiety", numeric=_AR_NUM, categorical=_AR_CAT,
                          models_dir="arab5_anxiety", subgroups=["EducationLevel", "FamilyIncome1"],
                          pretty="Arab-region (anxiety)"),
    "uganda": dict(label="y_anxiety_binary", numeric=_UG_NUM, categorical=_UG_CAT, models_dir="uganda",
                   subgroups=["education", "occupation"], pretty="Uganda (anxiety)"),
}


def features_of(tag):
    d = DATASETS[tag]
    return d["numeric"] + d["categorical"]


def get_env():
    """Project paths/config, assuming the notebook runs from notebooks/ (as in every earlier phase)."""
    import yaml
    root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    with open(os.path.join(root, "config", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    env = dict(
        ROOT=root, CFG=cfg, SEED=cfg["seed"],
        RAW=os.path.join(root, cfg["paths"]["raw_dir"]),
        INTERIM=os.path.join(root, cfg["paths"]["interim_dir"]),
        PROCESSED=os.path.join(root, cfg["paths"]["processed_dir"]),
        EXTERNAL=os.path.join(root, cfg["paths"]["external_dir"]),
        MODELS=os.path.join(root, cfg["paths"]["models_dir"]),
        TABLES=os.path.join(root, cfg["paths"]["tables_dir"]),
        N_BOOT=cfg["bootstrap"]["n_resamples"],
    )
    env["REV"] = os.path.join(env["TABLES"], "revision")
    os.makedirs(env["REV"], exist_ok=True)
    return env


def load_split(tag, env):
    """Return dict with train/cal/test frames, saved base predictions and OOF predictions."""
    d = os.path.join(env["PROCESSED"], tag)
    out = dict(
        train=pd.read_csv(os.path.join(d, "train.csv")),
        cal=pd.read_csv(os.path.join(d, "calibration.csv")),
        test=pd.read_csv(os.path.join(d, "test.csv")),
    )
    for key, fname in [("pred_cal", "base_predictions_calibration.csv"),
                       ("pred_test", "base_predictions_test.csv"),
                       ("oof", "oof_predictions_train.csv")]:
        p = os.path.join(d, fname)
        out[key] = pd.read_csv(p) if os.path.exists(p) else None
    return out


def load_rf_params(tag, env):
    with open(os.path.join(env["MODELS"], DATASETS[tag]["models_dir"], "best_hyperparameters.json")) as f:
        return json.load(f)


def load_final_threshold(tag, env, model="rf"):
    """Threshold of the proposed model as stored by notebook 07 (proposed_model_final_results.csv)."""
    df = pd.read_csv(os.path.join(env["TABLES"], "proposed_model_final_results.csv"))
    return float(df.loc[df["dataset"] == tag, "threshold"].values[0])


# ----------------------------------------------------------------------------------------
# Basic metrics and bootstrap machinery
# ----------------------------------------------------------------------------------------
def safe_auc(y, p):
    y = np.asarray(y)
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def _div(a, b):
    return a / b if b > 0 else np.nan


def operating_metrics(y, p, thr):
    """Threshold-dependent metrics + threshold-free AUC / PR-AUC for one sample."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    return dict(
        auc=safe_auc(y, p),
        pr_auc=average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        accuracy=(tp + tn) / len(y),
        balanced_accuracy=np.nanmean([sens, spec]),
        sensitivity=sens, specificity=spec,
        ppv=_div(tp, tp + fp), npv=_div(tn, tn + fn),
        macro_f1=f1_score(y, pred, average="macro"),
        predicted_positive_rate=pred.mean(),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )


def bootstrap_table(y, p, fn_dict, B=2000, seed=42, ci=0.95):
    """
    Generic nonparametric bootstrap. fn_dict(y_sample, p_sample) -> dict of scalar metrics.
    Returns DataFrame [metric, estimate, ci_lower, ci_upper, n_boot_valid]; the estimate is
    computed on the FULL sample (percentile CI from resamples).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    point = fn_dict(y, p)
    store = {k: [] for k in point}
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        r = fn_dict(y[idx], p[idx])
        for k, v in r.items():
            store[k].append(v)
    a = (1 - ci) / 2 * 100
    rows = []
    for k, v in point.items():
        arr = np.array(store[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        rows.append(dict(metric=k, estimate=v,
                         ci_lower=np.percentile(arr, a) if len(arr) else np.nan,
                         ci_upper=np.percentile(arr, 100 - a) if len(arr) else np.nan,
                         n_boot_valid=len(arr)))
    return pd.DataFrame(rows)


def bootstrap_ci_full(y, p, fn=safe_auc, B=2000, seed=42, ci=0.95):
    """(estimate on full sample, percentile lower, percentile upper) for a single metric."""
    t = bootstrap_table(y, p, lambda yy, pp: {"m": fn(yy, pp)}, B=B, seed=seed, ci=ci)
    r = t.iloc[0]
    return float(r.estimate), float(r.ci_lower), float(r.ci_upper)


def paired_bootstrap_delta(y, pa, pb, fn=safe_auc, B=2000, seed=42, ci=0.95):
    """
    Paired bootstrap of fn(pa) - fn(pb): the SAME resampled rows are used for both models in
    every iteration. Returns dict(delta, ci_lower, ci_upper, p_two_sided, n_boot_valid).
    p_two_sided = 2*min(P(delta*<=0), P(delta*>=0)), floored at 1/(B+1).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    pa = np.asarray(pa)
    pb = np.asarray(pb)
    n = len(y)
    d = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        va, vb = fn(y[idx], pa[idx]), fn(y[idx], pb[idx])
        if np.isfinite(va) and np.isfinite(vb):
            d.append(va - vb)
    d = np.array(d)
    a = (1 - ci) / 2 * 100
    p_two = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    p_two = max(p_two, 1.0 / (len(d) + 1))
    return dict(delta=float(fn(y, pa) - fn(y, pb)), ci_lower=float(np.percentile(d, a)),
                ci_upper=float(np.percentile(d, 100 - a)), p_two_sided=float(p_two), n_boot_valid=len(d))


def holm_adjust(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def wilson_ci(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    denom = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half, centre + half)


# ----------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p):
    """
    Steyerberg-style: slope = coefficient of logit(p) in a logistic recalibration model;
    intercept (calibration-in-the-large) = intercept estimated with the slope FIXED at 1
    (logit(p) as offset). Ideal: intercept 0, slope 1.
    """
    y = np.asarray(y).astype(float)
    z = _logit(p)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(z.reshape(-1, 1), y)
    slope = float(lr.coef_[0][0])

    def nll(a):
        eta = z + a
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    res = minimize_scalar(nll, bounds=(-10, 10), method="bounded")
    return float(res.x), slope


def calibration_metrics_fn(y, p):
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    prev = y.mean()
    null = prev * (1 - prev)
    a, b = calibration_intercept_slope(y, p)
    return dict(brier=brier, scaled_brier=(1 - brier / null) if null > 0 else np.nan,
                calibration_intercept=a, calibration_slope=b,
                mean_predicted=float(p.mean()), observed_prevalence=float(prev))


def platt_recalibrator(y_cal, p_cal):
    """Fit logistic recalibration on the CALIBRATION partition only; returns a callable p -> p'."""
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    lr.fit(_logit(p_cal).reshape(-1, 1), np.asarray(y_cal).astype(int))
    return lambda p: lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def binned_reliability(y, p, n_bins=10):
    """Quantile bins with Wilson CI for the observed fraction in each bin."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    rows = []
    for _, g in df.groupby("bin", observed=True):
        k, n = int(g["y"].sum()), len(g)
        lo, hi = wilson_ci(k, n)
        rows.append(dict(n=n, mean_predicted=g["p"].mean(), observed=k / n, ci_lower=lo, ci_upper=hi))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Decision curve analysis
# ----------------------------------------------------------------------------------------
def net_benefit_arrays(y, p, thresholds):
    y = np.asarray(y)
    n = len(y)
    prev = y.mean()
    nb_m, nb_a = [], []
    for pt in thresholds:
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        w = pt / (1 - pt)
        nb_m.append(tp / n - fp / n * w)
        nb_a.append(prev - (1 - prev) * w)
    return np.array(nb_m), np.array(nb_a)


def dca_with_ci(y, p, thresholds, B=2000, seed=42):
    """
    Net benefit of model, treat-all and the (model - treat-all) difference with percentile CI
    over the bootstrap. 'robust' = lower CI of (model - treat-all) > 0 AND model NB > 0.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    nb_m, nb_a = net_benefit_arrays(y, p, thresholds)
    diffs, nbs = [], []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        m, a = net_benefit_arrays(y[idx], p[idx], thresholds)
        diffs.append(m - a)
        nbs.append(m)
    diffs, nbs = np.array(diffs), np.array(nbs)
    out = pd.DataFrame(dict(
        threshold=thresholds, nb_model=nb_m, nb_treat_all=nb_a, nb_treat_none=0.0,
        delta_vs_treat_all=nb_m - nb_a,
        delta_ci_lower=np.percentile(diffs, 2.5, axis=0), delta_ci_upper=np.percentile(diffs, 97.5, axis=0),
        nb_model_ci_lower=np.percentile(nbs, 2.5, axis=0), nb_model_ci_upper=np.percentile(nbs, 97.5, axis=0)))
    out["point_estimate_beats_both"] = (out.nb_model > out.nb_treat_all) & (out.nb_model > 0)
    out["robust_beats_both"] = (out.delta_ci_lower > 0) & (out.nb_model_ci_lower > 0)
    return out


def contiguous_range(thresholds, flags):
    """(min, max) of the thresholds flagged True, or (nan, nan)."""
    t = np.asarray(thresholds)[np.asarray(flags, dtype=bool)]
    return (float(t.min()), float(t.max())) if len(t) else (np.nan, np.nan)


# ----------------------------------------------------------------------------------------
# Subgroup (fairness) analysis
# ----------------------------------------------------------------------------------------
def subgroup_table(y, p, thr, g, min_cases=10, B_auc=1000, seed=42):
    """
    Per-level error-rate table. Reliability is judged per quantity:
      TPR needs n_pos >= min_cases, FPR/specificity need n_neg >= min_cases.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    g = np.asarray(g)
    pred = (p >= thr).astype(int)
    rows = []
    for lev in pd.unique(g[~pd.isna(g)]):
        m = g == lev
        yy, pp, pr = y[m], p[m], pred[m]
        n, npos = int(m.sum()), int(yy.sum())
        nneg = n - npos
        tp = int(np.sum((pr == 1) & (yy == 1)))
        fn_ = npos - tp
        fp = int(np.sum((pr == 1) & (yy == 0)))
        tn = nneg - fp
        tpr_lo, tpr_hi = wilson_ci(tp, npos)
        fpr_lo, fpr_hi = wilson_ci(fp, nneg)
        ppv_lo, ppv_hi = wilson_ci(tp, tp + fp)
        if npos > 0 and nneg > 0:
            a_est, a_lo, a_hi = bootstrap_ci_full(yy, pp, B=B_auc, seed=seed)
        else:
            a_est = a_lo = a_hi = np.nan
        rows.append(dict(
            subgroup_level=lev, n=n, n_positive=npos, n_negative=nneg,
            prevalence=npos / n, predicted_positive_rate=pr.mean(), mean_predicted_risk=pp.mean(),
            observed_minus_predicted=npos / n - pp.mean(),
            tpr_recall=_div(tp, npos), tpr_ci_lower=tpr_lo, tpr_ci_upper=tpr_hi,
            fpr=_div(fp, nneg), fpr_ci_lower=fpr_lo, fpr_ci_upper=fpr_hi,
            specificity=_div(tn, nneg), ppv_precision=_div(tp, tp + fp),
            ppv_ci_lower=ppv_lo, ppv_ci_upper=ppv_hi,
            auc=a_est, auc_ci_lower=a_lo, auc_ci_upper=a_hi,
            tpr_reliable=npos >= min_cases, fpr_reliable=nneg >= min_cases,
            tp=tp, fn=fn_, fp=fp, tn=tn))
    out = pd.DataFrame(rows)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def subgroup_gaps(y, p, thr, g, min_cases=10, B=2000, seed=42):
    """
    Between-group gap in TPR and FPR (max - min over RELIABLE levels) with bootstrap CI, and chi-square
    heterogeneity tests on TP/FN and FP/TN counts. Levels are fixed from the observed data.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    g = np.asarray(g)
    pred = (p >= thr).astype(int)
    levels = list(pd.unique(g[~pd.isna(g)]))

    def rates(idx):
        yy, pr, gg = y[idx], pred[idx], g[idx]
        tpr, fpr = {}, {}
        for lev in levels:
            m = gg == lev
            pos, neg = np.sum(yy[m] == 1), np.sum(yy[m] == 0)
            if pos > 0:
                tpr[lev] = np.sum((pr[m] == 1) & (yy[m] == 1)) / pos
            if neg > 0:
                fpr[lev] = np.sum((pr[m] == 1) & (yy[m] == 0)) / neg
        return tpr, fpr

    # levels reliable on the observed data
    rel_tpr = [l for l in levels if np.sum((g == l) & (y == 1)) >= min_cases]
    rel_fpr = [l for l in levels if np.sum((g == l) & (y == 0)) >= min_cases]

    def gap(d, rel):
        v = [d[l] for l in rel if l in d]
        return (max(v) - min(v)) if len(v) >= 2 else np.nan

    idx_all = np.arange(len(y))
    t0, f0 = rates(idx_all)
    est_t, est_f = gap(t0, rel_tpr), gap(f0, rel_fpr)
    rng = np.random.RandomState(seed)
    gt, gf = [], []
    for _ in range(B):
        idx = rng.randint(0, len(y), len(y))
        t, f = rates(idx)
        gt.append(gap(t, rel_tpr))
        gf.append(gap(f, rel_fpr))
    gt, gf = np.array(gt, dtype=float), np.array(gf, dtype=float)
    gt, gf = gt[np.isfinite(gt)], gf[np.isfinite(gf)]

    def chi(pos_col, neg_col, rel):
        rows = []
        for l in rel:
            m = g == l
            a = int(np.sum((pred[m] == 1) & (y[m] == pos_col)))
            b = int(np.sum(m & (y == pos_col))) - a
            rows.append([a, b])
        tab = np.array(rows)
        if len(rows) < 2 or (tab.sum(axis=0) == 0).any():
            return np.nan
        try:
            return float(stats.chi2_contingency(tab)[1])
        except Exception:
            return np.nan

    return dict(
        tpr_gap=est_t, tpr_gap_ci_lower=np.percentile(gt, 2.5) if len(gt) else np.nan,
        tpr_gap_ci_upper=np.percentile(gt, 97.5) if len(gt) else np.nan,
        fpr_gap=est_f, fpr_gap_ci_lower=np.percentile(gf, 2.5) if len(gf) else np.nan,
        fpr_gap_ci_upper=np.percentile(gf, 97.5) if len(gf) else np.nan,
        p_heterogeneity_tpr=chi(1, 0, rel_tpr), p_heterogeneity_fpr=chi(0, 1, rel_fpr),
        n_levels_tpr=len(rel_tpr), n_levels_fpr=len(rel_fpr))


# ----------------------------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------------------------
def odds_ratio_ci(a, b, c, d, z=1.959964):
    """2x2: a=exposed cases, b=exposed non-cases, c=unexposed cases, d=unexposed non-cases (Haldane +0.5 if any 0)."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    orr = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, np.exp(np.log(orr) - z * se), np.exp(np.log(orr) + z * se)


def save_table(df, env, name, index=False):
    path = os.path.join(env["REV"], name)
    df.to_csv(path, index=index)
    print(f"Saved: {path}  ({len(df)} rows)")
    return path


def fmt_ci(est, lo, hi, nd=3):
    return f"{est:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


# ----------------------------------------------------------------------------------------
# One-standard-error selection and ensemble-strategy predictions (mirror ensemble.py exactly)
# ----------------------------------------------------------------------------------------
def one_se_select(comp):
    """
    Re-implements the selection step of ensemble.compare_ensemble_strategies on the saved comparison table:
      best = candidate with highest bootstrap-mean calibration AUC; SE = bootstrap SD of THAT candidate's AUC;
      eligible = candidates whose bootstrap mean >= best_mean - SE;
      winner = lowest complexity_rank among eligible (single=0 < equal_weight=1 < fitted=2),
               ties broken by highest bootstrap-mean AUC.
    Returns (winner_name, best_row_index, cutoff, eligible_frame).
    """
    best_i = comp["bootstrap_mean_auc"].idxmax()
    best_mean = comp.loc[best_i, "bootstrap_mean_auc"]
    se = comp.loc[best_i, "bootstrap_se"]
    cutoff = best_mean - se
    elig = comp[comp["bootstrap_mean_auc"] >= cutoff].sort_values(
        ["complexity_rank", "bootstrap_mean_auc"], ascending=[True, False])
    return elig.iloc[0]["strategy"], best_i, cutoff, elig


def strategy_proba(name, oof, target, seed=42, model_names=MODEL_NAMES):
    """
    Probabilities on `target` (DataFrame with one column per base learner) for any strategy name used in
    notebook 07. All weights / meta-learners are fitted on the TRAINING out-of-fold predictions only.
    """
    import ensemble as E
    if name.startswith("single_"):
        return target[name.replace("single_", "")].values
    if name == "nnls":
        w, _ = E.fit_nnls_weights(oof, model_names)
        return E.apply_ensemble(target, model_names, w)
    if name == "equal_weight":
        return E.apply_ensemble(target, model_names, E.equal_weight_ensemble(model_names))
    if name == "auc_optimized":
        w = E.fit_auc_optimized_weights(oof, model_names, seed=seed)
        return E.apply_ensemble(target, model_names, w)
    if name == "logistic_stacking":
        meta = E.fit_logistic_stacking(oof, model_names, seed=seed)
        return meta.predict_proba(target[model_names].values)[:, 1]
    if name == "greedy_selection":
        w = E.greedy_ensemble_selection(oof, model_names, seed=seed)
        return E.apply_ensemble(target, model_names, w)
    raise ValueError(name)


# Human-readable level labels for coded subgroup variables (from harmonization.py / variable descriptions)
LEVEL_LABELS = {
    "EducationLevel": {1: "secondary or less", 2: "above secondary"},
    "FamilyIncome1": {1: "income decreased (strain)", 2: "increased / unchanged"},
    "Work": {1: "employed", 2: "not employed"},
    "part1_country": {1: "Jordan", 2: "Palestine", 4: "Lebanon", 6: "Saudi Arabia", 8: "Bahrain"},
}


def pretty_levels(col, series):
    m = LEVEL_LABELS.get(col)
    if m is None:
        return series.astype(str).values
    return series.map(lambda v: m.get(v, str(v))).values
