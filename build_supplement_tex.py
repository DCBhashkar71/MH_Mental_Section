"""
build_supplement_tex.py -- builds Supplementary Tables S1-S11 (Study 2 revision) as LaTeX straight from the CSV files written by
phases 16-28 (reports/tables/revision/). No numbers are typed by hand.

Usage (from the project root, or pass the folder):
    python build_supplement_tex.py reports/tables/revision supplementary_study2.tex
Then upload supplementary_study2.tex to Overleaf (or compile it as a separate document) -- it is a standalone article.
"""
import os, sys
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "reports/tables/revision"
OUT = sys.argv[2] if len(sys.argv) > 2 else "supplementary_study2.tex"

def esc(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s

def cell(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", "{,}") if abs(v) >= 10000 else str(v)
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000: return f"{v:,.0f}".replace(",", "{,}")
        return f"{v:.{nd}f}".replace("-", "$-$")
    return esc(v)

def table(fname, caption, label, cols=None, rename=None, nd=3, size=r"\tiny", head=None, filt=None):
    p = os.path.join(SRC, fname)
    if not os.path.exists(p):
        return f"% missing {fname}\n"
    df = pd.read_csv(p)
    if filt is not None: df = filt(df)
    if cols: df = df[[c for c in cols if c in df.columns]]
    if head: df = df.head(head)
    rename = rename or {}
    spec = []
    for c in df.columns:
        if df[c].dtype == object:
            L = df[c].astype(str).str.len().max()
            spec.append("p{3.4cm}" if L > 32 else "l")
        else:
            spec.append("r")
    hdr = " & ".join(r"\textbf{%s}" % esc(rename.get(c, c)) for c in df.columns) + r" \\"
    rows = "\n".join(" & ".join(cell(v, nd) for v in r) + r" \\" for r in df.itertuples(index=False))
    return (f"{size}\n\\setlength{{\\tabcolsep}}{{2pt}}\n\\begin{{longtable}}{{{''.join(spec)}}}\n"
            f"\\caption*{{\\textbf{{Supplementary Table {label.upper()}.}} {caption}}}\\\\\n\\toprule\n{hdr}\n\\midrule\n\\endfirsthead\n"
            f"\\toprule\n{hdr}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n{rows}\n\\bottomrule\n\\end{{longtable}}\n\\normalsize\n")

doc = [r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[landscape,margin=1.5cm]{geometry}
\usepackage{booktabs,longtable,amsmath}
\usepackage{caption}
\begin{document}
\title{Supplementary Material: Study 2 (maternal mental health risk)}
\author{}\date{}
\maketitle
\noindent All tables are generated directly from the analysis outputs (reports/tables/revision) by build\_supplement\_tex.py.
Confidence intervals are 95\% percentile bootstrap intervals (2000 resamples) unless stated otherwise. AUC = area under the receiver operating characteristic curve;
RF = random forest; CI = confidence interval; FPR = false-positive rate; TPR = true-positive rate.
"""]
add = doc.append

# S1
add(table("S2_participant_flow.csv", "Participant flow and partition characteristics (Study 2). The partition rows give n, percentage and prevalence.", "s1a"))
add(table("S2_split_audit.csv", "Split audit: partition sizes, class balance, distinct feature-profile signatures and cross-partition overlap. Rows labelled OVERLAP give the number of signatures shared by two partitions (n, n\\_unique\\_signatures) and the number of rows in the later partition carrying such a signature (n\\_positive).", "s1b",
         cols=["dataset","partition","n","pct_of_analytic","n_positive","prevalence","prevalence_diff_vs_overall","n_unique_signatures","pct_rows_in_duplicated_profile","max_rows_sharing_one_signature"]))
# S2
add(table("S2_signature_definition.csv", "Variables entering the feature-profile signature (values used as stored, no rounding or binning; missing values written as MISSING).", "s2a"))
add(table("S2_split_audit.csv", "Cross-partition overlap of feature-profile signatures. n and n\\_unique\\_signatures = number of signatures shared by the two partitions; n\\_positive = number of rows in the later partition carrying a shared signature (0 means no overlap).", "s2b",
         cols=["dataset","partition","n","n_positive","n_unique_signatures"], filt=lambda d: d[d["partition"].astype(str).str.startswith("OVERLAP")]))
# S3
add(table("S2_validation_strategy_table.csv", "Validation-strategy comparison: random forest AUC under ordinary and group-aware validation. `tuned' = hyperparameters selected under group-aware cross-validation; `default' = 300 trees, unlimited depth. signature\\_exposure = share of validation rows whose exact feature signature also occurs in the training rows.", "s3"))
# S4
add(table("S2_selection_candidates_all.csv", "All twelve candidates per analysis on the calibration partition (bootstrap mean AUC, standard error, one-standard-error cutoff, eligibility and selection).", "s4a",
         cols=["dataset","strategy","calibration_auc","bootstrap_mean_auc","bootstrap_se","complexity_rank","rank_by_bootstrap_mean","one_se_cutoff","within_1se_of_best","selected"]))
add(table("S2_selection_oof_sensitivity.csv", "One-standard-error rule repeated on training out-of-fold predictions (single base learners only).", "s4b"))
add(table("S2_model_selection_audit.csv", "Model-selection audit per analysis.", "s4c"))
add(table("S2_rf_vs_selected_test.csv", "Test-partition AUC of the selected candidate and of random forest, with paired difference.", "s4d"))
# S5
add(table("S2_ensemble_vs_fixed_comparator.csv", "Candidates versus a comparator fixed on training out-of-fold AUC (paired bootstrap, identical resamples; Holm-adjusted $p$ within each analysis).", "s5a"))
add(table("S2_test_auc_all_strategies.csv", "Test AUC of every candidate strategy.", "s5b"))
# S6
add(table("S2_calibration_metrics.csv", "Calibration metrics of the proposed random forest, raw and after logistic recalibration fitted on the calibration partition.", "s6a"))
add(table("S2_calibration_bins.csv", "Reliability bins (quantile bins; Wilson 95\\% intervals).", "s6b"))
# S7, S8
add(table("S2_subgroup_extended.csv", "Subgroup error rates, calibration and within-group AUC (proposed random forest, test partitions). Strata with fewer than 10 positives (TPR) or negatives (FPR) are flagged unreliable.", "s7",
         cols=["dataset","variable","subgroup_level","n","n_positive","prevalence","predicted_positive_rate","mean_predicted_risk","tpr_recall","tpr_ci_lower","tpr_ci_upper","fpr","fpr_ci_lower","fpr_ci_upper","ppv_precision","auc","tpr_reliable","fpr_reliable"]))
add(table("S2_subgroup_gaps.csv", "Between-group gaps in TPR and FPR with bootstrap intervals and chi-square heterogeneity tests.", "s8a"))
add(table("S2_subgroup_ablation_gaps.csv", "Gaps after refitting without the stratifying variable (hyperparameters fixed, threshold re-tuned on the calibration partition).", "s8b"))
# S9
add(table("S2_uganda_cutoff_sensitivity.csv", "Uganda: sensitivity of discrimination to the dichotomisation cutoff of the anxiety score.", "s9a"))
add(table("S2_uganda_continuous_model.csv", "Uganda: random forest regression on the continuous anxiety score (test partition).", "s9b"))
add(table("S2_uganda_score_distribution.csv", "Uganda: distribution of the anxiety score.", "s9c"))
# S10
add(table("S2_xnat_feature_odds_ratios.csv", "Cross-national diagnostics: odds ratio of depression for each harmonised construct in each cohort.", "s10a"))
add(table("S2_xnat_logistic_transfer.csv", "Cross-national transfer of a plain logistic regression on the five harmonised features.", "s10b"))
add(table("S2_xnat_single_feature_transfer.csv", "Cross-national transfer of each construct alone, using the source-cohort direction of association.", "s10c"))
add(table("S2_xnat_profile_agreement.csv", "Profile-level prevalence in the two cohorts (profiles with at least 10 women in both cohorts).", "s10d"))
# S11
add(table("S2_environment.csv", "Software environment.", "s11a"))
add(table("S2_trial_budget.csv", "Hyperparameter-search budget per learner.", "s11b"))
add(table("S2_optuna_search_spaces.csv", "Optuna search spaces.", "s11c"))
add(table("S2_hyperparameters_all_models.csv", "Selected hyperparameters of all models.", "s11d"))
add(table("S2_outcome_definitions.csv", "Outcome definitions.", "s11e"))
add(table("S2_feature_inventory.csv", "Feature inventory with dataset variable names.", "s11f"))
add(table("S2_code_manifest.csv", "Code manifest (file names and checksums).", "s11g"))
add(table("S2_split_reproduction_check.csv", "Regeneration check: splits rebuilt from the interim files compared with the saved partitions.", "s11h"))
add("\\noindent Hashed split manifests for each analysis (S2\\_split\\_manifest\\_*.csv) are provided in the repository.\n\\end{document}")
open(OUT, "w", encoding="utf-8").write("\n".join(doc))
print("wrote", OUT)
