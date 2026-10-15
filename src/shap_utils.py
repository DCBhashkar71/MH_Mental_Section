"""
SHAP explainability helpers for Phase 9. Computes SHAP values for the
Random Forest proposed model (Phase 7) on each dataset, matching the
reference manuscript's approach (Section 2.10/3.6: SHAP on the tree-based
component of the final model).

NOTE ON TESTING: this module was NOT execution-tested (no shap package in
this environment, no internet to install it). shap's API has changed shape
conventions across versions for binary classifiers (list of two arrays vs.
a single 2D array vs. a 3D array), so compute_shap_values() handles all
three defensively -- but if you get an unexpected shape/type error, paste
the full traceback and the output of `print(shap.__version__)` so I can
pinpoint which convention your installed version uses.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_shap_values(pipeline, X_raw, sample_size=500, seed=42):
    """
    Compute SHAP values for the positive class from a fitted pipeline's
    tree-based classifier step (expects pipeline.named_steps["pre"] and
    pipeline.named_steps["clf"], matching this project's build_model_pipeline
    / build_preprocessor convention).

    Subsamples X_raw to sample_size rows for tractability (SHAP on trees is
    fast, but exact computation still scales with n_samples x n_trees).

    Returns: (shap_values_positive_class, feature_names, X_transformed, X_sample)
    """
    import shap

    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]

    if len(X_raw) > sample_size:
        X_sample = X_raw.sample(n=sample_size, random_state=seed)
    else:
        X_sample = X_raw.copy()

    X_transformed = pre.transform(X_sample)
    feature_names = pre.get_feature_names_out()

    explainer = shap.TreeExplainer(clf)
    raw_shap = explainer.shap_values(X_transformed)

    # Defensive handling of shap's differing output conventions across versions:
    if isinstance(raw_shap, list):
        # Older shap: list of arrays, one per class -- take positive class (index 1)
        shap_values_pos = raw_shap[1]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        # Newest shap: (n_samples, n_features, n_classes)
        shap_values_pos = raw_shap[:, :, 1]
    else:
        # Already a single (n_samples, n_features) array for the positive class
        shap_values_pos = raw_shap

    return shap_values_pos, feature_names, X_transformed, X_sample


def get_top_features(shap_values, feature_names, n=15):
    """Rank features by mean absolute SHAP value, matching the reference
    manuscript's Section 3.6 ranking table."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df.head(n)


def plot_shap_summary(shap_values, X_transformed, feature_names, project_root, dataset,
                       max_display=15, save_fig_fn=None):
    """
    Beeswarm-style SHAP summary plot, saved via the project's save_fig
    convention if save_fig_fn is provided (pass plotting_utils.save_fig).
    """
    import shap

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                       max_display=max_display, show=False)
    plt.tight_layout()

    if save_fig_fn is not None:
        save_fig_fn(plt.gcf(), f"{dataset}_shap_summary.png", project_root, dataset=dataset)

    plt.show()
    plt.close()


def plot_shap_bar(top_features_df, project_root, dataset, save_fig_fn=None):
    """Simple horizontal bar chart of mean |SHAP| per feature (cleaner for
    a manuscript figure than the beeswarm plot, which is denser)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ordered = top_features_df.sort_values("mean_abs_shap")
    ax.barh(ordered["feature"], ordered["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top predictors — {dataset}")
    plt.tight_layout()

    if save_fig_fn is not None:
        save_fig_fn(fig, f"{dataset}_shap_bar.png", project_root, dataset=dataset)

    plt.show()
    return fig
