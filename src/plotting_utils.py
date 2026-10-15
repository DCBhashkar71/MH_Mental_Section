"""
Shared plotting helper — used by every notebook to save manuscript-required
figures consistently into reports/figures/, with a subfolder per dataset.
"""
import os


def save_fig(fig, filename, project_root, dataset="general"):
    """
    Save a matplotlib figure to reports/figures/<dataset>/<filename>.

    Parameters
    ----------
    fig : matplotlib Figure object (the `fig` returned by plt.subplots())
    filename : str, e.g. "class_balance.png"
    project_root : str, the PROJECT_ROOT variable already defined in each notebook
    dataset : str, subfolder name — e.g. "pakistan", "arab5", "uganda", "postnatal", "cross_country"
    """
    out_dir = os.path.join(project_root, "reports", "figures", dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    return out_path
