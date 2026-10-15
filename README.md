# Predicting Maternal Mental Health Risk

Machine learning pipeline for perinatal depression and anxiety risk prediction,
integrating clinical, symptom, and social-determinants-of-health (SDoH) data
across four countries (Pakistan, Jordan/Palestine/Lebanon/Saudi Arabia/Bahrain,
Uganda), with a formal data-leakage audit and cross-country external validation.

## Setup

**Option A — pip**
```bash
conda create -n matmh python=3.11 -y
conda activate matmh
pip install -r requirements.txt
python -m ipykernel install --user --name matmh --display-name "Python (matmh)"
```

**Option B — conda (one shot)**
```bash
conda env create -f environment.yml
conda activate matmh
python -m ipykernel install --user --name matmh --display-name "Python (matmh)"
```

Then:
```bash
jupyter notebook
```
Open `notebooks/00_environment_check.ipynb` first and run all cells — it verifies
your setup and creates the required (currently empty) data/model/report subfolders.

## Project layout

```
data/
  raw/                 # original files — never edited
  interim/             # cleaned, not yet feature-engineered
  processed/           # final model-ready train/calibration/test tables
  external_validation/ # harmonized cross-country feature tables
notebooks/              # all analysis, numbered in run order
src/                    # reusable code imported by notebooks
models/                 # saved fitted models, per dataset
reports/
  figures/
  tables/
config/
  config.yaml           # seeds, paths, CV/Optuna/bootstrap settings
```

## Data sources

| File | Source | Population | n |
|---|---|---|---|
| `pakistan_raw.csv` (`dataset__1_.csv`) | PERI_DEP (Zafar et al., 2025) | Lahore/Gujranwala, Pakistan | 14,008 |
| `pakistan_oversampled.csv` | PERI_DEP, pre-oversampled by original authors (reference only) | same | 24,660 |
| `arab5_data.xlsx` (`ML-DataSet.xlsx`) | Qasrawi et al., 2022, F1000Research | Jordan, Palestine, Lebanon, Saudi Arabia, Bahrain | 3,569 |
| `uganda.csv` (`records.csv`) | Ugandan perinatal anxiety screening program | Uganda | 14,325 |
| `postnatal.csv` (`post_natal_data.csv`) | EPDS-style checklist | unspecified | 1,503 |

## Reproducibility

- Random seed fixed via `config/config.yaml` (`seed: 42`) and used everywhere.
- All preprocessing (imputation, encoding, SMOTE) is fit inside
  `imblearn.pipeline.Pipeline` on training folds only — never on the full dataset
  before splitting.
- `data/raw/` is treated as read-only; all transformations are code, not manual edits.

## Status

- [x] Folder structure created
- [x] Environment files created
- [ ] `00_environment_check.ipynb` run
- [ ] Leakage audit (`02_leakage_audit.ipynb`)
- [ ] Preprocessing pipeline
- [ ] Base learners + Optuna tuning
- [ ] NNLS ensemble
- [ ] Deep learning comparison
- [ ] SHAP / calibration / DCA / fairness
- [ ] Cross-country external validation
