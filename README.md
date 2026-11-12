# urban-heat-mapping-research

Reproducing and stress-testing **UrbanSENSE** (Berk et al., UNC Chapel Hill /
PNNL, preprint 2026) — a stratified-sensor-network approach to hourly,
10m-resolution urban heat stress mapping — as groundwork for a similar
held-out-sensor microclimate prediction problem in Ramat Gan.

Paper: [`paper/essoar.15007011_v1.pdf`](paper/essoar.15007011_v1.pdf)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
scripts/download_data.sh   # downloads the UrbanSENSE dataset (CC0) from UNC Dataverse
```

Then run the notebooks in order from `notebooks/` (each can also run
standalone).

## Contents

- `notebooks/01_data_exploration.ipynb` — sanity-checks the dataset against
  the paper and reproduces its descriptive findings (diurnal cycles by
  land-cover/shade, SVF/vegetation correlations, the night-time
  shaded-impervious anomaly, Heat Index).
- `notebooks/02_baselines.ipynb` — simple baselines (zero model, mean,
  stratified mean, linear regression) on the paper's own sensor-level
  train/test split, as a floor for judging the real model.
- `notebooks/03_model_reproduction.ipynb` — reproduces the authors' exact
  XGBoost pipeline (same features, split, hyperparameters) and compares
  numbers directly to the paper and to the notebook 02 baselines.
- `notebooks/04_weakness_analysis.ipynb` — residuals by sensor/stratum,
  feature importance, SHAP, and an extrapolation check, to find where the
  reproduced model is actually weak.
- `src/urbansense_common.py` — shared data loading, the sensor-level
  train/test split, feature lists and XGBoost hyperparameters, used
  identically by notebooks 02-04 so their numbers are directly comparable.
- `reference/UrbanSENSE_ML/` — the authors' own published code (notebooks,
  scripts), kept for comparison against our reproduction.
- `data/` — not committed (see `.gitignore`); recreated by
  `scripts/download_data.sh`.

## Dataset

UNC Dataverse, DOI [10.15139/S3/5EHFTA](https://doi.org/10.15139/S3/5EHFTA),
CC0. The dataset's own landing page blocks automated/browser access; the
Dataverse REST API (used by `scripts/download_data.sh`) does not.
