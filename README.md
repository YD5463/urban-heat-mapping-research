# urban-heat-mapping-research

Reproducing and stress-testing **UrbanSENSE** (Berk et al., UNC Chapel Hill /
PNNL, preprint 2026) — a stratified-sensor-network approach to hourly,
10m-resolution urban heat stress mapping — as groundwork for a similar
held-out-sensor microclimate prediction problem in Ramat Gan.

Paper: [`paper/essoar.15007011_v1.pdf`](paper/essoar.15007011_v1.pdf)

## Setup

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
scripts/download_data.sh        # downloads the UrbanSENSE dataset (CC0) from UNC Dataverse
python scripts/build_sensor_info.py   # derives data/processed/sensor_info.csv from the raw table
```

`scripts/download_data.sh` needs `curl` and `unzip` (not `rsync`) on the
PATH. Then run the notebooks in order from `notebooks/` (each can also run
standalone; 05-09 cache intermediate results under `data/results/` so
re-running is fast after the first pass).

## Contents

**Reproduction (01-04):**
- `notebooks/01_data_exploration.ipynb` — sanity-checks the dataset against
  the paper and reproduces its descriptive findings (diurnal cycles by
  land-cover/shade, SVF/vegetation correlations, the night-time
  shaded-impervious anomaly, Heat Index).
- `notebooks/02_baselines.ipynb` — simple baselines (zero model, mean,
  stratified mean, linear regression) on the paper's own sensor-level
  train/test split, as a floor for judging the real model.
- `notebooks/03_model_reproduction.ipynb` — reproduces the authors' exact
  XGBoost pipeline (same features, split, hyperparameters): day RMSE
  1.169°C, night RMSE 0.691°C, matching the paper.
- `notebooks/04_weakness_analysis.ipynb` — residuals by sensor/stratum,
  feature importance, SHAP, and an extrapolation check.

**Network-signal hypothesis (05-09):** does reading N other sensors at the
same hour improve prediction at an unseen N+1th location, and how many
sensors does a new deployment (e.g. Ramat Gan) actually need?
- `notebooks/05_network_signal.ipynb` — go/no-go: variance decomposition and
  variogram of the reproduced model's residuals, restricted to the 27
  development sensors (the 12-sensor lockbox is never touched until 07).
  Result: **GO** — residuals show significant spatial correlation
  (day p=0.0016, night p<1e-12) even though the model already absorbs
  almost all of the shared hourly offset implicitly.
- `notebooks/06_network_models_dev.ipynb` — builds network features and
  ablates B0/B1/B2-B4/M1(XGB+network)/M2(Ridge+network) via inner LOSO on
  the 27 development sensors, plus an overfitting diagnostic and a
  time-shuffle negative control. Day: M1 beats B1 by 14.5% median
  per-sensor RMSE (p=4.6e-6), confirmed not an artifact of the shuffle
  control. Night: no significant gain — consistent with notebook 05's
  variogram showing little spatial headroom at night.
- `notebooks/07_confirmatory.ipynb` — opens the 12-sensor lockbox once with
  the pipeline frozen (M1 for day, plain B1 for night): day RMSE 1.101°C vs.
  B1's 1.169°C (~5.8% RMSE / 11.4% MSE-skill improvement); also runs S-LOSO
  across all 39 sensors for extra power.
- `notebooks/08_stress_tests.ipynb` — buffered LOSO (gain survives out to a
  1km exclusion radius), leave-one-stratum-out and spatial-cluster holdouts
  (gain degrades substantially with no representative network sensor
  nearby), and dropout robustness (stable up to 50% hourly network dropout).
- `notebooks/09_sensor_count.ipynb` — the sensor-count curve: nested,
  common-random-numbers permutations of network subsets at fixed outer test
  sensors, a saturation-curve fit for N* (not the noisy curve's argmin), an
  inner-LOSO-vs-outer-test bias check, and density/dissimilarity context for
  translating Durham's N* to a different city.

**Shared code:**
- `src/urbansense_common.py` — data loading, the sensor-level train/test
  split, feature lists, XGBoost hyperparameters and `fit_xgb`, used
  identically across every notebook.
- `src/network_features.py` — leakage-safe network aggregate features
  (leave-self-out enforced for every row; a held-out sensor's data is never
  in another row's feature pool); vectorized per-hour across targets.
- `src/temporal_features.py` — causal per-sensor dynamics/thermo-history
  features (F1).
- `src/spatial_cv.py` — sensor-level fold generators (LOSO, buffered,
  leave-one-stratum-out, spatial clusters, nested N-subsets) and evaluation
  statistics (per-sensor RMSE, paired Wilcoxon, sensor-level bootstrap CI).
- `src/sensor_count.py` — saturation-curve fit, N* estimation (avoids
  picking the noisy curve's argmin), and the Area-of-Applicability-style
  dissimilarity index; synthetic-data validated.
- `scripts/build_sensor_info.py` — derives per-sensor static metadata from
  the raw table (no code in the original release produces this file).
- `scripts/_build_nb0{5,6,7,8,9}.py` — generate notebooks 05-09 from code
  (nbformat); re-run one of these after editing it, then re-execute the
  corresponding notebook, rather than hand-editing the `.ipynb` JSON.
- `reference/UrbanSENSE_ML/` — the authors' own published code (notebooks,
  scripts), kept for comparison against our reproduction.
- `data/` — not committed (see `.gitignore`); recreated by
  `scripts/download_data.sh` and `scripts/build_sensor_info.py`.

**Scope notes:** F3-F7 feature families (multi-scale static predictors,
morphology, topography, LST, boundary-layer height) require external data
(GEE/Planetary Computer extraction, DEM, WUDAPT LCZ, a CDS account) and were
deferred — the hypothesis test above uses only F1/F2-lite network and
temporal features, which were already sufficient to confirm the day-time
result. Regression-kriging (RK) was designed in the project plan but not
implemented in this pass. Notebook 09's nested design is run at reduced
scale (4 outer folds x 1 seed x 5 permutations, fixed hyperparameters per N)
rather than the full 2-seed x 10-permutation x inner-hyperparameter-search
design, per the plan's own compute-budget fallback.

## Dataset

UNC Dataverse, DOI [10.15139/S3/5EHFTA](https://doi.org/10.15139/S3/5EHFTA),
CC0. The dataset's own landing page blocks automated/browser access; the
Dataverse REST API (used by `scripts/download_data.sh`) does not.
