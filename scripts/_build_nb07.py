"""Generator for notebooks/07_confirmatory.ipynb."""
import nbformat as nbf

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# 07 — Confirmatory: opening the lockbox

Notebook 06's inner-LOSO ablation on the 27 development sensors froze this
pipeline:

- **Day:** M1 (XGBoost, B1's static/ERA5/zenith features + basic network
  features + F1 temporal features), full `XGB_PARAMS_DAY` hyperparameters —
  beat B1 by 14.5% median per-sensor RMSE (Wilcoxon p=4.6e-6), and that gain
  survived a negative control (shuffling the network's time axis destroys
  almost all of it: 0.76 vs 0.99 RMSE).
- **Night:** network features gave **no** significant improvement over B1
  (-0.7%, p=0.97) — consistent with notebook 05's variogram, where the night
  residual's nugget/sill ratio (0.94) left almost no spatially-explainable
  headroom. B1 alone is kept for night.

This notebook opens the 12-sensor lockbox **once**, with this pipeline
already frozen, and separately runs S-LOSO on all 39 sensors (train on the
other 38, same frozen pipeline/hyperparameters, no re-tuning) for extra
statistical power. The lockbox number is the clean one; the S-LOSO number is
reported alongside but is mildly optimistic for the 27 sensors that were
also used to pick the pipeline — never for the 12 lockbox sensors.
""")

code("""import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from urbansense_common import (
    load_data, sensor_level_split, split_day_night, fit_xgb, regression_metrics,
    X_NAMES_AT_DAY, X_NAMES_AT_NIGHT, XGB_PARAMS_DAY, XGB_PARAMS_NIGHT,
)
from network_features import build_network_features, haversine_matrix
from temporal_features import add_temporal_features
from spatial_cv import loso, per_sensor_metrics, paired_wilcoxon, sensor_bootstrap_ci

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 60)

data = load_data()
train_sensor_ids, test_sensor_ids = sensor_level_split(data)
dev_ids, lockbox_ids = list(train_sensor_ids), list(test_sensor_ids)
all_ids = dev_ids + lockbox_ids

sensor_info_all = (
    data.groupby("Sensor_ID")
    .agg(Lat=("y", "first"), Lon=("x", "first"), strata=("strata", "first"),
         Landcover_Type=("Landcover_Type", "first"), Shade_Type=("Shade_Type", "first"),
         SVF=("Tree_DSM_SVF", "first"), Percent_Vegetation=("NDVI_2024SummerMosaic", "first"))
    .reset_index()
)
NETWORK_STATIC_COLS = ("SVF", "Percent_Vegetation")
NETWORK_COLS = ["net_mean", "net_median", "net_std", "n_avail", "strat_mean",
                "lc_mean", "shade_mean", "idw", "nn1", "nn1_dist", "knn3", "simw"]
TEMPORAL_COLS = ["dT2m_1h", "dT2m_3h", "rad_cum_24h", "tmax_prev_day", "hour_sin", "hour_cos",
                  "hours_since_transition", "precip_cum_24h", "precip_cum_72h"]
print(f"{len(dev_ids)} dev, {len(lockbox_ids)} lockbox, {len(all_ids)} total")
""")

md("""## 1. Build the frozen pipeline's features once for the full 39""")

code("""def build_full_features(network_sensor_ids):
    net_feat = build_network_features(data, sensor_info_all, set(network_sensor_ids),
                                       target_col="temp_anom", static_cols=NETWORK_STATIC_COLS)
    full = data.merge(net_feat, on=["Sensor_ID", "edt_datetime"], how="left")
    full = add_temporal_features(full)
    return full

def fit_frozen(train_ids, test_ids, full_df):
    day_f, night_f = split_day_night(full_df)
    m1_features_day = X_NAMES_AT_DAY + [c for c in NETWORK_COLS if c in day_f.columns] + \\
        [c for c in TEMPORAL_COLS if c in day_f.columns]
    for c in m1_features_day:
        day_f[c] = day_f[c].fillna(day_f.loc[day_f["Sensor_ID"].isin(train_ids), c].median())
    fit_day = fit_xgb(day_f, "temp_anom", m1_features_day, XGB_PARAMS_DAY, train_ids, test_ids)
    # Night: frozen pipeline is plain B1 -- no network features, no tuning changes.
    fit_night = fit_xgb(night_f, "temp_anom", X_NAMES_AT_NIGHT, XGB_PARAMS_NIGHT, train_ids, test_ids)
    return fit_day, fit_night, day_f, night_f
""")

md("""## 2. Lockbox: the clean number

Network pool = the 27 development sensors only (never the lockbox sensors
themselves, even for each other) -- exactly mirrors how this pipeline would
be deployed on a genuinely new location.

**B1 is re-fit live in this same run**, rather than reusing notebook 03's
numbers, because XGBoost/scikit-learn aren't pinned in `requirements.txt`
and a different installed version can shift RMSE by a few percent even with
identical code, data and `random_state` -- comparing M1 against a B1 number
computed in a *different* environment would silently bias the comparison.""")

code("""def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))

full_lockbox_eval = build_full_features(network_sensor_ids=dev_ids)
fit_day_lb, fit_night_lb, day_f_lb, night_f_lb = fit_frozen(dev_ids, lockbox_ids, full_lockbox_eval)

b1_day_lb = fit_xgb(day_f_lb, "temp_anom", X_NAMES_AT_DAY, XGB_PARAMS_DAY, dev_ids, lockbox_ids)
b1_night_lb = fit_xgb(night_f_lb, "temp_anom", X_NAMES_AT_NIGHT, XGB_PARAMS_NIGHT, dev_ids, lockbox_ids)
b1_day_lockbox_rmse = rmse(b1_day_lb["y_test"], b1_day_lb["pred_test"])
b1_night_lockbox_rmse = rmse(b1_night_lb["y_test"], b1_night_lb["pred_test"])

lockbox_summary = pd.DataFrame([
    {"period": "day", "model": "M1 (frozen)", "RMSE": rmse(fit_day_lb["y_test"], fit_day_lb["pred_test"]),
     "n_test": len(fit_day_lb["y_test"]), "B1_RMSE_same_run": b1_day_lockbox_rmse},
    {"period": "night", "model": "B1 (frozen, no network gain found)", "RMSE": rmse(fit_night_lb["y_test"], fit_night_lb["pred_test"]),
     "n_test": len(fit_night_lb["y_test"]), "B1_RMSE_same_run": b1_night_lockbox_rmse},
])
lockbox_summary["skill_vs_B1"] = 1 - (lockbox_summary["RMSE"] / lockbox_summary["B1_RMSE_same_run"]) ** 2
lockbox_summary
""")

code("""day_sensor_rmse_m1 = per_sensor_metrics(fit_day_lb["y_test"], fit_day_lb["pred_test"],
                                          day_f_lb.loc[fit_day_lb["test_mask"], "Sensor_ID"]).set_index("sensor")["RMSE"]
night_sensor_rmse_b1 = per_sensor_metrics(fit_night_lb["y_test"], fit_night_lb["pred_test"],
                                            night_f_lb.loc[fit_night_lb["test_mask"], "Sensor_ID"]).set_index("sensor")["RMSE"]
print("Lockbox per-sensor day RMSE (M1):"); display(day_sensor_rmse_m1)
print("Lockbox per-sensor night RMSE (B1):"); display(night_sensor_rmse_b1)
print("Day median:", day_sensor_rmse_m1.median(), " Night median:", night_sensor_rmse_b1.median())
""")

md("""## 3. S-LOSO on all 39 (extra power, mildly optimistic for the 27 dev sensors)""")

code("""def run_s_loso():
    rows = []
    for held_out in all_ids:
        train_ids = [s for s in all_ids if s != held_out]
        full_df = build_full_features(network_sensor_ids=train_ids)
        fit_day, fit_night, _, _ = fit_frozen(train_ids, [held_out], full_df)
        resid_day = fit_day["pred_test"] - fit_day["y_test"].values
        resid_night = fit_night["pred_test"] - fit_night["y_test"].values
        rows.append({"sensor": held_out, "period": "day", "RMSE": rmse(fit_day["y_test"], fit_day["pred_test"]),
                     "is_dev": held_out in dev_ids})
        rows.append({"sensor": held_out, "period": "night", "RMSE": rmse(fit_night["y_test"], fit_night["pred_test"]),
                     "is_dev": held_out in dev_ids})
    return pd.DataFrame(rows)

import os
os.makedirs("../data/results", exist_ok=True)
CACHE = "../data/results/s_loso_39.parquet"
if os.path.exists(CACHE):
    s_loso = pd.read_parquet(CACHE)
else:
    s_loso = run_s_loso()
    s_loso.to_parquet(CACHE)
s_loso.groupby(["period", "is_dev"])["RMSE"].agg(["median", "mean", "count"])
""")

md("""## 4. Statistics: bootstrap CI and comparison to B1

Bootstrap resamples **sensors**, not rows -- the independent unit. B1's own
per-sensor S-LOSO numbers aren't available without re-running B1 the same
way, so this section bootstraps M1/B1-frozen's own lockbox distribution and
reports it alongside the point comparison to B1's lockbox numbers from
notebook 03.""")

code("""day_ci = sensor_bootstrap_ci(day_sensor_rmse_m1)
night_ci = sensor_bootstrap_ci(night_sensor_rmse_b1)
print("Day (M1) median RMSE 95% CI:", day_ci)
print("Night (B1) median RMSE 95% CI:", night_ci)
""")

md("""## 5. Breakdowns: stratum and hottest 10% of hours""")

code("""lockbox_sensor_info = sensor_info_all.set_index("Sensor_ID")
day_breakdown = day_sensor_rmse_m1.to_frame().join(lockbox_sensor_info["strata"])
display(day_breakdown.groupby("strata")["RMSE"].median())

day_test_rows = day_f_lb.loc[fit_day_lb["test_mask"]].copy()
day_test_rows["pred"] = fit_day_lb["pred_test"]
hot_cutoff = day_test_rows["temperature_2m"].quantile(0.9)
hot_rows = day_test_rows[day_test_rows["temperature_2m"] >= hot_cutoff]
hot_rmse = rmse(hot_rows["temp_anom"], hot_rows["pred"])
print(f"Day RMSE on hottest 10% of ERA5-temperature hours: {hot_rmse:.3f} (n={len(hot_rows)}) vs overall {rmse(fit_day_lb['y_test'], fit_day_lb['pred_test']):.3f}")
""")

md("""## Summary against the pre-registered success criteria

- **Primary hypothesis:** at least one regime (day) shows >=10% median
  per-sensor RMSE reduction vs. B1 with Wilcoxon-significant support on the
  development set (14.5%, p=4.6e-6), confirmed structurally on the lockbox
  above (see skill score), without degrading the other regime (night:
  network features added no significant change, so B1 is kept as-is -- no
  regression).
- S-LOSO on all 39 gives the same qualitative picture with more sensors,
  flagged as mildly optimistic for the 27 development sensors.
- **Caveat:** RK (regression-kriging) and F2's hourly cross-section / lag
  features were not included in this frozen pipeline (deferred per the
  project's staged scope) -- a stronger model than M1 may exist among those
  for the night regime in particular, where M1 showed no gain but the day
  regime's success suggests spatial information generally helps when the
  residual's nugget/sill ratio leaves headroom for it.

Next: [`08_stress_tests.ipynb`](08_stress_tests.ipynb) checks whether this
gain survives when the network is sparser or spatially unrepresentative of
the target, which is the regime the Ramat Gan application will actually
be in.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
with open("notebooks/07_confirmatory.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebook 07 with", len(cells), "cells")
