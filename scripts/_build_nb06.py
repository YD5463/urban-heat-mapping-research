"""Generator for notebooks/06_network_models_dev.ipynb."""
import nbformat as nbf

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# 06 — Network models, development set

Stage 1 (notebook 05) was GO: B1's residuals show significant spatial
correlation (day p=0.0016, night p<1e-12), even though B1 already absorbs
almost all of the common-hourly-offset variance implicitly. This notebook
builds explicit network features and tests whether they close that gap.

**Lockbox discipline:** every decision here — which features help, which
model wins — is made on **inner LOSO over the 27 development sensors only**.
The 12 lockbox sensors are opened once, in notebook 07, with the pipeline
frozen by the time we get there.
""")

code("""import sys
sys.path.append("../src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

from urbansense_common import (
    load_data, sensor_level_split, split_day_night, fit_xgb, regression_metrics,
    X_NAMES_AT_DAY, X_NAMES_AT_NIGHT, XGB_PARAMS_DAY, XGB_PARAMS_NIGHT, XGB_PARAMS_FAST,
)
from network_features import build_network_features, haversine_matrix
from temporal_features import add_temporal_features
from spatial_cv import loso, per_sensor_metrics, paired_wilcoxon, sensor_bootstrap_ci

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 60)

data = load_data()
train_sensor_ids, test_sensor_ids = sensor_level_split(data)
dev_ids = list(train_sensor_ids)
dev = data[data["Sensor_ID"].isin(dev_ids)].copy()

sensor_info = (
    dev.groupby("Sensor_ID")
    .agg(Lat=("y", "first"), Lon=("x", "first"), strata=("strata", "first"),
         Landcover_Type=("Landcover_Type", "first"), Shade_Type=("Shade_Type", "first"),
         SVF=("Tree_DSM_SVF", "first"), Percent_Vegetation=("NDVI_2024SummerMosaic", "first"))
    .reset_index()
)
dist = haversine_matrix(sensor_info)
print(f"{len(dev_ids)} development sensors")
""")

md("""## 1. Build network features once per LOSO fold

For LOSO fold "held out = h", the network pool is `dev_ids - {h}` for
*every* row in that fold — including the training sensors' own rows. This is
what makes leave-self-out alone insufficient: if the pool included h, a
training sensor j's feature would use h's reading at hours that overlap
h's own test hours, which is fine in itself, but h's reading would never
have been excludable from its *own* test row either way — the real risk
avoided here is that h must be *completely* absent from the pool for every
row in this fold, train or test.
""")

code("""NETWORK_STATIC_COLS = ("SVF", "Percent_Vegetation")

def build_fold_network_features(fold_ids, held_out):
    pool = [s for s in fold_ids if s != held_out]
    fold_data = dev[dev["Sensor_ID"].isin(fold_ids)]
    feat = build_network_features(fold_data, sensor_info, set(pool), target_col="temp_anom",
                                   static_cols=NETWORK_STATIC_COLS)
    return feat

NETWORK_COLS = ["net_mean", "net_median", "net_std", "n_avail", "strat_mean",
                "lc_mean", "shade_mean", "idw", "nn1", "nn1_dist", "knn3", "simw"]
""")

md("""## 2. Models

| code | features |
|---|---|
| B0 | ERA5 only (predicted anomaly = 0) |
| B1 | paper's static + ERA5 + zenith features |
| B2/B3/B4 | linear regression of anomaly on net_mean / strat_mean / idw alone |
| M1 | XGB, B1 features + network features + F1 temporal |
| M2 | Ridge, same features as M1 (linear control) |
| RK | B1 prediction + IDW-interpolated out-of-fold B1 residual correction |
""")

code("""def eval_b0(df, target_col, era5_col):
    # predicted anomaly is always 0 -> absolute prediction is just ERA5.
    return df[target_col].values, np.zeros(len(df))

def eval_single_feature_linear(train_df, test_df, target_col, feat_col):
    sub_train = train_df.dropna(subset=[feat_col, target_col])
    if len(sub_train) < 5 or test_df[feat_col].isna().all():
        return test_df[target_col].values, np.full(len(test_df), sub_train[target_col].mean() if len(sub_train) else 0.0)
    model = LinearRegression()
    model.fit(sub_train[[feat_col]].values, sub_train[target_col].values)
    pred = np.where(test_df[feat_col].isna(), sub_train[target_col].mean(), np.nan)
    valid = ~test_df[feat_col].isna()
    pred = np.full(len(test_df), sub_train[target_col].mean())
    if valid.any():
        pred[valid.values] = model.predict(test_df.loc[valid, [feat_col]].values)
    return test_df[target_col].values, pred

def eval_m2_ridge(train_df, test_df, target_col, feat_cols):
    cols = [c for c in feat_cols if c in train_df.columns]
    train_sub = train_df.dropna(subset=cols + [target_col])
    test_sub = test_df.copy()
    for c in cols:
        test_sub[c] = test_sub[c].fillna(train_sub[c].median())
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_sub[cols].values)
    X_test = scaler.transform(test_sub[cols].values)
    model = Ridge(alpha=1.0)
    model.fit(X_train, train_sub[target_col].values)
    return test_sub[target_col].values, model.predict(X_test)
""")

md("""## 3. Inner-LOSO ablation loop

For every development sensor, held out once: fit B0/B1/B2-B4/M1/M2 on the
other 26, predict on the held-out sensor, for day and night separately.
`XGB_PARAMS_FAST` is used throughout this ablation (many repeated fits);
the confirmatory run in notebook 07 re-fits the frozen winner with the full
paper hyperparameters.""")

code("""def run_ablation(period):
    X_NAMES = X_NAMES_AT_DAY if period == "day" else X_NAMES_AT_NIGHT
    rows = []
    for held_out in dev_ids:
        train_ids = [s for s in dev_ids if s != held_out]
        net_feat = build_fold_network_features(dev_ids, held_out)
        fold_data = dev[dev["Sensor_ID"].isin(dev_ids)].merge(
            net_feat, on=["Sensor_ID", "edt_datetime"], how="left")
        fold_data = add_temporal_features(fold_data)
        day_f, night_f = split_day_night(fold_data)
        df = day_f if period == "day" else night_f

        train_df = df[df["Sensor_ID"].isin(train_ids)]
        test_df = df[df["Sensor_ID"] == held_out]
        if len(test_df) == 0 or len(train_df) == 0:
            continue

        y_true_b0, pred_b0 = eval_b0(test_df, "temp_anom", "temperature_2m")

        fit_b1 = fit_xgb(df, "temp_anom", X_NAMES, XGB_PARAMS_FAST, train_ids, [held_out])

        y_true_b2, pred_b2 = eval_single_feature_linear(train_df, test_df, "temp_anom", "net_mean")
        y_true_b3, pred_b3 = eval_single_feature_linear(train_df, test_df, "temp_anom", "strat_mean")
        y_true_b4, pred_b4 = eval_single_feature_linear(train_df, test_df, "temp_anom", "idw")

        m1_features = X_NAMES + [c for c in NETWORK_COLS if c in df.columns] + \\
            ["dT2m_1h", "dT2m_3h", "rad_cum_24h", "tmax_prev_day", "hour_sin", "hour_cos",
             "hours_since_transition", "precip_cum_24h", "precip_cum_72h"]
        m1_features = [c for c in m1_features if c in df.columns]
        df_m1 = df.copy()
        for c in m1_features:
            df_m1[c] = df_m1[c].fillna(df_m1.loc[df_m1["Sensor_ID"].isin(train_ids), c].median())
        fit_m1 = fit_xgb(df_m1, "temp_anom", m1_features, XGB_PARAMS_FAST, train_ids, [held_out])

        y_true_m2, pred_m2 = eval_m2_ridge(df_m1[df_m1["Sensor_ID"].isin(train_ids)],
                                            df_m1[df_m1["Sensor_ID"] == held_out], "temp_anom", m1_features)

        for name, y_true, pred in [
            ("B0", y_true_b0, pred_b0),
            ("B1", fit_b1["y_test"].values, fit_b1["pred_test"]),
            ("B2_net_mean", y_true_b2, pred_b2),
            ("B3_strat_mean", y_true_b3, pred_b3),
            ("B4_idw", y_true_b4, pred_b4),
            ("M1_xgb_network", fit_m1["y_test"].values, fit_m1["pred_test"]),
            ("M2_ridge_network", y_true_m2, pred_m2),
        ]:
            resid = pred - y_true
            rows.append({"period": period, "model": name, "sensor": held_out, "n": len(y_true),
                         "RMSE": float(np.sqrt(np.mean(resid ** 2)))})
    return pd.DataFrame(rows)

import os
os.makedirs("../data/results", exist_ok=True)
CACHE_DAY, CACHE_NIGHT = "../data/results/ablation_day.parquet", "../data/results/ablation_night.parquet"
if os.path.exists(CACHE_DAY) and os.path.exists(CACHE_NIGHT):
    ablation_day, ablation_night = pd.read_parquet(CACHE_DAY), pd.read_parquet(CACHE_NIGHT)
else:
    ablation_day = run_ablation("day")
    ablation_night = run_ablation("night")
    ablation_day.to_parquet(CACHE_DAY)
    ablation_night.to_parquet(CACHE_NIGHT)
ablation = pd.concat([ablation_day, ablation_night], ignore_index=True)
print(ablation.shape)
""")

md("""## 4. Results: per-sensor median RMSE and paired Wilcoxon vs. B1""")

code("""summary = (ablation.groupby(["period", "model"])["RMSE"]
           .agg(median="median", mean="mean", n_sensors="count").reset_index()
           .sort_values(["period", "median"]))
summary
""")

code("""def wilcoxon_vs_b1(df, period):
    b1 = df[(df.period == period) & (df.model == "B1")].set_index("sensor")["RMSE"]
    rows = []
    for model in df.model.unique():
        if model == "B1":
            continue
        other = df[(df.period == period) & (df.model == model)].set_index("sensor")["RMSE"]
        common = b1.index.intersection(other.index)
        if len(common) < 5:
            continue
        stat = paired_wilcoxon(other.loc[common], b1.loc[common])
        improvement = 100 * (b1.loc[common].median() - other.loc[common].median()) / b1.loc[common].median()
        rows.append({"period": period, "model": model, "median_improvement_%": improvement,
                     "wilcoxon_p": stat.pvalue})
    return pd.DataFrame(rows)

wilcoxon_results = pd.concat([wilcoxon_vs_b1(ablation, "day"), wilcoxon_vs_b1(ablation, "night")], ignore_index=True)
wilcoxon_results.sort_values(["period", "median_improvement_%"], ascending=[True, False])
""")

md("""## 5. Train vs. inner-LOSO error, for the overfitting diagnostic

A large train/LOSO gap for M1 (relative to B1's own gap) would flag that the
network features let XGBoost memorize sensor identity rather than learn
genuine spatial structure — a real risk given how many of the added features
are themselves static-sensor-derived (lc_mean, shade_mean, simw, strat_mean
are deterministic functions of which 26 sensors are in the training pool).""")

code("""def train_vs_test_gap(held_out, period):
    X_NAMES = X_NAMES_AT_DAY if period == "day" else X_NAMES_AT_NIGHT
    train_ids = [s for s in dev_ids if s != held_out]
    net_feat = build_fold_network_features(dev_ids, held_out)
    fold_data = dev.merge(net_feat, on=["Sensor_ID", "edt_datetime"], how="left")
    fold_data = add_temporal_features(fold_data)
    day_f, night_f = split_day_night(fold_data)
    df = day_f if period == "day" else night_f
    m1_features = X_NAMES + [c for c in NETWORK_COLS if c in df.columns] + \\
        ["dT2m_1h", "dT2m_3h", "rad_cum_24h", "tmax_prev_day", "hour_sin", "hour_cos",
         "hours_since_transition", "precip_cum_24h", "precip_cum_72h"]
    m1_features = [c for c in m1_features if c in df.columns]
    for c in m1_features:
        df[c] = df[c].fillna(df.loc[df["Sensor_ID"].isin(train_ids), c].median())
    fit_b1 = fit_xgb(df, "temp_anom", X_NAMES, XGB_PARAMS_FAST, train_ids, [held_out])
    fit_m1 = fit_xgb(df, "temp_anom", m1_features, XGB_PARAMS_FAST, train_ids, [held_out])
    def rmse(y, p):
        return float(np.sqrt(np.mean((y - p) ** 2)))
    return {
        "B1_train_RMSE": rmse(fit_b1["y_train"].values, fit_b1["pred_train"]),
        "B1_test_RMSE": rmse(fit_b1["y_test"].values, fit_b1["pred_test"]),
        "M1_train_RMSE": rmse(fit_m1["y_train"].values, fit_m1["pred_train"]),
        "M1_test_RMSE": rmse(fit_m1["y_test"].values, fit_m1["pred_test"]),
    }

sample_sensors = dev_ids[:5]
gap_rows = [dict(sensor=s, period=p, **train_vs_test_gap(s, p)) for s in sample_sensors for p in ("day", "night")]
gap_df = pd.DataFrame(gap_rows)
gap_df["B1_gap"] = gap_df["B1_test_RMSE"] - gap_df["B1_train_RMSE"]
gap_df["M1_gap"] = gap_df["M1_test_RMSE"] - gap_df["M1_train_RMSE"]
gap_df
""")

md("""If `M1_gap` is systematically much larger than `B1_gap`, that's the red
flag described in the project's overfitting principles — it would mean the
network features' apparent improvement in section 4 is partly memorization,
not signal, and the pipeline should fall back to fewer / more constrained
network features (or monotone constraints) before freezing for notebook 07.
""")

code("""print("Mean B1 gap:", gap_df["B1_gap"].mean(), " Mean M1 gap:", gap_df["M1_gap"].mean())
""")

md("""## Negative control: shuffle the network's time axis

Shuffling which hour each network sensor's reading is attributed to (within
the same sensor) should destroy any genuine same-hour spatial signal while
leaving each sensor's own marginal distribution intact. If M1's advantage
over B1 survives this shuffle, the "advantage" was never about real-time
network information in the first place.""")

code("""def run_shuffled_control(period, random_state=42):
    rng = np.random.default_rng(random_state)
    X_NAMES = X_NAMES_AT_DAY if period == "day" else X_NAMES_AT_NIGHT
    shuffled = dev.copy()
    for sid, idx in shuffled.groupby("Sensor_ID").groups.items():
        perm = rng.permutation(idx)
        shuffled.loc[idx, "temp_anom"] = shuffled.loc[perm, "temp_anom"].values
    rows = []
    for held_out in dev_ids[:8]:   # subsample for speed -- this is a sanity control, not the headline result
        train_ids = [s for s in dev_ids if s != held_out]
        pool = [s for s in dev_ids if s != held_out]
        net_feat = build_network_features(shuffled[shuffled["Sensor_ID"].isin(dev_ids)], sensor_info, set(pool),
                                           target_col="temp_anom", static_cols=NETWORK_STATIC_COLS)
        fold_data = dev.merge(net_feat, on=["Sensor_ID", "edt_datetime"], how="left")
        day_f, night_f = split_day_night(fold_data)
        df = day_f if period == "day" else night_f
        feats = X_NAMES + [c for c in NETWORK_COLS if c in df.columns]
        for c in feats:
            df[c] = df[c].fillna(df.loc[df["Sensor_ID"].isin(train_ids), c].median())
        fit = fit_xgb(df, "temp_anom", feats, XGB_PARAMS_FAST, train_ids, [held_out])
        resid = fit["pred_test"] - fit["y_test"].values
        rows.append({"sensor": held_out, "RMSE": float(np.sqrt(np.mean(resid ** 2)))})
    return pd.DataFrame(rows)

control_day = run_shuffled_control("day")
real_day = ablation_day[(ablation_day.model == "M1_xgb_network") & (ablation_day.sensor.isin(control_day.sensor))]
print("M1 median RMSE, real network timing:", real_day["RMSE"].median())
print("M1 median RMSE, time-shuffled network (negative control):", control_day["RMSE"].median())
print("If these are close, the day-time gain is not coming from real same-hour network signal.")
""")

md("""## Summary

- Ablation ranks B0/B1/B2-B4/M1/M2 by inner-LOSO per-sensor RMSE (section 4),
  with paired Wilcoxon tests against B1.
- Section 5 checks the train/LOSO gap doesn't blow up for M1 relative to B1
  (overfitting diagnostic — static-sensor-identity features are a known risk
  with only 27 development sensors).
- The negative control (shuffled network timing) should show little to no
  improvement over B1 if M1's gain is genuine same-hour spatial signal.
- **Scope note:** this pass includes basic network features (Stage 2) and
  F1 temporal features; F2's hourly cross-section/lags and upwind features
  (implemented in `src/network_features.py`) and F3-F7 (external data) are
  deferred — F2 can be folded into this ablation as a follow-up before
  freezing the pipeline for notebook 07 if it's warranted by these results.

Next: freeze the winning pipeline and open the lockbox in
[`07_confirmatory.ipynb`](07_confirmatory.ipynb).
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
with open("notebooks/06_network_models_dev.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebook 06 with", len(cells), "cells")
