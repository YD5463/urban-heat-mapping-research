"""
Shared data-prep / split / hyperparameter definitions for notebooks 02-04.

This module exists so that the sensor-level train/test split, the feature
lists, and the XGBoost hyperparameters are defined in exactly one place and
therefore identical across the baseline, model-reproduction and
weakness-analysis notebooks — otherwise a small accidental difference in how
each notebook re-derives the split would make the cross-notebook comparisons
meaningless.

Everything here is a direct transcription from the authors' own code in
``reference/UrbanSENSE_ML/2_Running_ML_for_City/Model_Fit_XGB.ipynb`` and
``reference/UrbanSENSE_ML/3_Analysis_Plots/AT_ML_performance.ipynb`` /
``RH_ML_performance.ipynb``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURE_PATH = "../data/processed/ERA5_everyday_merged.csv"
SENSOR_INFO_PATH = "../data/processed/sensor_info.csv"

# Durham, NC — used by the authors to compute solar zenith angle via pvlib.
DURHAM_LAT = 35.99260
DURHAM_LON = -78.89892

# Static variables first, then dynamic — order doesn't matter for the model,
# kept matching the source notebook for easy comparison.
X_NAMES_AT_DAY = [
    "NDBI_2024SummerMosaic", "NDVI_2024SummerMosaic", "ESA_50",
    "nearest_motorway_thru_primary_dist", "NDMI", "Tree_DSM_SVF", "Albedo_2024SummerMosaic",
    "zenith_angle", "surface_net_thermal_radiation_hourly", "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly", "surface_solar_radiation_downwards_accumulation",
    "u_component_of_wind_10m", "v_component_of_wind_10m", "windspeed_10m",
    "relative_humidity_2m", "temperature_2m", "total_precipitation_hourly",
]
X_NAMES_AT_NIGHT = [
    "NDBI_2024SummerMosaic", "NDVI_2024SummerMosaic", "ESA_50",
    "nearest_motorway_thru_primary_dist", "Tree_DSM_SVF",
    "zenith_angle", "surface_net_thermal_radiation_hourly",
    "surface_thermal_radiation_downwards_hourly", "surface_solar_radiation_downwards_accumulation",
    "u_component_of_wind_10m", "v_component_of_wind_10m", "windspeed_10m",
    "relative_humidity_2m", "temperature_2m", "total_precipitation_hourly",
]
X_NAMES_RH_DAY = [
    "NDBI_2024SummerMosaic", "ESA_50",
    "NDMI", "Tree_DSM_SVF", "Albedo_2024SummerMosaic",
    "zenith_angle", "surface_net_thermal_radiation_hourly", "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly", "surface_solar_radiation_downwards_accumulation",
    "u_component_of_wind_10m", "v_component_of_wind_10m", "windspeed_10m",
    "relative_humidity_2m", "temperature_2m", "total_precipitation_hourly",
]
X_NAMES_RH_NIGHT = [
    "NDBI_2024SummerMosaic", "ESA_50",
    "nearest_motorway_thru_primary_dist", "NDMI", "Tree_DSM_SVF",
    "zenith_angle", "surface_net_thermal_radiation_hourly",
    "surface_thermal_radiation_downwards_hourly", "surface_solar_radiation_downwards_accumulation",
    "u_component_of_wind_10m", "v_component_of_wind_10m", "windspeed_10m",
    "relative_humidity_2m", "temperature_2m", "total_precipitation_hourly",
]

# Hardcoded in the authors' Model_Fit_XGB.ipynb (same for the AT and RH models).
# n_jobs is capped (rather than left at XGBoost's default of "all cores") so a
# single fit doesn't oversubscribe a shared/busy machine - the fits are small
# enough (tens of thousands of rows) that this costs no real speed.
XGB_PARAMS_DAY = dict(
    objective="reg:squarederror", random_state=42, n_jobs=4,
    n_estimators=883, max_depth=8, learning_rate=0.059,
    subsample=0.6, colsample_bytree=0.61, min_child_weight=17,
    gamma=0.001, reg_alpha=0.0001, reg_lambda=80,
)
XGB_PARAMS_NIGHT = dict(
    objective="reg:squarederror", random_state=42, n_jobs=4,
    n_estimators=700, max_depth=7, learning_rate=0.2,
    subsample=0.9, colsample_bytree=0.4, min_child_weight=26,
    gamma=0.014, reg_alpha=0.002, reg_lambda=57,
)

# A cheap stand-in for the above, used for the many repeated fits in the
# network-feature ablation and the sensor-count curve (notebooks 06, 09),
# where sheer fit count matters more than squeezing out the last 0.01°C.
XGB_PARAMS_FAST = dict(
    objective="reg:squarederror", random_state=42, n_jobs=4, tree_method="hist",
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
)


def load_data(geo: bool = False) -> pd.DataFrame:
    """Load the hourly feature table, merge in sensor metadata, and add the
    derived columns every downstream notebook needs (zenith angle, strata,
    anomaly targets).

    `geo=True` additionally left-merges in the geo-feature pipeline's output
    (notebook 10: topography/morphology/wind/shadow/roof features from
    scripts/build_static_features.py), by Sensor_ID. Default is False and
    that default path is unchanged from before this parameter existed -
    notebooks 01-09 must keep reproducing exactly."""
    data = pd.read_csv(FEATURE_PATH)
    sensor_info = pd.read_csv(SENSOR_INFO_PATH)
    data = data.merge(
        sensor_info[["Sensor_ID", "Landcover_Type", "Shade_Type"]],
        on="Sensor_ID", how="left",
    )

    if geo:
        from city_config import DURHAM

        # city_config paths are repo-root-relative; every notebook here runs
        # with cwd=notebooks/ (matching FEATURE_PATH/SENSOR_INFO_PATH above).
        geo_df = pd.read_parquet("../" + DURHAM.processed_geo_path).reset_index()
        overlap = (set(geo_df.columns) & set(data.columns)) - {"Sensor_ID"}
        if overlap:
            raise ValueError(f"geo feature table has columns colliding with the existing table: {overlap}")
        data = data.merge(geo_df, on="Sensor_ID", how="left")

    # Solar zenith angle, used as a feature (same approach as the authors':
    # convert local timestamps to UTC, then pvlib.solarposition at the
    # Durham lat/lon since all sensors are close enough together that a
    # single city-wide solar position is an acceptable approximation).
    data["edt_datetime"] = pd.to_datetime(data["edt_datetime"]).dt.tz_localize("US/Eastern")
    data["utc_datetime"] = data["edt_datetime"].dt.tz_convert("UTC")
    solpos = pvlib.solarposition.get_solarposition(
        time=data["utc_datetime"], latitude=DURHAM_LAT, longitude=DURHAM_LON
    )
    data["zenith_angle"] = solpos["zenith"].values

    # Anomaly targets: sensor reading minus the ERA5-Land city-wide mean for
    # that hour. This removes the diurnal cycle and synoptic weather, leaving
    # only the *local* deviation the static predictors should explain.
    data["temp_anom"] = data["Temperature"] - data["temperature_2m"]
    data["rh_anom"] = data["Relative_Humidity"] - data["relative_humidity_2m"]

    # Composite stratum used for the sensor-level split.
    data["strata"] = data["Landcover_Type"].astype(str) + "_" + data["Shade_Type"].astype(str)

    return data


def sensor_level_split(data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """Reproduces the authors' held-out evaluation split exactly:
    StratifiedShuffleSplit at the *sensor* level (not the row level!),
    stratified on Landcover x Shade, so both train and test sets contain
    sensors from every stratum. Returns (train_sensor_ids, test_sensor_ids).
    """
    sensor_strata = data.groupby("Sensor_ID")["strata"].first()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(sensor_strata.index, sensor_strata.values))
    return sensor_strata.index[train_idx], sensor_strata.index[test_idx]


def split_day_night(data: pd.DataFrame):
    """The authors treat "day" / "night" as "any incoming shortwave
    radiation at all" rather than a fixed local-time window — this lines up
    each sensor's model switch with the actual solar day, not clock time."""
    day = data[data["surface_solar_radiation_downwards_hourly"] > 1].copy().reset_index(drop=True)
    night = data[data["surface_solar_radiation_downwards_hourly"] < 1].copy().reset_index(drop=True)
    return day, night


def compute_heat_index_F(T, R):
    """Rothfusz regression + NOAA adjustments. T in degF, R in % RH.
    Transcribed from reference/UrbanSENSE_ML/2_Running_ML_for_City/compute_HI.py
    (numpy instead of xarray)."""
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094)  # Steadman
    mask = HI >= 80
    HI_roth = (-42.379 + 2.04901523 * T + 10.14333127 * R
               - 0.22475541 * T * R - 6.83783e-3 * T ** 2
               - 5.481717e-2 * R ** 2 + 1.22874e-3 * T ** 2 * R
               + 8.5282e-4 * T * R ** 2 - 1.99e-6 * T ** 2 * R ** 2)
    HI = np.where(mask, HI_roth, HI)

    adj1_mask = (R < 13) & (T >= 80) & (T <= 112)
    adj1 = ((13 - R) / 4.0) * np.sqrt(np.clip((17 - np.abs(T - 95.0)) / 17.0, 0, None))
    HI = np.where(adj1_mask, HI - adj1, HI)

    adj2_mask = (R > 85) & (T >= 80) & (T <= 87)
    adj2 = ((R - 85) / 10.0) * ((87 - T) / 5.0)
    HI = np.where(adj2_mask, HI + adj2, HI)
    return HI


def F_to_C(F):
    return (np.asarray(F, dtype=float) - 32.0) * 5.0 / 9.0


def C_to_F(C):
    return np.asarray(C, dtype=float) * 9.0 / 5.0 + 32.0


def fit_xgb(data: pd.DataFrame, target_col: str, X_names: list, hyperparams: dict,
            train_ids, test_ids) -> dict:
    """Fits one XGBRegressor on the rows whose Sensor_ID is in `train_ids`
    and evaluates on `test_ids`. `test_ids` may be a single held-out sensor
    (LOSO) or the full lockbox — the function itself is agnostic, the
    leakage discipline is entirely in what the caller passes as train/test
    ids. Scaler is fit on train rows only, as in the authors' code."""
    train_mask = data["Sensor_ID"].isin(train_ids)
    test_mask = data["Sensor_ID"].isin(test_ids)

    X_train = data.loc[train_mask, X_names]
    X_test = data.loc[test_mask, X_names]
    y_train = data.loc[train_mask, target_col]
    y_test = data.loc[test_mask, target_col]

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train.values)
    X_test_norm = scaler.transform(X_test.values)

    model = XGBRegressor(**hyperparams)
    model.fit(X_train_norm, y_train)

    return {
        "model": model,
        "scaler": scaler,
        "y_train": y_train, "y_test": y_test,
        "pred_train": model.predict(X_train_norm),
        "pred_test": model.predict(X_test_norm),
        "train_mask": train_mask, "test_mask": test_mask,
        "X_names": X_names,
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    resid = y_pred - y_true
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    mbe = float(np.mean(resid))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MBE": mbe, "R2": r2}
