"""Sensor-level cross-validation protocols and evaluation statistics.

Every fold generator here yields (train_sensor_ids, test_sensor_ids) — never
row indices — because the independent unit in this dataset is the sensor
(39 of them), not the hour. Mixing up row-level and sensor-level CV is the
single easiest way to silently leak information between train and test.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans


def loso(sensor_ids) -> list:
    """Leave-one-sensor-out: each sensor is the test set once, trained on
    all the others in `sensor_ids`."""
    sensor_ids = list(sensor_ids)
    return [([s for s in sensor_ids if s != held_out], [held_out]) for held_out in sensor_ids]


def inner_loso(sensor_ids) -> list:
    """Alias for loso restricted to a development pool — kept as a separate
    name so call sites document *which* pool (dev vs. full 39) they mean."""
    return loso(sensor_ids)


def fixed_split(train_ids, test_ids) -> list:
    """A single, non-rotating split (e.g. the paper's 27/12 lockbox split)."""
    return [(list(train_ids), list(test_ids))]


def leave_one_stratum_out(sensor_info: pd.DataFrame, id_col: str = "Sensor_ID",
                           strata_col: str = "strata") -> list:
    """Each fold holds out every sensor in one stratum; trains on all sensors
    from the other strata. Harder than plain LOSO: the test sensors share no
    stratum-mate with the training set at all."""
    folds = []
    for stratum, group in sensor_info.groupby(strata_col):
        test_ids = group[id_col].tolist()
        train_ids = sensor_info.loc[sensor_info[strata_col] != stratum, id_col].tolist()
        folds.append((train_ids, test_ids))
    return folds


def buffered(sensor_info: pd.DataFrame, dist: pd.DataFrame, radius_m: float,
             id_col: str = "Sensor_ID") -> list:
    """LOSO, but additionally drops from the training pool every sensor
    within `radius_m` of the held-out sensor (not just the held-out sensor
    itself). radius_m=0 reduces to plain LOSO."""
    ids = sensor_info[id_col].tolist()
    folds = []
    for held_out in ids:
        near = dist.loc[held_out]
        excluded = set(near.index[near <= radius_m]) | {held_out}
        train_ids = [s for s in ids if s not in excluded]
        folds.append((train_ids, [held_out]))
    return folds


def spatial_clusters(sensor_info: pd.DataFrame, k: int = 4, random_state: int = 42,
                      id_col: str = "Sensor_ID", lat_col: str = "Lat", lon_col: str = "Lon") -> list:
    """Each fold holds out one KMeans spatial cluster of sensors (on raw
    lat/lon — fine for CV-fold assignment, never used as a model feature)."""
    coords = sensor_info[[lat_col, lon_col]].values
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(coords)
    folds = []
    for c in np.unique(labels):
        test_ids = sensor_info.loc[labels == c, id_col].tolist()
        train_ids = sensor_info.loc[labels != c, id_col].tolist()
        folds.append((train_ids, test_ids))
    return folds


def sensor_x_time_blocks(data: pd.DataFrame, held_out_sensor, time_col: str = "edt_datetime",
                          id_col: str = "Sensor_ID", train_frac: float = 0.7):
    """ST protocol for one held-out sensor: train on the other sensors over
    *all* hours plus (optionally) nothing of the held-out sensor; test on the
    held-out sensor's hours *after* a time cutoff, so the test hours don't
    overlap the training sensors' same-hour common offset as cleanly as in
    protocol S. Returns (train_mask, test_mask) boolean arrays over `data`."""
    other = data[id_col] != held_out_sensor
    cutoff = data[time_col].quantile(train_frac)
    train_mask = other & (data[time_col] <= cutoff)
    test_mask = (data[id_col] == held_out_sensor) & (data[time_col] > cutoff)
    return train_mask, test_mask


def nested_n_curve(pool_ids, n_values, n_permutations=10, random_state=42, stratified_info=None):
    """Generates the (permutation, N) -> network_sensor_subset table used by
    the sensor-count curve. Each permutation of `pool_ids` is fixed once;
    for a given N, the subset is always "the first N ids of that
    permutation" (common random numbers), so RMSE(N) and RMSE(N+1) differ
    only by the one sensor added, not by resampling noise.

    If `stratified_info` (a Series Sensor_ID -> stratum, indexed like
    pool_ids) is given, half the permutations are built round-robin across
    strata instead of uniformly at random.
    """
    rng = np.random.default_rng(random_state)
    pool_ids = list(pool_ids)
    perms = []
    n_random = n_permutations // 2
    n_strat = n_permutations - n_random
    for _ in range(n_random):
        perms.append(list(rng.permutation(pool_ids)))
    if stratified_info is not None and n_strat > 0:
        strata = stratified_info.reindex(pool_ids)
        groups = {s: list(rng.permutation(strata.index[strata == s])) for s in strata.unique()}
        for _ in range(n_strat):
            order = []
            keys = list(groups.keys())
            idxs = {k: 0 for k in keys}
            local_groups = {k: list(rng.permutation(v)) for k, v in groups.items()}
            while len(order) < len(pool_ids):
                for k in keys:
                    if idxs[k] < len(local_groups[k]):
                        order.append(local_groups[k][idxs[k]])
                        idxs[k] += 1
            perms.append(order)
    else:
        for _ in range(n_strat):
            perms.append(list(rng.permutation(pool_ids)))

    table = []
    for p_idx, perm in enumerate(perms):
        for n in n_values:
            table.append({"permutation": p_idx, "N": n, "network_ids": tuple(perm[:n])})
    return table


def run_protocol(fit_fn, folds, cache_path: str | None = None, force: bool = False) -> pd.DataFrame:
    """Runs `fit_fn(train_ids, test_ids) -> dict-of-scalars` over every fold,
    returns a DataFrame, and (if cache_path given) caches to parquet so
    re-running a notebook doesn't redo expensive fits."""
    if cache_path and os.path.exists(cache_path) and not force:
        return pd.read_parquet(cache_path)

    rows = []
    for train_ids, test_ids in folds:
        result = fit_fn(train_ids, test_ids)
        rows.append(result)
    out = pd.DataFrame(rows)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out.to_parquet(cache_path)
    return out


def per_sensor_metrics(y_true, y_pred, sensor_ids) -> pd.DataFrame:
    """RMSE/MAE/MBE per held-out sensor."""
    df = pd.DataFrame({"sensor": sensor_ids, "y_true": y_true, "y_pred": y_pred})
    rows = []
    for sensor, g in df.groupby("sensor"):
        resid = g["y_pred"] - g["y_true"]
        rows.append({
            "sensor": sensor, "n": len(g),
            "RMSE": float(np.sqrt(np.mean(resid ** 2))),
            "MAE": float(np.mean(np.abs(resid))),
            "MBE": float(np.mean(resid)),
        })
    return pd.DataFrame(rows)


def paired_wilcoxon(rmse_a: pd.Series, rmse_b: pd.Series):
    """Paired Wilcoxon signed-rank test on per-sensor RMSE, model A vs B
    (both indexed by sensor, same sensors). Returns the scipy result."""
    common = rmse_a.index.intersection(rmse_b.index)
    return wilcoxon(rmse_a.loc[common], rmse_b.loc[common])


def sensor_bootstrap_ci(per_sensor_rmse: pd.Series, n_boot: int = 2000, random_state: int = 42,
                         ci: float = 0.95):
    """Bootstraps the *median* per-sensor RMSE by resampling sensors (not
    rows) with replacement — the sensor is the independent unit, so this is
    the correct resampling level for a confidence interval on "typical
    sensor performance"."""
    rng = np.random.default_rng(random_state)
    vals = per_sensor_rmse.values
    boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
    medians = np.median(boot, axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(medians, [alpha, 1 - alpha])
    return {"median": float(np.median(vals)), "ci_low": float(lo), "ci_high": float(hi)}
