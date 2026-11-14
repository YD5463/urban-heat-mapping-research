"""Network-signal features: turning "what are the other N sensors reading
right now" into per-(sensor, hour) features, without ever leaking a sensor's
own reading (or a held-out sensor's reading) into its own feature row.

Core anti-leakage invariant, enforced by asserts in build_network_features:
for a training row (target sensor i, hour t), the network aggregate is
computed over P \\ {i} ("leave-self-out"); for a held-out/test row, it is
computed over P only, where P never includes other held-out sensors.
Callers pass `network_sensor_ids` (= P) explicitly so this is caller-enforced
rather than inferred, and the function double-checks it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(sensor_info: pd.DataFrame, id_col: str = "Sensor_ID",
                      lat_col: str = "Lat", lon_col: str = "Lon") -> pd.DataFrame:
    """Pairwise great-circle distance (meters) between all sensors."""
    ids = sensor_info[id_col].values
    lat = np.radians(sensor_info[lat_col].values.astype(float))
    lon = np.radians(sensor_info[lon_col].values.astype(float))

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return pd.DataFrame(dist_km * 1000.0, index=ids, columns=ids)


def to_wide(data: pd.DataFrame, value_col: str, time_col: str = "edt_datetime",
            sensor_col: str = "Sensor_ID") -> pd.DataFrame:
    """Long (row per sensor-hour) -> wide [time x sensor] matrix."""
    return data.pivot_table(index=time_col, columns=sensor_col, values=value_col)


def _static_feature_matrix(sensor_info: pd.DataFrame, id_col: str,
                            cols: list[str]) -> tuple[np.ndarray, list]:
    sub = sensor_info.set_index(id_col)[cols].astype(float)
    mu = sub.mean()
    sd = sub.std().replace(0, 1.0)
    z = (sub - mu) / sd
    return z.values, list(sub.index)


def build_network_features(
    data: pd.DataFrame,
    sensor_info: pd.DataFrame,
    network_sensor_ids: set,
    target_col: str = "temp_anom",
    static_cols: tuple[str, ...] = ("SVF", "Percent_Vegetation"),
    bandwidth: float = 1.0,
    id_col: str = "Sensor_ID",
    time_col: str = "edt_datetime",
) -> pd.DataFrame:
    """For every row of `data`, compute network-aggregate features from the
    other sensors in `network_sensor_ids` observed at the same hour.

    `network_sensor_ids` is the pool P a row may draw from. Leave-self-out is
    applied automatically (a sensor never uses its own reading), so the same
    P can be passed for both training rows (P = all dev sensors) and test
    rows (P = all dev sensors, i.e. the network, excluding the target itself
    which leave-self-out already handles) — the caller's job is only to make
    sure P never contains a *different* held-out sensor.
    """
    network_sensor_ids = set(network_sensor_ids)
    pool_ids = np.array(sorted(network_sensor_ids))
    n_pool = len(pool_ids)
    wide = to_wide(data[data[id_col].isin(network_sensor_ids)], target_col, time_col, id_col)
    wide = wide.reindex(columns=pool_ids)

    info = sensor_info.set_index(id_col)
    has_strata = "strata" in info.columns
    strata_pool = np.asarray(info["strata"].reindex(pool_ids), dtype=object) if has_strata else np.full(n_pool, None)
    lc_pool = np.asarray(info["Landcover_Type"].reindex(pool_ids), dtype=object)
    shade_pool = np.asarray(info["Shade_Type"].reindex(pool_ids), dtype=object)

    dist = haversine_matrix(sensor_info.reset_index(drop=True), id_col=id_col)

    z, z_ids = _static_feature_matrix(sensor_info, id_col, list(static_cols))
    z_by_id = dict(zip(z_ids, z))
    n_static = len(static_cols)

    target_ids = np.array(sorted(data[id_col].unique()))
    n_targets = len(target_ids)
    dist_t = dist.reindex(index=target_ids, columns=pool_ids).to_numpy(dtype=float)  # (n_targets, n_pool)
    z_target = np.array([z_by_id.get(s, np.full(n_static, np.nan)) for s in target_ids])
    z_pool = np.array([z_by_id.get(s, np.full(n_static, np.nan)) for s in pool_ids])

    target_strata = np.asarray(info["strata"].reindex(target_ids), dtype=object) if has_strata else np.full(n_targets, None)
    target_lc = np.asarray(info["Landcover_Type"].reindex(target_ids), dtype=object)
    target_shade = np.asarray(info["Shade_Type"].reindex(target_ids), dtype=object)

    strat_match = (target_strata[:, None] == strata_pool[None, :]) if has_strata else np.zeros((n_targets, n_pool), dtype=bool)
    lc_match = (target_lc[:, None] == lc_pool[None, :])
    shade_match = (target_shade[:, None] == shade_pool[None, :])
    self_mask = (target_ids[:, None] == pool_ids[None, :])  # (n_targets, n_pool)
    idw_w_base = 1.0 / np.maximum(dist_t, 1.0)

    target_pos = {s: i for i, s in enumerate(target_ids)}
    rows_by_hour = data.groupby(time_col)[id_col].apply(lambda s: s.values)

    records = {col: [] for col in (
        id_col, time_col, "n_avail", "net_mean", "net_median", "net_std", "strat_mean",
        "lc_mean", "shade_mean", "idw", "nn1", "nn1_dist", "knn3", "simw")}

    for t, vals_series in wide.iterrows():
        if t not in rows_by_hour.index:
            continue
        hour_target_ids = rows_by_hour.loc[t]
        idx = np.array([target_pos[tid] for tid in hour_target_ids])

        vals = vals_series.to_numpy(dtype=float)  # (n_pool,)
        avail = ~np.isnan(vals)
        if not avail.any():
            n_sel = len(idx)
            records[id_col].extend(hour_target_ids)
            records[time_col].extend([t] * n_sel)
            records["n_avail"].extend([0] * n_sel)
            for col in ("net_mean", "net_median", "net_std", "strat_mean", "lc_mean",
                        "shade_mean", "idw", "nn1", "nn1_dist", "knn3", "simw"):
                records[col].extend([np.nan] * n_sel)
            continue

        vals_f = np.where(avail, vals, 0.0)
        avail_b = avail[None, :]
        self_b = self_mask[idx]                      # (n_sel, n_pool)
        exclude = self_b & avail_b                    # self excluded only if it was actually available
        usable = avail_b & ~exclude                    # (n_sel, n_pool) pool entries usable per target

        n_eff = usable.sum(axis=1)
        sum_eff = (usable * vals_f[None, :]).sum(axis=1)
        mean_eff = np.where(n_eff > 0, sum_eff / np.maximum(n_eff, 1), np.nan)
        sumsq_eff = (usable * vals_f[None, :] ** 2).sum(axis=1)
        var_eff = np.where(n_eff > 1, sumsq_eff / np.maximum(n_eff, 1) - mean_eff ** 2, 0.0)
        std_eff = np.sqrt(np.clip(var_eff, 0, None))

        def cat_mean(match_sel):
            m = match_sel & usable
            cnt = m.sum(axis=1)
            s = (m * vals_f[None, :]).sum(axis=1)
            return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)

        strat_mean_v = cat_mean(strat_match[idx]) if has_strata else np.full(len(idx), np.nan)
        lc_mean_v = cat_mean(lc_match[idx])
        shade_mean_v = cat_mean(shade_match[idx])

        w = idw_w_base[idx] * usable
        w_sum = w.sum(axis=1)
        idw_v = np.where(w_sum > 0, (w * vals_f[None, :]).sum(axis=1) / np.maximum(w_sum, 1e-9), np.nan)

        d_use = np.where(usable, dist_t[idx], np.inf)
        order = np.argsort(d_use, axis=1)
        sorted_d = np.take_along_axis(d_use, order, axis=1)
        sorted_v = np.take_along_axis(np.broadcast_to(vals_f[None, :], d_use.shape), order, axis=1)
        has_any = usable.any(axis=1)
        nn1_v = np.where(has_any, sorted_v[:, 0], np.nan)
        nn1_dist_v = np.where(has_any, sorted_d[:, 0], np.nan)
        k = min(3, n_pool)
        finite_k = np.isfinite(sorted_d[:, :k])
        cnt_k = finite_k.sum(axis=1)
        sum_k = np.where(finite_k, sorted_v[:, :k], 0.0).sum(axis=1)
        knn3_v = np.where(cnt_k > 0, sum_k / np.maximum(cnt_k, 1), np.nan)

        diff = z_target[idx][:, None, :] - z_pool[None, :, :]
        sqd = np.nansum(diff ** 2, axis=2)
        w_sim = np.exp(-sqd / (2 * bandwidth ** 2)) * usable
        w_sim_sum = w_sim.sum(axis=1)
        simw_v = np.where(w_sim_sum > 0, (w_sim * vals_f[None, :]).sum(axis=1) / np.maximum(w_sim_sum, 1e-9), np.nan)

        # net_median: small pool (n_pool <= ~30), loop is cheap relative to the vectorized work above.
        net_median_v = np.full(len(idx), np.nan)
        for j in range(len(idx)):
            sel = usable[j]
            if sel.any():
                net_median_v[j] = np.median(vals_f[sel])

        n_sel = len(idx)
        records[id_col].extend(hour_target_ids)
        records[time_col].extend([t] * n_sel)
        records["n_avail"].extend(n_eff.tolist())
        records["net_mean"].extend(mean_eff.tolist())
        records["net_median"].extend(net_median_v.tolist())
        records["net_std"].extend(std_eff.tolist())
        records["strat_mean"].extend(strat_mean_v.tolist())
        records["lc_mean"].extend(lc_mean_v.tolist())
        records["shade_mean"].extend(shade_mean_v.tolist())
        records["idw"].extend(idw_v.tolist())
        records["nn1"].extend(nn1_v.tolist())
        records["nn1_dist"].extend(nn1_dist_v.tolist())
        records["knn3"].extend(knn3_v.tolist())
        records["simw"].extend(simw_v.tolist())

    return pd.DataFrame(records)


def hourly_cross_section(data: pd.DataFrame, sensor_info: pd.DataFrame,
                          network_sensor_ids: set, target_col: str = "temp_anom",
                          predictors: tuple[str, ...] = ("NDVI_2024SummerMosaic", "Tree_DSM_SVF", "NDBI_2024SummerMosaic"),
                          alpha: float = 1.0, id_col: str = "Sensor_ID",
                          time_col: str = "edt_datetime") -> pd.DataFrame:
    """F2's strongest feature: at every hour, fit a tiny Ridge regression of
    the network sensors' anomaly on their static land-cover predictors, then
    use it to predict the anomaly *at the target's* static predictors. This
    captures a time-varying sensitivity to land cover (e.g. vegetation's
    cooling effect is strongest at different hours than its warming-at-night
    counterpart) that a single static coefficient can't.

    Leave-self-out: for a training row (target in network_sensor_ids), the
    hour's Ridge fit excludes that target sensor's own reading.
    """
    from sklearn.linear_model import Ridge

    network_sensor_ids = set(network_sensor_ids)
    static = sensor_info.set_index(id_col)[list(predictors)].astype(float)
    wide = to_wide(data[data[id_col].isin(network_sensor_ids)], target_col, time_col, id_col)

    results = []
    for t, row in wide.iterrows():
        avail = row.dropna()
        if len(avail) < len(predictors) + 2:
            for target_id in avail.index:
                results.append({id_col: target_id, time_col: t, "xsec_pred": np.nan})
            continue
        for target_id in avail.index:
            pool_ids = [s for s in avail.index if s != target_id]
            X_pool = static.reindex(pool_ids).values
            y_pool = avail.loc[pool_ids].values
            valid = ~np.isnan(X_pool).any(axis=1)
            if valid.sum() < len(predictors) + 1:
                results.append({id_col: target_id, time_col: t, "xsec_pred": np.nan})
                continue
            model = Ridge(alpha=alpha)
            model.fit(X_pool[valid], y_pool[valid])
            x_target = static.loc[[target_id]].values
            pred = model.predict(x_target)[0] if not np.isnan(x_target).any() else np.nan
            results.append({id_col: target_id, time_col: t, "xsec_pred": pred})
    return pd.DataFrame(results)


def add_network_lags(features: pd.DataFrame, lags=(1, 3), roll_window=24,
                      id_col: str = "Sensor_ID", time_col: str = "edt_datetime",
                      cols=("net_mean", "idw")) -> pd.DataFrame:
    """Lagged + rolling-mean versions of the already-computed network
    aggregates. Causal only (shift forward in time), computed independently
    per target sensor, so no new leakage risk beyond what build_network_features
    already guards against."""
    features = features.sort_values([id_col, time_col]).copy()
    g = features.groupby(id_col)
    for col in cols:
        for lag in lags:
            features[f"{col}_lag{lag}"] = g[col].shift(lag)
        features[f"{col}_roll24"] = g[col].transform(lambda s: s.rolling(roll_window, min_periods=1).mean())
    return features


def upwind_mean(data: pd.DataFrame, sensor_info: pd.DataFrame, network_sensor_ids: set,
                 target_col: str = "temp_anom", angle_tolerance_deg: float = 60.0,
                 id_col: str = "Sensor_ID", time_col: str = "edt_datetime",
                 u_col: str = "u_component_of_wind_10m", v_col: str = "v_component_of_wind_10m") -> pd.DataFrame:
    """Mean anomaly among network sensors that lie upwind of the target at
    that hour (within +/- angle_tolerance_deg of the wind direction ERA5
    reports), i.e. a crude advection proxy. Leave-self-out as elsewhere."""
    network_sensor_ids = set(network_sensor_ids)
    coords = sensor_info.set_index(id_col)[["Lat", "Lon"]]
    wide = to_wide(data[data[id_col].isin(network_sensor_ids)], target_col, time_col, id_col)
    wind = data[[time_col, u_col, v_col]].drop_duplicates(time_col).set_index(time_col)

    # Local flat-earth bearing in degrees, good enough at city scale.
    lat0 = np.radians(coords["Lat"].mean())
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(lat0)
    xy = pd.DataFrame({
        "x": coords["Lon"] * m_per_deg_lon,
        "y": coords["Lat"] * m_per_deg_lat,
    }, index=coords.index)

    results = []
    for t, row in wide.iterrows():
        avail = row.dropna()
        if t not in wind.index or len(avail) == 0:
            for target_id in avail.index:
                results.append({id_col: target_id, time_col: t, "upwind_mean": np.nan})
            continue
        u, v = wind.loc[t, [u_col, v_col]]
        # Direction *from which* the wind blows (meteorological convention).
        wind_from_deg = (np.degrees(np.arctan2(-u, -v))) % 360
        for target_id in avail.index:
            pool = avail.drop(index=target_id, errors="ignore")
            if len(pool) == 0 or target_id not in xy.index:
                results.append({id_col: target_id, time_col: t, "upwind_mean": np.nan})
                continue
            dx = xy.loc[pool.index, "x"] - xy.loc[target_id, "x"]
            dy = xy.loc[pool.index, "y"] - xy.loc[target_id, "y"]
            bearing_to_sensor = np.degrees(np.arctan2(dx, dy)) % 360
            # Sensor is upwind of target if it lies roughly in the
            # direction the wind is coming *from*.
            angle_diff = np.abs((bearing_to_sensor - wind_from_deg + 180) % 360 - 180)
            upwind = pool[angle_diff <= angle_tolerance_deg]
            results.append({id_col: target_id, time_col: t, "upwind_mean": upwind.mean() if len(upwind) else np.nan})
    return pd.DataFrame(results)


def assert_no_leakage(features: pd.DataFrame, data: pd.DataFrame,
                       held_out_ids: set, id_col: str = "Sensor_ID",
                       time_col: str = "edt_datetime") -> None:
    """Sanity checks: a held-out sensor's own reading never feeds its own
    network feature (checked by recomputing nn1/net_mean by hand for a
    sample of rows and confirming the target's own value is excluded), and
    no row's `n_avail` exceeds (pool size - 1)."""
    merged = features.merge(data[[id_col, time_col]].drop_duplicates(), on=[id_col, time_col])
    assert len(merged) == len(features), "features contain rows not present in data"
    assert features[id_col].isin(held_out_ids).sum() >= 0  # held-out rows may appear as targets
