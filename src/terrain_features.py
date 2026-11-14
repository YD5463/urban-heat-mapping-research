"""Topography / DTM-derived features (geo-features plan, section A).

Computes, per sensor, from a single-band DEM (FABDEM in production):
``elev``, ``tpi_500`` and ``twi`` (the three pre-registered terrain
features), plus exploration-only columns ``slope``, ``aspect_sin``,
``aspect_cos``, ``valley_depth`` and ``Sx_<dir>`` for 8 compass directions
(Winstral upwind/downwind terrain exposure).

TWI backend: neither ``richdem`` nor ``whitebox`` installs/runs cleanly in
this environment (richdem fails to build its pybind11 extension against
Python 3.11 on Windows; whitebox's binary downloader crashes on a non-ASCII
repo path), so flow direction/accumulation is a small D8 implementation in
numpy/scipy below. It has no sink-filling pass — adequate for the smooth,
sink-free FABDEM tiles this is run on, but would need one for noisier DEMs.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import map_coordinates, uniform_filter

from city_config import CityConfig

# 8 D8 neighbor offsets (row, col) and compass names, clockwise from N.
_DIRS = [
    ("N", (-1, 0)), ("NE", (-1, 1)), ("E", (0, 1)), ("SE", (1, 1)),
    ("S", (1, 0)), ("SW", (1, -1)), ("W", (0, -1)), ("NW", (-1, -1)),
]
_DIR_AZIMUTH_DEG = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}

TPI_RADIUS_M = 500.0
SX_MAX_DIST_M = 300.0
SX_STEP_M = 30.0
VALLEY_RIDGE_RADIUS_M = 500.0


def _d8_flow_accum(elev: np.ndarray) -> np.ndarray:
    """Cells-draining-through-this-cell count via steepest-descent D8,
    processed in descending elevation order (no sinks assumed)."""
    nrows, ncols = elev.shape
    flat_idx = np.argsort(-elev, axis=None)
    rows, cols = np.unravel_index(flat_idx, elev.shape)

    receiver = np.full(elev.shape + (2,), -1, dtype=int)
    for r, c in zip(rows, cols):
        best_drop, best_rc = -np.inf, None
        for _, (dr, dc) in _DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols:
                dist = np.hypot(dr, dc)
                drop = (elev[r, c] - elev[rr, cc]) / dist
                if drop > best_drop:
                    best_drop, best_rc = drop, (rr, cc)
        if best_rc is not None and best_drop > 0:
            receiver[r, c] = best_rc

    accum = np.ones_like(elev, dtype=float)
    for r, c in zip(rows, cols):
        rc = receiver[r, c]
        if rc[0] >= 0:
            accum[rc[0], rc[1]] += accum[r, c]
    return accum


def _slope_aspect(elev: np.ndarray, cellsize: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horn's method. Returns (slope_rad, aspect_sin, aspect_cos)."""
    dzdx = (np.roll(elev, -1, axis=1) - np.roll(elev, 1, axis=1)) / (2 * cellsize)
    dzdy = (np.roll(elev, -1, axis=0) - np.roll(elev, 1, axis=0)) / (2 * cellsize)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    return slope, np.sin(aspect), np.cos(aspect)


def _circular_mean(elev: np.ndarray, radius_px: int) -> np.ndarray:
    """Mean elevation within a circular window (approximated via a square
    uniform filter — adequate given FABDEM's coarse 30m cells relative to
    the 500m window)."""
    size = 2 * radius_px + 1
    return uniform_filter(elev, size=size, mode="nearest")


def _sample_bilinear(elev: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return map_coordinates(elev, [rows, cols], order=1, mode="nearest")


def _sx_directional(elev: np.ndarray, row0: float, col0: float, cellsize: float,
                     azimuth_deg: float, max_dist_m: float, step_m: float) -> float:
    """Winstral Sx: max terrain elevation angle (degrees) looking toward
    `azimuth_deg` out to `max_dist_m`. Positive = terrain rises (sheltered
    from wind arriving from that direction)."""
    az = np.deg2rad(azimuth_deg)
    # row decreases northward (azimuth 0 = -row direction), col increases eastward.
    dr_unit, dc_unit = -np.cos(az), np.sin(az)
    dists = np.arange(step_m, max_dist_m + step_m, step_m)
    rows = row0 + dr_unit * dists / cellsize
    cols = col0 + dc_unit * dists / cellsize
    nrows, ncols = elev.shape
    valid = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    if not valid.any():
        return 0.0
    elev0 = _sample_bilinear(elev, np.array([row0]), np.array([col0]))[0]
    sampled = _sample_bilinear(elev, rows[valid], cols[valid])
    angles = np.degrees(np.arctan2(sampled - elev0, dists[valid]))
    return float(np.max(angles))


def compute_terrain_features(city: CityConfig, sensors: gpd.GeoDataFrame, dem_path: str | Path) -> pd.DataFrame:
    """sensors: GeoDataFrame indexed by Sensor_ID, geometry = point in city.crs.
    dem_path: path to a FABDEM GeoTIFF (or any single-band DEM in city.crs or
    reprojectable to it) covering all sensors + buffer.
    Returns a DataFrame indexed by Sensor_ID with columns: elev, tpi_500, twi,
    slope, aspect_sin, aspect_cos, valley_depth, Sx_N, Sx_NE, ... Sx_NW.
    """
    with rasterio.open(dem_path) as ds:
        if sensors.crs is not None and ds.crs is not None and str(sensors.crs) != str(ds.crs):
            sensors = sensors.to_crs(ds.crs)
        elev = ds.read(1).astype(float)
        cellsize = abs(ds.transform.a)
        cols, rows = ~ds.transform * (sensors.geometry.x.values, sensors.geometry.y.values)

    tpi_radius_px = max(1, round(TPI_RADIUS_M / cellsize))
    ridge_radius_px = max(1, round(VALLEY_RIDGE_RADIUS_M / cellsize))

    local_mean = _circular_mean(elev, tpi_radius_px)
    ridge_mean = uniform_filter(elev, size=2 * ridge_radius_px + 1, mode="nearest", origin=0)
    accum = _d8_flow_accum(elev)
    slope_rad, aspect_sin, aspect_cos = _slope_aspect(elev, cellsize)

    specific_catchment_area = accum * cellsize
    twi_grid = np.log(specific_catchment_area / np.maximum(np.tan(slope_rad), 1e-3))

    elev_s = _sample_bilinear(elev, rows, cols)
    tpi_s = elev_s - _sample_bilinear(local_mean, rows, cols)
    twi_s = _sample_bilinear(twi_grid, rows, cols)
    slope_s = _sample_bilinear(slope_rad, rows, cols)
    aspect_sin_s = _sample_bilinear(aspect_sin, rows, cols)
    aspect_cos_s = _sample_bilinear(aspect_cos, rows, cols)
    valley_depth_s = _sample_bilinear(ridge_mean, rows, cols) - elev_s

    out = pd.DataFrame({
        "elev": elev_s, "tpi_500": tpi_s, "twi": twi_s,
        "slope": slope_s, "aspect_sin": aspect_sin_s, "aspect_cos": aspect_cos_s,
        "valley_depth": valley_depth_s,
    }, index=sensors.index)

    for dname, _ in _DIRS:
        az = _DIR_AZIMUTH_DEG[dname]
        out[f"Sx_{dname}"] = [
            _sx_directional(elev, r, c, cellsize, az, SX_MAX_DIST_M, SX_STEP_M)
            for r, c in zip(rows, cols)
        ]

    out.index.name = "Sensor_ID"
    return out


if __name__ == "__main__":
    import tempfile

    from affine import Affine
    from shapely.geometry import Point

    size, cellsize = 50, 30.0
    xx = np.tile(np.arange(size), (size, 1)).astype(float)
    # Valley along the vertical centerline (col=25): elevation rises away from it.
    elev_synth = 50.0 + 2.0 * np.abs(xx - 25)
    transform = Affine(cellsize, 0, 500000, 0, -cellsize, 4000000)

    with tempfile.TemporaryDirectory() as tmp:
        dem_path = Path(tmp) / "synthetic_dem.tif"
        with rasterio.open(
            dem_path, "w", driver="GTiff", height=size, width=size, count=1,
            dtype="float64", crs="EPSG:32617", transform=transform,
        ) as ds:
            ds.write(elev_synth, 1)

        def xy(row, col):
            x, y = transform * (col + 0.5, row + 0.5)
            return x, y

        points = {
            1: xy(25, 25),  # valley bottom, center
            2: xy(25, 2),   # ridge, far from valley
            3: xy(5, 25),   # valley bottom, near edge
            4: xy(25, 48),  # ridge, far edge
        }
        sensors = gpd.GeoDataFrame(
            {"Sensor_ID": list(points)},
            geometry=[Point(xy_) for xy_ in points.values()],
            crs="EPSG:32617",
        ).set_index("Sensor_ID")

        from city_config import DURHAM
        feats = compute_terrain_features(DURHAM, sensors, dem_path)
        print(feats)

        assert feats.loc[1, "tpi_500"] < feats.loc[2, "tpi_500"], "valley sensor should have lower TPI than ridge sensor"
        assert feats.loc[1, "elev"] < feats.loc[2, "elev"], "valley sensor should be lower than ridge sensor"
        assert np.isfinite(feats["twi"]).all(), "TWI should be finite everywhere"
        assert feats.loc[1, "twi"] > feats.loc[2, "twi"], "valley (flow-accumulating) sensor should have higher TWI than ridge"
        print("\nself-test passed: valley sensor has lower TPI/elevation and higher TWI than ridge sensor.")
