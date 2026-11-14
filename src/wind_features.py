"""Dynamic (per sensor-hour) wind features: upwind building blockage, upwind
vegetation fetch, and a locally-reduced effective wind speed.

All geometry is done in the city's projected CRS (metres) from
`city_config.CityConfig`. Static ingredients (building footprints+heights,
NDVI raster, roughness length z0) come from morphology_features.py /
S2 rasters and are passed in rather than recomputed here.

Bearing convention matches `network_features.upwind_mean`: bearing from a
point towards another point is `degrees(atan2(dx, dy)) % 360` (0=north,
clockwise). `wind_dir_deg` is the meteorological "from" direction (ERA5
convention): a building is "upwind" of a sensor if it lies in roughly that
same compass direction from the sensor.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.affinity import rotate as shp_rotate
from shapely.geometry import LineString

from city_config import CityConfig

N_DIR_BINS = 16
DIR_BIN_WIDTH = 360.0 / N_DIR_BINS
DIR_BIN_CENTERS = np.arange(N_DIR_BINS) * DIR_BIN_WIDTH


def _nearest_dir_bin(deg: np.ndarray | float) -> np.ndarray:
    idx = np.round(np.asarray(deg, dtype=float) / DIR_BIN_WIDTH).astype(int) % N_DIR_BINS
    return DIR_BIN_CENTERS[idx]


def _frontal_area_index(sensor_xy: tuple[float, float], buildings: gpd.GeoDataFrame,
                         wind_from_deg: float, radius: float = 250.0,
                         angle_tolerance_deg: float = 60.0, height_col: str = "height") -> float:
    """Sum(building frontal area facing the wind) / buffer plan area, for
    buildings within `radius` that lie upwind of the sensor (within
    +/- angle_tolerance_deg of wind_from_deg)."""
    sx, sy = sensor_xy
    cand = buildings[buildings.distance(gpd_point(sx, sy)) <= radius]
    if cand.empty:
        return 0.0

    dx = cand.geometry.centroid.x.values - sx
    dy = cand.geometry.centroid.y.values - sy
    bearing = np.degrees(np.arctan2(dx, dy)) % 360
    angle_diff = np.abs((bearing - wind_from_deg + 180) % 360 - 180)
    upwind = cand[angle_diff <= angle_tolerance_deg]
    if upwind.empty:
        return 0.0

    # Rotate so the wind direction aligns with the rotated y-axis; the
    # footprint's extent along the rotated x-axis is its width facing the wind.
    total_frontal_area = 0.0
    for geom, height in zip(upwind.geometry, upwind[height_col]):
        rotated = shp_rotate(geom, -wind_from_deg, origin=(sx, sy), use_radians=False)
        xs = np.asarray(rotated.exterior.coords.xy[0])
        width = xs.max() - xs.min()
        total_frontal_area += width * height

    return total_frontal_area / (np.pi * radius ** 2)


def _fetch_ndvi(sensor_xy: tuple[float, float], ndvi_dataset, wind_from_deg: float,
                 length: float = 500.0, corridor_width: float = 50.0) -> float:
    """Mean NDVI along a corridor of `corridor_width` extending `length`
    metres upwind of the sensor (i.e. towards where the wind is coming from)."""
    sx, sy = sensor_xy
    theta = np.radians(wind_from_deg)
    end = (sx + length * np.sin(theta), sy + length * np.cos(theta))
    corridor = LineString([(sx, sy), end]).buffer(corridor_width / 2.0)
    try:
        arr, _ = rio_mask(ndvi_dataset, [corridor], crop=True, filled=True, nodata=np.nan)
    except ValueError:
        return np.nan  # corridor doesn't overlap the raster at all
    vals = arr[0]
    valid = vals[np.isfinite(vals)]
    return float(valid.mean()) if valid.size else np.nan


def gpd_point(x: float, y: float):
    from shapely.geometry import Point
    return Point(x, y)


def _log_wind_profile(wind_speed_10m: np.ndarray, z0: np.ndarray,
                       z_sensor: float = 2.0, z_ref: float = 10.0) -> np.ndarray:
    """Reduce ERA5's 10m wind speed to the sensor height via a logarithmic
    wind profile. z0 clipped well below z_sensor to keep both logs positive
    and the ratio bounded for very rough (large z0) built-up pixels."""
    z0_safe = np.clip(np.asarray(z0, dtype=float), 1e-4, z_sensor * 0.3)
    ratio = np.log(z_sensor / z0_safe) / np.log(z_ref / z0_safe)
    return np.asarray(wind_speed_10m, dtype=float) * np.clip(ratio, 0.0, 1.0)


def compute_wind_features_dynamic(
    city: CityConfig,
    sensors: gpd.GeoDataFrame,
    times_utc: pd.DatetimeIndex,
    wind_dir_deg: pd.Series,
    wind_speed_ms: pd.Series,
    buildings: gpd.GeoDataFrame,
    ndvi_path,
    z0: pd.Series,
    height_col: str = "height",
) -> pd.DataFrame:
    """sensors: GeoDataFrame indexed by Sensor_ID, geometry = point in city.crs.
    times_utc/wind_dir_deg/wind_speed_ms: aligned hourly ERA5 series (same index as times_utc).
    z0: Series indexed by Sensor_ID (roughness length in metres, from morphology_features.py).
    Returns a DataFrame indexed by (Sensor_ID, time) with columns:
    fai_upwind_250, fetch_ndvi_500, wind_eff.
    """
    assert sensors.crs == buildings.crs == city.crs or str(sensors.crs) == str(city.crs), \
        "sensors/buildings must be in city.crs"

    sensor_xy = {sid: (geom.x, geom.y) for sid, geom in zip(sensors.index, sensors.geometry)}

    # Geometry-dependent features only vary with wind *direction*, not time,
    # so precompute them once per (sensor, 16-sector bin) and look up per hour.
    fai_lookup: dict[tuple, float] = {}
    fetch_lookup: dict[tuple, float] = {}

    close_ndvi = isinstance(ndvi_path, (str, Path))
    ndvi_dataset = rasterio.open(ndvi_path) if close_ndvi else ndvi_path
    try:
        for sid, xy in sensor_xy.items():
            for d in DIR_BIN_CENTERS:
                fai_lookup[(sid, d)] = _frontal_area_index(xy, buildings, d, height_col=height_col)
                fetch_lookup[(sid, d)] = _fetch_ndvi(xy, ndvi_dataset, d)
    finally:
        if close_ndvi:
            ndvi_dataset.close()

    dir_bin = wind_dir_deg.reindex(times_utc).pipe(_nearest_dir_bin)
    dir_bin = pd.Series(dir_bin, index=times_utc)
    ws = wind_speed_ms.reindex(times_utc)

    records = []
    for sid in sensors.index:
        z0_s = float(z0.loc[sid])
        eff = _log_wind_profile(ws.values, np.full(len(ws), z0_s))
        for t, d, we in zip(times_utc, dir_bin.values, eff):
            records.append({
                "Sensor_ID": sid, "time": t,
                "fai_upwind_250": fai_lookup[(sid, d)],
                "fetch_ndvi_500": fetch_lookup[(sid, d)],
                "wind_eff": we,
            })

    out = pd.DataFrame(records).set_index(["Sensor_ID", "time"])
    return out


# --- WindNinja speedup lookup (table 14*, optional) -------------------------
# WindNinja (https://weather.firelab.org/windninja/) is a mass-conserving
# diagnostic wind model. We don't run its solver here; this just reads a
# precomputed stack of speedup-ratio rasters (one band per direction x speed
# combination, mass-conserving solves done ahead of time) and looks up the
# nearest direction/speed bin.

def speedup_lookup(grid_path, sensor_xy: tuple[float, float], wind_dir_deg: float,
                    wind_speed_ms: float, directions=DIR_BIN_CENTERS, speeds=(2.0, 5.0, 10.0)) -> float:
    """`grid_path` is a multiband raster, band = dir_idx * len(speeds) + speed_idx + 1
    (rasterio bands are 1-indexed), each band holding a speedup ratio
    (local speed / reference speed) for that (direction, speed) WindNinja run."""
    dir_idx = int(np.argmin(np.abs(np.asarray(directions) - wind_dir_deg)))
    speed_idx = int(np.argmin(np.abs(np.asarray(speeds) - wind_speed_ms)))
    band = dir_idx * len(speeds) + speed_idx + 1
    with rasterio.open(grid_path) as ds:
        row, col = ds.index(*sensor_xy)
        return float(ds.read(band, window=((row, row + 1), (col, col + 1)))[0, 0])


def run_windninja(dem_path, out_dir, windninja_cli: str = "WindNinja_cli",
                   directions=DIR_BIN_CENTERS, speeds=(2.0, 5.0, 10.0)) -> list[Path]:
    """Best-effort wrapper: runs WindNinja's CLI once per (direction, speed)
    to build the grid `speedup_lookup` reads. Untested against a real
    WindNinja install (not available in this environment) — raises a clear
    error instead of silently producing nothing if the binary is missing."""
    if shutil.which(windninja_cli) is None:
        raise RuntimeError(
            f"WindNinja not available: {windninja_cli!r} not found on PATH. "
            "Install from https://weather.firelab.org/windninja/ or skip the "
            "W+ (speedup) feature — it is optional in the ablation."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for d in directions:
        for s in speeds:
            tag = f"dir{int(d):03d}_spd{s:g}"
            cmd = [
                windninja_cli,
                "--elevation_file", str(dem_path),
                "--initialization_method", "domainAverageInitialization",
                "--input_speed", str(s), "--input_speed_units", "mps",
                "--input_direction", str(d),
                "--output_path", str(out_dir / tag),
                "--mesh_resolution", "30", "--units_mesh_resolution", "m",
            ]
            subprocess.run(cmd, check=True)
            outputs.append(out_dir / tag)
    return outputs


if __name__ == "__main__":
    import tempfile

    from shapely.geometry import Point, box

    crs = "EPSG:32617"
    sensors = gpd.GeoDataFrame(
        {"Sensor_ID": ["A", "B"]},
        geometry=[Point(0, 0), Point(1000, 0)],
        crs=crs,
    ).set_index("Sensor_ID")

    # A tall building cluster immediately west (upwind, wind_from=270) of A only.
    buildings = gpd.GeoDataFrame(
        {"height": [20.0]},
        geometry=[box(-150, -50, -50, 50)],
        crs=crs,
    )

    # NDVI raster: baseline 0.1 everywhere, a vegetated patch (0.9) west of B only.
    west, north, xres, yres = -600.0, 600.0, 20.0, 20.0
    width, height = 110, 60
    ndvi = np.full((height, width), 0.1, dtype="float32")
    transform = rasterio.transform.from_origin(west, north, xres, yres)

    def rc(x, y):
        col = int((x - west) / xres)
        row = int((north - y) / yres)
        return row, col

    r0, c0 = rc(500, 100)
    r1, c1 = rc(1000, -100)
    ndvi[r0:r1, c0:c1] = 0.9  # patch x in [500,1000], y in [-100,100]

    tmp = Path(tempfile.mkdtemp()) / "ndvi.tif"
    with rasterio.open(
        tmp, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(ndvi, 1)

    times = pd.DatetimeIndex(["2024-07-14 03:00", "2024-07-14 15:00"], tz="UTC")
    wind_dir = pd.Series([270.0, 270.0], index=times)  # wind blowing FROM the west
    wind_speed = pd.Series([3.0, 5.0], index=times)
    z0 = pd.Series({"A": 1.0, "B": 0.3})

    test_city = CityConfig(
        name="test", crs=crs, tz="UTC", lat=0.0, lon=0.0,
        boundary_path="", sensors_path=None, raw_geo_dir="", processed_geo_path="",
    )
    result = compute_wind_features_dynamic(
        city=test_city, sensors=sensors, times_utc=times, wind_dir_deg=wind_dir,
        wind_speed_ms=wind_speed, buildings=buildings, ndvi_path=tmp, z0=z0,
    )
    print(result)

    fai_a = result.loc["A"]["fai_upwind_250"].iloc[0]
    fai_b = result.loc["B"]["fai_upwind_250"].iloc[0]
    fetch_a = result.loc["A"]["fetch_ndvi_500"].iloc[0]
    fetch_b = result.loc["B"]["fetch_ndvi_500"].iloc[0]
    assert fai_a > fai_b, f"expected A (buildings upwind) to have higher FAI: {fai_a} vs {fai_b}"
    assert fetch_b > fetch_a, f"expected B (vegetation upwind) to have higher fetch NDVI: {fetch_b} vs {fetch_a}"
    assert result["wind_eff"].notna().all()
    print("self-test OK: fai_a=%.4f fai_b=%.4f fetch_a=%.3f fetch_b=%.3f" % (fai_a, fai_b, fetch_a, fetch_b))
