"""Orchestrates src/{terrain,morphology,wind,shadow,roof}_features.py against
real downloaded geo data for one city (geo-features plan, section 4),
producing data/processed/static_geo_<city>.parquet: one row per sensor, every
pre-registered + exploration-only column from all five feature families.

Usage: python scripts/build_static_features.py durham
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# This repo lives under a non-ASCII (Hebrew) directory name; Windows' default
# console codepage (cp1252) can't print it, crashing any print() of an
# absolute path right after the real work (e.g. to_parquet) already succeeded.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from city_config import get_city  # noqa: E402
from morphology_features import build_synthetic_dsm, compute_morphology_features  # noqa: E402
from roof_features import compute_roof_features  # noqa: E402
from shadow_features import compute_shadow_features_dynamic  # noqa: E402
from terrain_features import compute_terrain_features  # noqa: E402
from wind_features import compute_wind_features_dynamic  # noqa: E402

# Narrow-to-broadband albedo coefficients, transcribed from the authors' own
# reference/UrbanSENSE_ML/1_Creating_Input_Data/Extract_inputs_GEE.js
# (computeAlbedoNTB), so our S2 albedo matches Albedo_2024SummerMosaic's formula.
_ALBEDO_COEF = {"B02": 0.2266, "B03": 0.1236, "B04": 0.1573, "B08": 0.3417, "B11": 0.1170, "B12": 0.0338}

# Dynamic (wind/shadow) features are evaluated for one representative week
# rather than the full ~3-month record: the per-sensor ray-march (shadow) and
# per-direction-bin building/NDVI lookups (wind) are O(sensors x hours), and a
# week spans a realistic range of wind directions/speeds and solar angles
# without an hours-long run.
DYNAMIC_WEEK_START = "2024-07-14"
DYNAMIC_WEEK_HOURS = 24 * 7
SVF_RADIUS_M = 60.0
ROOF_RADIUS_M = 250.0


def _mosaic_fabdem(city) -> Path:
    """Durham's window spans 4 one-degree FABDEM tiles; Ramat Gan's spans 1.
    Mosaics them (still in FABDEM's native EPSG:4326) if there's more than one."""
    tiles = sorted((REPO_ROOT / city.raw_geo_dir / "fabdem").glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(f"no FABDEM tiles under {city.raw_geo_dir}/fabdem/ - run fetch_geo_{city.name}.py first")
    out_path = REPO_ROOT / city.raw_geo_dir / "fabdem_mosaic.tif"
    if len(tiles) == 1:
        shutil.copy(tiles[0], out_path)
        return out_path
    srcs = [rasterio.open(t) for t in tiles]
    mosaic, transform = rio_merge(srcs)
    profile = srcs[0].profile.copy()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    return out_path


def _reproject_dem_to_city_crs(city, dem_path_4326: Path) -> Path:
    """terrain_features.compute_terrain_features treats the DEM's own CRS as
    the working CRS for slope/TPI/TWI/Sx (metres assumed) - unlike
    morphology_features.build_synthetic_dsm, it does not reproject the raster
    itself, only the sensor points to match it. FABDEM ships in EPSG:4326
    (degrees), so it must be reprojected to the city's metric CRS first."""
    out_path = dem_path_4326.with_name(dem_path_4326.stem + "_proj.tif")
    with rasterio.open(dem_path_4326) as src:
        transform, width, height = calculate_default_transform(
            src.crs, city.crs, src.width, src.height, *src.bounds, resolution=30.0,
        )
        profile = src.profile.copy()
        profile.update(crs=city.crs, transform=transform, width=width, height=height)
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=city.crs,
                resampling=Resampling.bilinear,
            )
    return out_path


def _build_s2_albedo(city) -> Path:
    """Broadband albedo via computeAlbedoNTB (see module docstring); SWIR
    bands (20m) are resampled onto the 10m visible/NIR grid first."""
    s2_dir = REPO_ROOT / city.raw_geo_dir / "s2"
    with rasterio.open(s2_dir / "B02.tif") as ref:
        ref_profile, ref_shape, ref_transform, ref_crs = ref.profile.copy(), (ref.height, ref.width), ref.transform, ref.crs

    bands = {}
    for name in ("B02", "B03", "B04", "B08"):
        with rasterio.open(s2_dir / f"{name}.tif") as ds:
            bands[name] = ds.read(1).astype(np.float32) * 0.0001
    for name in ("B11", "B12"):
        with rasterio.open(s2_dir / f"{name}.tif") as ds:
            dst = np.zeros(ref_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(ds, 1), destination=dst,
                src_transform=ds.transform, src_crs=ds.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
            bands[name] = dst * 0.0001

    albedo = sum(_ALBEDO_COEF[b] * bands[b] for b in _ALBEDO_COEF)
    out_path = s2_dir / "albedo.tif"
    profile = ref_profile.copy()
    profile.update(dtype="float32", count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(albedo.astype(np.float32), 1)
    return out_path


def _build_ndvi(city) -> Path:
    s2_dir = REPO_ROOT / city.raw_geo_dir / "s2"
    out_path = s2_dir / "ndvi.tif"
    with rasterio.open(s2_dir / "B08.tif") as nir, rasterio.open(s2_dir / "B04.tif") as red:
        nir_arr, red_arr = nir.read(1).astype(np.float32), red.read(1).astype(np.float32)
        ndvi = (nir_arr - red_arr) / np.maximum(nir_arr + red_arr, 1e-6)
        profile = nir.profile.copy()
        profile.update(dtype="float32", count=1)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndvi.astype(np.float32), 1)
    return out_path


def _build_s2_rgb_u8(city) -> Path:
    """roof_features.py's generic-RGB interface assumes 0-255 imagery
    (matching Google Tiles, its intended source). S2's surface-reflectance DN
    is converted to 8-bit via a fixed 0-0.3-reflectance display stretch - a
    crude stand-in until a true high-resolution RGB source (Google tiles,
    once GOOGLE_MAPS_API_KEY is set, or an orthophoto) is available."""
    s2_dir = REPO_ROOT / city.raw_geo_dir / "s2"
    out_path = s2_dir / "rgb_u8.tif"
    with rasterio.open(s2_dir / "B04.tif") as r_ds, rasterio.open(s2_dir / "B03.tif") as g_ds, rasterio.open(s2_dir / "B02.tif") as b_ds:
        bands_u8 = []
        for ds in (r_ds, g_ds, b_ds):
            refl = ds.read(1).astype(np.float32) * 0.0001
            bands_u8.append(np.clip(refl / 0.3 * 255.0, 0, 255).astype("uint8"))
        profile = r_ds.profile.copy()
        profile.update(count=3, dtype="uint8")
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, arr in enumerate(bands_u8, start=1):
                dst.write(arr, i)
    return out_path


def _load_sensors(city) -> gpd.GeoDataFrame:
    df = pd.read_csv(REPO_ROOT / city.sensors_path)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["Lon"], df["Lat"]), crs="EPSG:4326",
    ).set_index("Sensor_ID").to_crs(city.crs)
    return gdf


def _load_buildings(city) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(REPO_ROOT / city.raw_geo_dir / "gba" / "buildings.parquet")
    return gdf.to_crs(city.crs)


def _load_wind_series(city) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
    """ERA5 u/v -> meteorological "from" wind direction, one row per hour
    (ERA5-Land's ~9km cells make every sensor's value for a given hour
    effectively identical; averaging across sensors is a harmless tie-break).
    Restricted to the DYNAMIC_WEEK_* window."""
    era5_path = REPO_ROOT / "data/processed/ERA5_everyday_merged.csv"
    if not era5_path.exists():
        raise FileNotFoundError(f"{era5_path} missing - run scripts/download_data.sh first")
    df = pd.read_csv(era5_path, usecols=[
        "edt_datetime", "u_component_of_wind_10m", "v_component_of_wind_10m", "windspeed_10m",
    ])
    df["edt_datetime"] = pd.to_datetime(df["edt_datetime"])
    hourly = df.groupby("edt_datetime")[
        ["u_component_of_wind_10m", "v_component_of_wind_10m", "windspeed_10m"]
    ].mean().dropna()
    hourly.index = hourly.index.tz_localize(city.tz, ambiguous="NaT", nonexistent="shift_forward")
    hourly = hourly[~hourly.index.isna()]

    week_start = pd.Timestamp(DYNAMIC_WEEK_START, tz=city.tz)
    week_mask = (hourly.index >= week_start) & (hourly.index < week_start + pd.Timedelta(hours=DYNAMIC_WEEK_HOURS))
    hourly = hourly.loc[week_mask]
    if hourly.empty:
        raise ValueError(f"no ERA5 rows in the {DYNAMIC_WEEK_HOURS}h window starting {DYNAMIC_WEEK_START}")

    times_utc = pd.DatetimeIndex(hourly.index).tz_convert("UTC")
    u, v = hourly["u_component_of_wind_10m"].values, hourly["v_component_of_wind_10m"].values
    wind_dir_deg = pd.Series(np.degrees(np.arctan2(-u, -v)) % 360, index=times_utc)
    wind_speed_ms = pd.Series(hourly["windspeed_10m"].values, index=times_utc)
    return times_utc, wind_dir_deg, wind_speed_ms


def build_static_features(city_name: str) -> pd.DataFrame:
    city = get_city(city_name)
    if not city.sensors_path:
        raise ValueError(f"{city.name} has no sensors_path yet - nothing to compute a per-sensor table for.")

    print(f"=== building static geo-features for {city.name} ===")
    sensors = _load_sensors(city)
    buildings = _load_buildings(city)
    print(f"{len(sensors)} sensors, {len(buildings)} building footprints")

    dem_path_4326 = _mosaic_fabdem(city)
    dem_path_proj = _reproject_dem_to_city_crs(city, dem_path_4326)
    s2_albedo_path = _build_s2_albedo(city)
    ndvi_path = _build_ndvi(city)
    rgb_path = _build_s2_rgb_u8(city)
    canopy_path = REPO_ROOT / city.raw_geo_dir / "canopy" / "canopy_avg.tif"
    canopy_path = canopy_path if canopy_path.exists() else None

    print("terrain features...")
    terrain = compute_terrain_features(city, sensors, dem_path_proj)

    print("morphology + roof-albedo (S2) features...")
    morph = compute_morphology_features(
        city, sensors, dem_path_4326, buildings, s2_albedo_path,
        canopy_path=canopy_path, buffer_radii={"roof": ROOF_RADIUS_M, "morph": 100.0},
    )

    print("roof features (R2, S2-RGB stand-in - GOOGLE_MAPS_API_KEY not set in this environment)...")
    roof = compute_roof_features(city, sensors, buildings, rgb_path)

    print("loading ERA5 wind series for the dynamic-feature week...")
    times_utc, wind_dir_deg, wind_speed_ms = _load_wind_series(city)

    print(f"wind features for {len(times_utc)}h x {len(sensors)} sensors...")
    wind = compute_wind_features_dynamic(
        city, sensors, times_utc, wind_dir_deg, wind_speed_ms, buildings, ndvi_path, z0=morph["z0"],
    )

    print("shadow features (per-sensor local DSM, same reasoning as the SVF fix in morphology_features.py)...")
    bld_sindex = buildings.sindex
    shadow_parts = []
    shadow_radius_m = 200.0  # matches compute_shadow_features_dynamic's default max_dist_m
    for sensor_id, geom in zip(sensors.index, sensors.geometry):
        local_bounds = (geom.x - shadow_radius_m, geom.y - shadow_radius_m,
                         geom.x + shadow_radius_m, geom.y + shadow_radius_m)
        local_buildings = buildings.iloc[list(bld_sindex.intersection(local_bounds))]
        dsm, dsm_transform, _ = build_synthetic_dsm(
            city, local_bounds, resolution_m=1.0, dem_path=dem_path_4326,
            buildings=local_buildings, canopy_path=canopy_path,
        )
        one_sensor = sensors.loc[[sensor_id]]
        shadow_parts.append(compute_shadow_features_dynamic(city, one_sensor, times_utc, dsm, dsm_transform))
    shadow = pd.concat(shadow_parts)

    dynamic = wind.join(shadow, how="outer")
    dynamic_summary = dynamic.groupby("Sensor_ID").mean(numeric_only=True)
    dynamic_summary.columns = [f"{c}_mean" for c in dynamic_summary.columns]

    static = terrain.join(morph, how="outer").join(roof, how="outer").join(dynamic_summary, how="outer")
    static.index.name = "Sensor_ID"

    out_path = REPO_ROOT / city.processed_geo_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    static.to_parquet(out_path)
    print(f"wrote {static.shape} to {out_path.relative_to(REPO_ROOT)}")

    dynamic_out = out_path.with_name(out_path.stem + "_dynamic.parquet")
    dynamic.reset_index().to_parquet(dynamic_out)
    print(f"wrote per-hour dynamic table {dynamic.shape} to {dynamic_out.relative_to(REPO_ROOT)}")

    return static


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: build_static_features.py <city_name>")
        sys.exit(1)
    build_static_features(sys.argv[1])
