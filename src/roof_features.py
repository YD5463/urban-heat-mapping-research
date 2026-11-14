"""Roof colour/material features from a generic RGB raster (family R2 in the
geo-features plan), on top of the GBA building-footprint mask.

Deliberately source-agnostic: `rgb_path` can point at Google Map Tiles
imagery today, or NAIP / PlanetScope / a surveying-agency orthophoto later,
without any code change here — only the raster changes. This complements
(does not replace) the Sentinel-2-based `roof_albedo_r` / `roof_dark_frac_r`
computed elsewhere (family R); R2 exists to test whether higher-resolution
imagery adds anything over S2 in the ablation (B1+R vs. B1+R2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

from city_config import CityConfig

DEFAULT_BUFFER_RADII = {"100": 100.0}

# Solar panels / dark water heaters read as low-brightness, low-saturation
# (near-black/dark-blue) relative to red-tile or light concrete roofs; these
# thresholds are a simple heuristic baseline (not a trained classifier) and
# should be re-checked visually (notebook 10) before being trusted.
_SOLAR_BRIGHTNESS_MAX = 0.35
_SOLAR_SATURATION_MAX = 0.25
_DARK_BRIGHTNESS_MAX = 0.40


def _rgb_to_brightness_saturation(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rgb: (3, H, W) array in [0, 1]. Returns (brightness, saturation), each (H, W)."""
    cmax = rgb.max(axis=0)
    cmin = rgb.min(axis=0)
    brightness = cmax
    saturation = np.where(cmax > 0, (cmax - cmin) / np.where(cmax > 0, cmax, 1.0), 0.0)
    return brightness, saturation


def _normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(float)
    if np.nanmax(band) > 1.0:
        band = band / 255.0
    return np.clip(band, 0.0, 1.0)


def compute_roof_features(
    city: CityConfig,
    sensors: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    rgb_path: str,
    buffer_radii: dict | None = None,
) -> pd.DataFrame:
    """sensors: GeoDataFrame indexed by Sensor_ID, geometry = point, crs == city.crs.
    buildings: GBA footprint polygons (any columns), crs == city.crs; used only
    as a roof mask, like the S2-based roof_albedo_r elsewhere.
    rgb_path: path to a readable-by-rasterio RGB(+) raster covering the
    sensor buffers, in or reprojectable to city.crs.

    Returns a DataFrame indexed by Sensor_ID with, per radius key r in
    buffer_radii (default {"100": 100m}):
      roof_bright_{r}     mean brightness (V of HSV) over roof-masked pixels
      roof_dark_frac_{r}  fraction of roof-masked pixels with brightness < _DARK_BRIGHTNESS_MAX
      roof_solar_frac_{r} fraction of roof-masked pixels classified as
                          solar-panel/dark-water-heater-like by the heuristic
                          brightness+saturation threshold above (NOT a
                          trained model — a documented heuristic baseline).

    Buffers with no roof-masked pixels (no building within radius, or no
    raster coverage) get NaN, not 0, so they don't silently bias the mean.
    """
    buffer_radii = buffer_radii or DEFAULT_BUFFER_RADII
    if sensors.crs is None or str(sensors.crs) != str(city.crs):
        sensors = sensors.to_crs(city.crs)
    if buildings.crs is None or str(buildings.crs) != str(city.crs):
        buildings = buildings.to_crs(city.crs)

    bld_union = buildings.geometry.unary_union

    with rasterio.open(rgb_path) as src:
        raster_crs = src.crs
        rows = []
        for sensor_id, geom in sensors.geometry.items():
            row = {"Sensor_ID": sensor_id}
            for suffix, radius in buffer_radii.items():
                buf = geom.buffer(radius)
                roof_in_buf = buf.intersection(bld_union)
                vals = _sample_roof_pixels(src, roof_in_buf, raster_crs, city.crs)
                if vals is None or vals.shape[1] == 0:
                    row[f"roof_bright_{suffix}"] = np.nan
                    row[f"roof_dark_frac_{suffix}"] = np.nan
                    row[f"roof_solar_frac_{suffix}"] = np.nan
                    continue
                rgb = vals[:3]
                brightness, saturation = _rgb_to_brightness_saturation(rgb)
                solar_like = (brightness < _SOLAR_BRIGHTNESS_MAX) & (saturation < _SOLAR_SATURATION_MAX)
                row[f"roof_bright_{suffix}"] = float(brightness.mean())
                row[f"roof_dark_frac_{suffix}"] = float((brightness < _DARK_BRIGHTNESS_MAX).mean())
                row[f"roof_solar_frac_{suffix}"] = float(solar_like.mean())
            rows.append(row)

    return pd.DataFrame(rows).set_index("Sensor_ID")


def _sample_roof_pixels(src, roof_geom, raster_crs, city_crs):
    if roof_geom.is_empty:
        return None
    geom = roof_geom
    if raster_crs is not None and str(raster_crs) != str(city_crs):
        geom = gpd.GeoSeries([roof_geom], crs=city_crs).to_crs(raster_crs).iloc[0]
    try:
        out, _ = rio_mask(src, [mapping(geom)], crop=True, filled=False)
    except ValueError:
        return None  # geometry doesn't overlap the raster at all
    bands = [_normalize_band(np.ma.filled(out[b].astype(float), np.nan)) for b in range(out.shape[0])]
    stacked = np.stack(bands)
    valid = ~np.isnan(stacked).any(axis=0)
    if not valid.any():
        return None
    return stacked[:, valid]
