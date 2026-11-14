"""City-level configuration (CRS, timezone, reference lat/lon, data paths)
for the geo-feature pipeline (notebook 10 and src/*_features.py).

Durham and Ramat Gan share every feature-extraction code path; only these
constants differ between them. ``urbansense_common.py`` keeps its own
hardcoded Durham lat/lon (it predates this module and nb01-09 must keep
reproducing unchanged) — this module is additive, used by the new geo
pipeline only.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CityConfig:
    name: str
    crs: str  # projected CRS (metres) used for all buffer/raster geometry ops
    tz: str  # IANA timezone, for local<->UTC conversion (pvlib solar position etc.)
    lat: float  # city-representative point, for single-point solar-position calcs
    lon: float
    boundary_path: str  # city/study-area boundary polygon (GeoJSON), relative to repo root
    sensors_path: str | None  # Sensor_ID, Lat, Lon (+ optional metadata) CSV, or None if not yet deployed
    raw_geo_dir: str  # downloaded geo rasters/vectors for this city
    processed_geo_path: str  # output of scripts/build_static_features.py


DURHAM = CityConfig(
    name="durham",
    crs="EPSG:32617",
    tz="US/Eastern",
    lat=35.99260,
    lon=-78.89892,
    boundary_path="data/raw/geo/durham/boundary.geojson",
    sensors_path="data/processed/sensor_info.csv",
    raw_geo_dir="data/raw/geo/durham",
    processed_geo_path="data/processed/static_geo_durham.parquet",
)

RAMAT_GAN = CityConfig(
    name="ramat_gan",
    crs="EPSG:2039",
    tz="Asia/Jerusalem",
    lat=32.0684,
    lon=34.8248,
    boundary_path="data/raw/geo/ramat_gan/boundary.geojson",
    sensors_path=None,
    raw_geo_dir="data/raw/geo/ramat_gan",
    processed_geo_path="data/processed/static_geo_ramat_gan.parquet",
)

CITIES = {"durham": DURHAM, "ramat_gan": RAMAT_GAN}


def get_city(name: str) -> CityConfig:
    try:
        return CITIES[name]
    except KeyError:
        raise ValueError(f"Unknown city {name!r}; expected one of {list(CITIES)}") from None
