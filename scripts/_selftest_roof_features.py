"""Synthetic-raster self-test for src/roof_features.py.

No network/API key needed — builds a tiny in-memory RGB raster with one
dark/solar-like roof and one bright/tile-like roof, and checks
compute_roof_features assigns the heuristic labels as expected. Run with:
    .venv/Scripts/python scripts/_selftest_roof_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from city_config import DURHAM  # noqa: E402
from roof_features import compute_roof_features  # noqa: E402

CRS = DURHAM.crs
SIZE = 40  # pixels, 1 m/pixel


def build_raster() -> np.ndarray:
    rgb = np.full((3, SIZE, SIZE), 200, dtype=np.uint8)  # bright background ("street/concrete")
    # Dark/solar-like roof: low brightness, low saturation, near (10,20,20)=0..255 scale.
    rgb[:, 5:15, 5:15] = np.array([20, 25, 22]).reshape(3, 1, 1)
    # Bright/tile-like roof: high brightness, mid saturation, reddish.
    rgb[:, 25:35, 25:35] = np.array([210, 90, 70]).reshape(3, 1, 1)
    return rgb


def main():
    rgb = build_raster()
    transform = from_origin(0, SIZE, 1, 1)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=SIZE, width=SIZE, count=3,
                       dtype=rgb.dtype, crs=CRS, transform=transform) as dst:
            dst.write(rgb)
        buildings = gpd.GeoDataFrame(
            {"bld_id": ["solar_roof", "tile_roof"]},
            geometry=[box(5, 25, 15, 35), box(25, 5, 35, 15)],  # note: row/col -> y flips vs. array indexing above
            crs=CRS,
        )
        sensors = gpd.GeoDataFrame(
            {"Sensor_ID": ["S_solar", "S_tile"]},
            geometry=[Point(10, 30), Point(30, 10)],
            crs=CRS,
        ).set_index("Sensor_ID")

        # mem.name is a GDAL vsimem path that rasterio.open() (inside compute_roof_features) can read directly.
        result = compute_roof_features(DURHAM, sensors, buildings, mem.name, buffer_radii={"100": 20.0})

    print(result)
    assert result.loc["S_solar", "roof_dark_frac_100"] > 0.5, "expected solar roof to read as mostly dark"
    assert result.loc["S_solar", "roof_solar_frac_100"] > 0.5, "expected solar roof to be flagged solar-like"
    assert result.loc["S_tile", "roof_solar_frac_100"] < result.loc["S_solar", "roof_solar_frac_100"], \
        "tile roof should be flagged solar-like less often than the solar roof"
    print("OK: compute_roof_features heuristic behaves as expected on synthetic data.")


if __name__ == "__main__":
    main()
