"""Download Google Map Tiles API satellite imagery covering a city's sensor
buffers (or full boundary), and stitch into one GeoTIFF for roof_features.py.

Usage:
    python scripts/fetch_google_tiles.py durham --buffer 300 --zoom 19
    python scripts/fetch_google_tiles.py ramat_gan --full-boundary --zoom 19

Requires the environment variable GOOGLE_MAPS_API_KEY (Map Tiles API v1).
Caches raw tiles under <city.raw_geo_dir>/google_tiles/z<zoom>/ (gitignored
via the repo's data/ ignore rule) and writes a single stitched GeoTIFF to
<city.raw_geo_dir>/google_tiles_mosaic.tif.

Per the project plan, usage of this script is the user's own call: Google's
imagery terms restrict derived-data extraction, and only aggregate
buffer-level roof features (not raw tiles or pixel-level derivatives) should
ever be published or committed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from city_config import CITIES  # noqa: E402

TILE_SESSION_URL = "https://tile.googleapis.com/v1/createSession"
TILE_URL_TEMPLATE = "https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}"
TILE_PX = 256


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds_lonlat(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2 ** zoom

    def lat_of(yy):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))

    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = lat_of(y)
    lat_s = lat_of(y + 1)
    return lon_w, lat_s, lon_e, lat_n


def create_session(api_key: str) -> str:
    resp = requests.post(
        TILE_SESSION_URL,
        params={"key": api_key},
        json={"mapType": "satellite", "language": "en-US", "region": "US"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["session"]


def fetch_tiles(city_name: str, buffer_m: float, zoom: int, full_boundary: bool) -> Path | None:
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("GOOGLE_MAPS_API_KEY is not set — set it in the environment to download "
              "Google Map Tiles imagery. Skipping (not an error).")
        return None

    city = CITIES[city_name]
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / city.raw_geo_dir / "google_tiles" / f"z{zoom}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if full_boundary:
        boundary_path = repo_root / city.boundary_path
        if not boundary_path.exists():
            print(f"--full-boundary given but {boundary_path} doesn't exist; "
                  f"falling back to a buffer around city.lat/lon instead.")
            lon0, lat0 = city.lon, city.lat
            import geopandas as gpd
            from shapely.geometry import Point
            bounds = gpd.GeoSeries([Point(lon0, lat0)], crs="EPSG:4326").to_crs(city.crs) \
                .buffer(buffer_m).to_crs("EPSG:4326").total_bounds
        else:
            import geopandas as gpd
            bounds = gpd.read_file(boundary_path).to_crs("EPSG:4326").total_bounds
    else:
        if not city.sensors_path:
            print(f"City {city_name!r} has no sensors_path and --full-boundary wasn't given; nothing to fetch.")
            return None
        sensors_path = repo_root / city.sensors_path
        if not sensors_path.exists():
            print(f"{sensors_path} doesn't exist yet — nothing to fetch for {city_name!r}.")
            return None
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point
        df = pd.read_csv(sensors_path)
        pts = gpd.GeoSeries([Point(lon, lat) for lat, lon in zip(df["Lat"], df["Lon"])], crs="EPSG:4326")
        bounds = pts.to_crs(city.crs).buffer(buffer_m).to_crs("EPSG:4326").total_bounds

    min_lon, min_lat, max_lon, max_lat = bounds
    x0, y1 = lonlat_to_tile(min_lon, min_lat, zoom)
    x1, y0 = lonlat_to_tile(max_lon, max_lat, zoom)
    x_lo, x_hi = sorted((x0, x1))
    y_lo, y_hi = sorted((y0, y1))
    n_tiles = (x_hi - x_lo + 1) * (y_hi - y_lo + 1)
    print(f"{city_name}: fetching {n_tiles} tiles at zoom {zoom} "
          f"(x {x_lo}-{x_hi}, y {y_lo}-{y_hi})")

    session = create_session(api_key)
    tile_paths = []
    for x in range(x_lo, x_hi + 1):
        for y in range(y_lo, y_hi + 1):
            dest = out_dir / f"{x}_{y}.png"
            tile_paths.append((dest, x, y))
            if dest.exists():
                continue
            url = TILE_URL_TEMPLATE.format(z=zoom, x=x, y=y)
            resp = requests.get(url, params={"session": session, "key": api_key}, timeout=30)
            if resp.status_code != 200:
                print(f"  tile {x},{y} failed: HTTP {resp.status_code}")
                continue
            dest.write_bytes(resp.content)

    return _stitch(tile_paths, zoom, city, repo_root)


def _stitch(tile_paths, zoom, city, repo_root) -> Path | None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds
    from PIL import Image

    present = [(p, x, y) for p, x, y in tile_paths if p.exists()]
    if not present:
        print("No tiles downloaded; nothing to stitch.")
        return None

    xs = sorted({x for _, x, _ in present})
    ys = sorted({y for _, _, y in present})
    x_lo, y_lo = xs[0], ys[0]
    width = (xs[-1] - x_lo + 1) * TILE_PX
    height = (ys[-1] - y_lo + 1) * TILE_PX
    mosaic = np.zeros((3, height, width), dtype=np.uint8)

    for path, x, y in present:
        img = np.array(Image.open(path).convert("RGB"))
        col0 = (x - x_lo) * TILE_PX
        row0 = (y - y_lo) * TILE_PX
        mosaic[:, row0:row0 + TILE_PX, col0:col0 + TILE_PX] = img.transpose(2, 0, 1)

    lon_w, _, _, lat_n = tile_bounds_lonlat(x_lo, y_lo, zoom)
    lon_e, lat_s, _, _ = tile_bounds_lonlat(xs[-1], ys[-1], zoom)
    transform = from_bounds(lon_w, lat_s, lon_e, lat_n, width, height)

    out_path = repo_root / city.raw_geo_dir / "google_tiles_mosaic.tif"
    with rasterio.open(
        out_path, "w", driver="GTiff", height=height, width=width, count=3,
        dtype=mosaic.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(mosaic)
    print(f"Wrote mosaic: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("city", choices=list(CITIES))
    parser.add_argument("--buffer", type=float, default=300.0, help="metres around each sensor")
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--full-boundary", action="store_true",
                         help="cover the whole city boundary instead of per-sensor buffers")
    args = parser.parse_args()
    fetch_tiles(args.city, args.buffer, args.zoom, args.full_boundary)


if __name__ == "__main__":
    main()
