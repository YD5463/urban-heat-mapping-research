"""Shared implementation for scripts/fetch_geo_<city>.py.

Downloads, for one city, the raw geo layers feeding src/terrain_features.py,
src/morphology_features.py, src/wind_features.py, src/shadow_features.py and
src/roof_features.py, into data/raw/geo/<city>/<source>/. Each source is
windowed to the city's sensor bbox (or geocoded boundary, if there is no
sensor file yet) plus a 2km buffer, fetched directly off its host via GDAL's
/vsicurl//vsizip/ remote-range-read or a bbox-pruned remote query, so we never
pull a whole multi-GB global tile/partition to disk just to crop it locally.

Each downloader prints one summary line and returns True/False; a failed
source never aborts the others (research-grade: partial results are still
useful, and a human reads the log to see what to patch up by hand).
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

# This repo lives under a non-ASCII (Hebrew) directory name; Windows' default
# console codepage (cp1252) can't print it, crashing any print() of an
# absolute path right after the real work (e.g. a successful download)
# already completed.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# GDAL/curl's Windows schannel backend cannot open a CA bundle path containing
# non-ASCII characters (this repo lives under a Hebrew directory name) and
# fails every HTTPS request with a cryptic "filename ... is incorrect" error.
# Copy the CA bundle to a plain-ASCII temp path once, before rasterio/requests
# touch the network, and point every relevant env var at that copy.
try:
    import certifi

    _ascii_cacert = Path(tempfile.gettempdir()) / "urbansense_cacert.pem"
    if not _ascii_cacert.exists():
        shutil.copy(certifi.where(), _ascii_cacert)
    for _var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "GDAL_CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        os.environ[_var] = str(_ascii_cacert)
except Exception:
    pass

import duckdb
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from city_config import get_city  # noqa: E402

BUFFER_M = 2000.0


def _log(source: str, msg: str) -> None:
    print(f"[{source}] {msg}", flush=True)


def _clip_window(win: Window, src) -> Window:
    """`from_bounds` can return a window that partially or fully falls
    outside the dataset (e.g. a 1-degree FABDEM tile only covers part of a
    multi-tile download window). `read(window=...)` silently clips such a
    window to the valid extent, but `window_transform(win)` on the
    *unclipped* window still anchors the array at the requested (possibly
    out-of-bounds) corner - georeferencing the returned (smaller) array to
    the wrong location. Intersecting first keeps the transform and the data
    shape in agreement."""
    full = Window(0, 0, src.width, src.height)
    return win.intersection(full)


def get_window(city) -> tuple[gpd.GeoDataFrame, tuple[float, float, float, float]]:
    """Returns (window_gdf in EPSG:4326, (west, south, east, north) in degrees)."""
    if city.sensors_path and (REPO_ROOT / city.sensors_path).exists():
        df = pd.read_csv(REPO_ROOT / city.sensors_path)
        pts = gpd.GeoSeries(gpd.points_from_xy(df["Lon"], df["Lat"]), crs="EPSG:4326")
        area = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")
        hull = area.to_crs(city.crs).union_all().convex_hull.buffer(BUFFER_M)
        window = gpd.GeoDataFrame(geometry=[hull], crs=city.crs).to_crs("EPSG:4326")
    else:
        import osmnx as ox

        boundary = ox.geocode_to_gdf(f"{city.name.replace('_', ' ').title()}, Israel")
        hull = boundary.to_crs(city.crs).union_all().buffer(BUFFER_M)
        window = gpd.GeoDataFrame(geometry=[hull], crs=city.crs).to_crs("EPSG:4326")

    out_path = REPO_ROOT / city.boundary_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window.to_file(out_path, driver="GeoJSON")
    bounds = window.total_bounds  # west, south, east, north
    return window, tuple(bounds)


# ---------------------------------------------------------------- FABDEM ---
def fetch_fabdem(city, bounds) -> bool:
    west, south, east, north = bounds
    out_dir = REPO_ROOT / city.raw_geo_dir / "fabdem"
    out_dir.mkdir(parents=True, exist_ok=True)

    # FABDEM is shipped as 10x10-degree zips, each containing per-1-degree
    # tiles. Find the 10x10 bundle covering our window's SW corner (windows
    # this small never cross a bundle boundary in our two cities).
    lat10 = math.floor(south / 10) * 10
    lon10 = math.floor(west / 10) * 10

    def lat_tag(v):
        return f"{'N' if v >= 0 else 'S'}{abs(int(v)):02d}"

    def lon_tag(v):
        return f"{'E' if v >= 0 else 'W'}{abs(int(v)):03d}"

    bundle = f"{lat_tag(lat10)}{lon_tag(lon10)}-{lat_tag(lat10 + 10)}{lon_tag(lon10 + 10)}"
    zip_url = f"https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/{bundle}_FABDEM_V1-2.zip"

    # Individual tiles inside the bundle are named by their own 1-degree SW corner.
    lat_lo, lat_hi = math.floor(south), math.floor(north)
    lon_lo, lon_hi = math.floor(west), math.floor(east)
    n_ok = 0
    for lat in range(lat_lo, lat_hi + 1):
        for lon in range(lon_lo, lon_hi + 1):
            tile = f"{lat_tag(lat)}{lon_tag(lon)}_FABDEM_V1-2.tif"
            vsi_path = f"/vsizip//vsicurl/{zip_url}/{tile}"
            dest = out_dir / tile
            try:
                with rasterio.open(vsi_path) as src:
                    win = _clip_window(from_bounds(west, south, east, north, src.transform), src)
                    data = src.read(1, window=win)
                    profile = src.profile.copy()
                    profile.update(
                        height=data.shape[0], width=data.shape[1],
                        transform=src.window_transform(win),
                    )
                with rasterio.open(dest, "w", **profile) as dst:
                    dst.write(data, 1)
                n_ok += 1
            except Exception as e:
                _log("fabdem", f"tile {tile} unavailable/out of window ({e})")
    if n_ok:
        _log("fabdem", f"wrote {n_ok} clipped tile(s) to {out_dir} (bundle {bundle})")
        return True
    _log("fabdem", "no tiles retrieved")
    return False


# ------------------------------------------------------------------ GBA ---
def fetch_gba(city, bounds) -> bool:
    west, south, east, north = bounds
    out_dir = REPO_ROOT / city.raw_geo_dir / "gba"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "buildings.parquet"

    lat5_lo, lat5_hi = math.floor(south / 5) * 5, math.ceil(north / 5) * 5
    lon5_lo, lon5_hi = math.floor(west / 5) * 5, math.ceil(east / 5) * 5

    def lat_tag(v):
        return f"{'n' if v >= 0 else 's'}{abs(int(v)):02d}"

    def lon_tag(v):
        return f"{'e' if v >= 0 else 'w'}{abs(int(v)):03d}"

    urls = []
    for lat0 in range(lat5_lo, lat5_hi, 5):
        for lon0 in range(lon5_lo, lon5_hi, 5):
            name = f"{lon_tag(lon0)}_{lat_tag(lat0 + 5)}_{lon_tag(lon0 + 5)}_{lat_tag(lat0)}.parquet"
            urls.append(f"https://data.source.coop/tge-labs/globalbuildingatlas-lod1/{name}")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")
    frames = []
    for url in urls:
        try:
            t0 = time.time()
            q = f"""
                SELECT id, height, var, region,
                       ST_AsWKB(geometry) AS geometry_wkb
                FROM read_parquet('{url}')
                WHERE bbox.xmin <= {east} AND bbox.xmax >= {west}
                  AND bbox.ymin <= {north} AND bbox.ymax >= {south}
            """
            df = con.execute(q).fetchdf()
            _log("gba", f"{url.rsplit('/', 1)[-1]}: {len(df)} buildings ({time.time() - t0:.1f}s, bbox-pruned read)")
            frames.append(df)
        except Exception as e:
            _log("gba", f"{url.rsplit('/', 1)[-1]} failed: {e}")
    if not frames:
        return False
    all_df = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(
        all_df.drop(columns=["geometry_wkb"]),
        geometry=gpd.GeoSeries.from_wkb(all_df["geometry_wkb"].apply(bytes)),
        crs="EPSG:4326",
    )
    gdf.to_parquet(out_path)
    _log("gba", f"wrote {len(gdf)} building footprints to {out_path}")
    return True


# --------------------------------------------------------------- CANOPY ---
def fetch_canopy(city, bounds) -> bool:
    west, south, east, north = bounds
    out_dir = REPO_ROOT / city.raw_geo_dir / "canopy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename lat/lon is the *top-left* corner of a 10x10-degree cell.
    lat10 = math.ceil(north / 10) * 10
    lon10 = math.floor(west / 10) * 10
    base = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float_epsg4326_v3_10deg"
    ok = True
    for param in ("avg", "cover"):
        url = f"{base}/meta_chm_lat={lat10}.0_lon={lon10}.0_{param}.tif"
        dest = out_dir / f"canopy_{param}.tif"
        try:
            with rasterio.open(f"/vsicurl/{url}") as src:
                win = _clip_window(from_bounds(west, south, east, north, src.transform), src)
                data = src.read(1, window=win)
                profile = src.profile.copy()
                profile.update(
                    height=data.shape[0], width=data.shape[1],
                    transform=src.window_transform(win),
                )
            with rasterio.open(dest, "w", **profile) as dst:
                dst.write(data, 1)
            _log("canopy", f"{param}: clipped window {data.shape} -> {dest}")
        except Exception as e:
            _log("canopy", f"{param} failed: {e}")
            ok = False
    return ok


# ------------------------------------------------------------------- S2 ---
def fetch_s2(city, bounds) -> bool:
    west, south, east, north = bounds
    out_dir = REPO_ROOT / city.raw_geo_dir / "s2"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import planetary_computer
        from pystac_client import Client
    except ImportError as e:
        _log("s2", f"missing dependency: {e}")
        return False

    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=[west, south, east, north],
            datetime="2024-06-01/2024-09-30",
            query={"eo:cloud_cover": {"lt": 20}},
        )
        items = list(search.item_collection())
        if not items:
            _log("s2", "no low-cloud scene found in summer 2024 window")
            return False

        # A single MGRS tile can leave most of a multi-tile window uncovered
        # (Durham's ~22x25km window straddles 4 S2 tiles) - pick the
        # acquisition date whose same-pass tiles together cover the window,
        # then mosaic every tile from that date per band, rather than just
        # the single globally-lowest-cloud item (which may only touch a
        # corner of the window).
        window_poly = box(west, south, east, north)
        by_date: dict[str, list] = {}
        for it in items:
            by_date.setdefault(it.properties["datetime"][:10], []).append(it)

        def _covers_window(group) -> bool:
            union = None
            for it in group:
                g = shape(it.geometry)
                union = g if union is None else union.union(g)
            return union is not None and union.contains(window_poly)

        full_coverage = [(d, g) for d, g in by_date.items() if _covers_window(g)]
        candidates = full_coverage or list(by_date.items())
        candidates.sort(key=lambda dg: (
            -len(dg[1]), max(it.properties.get("eo:cloud_cover", 100) for it in dg[1]),
        ))
        date, group = candidates[0]
        group = sorted(group, key=lambda it: it.properties.get("eo:cloud_cover", 100))
        if not full_coverage:
            _log("s2", f"WARNING: no single date fully covers the window - using best partial match {date}")
        _log("s2", f"using {len(group)} tile(s) from {date}: " + ", ".join(
            f"{it.properties.get('s2:mgrs_tile', '?')}({it.properties.get('eo:cloud_cover', float('nan')):.2f}%)"
            for it in group
        ))

        bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
        for band in bands:
            tmp_paths = []
            for it in group:
                href = it.assets[band].href
                with rasterio.open(href) as src:
                    b_west, b_south, b_east, b_north = transform_bounds(
                        "EPSG:4326", src.crs, west, south, east, north
                    )
                    win = _clip_window(from_bounds(b_west, b_south, b_east, b_north, src.transform), src)
                    if win.width <= 0 or win.height <= 0:
                        continue
                    data = src.read(1, window=win)
                    profile = src.profile.copy()
                    profile.update(height=data.shape[0], width=data.shape[1], transform=src.window_transform(win))
                tmp_path = out_dir / f"_tmp_{band}_{it.properties.get('s2:mgrs_tile', len(tmp_paths))}.tif"
                with rasterio.open(tmp_path, "w", **profile) as dst:
                    dst.write(data, 1)
                tmp_paths.append(tmp_path)

            if not tmp_paths:
                _log("s2", f"{band}: no tile in {date}'s group overlapped the window")
                continue

            dest = out_dir / f"{band}.tif"
            if len(tmp_paths) == 1:
                shutil.copy(tmp_paths[0], dest)
                n_mosaicked = 1
            else:
                open_srcs = [rasterio.open(p) for p in tmp_paths]
                mosaic, mosaic_transform = rio_merge(open_srcs)
                profile = open_srcs[0].profile.copy()
                profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=mosaic_transform)
                with rasterio.open(dest, "w", **profile) as dst:
                    dst.write(mosaic)
                for s in open_srcs:
                    s.close()
                n_mosaicked = len(tmp_paths)
            for p in tmp_paths:
                p.unlink()

            with rasterio.open(dest) as ds:
                _log("s2", f"{band}: mosaicked {n_mosaicked} tile(s) -> {ds.shape} -> {dest}")
        return True
    except Exception as e:
        _log("s2", f"failed: {e}")
        traceback.print_exc()
        return False


SOURCES = {
    "fabdem": fetch_fabdem,
    "gba": fetch_gba,
    "canopy": fetch_canopy,
    "s2": fetch_s2,
}


def main(city_name: str) -> None:
    city = get_city(city_name)
    print(f"=== fetching geo layers for {city.name} ===")
    _, bounds = get_window(city)
    print(f"window (lon/lat): {bounds}")

    results = {}
    for name, fn in SOURCES.items():
        try:
            results[name] = fn(city, bounds)
        except Exception as e:
            _log(name, f"unhandled error: {e}")
            traceback.print_exc()
            results[name] = False

    print("=== summary ===")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED/SKIPPED'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: fetch_geo.py <city_name>")
        sys.exit(1)
    main(sys.argv[1])
