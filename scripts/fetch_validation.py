"""Downloads *validation-only* layers — never used as model features, only
to sanity-check the feature-source rasters/vectors fetched by fetch_geo.py:

- Durham: USGS 3DEP LiDAR DSM (1m), to validate FABDEM elevation and the
  GBA building heights against a ground-truth LiDAR surface.
- Ramat Gan: OSM buildings carrying a `building:levels` tag (via Overpass),
  to validate GBA building heights (height ~= levels * 3m) where no LiDAR
  exists in Israel (see the project plan's section 2 for why).

Usage: python scripts/fetch_validation.py <durham|ramat_gan>
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import certifi

    _ascii_cacert = Path(tempfile.gettempdir()) / "urbansense_cacert.pem"
    if not _ascii_cacert.exists():
        shutil.copy(certifi.where(), _ascii_cacert)
    for _var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "GDAL_CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE"):
        os.environ[_var] = str(_ascii_cacert)
except Exception:
    pass

import geopandas as gpd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from city_config import get_city  # noqa: E402


def _window_bounds(city) -> tuple[float, float, float, float]:
    gdf = gpd.read_file(REPO_ROOT / city.boundary_path)
    return tuple(gdf.total_bounds)  # west, south, east, north


def fetch_durham_lidar(city) -> bool:
    west, south, east, north = _window_bounds(city)
    out_dir = REPO_ROOT / city.raw_geo_dir / "validation" / "lidar_3dep"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import py3dep

        dem = py3dep.get_map("DEM", (west, south, east, north), resolution=1, crs="EPSG:4326")
        dest = out_dir / "3dep_dsm_1m.tif"
        dem.rio.to_raster(dest)
        print(f"[3dep] wrote {dem.shape} raster to {dest}")
        return True
    except Exception as e:
        print(f"[3dep] failed: {e}")
        return False


def fetch_ramat_gan_osm_levels(city) -> bool:
    west, south, east, north = _window_bounds(city)
    out_dir = REPO_ROOT / city.raw_geo_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "osm_building_levels.geojson"

    query = f"""
    [out:json][timeout:60];
    (
      way["building"]["building:levels"]({south},{west},{north},{east});
      relation["building"]["building:levels"]({south},{west},{north},{east});
    );
    out geom;
    """
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": "urbansense-research/1.0"},
            timeout=90,
        )
        resp.raise_for_status()
        osm = resp.json()
        elements = osm.get("elements", [])

        features = []
        for el in elements:
            if "geometry" not in el:
                continue
            coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
            features.append({
                "type": "Feature",
                "properties": {
                    "osm_id": el["id"],
                    "building_levels": el.get("tags", {}).get("building:levels"),
                    "building": el.get("tags", {}).get("building"),
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            })
        geojson = {"type": "FeatureCollection", "features": features}
        dest.write_text(json.dumps(geojson))
        print(f"[osm_levels] wrote {len(features)} tagged buildings to {dest}")
        return True
    except Exception as e:
        print(f"[osm_levels] failed: {e}")
        return False


def main(city_name: str) -> None:
    city = get_city(city_name)
    print(f"=== fetching validation layers for {city.name} ===")
    if city.name == "durham":
        ok = fetch_durham_lidar(city)
    elif city.name == "ramat_gan":
        ok = fetch_ramat_gan_osm_levels(city)
    else:
        raise ValueError(f"no validation layer defined for {city.name!r}")
    print("=== summary ===")
    print(f"  {'OK' if ok else 'FAILED/SKIPPED'}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("durham", "ramat_gan"):
        print("usage: fetch_validation.py <durham|ramat_gan>")
        sys.exit(1)
    main(sys.argv[1])
