"""Regenerates report/figures/netatmo_check_ramat_gan.png.

One-off documentation figure for a decision already made: a live check
against Netatmo's public /getpublicdata API (see
data/raw/netatmo_ramat_gan_public_check.json for the raw responses) found
only 4 public stations inside Ramat Gan's own municipal area, all on the
western edge, none inside the three coverage gaps identified in
notebooks/11_ramat_gan_data_intake.ipynb (Tel HaShomer, Ramat Shikma, Ramat
Yitzhak). This figure makes that visually obvious for the summary.tex
writeup / for showing Oren -- it does not feed into any modeling pipeline.

Usage: python scripts/_build_netatmo_check_figure.py
"""
import json
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

GIS_PATH = REPO_ROOT / "ramat_gan" / "wetag-gis-2026-11-15.geojson"
NETATMO_CHECK_PATH = REPO_ROOT / "data" / "raw" / "netatmo_ramat_gan_public_check.json"
OUT_PATH = REPO_ROOT / "report" / "figures" / "netatmo_check_ramat_gan.png"

DROP_SERIALS = {12, 16}  # matches Phase A's cleaning decision; 13 was never in the GIS file

# Voronoi-gap centroids identified in notebooks/11_ramat_gan_data_intake.ipynb
# section 3 (largest "service area" cells against the real municipal boundary).
GAP_ZONES = [
    {"name": "Ramat Shikma", "lon": 34.8247, "lat": 32.0491},
    {"name": "Tel HaShomer", "lon": 34.8441, "lat": 32.0461},
    {"name": "Ramat Yitzhak", "lon": 34.8230, "lat": 32.0740},
]

sensors = gpd.read_file(GIS_PATH)
sensors["serial"] = sensors["מספר סידורי"].astype(int)
sensors = sensors[~sensors["serial"].isin(DROP_SERIALS)]
print(f"{len(sensors)} WeTag sensors plotted (dropped {sorted(DROP_SERIALS)})")

with open(NETATMO_CHECK_PATH, encoding="utf-8") as f:
    netatmo_check = json.load(f)
tight_query = next(q for q in netatmo_check["queries"] if q["label"] == "ramat_gan_municipal_bbox")
netatmo_stations = pd.DataFrame(
    [
        {
            "lon": s["place"]["location"][0],
            "lat": s["place"]["location"][1],
            "city": s["place"]["city"],
        }
        for s in tight_query["response"]["body"]
    ]
)
print(f"{len(netatmo_stations)} Netatmo public stations in Ramat Gan's own bbox")
netatmo_gdf = gpd.GeoDataFrame(
    netatmo_stations,
    geometry=gpd.points_from_xy(netatmo_stations["lon"], netatmo_stations["lat"]),
    crs=4326,
)

# Real municipal boundary, not the +2km buffered study-area extent in
# data/raw/geo/ramat_gan/boundary.geojson (that one is for raster downloads,
# see scripts/fetch_geo.py's BUFFER_M -- using it here would make the gaps
# look like they're well inside Ramat Gan when the buffer actually extends
# past the true city border).
boundary = ox.geocode_to_gdf("Ramat Gan, Israel")

sensors_3857 = sensors.to_crs(3857)
netatmo_3857 = netatmo_gdf.to_crs(3857)
boundary_3857 = boundary.to_crs(3857)
gap_3857 = gpd.GeoDataFrame(
    pd.DataFrame(GAP_ZONES),
    geometry=gpd.points_from_xy([g["lon"] for g in GAP_ZONES], [g["lat"] for g in GAP_ZONES]),
    crs=4326,
).to_crs(3857)

fig, ax = plt.subplots(figsize=(9, 9))
boundary_3857.boundary.plot(ax=ax, color="yellow", linewidth=1.5, zorder=2)

sensors_3857.plot(ax=ax, color="deepskyblue", markersize=25, edgecolor="white",
                   linewidth=0.5, zorder=3, label=f"WeTag sensors ({len(sensors_3857)})")

netatmo_3857.plot(ax=ax, color="red", markersize=140, marker="*", edgecolor="white",
                   linewidth=0.8, zorder=4, label=f"Netatmo public stations ({len(netatmo_3857)})")
for _, row in netatmo_3857.iterrows():
    ax.annotate(row["city"], (row.geometry.x, row.geometry.y), fontsize=8, color="white",
                weight="bold", xytext=(6, 6), textcoords="offset points", zorder=5)

gap_3857.plot(ax=ax, color="none", markersize=400, marker="o", edgecolor="orange",
              linewidth=2, zorder=3, label="known coverage gaps")
for _, row in gap_3857.iterrows():
    ax.annotate(row["name"], (row.geometry.x, row.geometry.y), fontsize=9, color="orange",
                weight="bold", xytext=(10, -14), textcoords="offset points", zorder=5)

ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=sensors_3857.crs.to_string())
ax.set_title("Netatmo public-station coverage check vs. Ramat Gan's own sensor gaps")
ax.set_axis_off()
ax.legend(loc="lower left", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"wrote {OUT_PATH}")
