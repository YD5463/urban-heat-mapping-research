"""Regenerates report/figures/sensor_map.png as a real street map of Durham,
NC, with development sensors marked as circles and lockbox sensors as
triangles (replacing the earlier plain lat/lon scatter plot).

Requires osmnx + geopandas (project .venv has both); fetches the Durham
city boundary and a local street network from OpenStreetMap, so it needs
network access.

Usage: python scripts/_build_sensor_map.py
"""
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
MARGIN_DEG = 0.025

sensor_info = pd.read_csv(REPO_ROOT / "data/processed/sensor_info.csv")
feature_ids = pd.read_csv(
    REPO_ROOT / "data/processed/ERA5_everyday_merged.csv", usecols=["Sensor_ID"]
)["Sensor_ID"].unique()
sensor_info = sensor_info[sensor_info["Sensor_ID"].isin(feature_ids)].reset_index(drop=True)
assert len(sensor_info) == 39, f"expected the 39-sensor network, got {len(sensor_info)}"

# Reproduces urbansense_common.sensor_level_split (test_size=0.3, random_state=42)
# at the sensor level: same stratified dev/lockbox split used throughout.
sensor_info["strata"] = sensor_info["Landcover_Type"] + "_" + sensor_info["Shade_Type"]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(sss.split(sensor_info["Sensor_ID"], sensor_info["strata"]))
dev = sensor_info.iloc[train_idx]
lockbox = sensor_info.iloc[test_idx]
print(f"{len(dev)} development sensors, {len(lockbox)} lockbox sensors")

west, south = sensor_info["Lon"].min() - MARGIN_DEG, sensor_info["Lat"].min() - MARGIN_DEG
east, north = sensor_info["Lon"].max() + MARGIN_DEG, sensor_info["Lat"].max() + MARGIN_DEG

boundary = ox.geocode_to_gdf("Durham, North Carolina, USA")
ox.settings.timeout = 180
graph = ox.graph_from_bbox((west, south, east, north), network_type="drive")

fig, ax = plt.subplots(figsize=(8, 8))
ox.plot_graph(
    graph, ax=ax, show=False, close=False,
    node_size=0, edge_color="#bbbbbb", edge_linewidth=0.5, bgcolor="white",
)
boundary.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=2)

ax.scatter(
    dev["Lon"], dev["Lat"], marker="o", s=70, c="tab:blue",
    edgecolor="black", linewidth=0.6, zorder=3, label=f"Development ({len(dev)})",
)
ax.scatter(
    lockbox["Lon"], lockbox["Lat"], marker="^", s=90, c="tab:red",
    edgecolor="black", linewidth=0.6, zorder=3, label=f"Lockbox ({len(lockbox)})",
)

ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.set_aspect(1 / math.cos(math.radians(sensor_info["Lat"].mean())))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Durham, NC — 39-sensor UrbanSENSE network")
ax.legend(loc="lower right", frameon=True, fontsize=9)
fig.tight_layout()

out_path = REPO_ROOT / "report/figures/sensor_map.png"
fig.savefig(out_path, dpi=200)
print("saved figure to report/figures/sensor_map.png")
