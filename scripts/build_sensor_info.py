"""Builds data/processed/sensor_info.csv from the raw 10-min sensor table.

load_data() in src/urbansense_common.py needs one static metadata row per
Sensor_ID (Lat, Lon, Landcover_Type, Shade_Type, SVF, Percent_Vegetation,
Decile_Group), but no script in this repo or the authors' reference code
produces that file — it has to be derived by taking the first raw reading
per sensor, since these fields are static per sensor in the raw table.
"""
import pandas as pd

RAW_PATH = "data/raw/Sensor_Data_Sensor_Info.csv"
OUT_PATH = "data/processed/sensor_info.csv"

COLUMNS = [
    "Sensor_ID", "Lat", "Lon", "Landcover_Type", "Shade_Type",
    "SVF", "Percent_Vegetation", "Decile_Group",
]


def main():
    raw = pd.read_csv(RAW_PATH, sep="\t")
    raw["Landcover_Type"] = raw["Landcover_Type"].str.strip('"')
    raw["Shade_Type"] = raw["Shade_Type"].str.strip('"')
    sensor_info = raw.groupby("Sensor_ID", as_index=False)[COLUMNS[1:]].first()
    sensor_info = sensor_info[COLUMNS]
    sensor_info.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(sensor_info)} sensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
