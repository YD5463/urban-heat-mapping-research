"""Cleans the raw Ramat Gan WeTag sensor export into modeling-ready tables.

Drops sensors 12 (dies after 2.5 weeks), 13 (no GIS location), and 16
(1.5m from sensor 15 -- same pole), then resamples the remaining sensors'
irregular 15/30-min readings onto a uniform hourly grid per sensor, leaving
real gaps as NaN rather than interpolating across them.

Usage: python scripts/build_ramat_gan_data.py
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

GIS_PATH = REPO_ROOT / "ramat_gan" / "wetag-gis-2026-11-15.geojson"
TEMP_PATH = (
    REPO_ROOT / "ramat_gan"
    / "חיישן טמפרטורה ולחות - sensor data (raw) - 2026-08-17 to 2026-11-15.csv"
)
SENSOR_INFO_OUT = REPO_ROOT / "data" / "processed" / "sensor_info_ramat_gan.csv"
HOURLY_OUT = REPO_ROOT / "data" / "processed" / "ramat_gan_hourly.parquet"

DROP_SERIALS = {12, 13, 16}
TZ = "Asia/Jerusalem"


def load_raw_gis(gis_path=GIS_PATH) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gis_path)
    gdf["serial"] = gdf["מספר סידורי"].astype(int)
    return gdf


def load_raw_temps(csv_path=TEMP_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"מספר סידורי": "serial", "temperature (°C)": "temp_c"})
    df["ts_utc"] = pd.to_datetime(df["timestamp_local"], utc=True)
    return df[["serial", "ts_utc", "temp_c"]]


def build_sensor_info(gis: gpd.GeoDataFrame, drop_serials=DROP_SERIALS) -> pd.DataFrame:
    kept = gis[~gis["serial"].isin(drop_serials)].copy()
    kept["Lat"] = kept.geometry.y
    kept["Lon"] = kept.geometry.x
    info = kept.rename(
        columns={
            "serial": "Sensor_ID",
            "מספר עמוד": "pole_id",
            "מקום התקנה": "mount_type",
            "הערות": "notes",
        }
    )[["Sensor_ID", "Lat", "Lon", "pole_id", "mount_type", "notes"]]
    return info.sort_values("Sensor_ID").reset_index(drop=True)


def build_hourly_readings(temps: pd.DataFrame, drop_serials=DROP_SERIALS, freq="1h") -> pd.DataFrame:
    kept = temps[~temps["serial"].isin(drop_serials)].copy()
    kept["local_datetime"] = kept["ts_utc"].dt.tz_convert(TZ)

    frames = []
    for serial, sub in kept.groupby("serial"):
        hourly = (
            sub.set_index("local_datetime")["temp_c"]
            .resample(freq)
            .mean()
        )
        hourly = hourly.reindex(
            pd.date_range(hourly.index.min(), hourly.index.max(), freq=freq, tz=TZ)
        )
        frames.append(pd.DataFrame({
            "Sensor_ID": serial,
            "local_datetime": hourly.index,
            "Temperature": hourly.values,
        }))

    return pd.concat(frames, ignore_index=True).sort_values(["Sensor_ID", "local_datetime"])


def main():
    gis = load_raw_gis()
    temps = load_raw_temps()

    sensor_info = build_sensor_info(gis)
    hourly = build_hourly_readings(temps)

    SENSOR_INFO_OUT.parent.mkdir(parents=True, exist_ok=True)
    sensor_info.to_csv(SENSOR_INFO_OUT, index=False)
    hourly.to_parquet(HOURLY_OUT, index=False)

    n_valid = hourly["Temperature"].notna().sum()
    print(f"{len(sensor_info)} sensors kept (dropped {sorted(DROP_SERIALS)})")
    print(f"wrote {SENSOR_INFO_OUT}")
    print(f"wrote {HOURLY_OUT}: {len(hourly)} rows, {n_valid} with a reading "
          f"({n_valid / len(hourly):.1%})")


if __name__ == "__main__":
    main()
