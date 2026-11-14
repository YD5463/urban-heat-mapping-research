"""Dynamic shadow features (table rows 12-13 of the geo-features plan):
whether a sensor's pixel is shadowed at time t by terrain/buildings/trees,
and the preceding-hours shaded fraction (thermal-inertia proxy).

Shadow is determined by ray-marching from the sensor toward the sun's
azimuth and checking whether any DSM pixel along that ray rises above the
sensor-to-sun line of sight — the standard horizon-shadow test, applied to
a synthetic DSM (FABDEM + building heights + canopy, see
``morphology_features.build_synthetic_dsm``) rather than true LiDAR, since
no LiDAR DSM is available for both cities (see plan section 2).

Solar position reuses the authors' own approach (see
``urbansense_common.load_data``): a single city-representative lat/lon via
``pvlib.solarposition.get_solarposition``, rather than a per-sensor solar
position — sensors within one city are close enough together that this is
an acceptable approximation, and it keeps the two pipelines consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib
import rasterio
from rasterio.transform import rowcol

from city_config import CityConfig

try:
    from morphology_features import build_synthetic_dsm  # noqa: F401
except ModuleNotFoundError:
    pass  # not yet written when this module is developed in parallel; only needed by callers of build_synthetic_dsm, not by this file itself


def _solar_position(city: CityConfig, times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    return pvlib.solarposition.get_solarposition(
        time=times_utc, latitude=city.lat, longitude=city.lon
    )


def _is_shaded_at(
    dsm: np.ndarray,
    transform,
    sensor_row: int,
    sensor_col: int,
    sensor_height: float,
    azimuth_deg: float,
    elevation_deg: float,
    resolution_m: float,
    max_dist_m: float,
) -> bool:
    """Horizon-shadow test: step outward from the sensor toward the sun's
    azimuth; shaded if any DSM pixel on that ray is higher than the
    sensor-to-sun sightline at that distance."""
    if elevation_deg <= 0:
        return True  # sun below horizon: no direct beam to be shaded from

    az_rad = np.radians(azimuth_deg)
    # unit step toward the sun in world (x, y); world y = -row*res, x = col*res
    dx = np.sin(az_rad) * resolution_m
    dy = np.cos(az_rad) * resolution_m
    sun_x, sun_y = rasterio.transform.xy(transform, sensor_row, sensor_col)

    tan_elev = np.tan(np.radians(elevation_deg))
    n_steps = int(max_dist_m / resolution_m)
    rows, cols = dsm.shape
    for step in range(1, n_steps + 1):
        x = sun_x + dx * step
        y = sun_y + dy * step
        r, c = rowcol(transform, x, y)
        if not (0 <= r < rows and 0 <= c < cols):
            break
        dist = step * resolution_m
        los_height = sensor_height + dist * tan_elev
        if dsm[r, c] > los_height:
            return True
    return False


def compute_shadow_features_dynamic(
    city: CityConfig,
    sensors,
    times_utc: pd.DatetimeIndex,
    dsm: np.ndarray,
    transform,
    resolution_m: float = 1.0,
    max_dist_m: float = 200.0,
) -> pd.DataFrame:
    """sensors: GeoDataFrame indexed by Sensor_ID, geometry = point in city.crs.
    times_utc: tz-aware (UTC) DatetimeIndex of hours to evaluate.
    dsm, transform: synthetic DSM array (metres) and its rasterio Affine
    transform, as returned by ``morphology_features.build_synthetic_dsm``.

    Returns a DataFrame indexed by (Sensor_ID, time) with columns:
    shaded, shade_frac_3h, shade_hours_cum.
    """
    solpos = _solar_position(city, times_utc)

    records = []
    for sensor_id, geom in zip(sensors.index, sensors.geometry):
        row, col = rowcol(transform, geom.x, geom.y)
        sensor_height = float(dsm[row, col])
        for t, azimuth, elevation in zip(times_utc, solpos["azimuth"], solpos["apparent_elevation"]):
            shaded = _is_shaded_at(
                dsm, transform, row, col, sensor_height,
                azimuth, elevation, resolution_m, max_dist_m,
            )
            records.append((sensor_id, t, bool(shaded), float(elevation)))

    out = pd.DataFrame(records, columns=["Sensor_ID", "time", "shaded", "_elevation"])
    out = out.sort_values(["Sensor_ID", "time"])

    out["shade_frac_3h"] = (
        out.groupby("Sensor_ID")["shaded"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )

    local_date = out["time"].dt.tz_convert(city.tz).dt.date
    daylight_shaded = out["shaded"].astype(float) * (out["_elevation"] > 0)
    out["shade_hours_cum"] = (
        daylight_shaded.groupby([out["Sensor_ID"], local_date]).cumsum()
    )

    out = out.drop(columns="_elevation").set_index(["Sensor_ID", "time"])
    return out


if __name__ == "__main__":
    import geopandas as gpd
    from shapely.geometry import Point
    from city_config import DURHAM

    res = 1.0
    size = 61
    dsm = np.zeros((size, size), dtype=float)
    building_row, building_col = 30, 30
    dsm[building_row - 1:building_row + 2, building_col - 1:building_col + 2] = 20.0  # 20m "building"
    transform = rasterio.transform.from_origin(0, 0, res, res)

    # Sensor A: 10 px west of the building -> shadowed by morning sun (rising in the east).
    sensor_a_row, sensor_a_col = building_row, building_col - 10
    xa, ya = rasterio.transform.xy(transform, sensor_a_row, sensor_a_col)
    # Sensor B: 10 px north of the building, unaffected by an easterly sun.
    sensor_b_row, sensor_b_col = building_row - 10, building_col
    xb, yb = rasterio.transform.xy(transform, sensor_b_row, sensor_b_col)

    sensors = gpd.GeoDataFrame(
        {"Sensor_ID": ["A", "B"]},
        geometry=[Point(xa, ya), Point(xb, yb)],
        crs=DURHAM.crs,
    ).set_index("Sensor_ID")

    times = pd.date_range("2024-07-15 09:00", periods=5, freq="h", tz="US/Eastern").tz_convert("UTC")

    result = compute_shadow_features_dynamic(DURHAM, sensors, times, dsm, transform, resolution_m=res)

    solpos = _solar_position(DURHAM, times)
    print(solpos[["azimuth", "apparent_elevation"]])
    print(result)

    morning_shaded_a = result.loc[("A", times[0]), "shaded"]
    morning_shaded_b = result.loc[("B", times[0]), "shaded"]
    assert morning_shaded_a == True, "sensor A (west of the building) should be shaded by the morning sun"
    assert morning_shaded_b == False, "sensor B (north of the building) should not be shaded by an easterly sun"
    assert result["shade_frac_3h"].between(0, 1).all()
    assert (result["shade_hours_cum"] >= 0).all()
    print("self-test passed")
