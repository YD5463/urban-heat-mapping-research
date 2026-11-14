"""F1 — dynamics and thermo-history features, computed per sensor from its
own ERA5 time series. These are "free" (column [0]: nothing beyond the
existing hourly table is needed) and purely causal (each row only looks
backward in time), so there's no train/test leakage risk to guard against
here the way there is for network features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(data: pd.DataFrame, id_col: str = "Sensor_ID",
                           time_col: str = "edt_datetime") -> pd.DataFrame:
    data = data.sort_values([id_col, time_col]).copy()
    g = data.groupby(id_col)

    t2m = g["temperature_2m"]
    data["dT2m_1h"] = t2m.diff(1)
    data["dT2m_3h"] = t2m.diff(3)

    rad = g["surface_solar_radiation_downwards_hourly"]
    data["rad_cum_24h"] = rad.transform(lambda s: s.rolling(24, min_periods=1).sum())

    # Previous day's peak ERA5 temperature: urban materials store daytime
    # heat and release it overnight, so yesterday's Tmax — not today's
    # instantaneous ERA5 value — is what should predict tonight's UHI.
    daily_date = data[time_col].dt.date
    daily_max = data.groupby([id_col, daily_date])["temperature_2m"].transform("max")
    data["tmax_today"] = daily_max
    data["tmax_prev_day"] = data.groupby(id_col)["tmax_today"].shift(24)

    hour = data[time_col].dt.hour + data[time_col].dt.minute / 60.0
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    is_day = (data["surface_solar_radiation_downwards_hourly"] > 1).astype(int)
    transition = is_day.groupby(data[id_col]).diff().fillna(0) != 0
    data["_transition_idx"] = np.where(transition, data.groupby(id_col).cumcount(), np.nan)
    data["_transition_idx"] = data.groupby(id_col)["_transition_idx"].ffill()
    row_idx = data.groupby(id_col).cumcount()
    data["hours_since_transition"] = (row_idx - data["_transition_idx"]).fillna(0)
    data.drop(columns=["_transition_idx"], inplace=True)

    precip = g["total_precipitation_hourly"]
    data["precip_cum_24h"] = precip.transform(lambda s: s.rolling(24, min_periods=1).sum())
    data["precip_cum_72h"] = precip.transform(lambda s: s.rolling(72, min_periods=1).sum())

    return data
