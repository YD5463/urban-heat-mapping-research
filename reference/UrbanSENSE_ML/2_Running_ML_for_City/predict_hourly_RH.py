# -*- coding: utf-8 -*-
"""
predict_hourly_RH.py
Script to create relative humidity predictions for Durham using the XGB ML model 
July to August Hourly Predictions 
Author: Sarah Berk sberk@unc.edu
"""
#!/usr/bin/env python3
"""
predict_hourly_RH.py
Inputs:
 - trained model (.sav)and scaler (.joblib)
 - static rasters (svf, albedo, landcover, ndbi, ndvi, ndmi, road_dist)
 - hourly meteorology CSV with a 'time' column
Maps array_index -> hour in the start to end period, predicts full raster for each hour,
and writes a single GeoTIFF per hour: e.g. pred_20240701_00.tif
"""

import argparse
import os
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
import joblib
#from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--array_index", type=int, required=True)
    p.add_argument("--day_model", required=True)
    p.add_argument("--day_scaler", required=True)
    p.add_argument("--night_model", required=True)
    p.add_argument("--night_scaler", required=True)
    p.add_argument("--met", required=True)
    p.add_argument("--svf", required=True)
    p.add_argument("--albedo", required=True)
    p.add_argument("--landcover", required=True)
    p.add_argument("--ndbi", required=True)
    p.add_argument("--ndvi", required=True)
    p.add_argument("--road_dist", required=True)
    p.add_argument("--ndmi", required=True)
    p.add_argument("--outdir", default="Predictions_RH")
    return p.parse_args()

def timestamp_for_index(start, end, idx):
    hours = pd.date_range(start, end, freq="h")
    if idx < 0 or idx >= len(hours):
        raise IndexError("Array index out of range")
    return hours[idx]

#def load_static_raster_as_array(path, template=None):
 #   """
 #   Load raster as numpy array, optionally reproject/resample to template's grid.
 #   template: xarray DataArray or rasterio dataset path (we accept rasterio env).
 #   Returns (arr, profile)
 #   """
 #   with rasterio.open(path) as src:
 #       if template is None:
 #           arr = src.read(1, masked=True).astype(np.float32)
 #          prof = src.profile
 #           return arr, prof
 #       else:
 #           # use rioxarray reproject_match for robust handling of geotransforms
 #           da = rxr.open_rasterio(path, masked=True)
 #           da_match = da.rio.reproject_match(template)
 #           arr = da_match.values[0].astype(np.float32)
 #           prof = rasterio.open(path).profile  # base profile (we'll adapt from template for writing)
 #           return arr, prof

def main():
    args = parse_args()
    ts = timestamp_for_index(args.start, args.end, args.array_index)
    
    os.makedirs(args.outdir, exist_ok=True)


    # Load meteorology CSV and fetch the row for this timestamp
    met = pd.read_csv(args.met, parse_dates=["time"])
    row = met.loc[met["time"] == ts]
    if row.empty:
        raise ValueError(f"No meteorology row for {ts}")
    row = row.iloc[0]

    # Load static rasters (template: SVF)
    svf_da = rxr.open_rasterio(args.svf, masked=True)
    svf = svf_da.values[0].astype(np.float32)
    #profile = svf_da.rio.write_crs(svf_da.rio.crs).rio.to_rasterio().profile
    # esure CRS is set
    svf_da = svf_da.rio.write_crs(svf_da.rio.crs)
    # get profile 
    with rasterio.open(args.svf) as src:
        profile = src.profile.copy()
    
    # Load all other static rasters, reproject to SVF grid
    def load_match(path): return rxr.open_rasterio(path, masked=True).rio.reproject_match(svf_da).values[0].astype(np.float32)    
    
    ndbi = load_match(args.ndbi)
    ndvi = load_match(args.ndvi)
    landcover = load_match(args.landcover)
    road_dist = load_match(args.road_dist)
    ndmi = load_match(args.ndmi)
    albedo = load_match(args.albedo)

    # Load day/night models
    day_model = joblib.load(args.day_model)
    day_scaler = joblib.load(args.day_scaler)
    night_model = joblib.load(args.night_model)
    night_scaler = joblib.load(args.night_scaler)

    # Choose model based on solar radiation
    if row["surface_solar_radiation_downwards_hourly"] >= 1:
        model, scaler = day_model, day_scaler
        static_order = ["ndbi","landcover", "ndmi","svf","albedo"]
        dynamic_order = ['zenith_angle',
                     'surface_net_thermal_radiation_hourly',
                     'surface_solar_radiation_downwards_hourly',
                     'surface_thermal_radiation_downwards_hourly',
                     'surface_solar_radiation_downwards_accumulation',
                     'u_component_of_wind_10m',
                     'v_component_of_wind_10m',
                     'windspeed_10m',
                     'relative_humidity_2m',
                     'temperature_2m',
                     'total_precipitation_hourly']
    else:
        model, scaler = night_model, night_scaler
        static_order = ["ndbi","ndvi","landcover", "road_dist","svf"]
        dynamic_order = ['zenith_angle',
                     'surface_net_thermal_radiation_hourly',
                     'surface_thermal_radiation_downwards_hourly',
                     'surface_solar_radiation_downwards_accumulation',
                     'u_component_of_wind_10m',
                     'v_component_of_wind_10m',
                     'windspeed_10m',
                     'relative_humidity_2m',
                     'temperature_2m',
                     'total_precipitation_hourly']

    # Create static feature stack
    static_map = dict(ndbi=ndbi, ndvi=ndvi, landcover=landcover, road_dist=road_dist,
                  svf=svf, albedo=albedo, ndmi=ndmi)
    static_stack = np.stack([static_map[k] for k in static_order], axis=-1)
    valid_mask = np.all(np.isfinite(static_stack), axis=-1)
    H, W, F_static = static_stack.shape

    # Create dynamic feature stack
    dyn_vec = np.array([float(row[name]) for name in dynamic_order], dtype=np.float32)
    F_dyn = dyn_vec.shape[0]
    dyn_stack = np.broadcast_to(dyn_vec.reshape(1,1,F_dyn), (H,W,F_dyn))

    # Combine static + dynamic features for prediction
    X = np.concatenate([static_stack, dyn_stack], axis=-1).reshape(-1, F_static + F_dyn)
    flat_mask = valid_mask.reshape(-1)
    preds = np.full(flat_mask.shape[0], np.nan, dtype=np.float32)

    # Apply scaler and predict
    if flat_mask.any():
        X_valid = X[flat_mask]
        if scaler is not None:
            X_valid = scaler.transform(X_valid)
        preds[flat_mask] = model.predict(X_valid).astype(np.float32)

    pred_raster_anomaly = preds.reshape(H, W)
    city_mean = float(row["relative_humidity_2m"])
    pred_raster_absolute = pred_raster_anomaly + city_mean
    
    # Clip predictions to 0-100% relative humidity
    pred_raster_absolute = np.clip(pred_raster_absolute, 0, 100)
   
    # Output profile
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, compress="lzw", BIGTIFF="IF_SAFER")

    ts_str = ts.strftime('%Y%m%d_%H')

    # Write rasters
    #out_fp_anom = os.path.join(args.outdir, f"pred_anom_{ts_str}.tif")
    #with rasterio.open(out_fp_anom, "w", **out_profile) as dst:
    #    dst.write(pred_raster_anomaly, 1)
    #print(f"Wrote {out_fp_anom}")

    out_fp_abs = os.path.join(args.outdir, f"pred_abs_{ts_str}.tif")
    with rasterio.open(out_fp_abs, "w", **out_profile) as dst:
        dst.write(pred_raster_absolute, 1)
    print(f"Wrote {out_fp_abs}")


if __name__ == "__main__":
    main()