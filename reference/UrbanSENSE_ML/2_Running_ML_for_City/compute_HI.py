import math
import rioxarray as rxr
import xarray as xr
import numpy as np
import os
import sys

def compute_heat_index(T, R):
    HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094) # steadman
    mask = HI >= 80 #use roth for heat index over 80
    HI_roth = (-42.379 + 2.04901523 * T + 10.14333127 * R
               - 0.22475541 * T * R - 6.83783e-3 * T**2
               - 5.481717e-2 * R**2 + 1.22874e-3 * T**2 * R
               + 8.5282e-4 * T * R**2 - 1.99e-6 * T**2 * R**2)
    HI = xr.where(mask, HI_roth, HI)
    # adjustments, in line with NOAA
    adj1_mask = (R < 13) & (T >= 80) & (T <= 112)
    adj1 = ((13 - R) / 4.0) * np.sqrt((17 - np.abs(T - 95.0)) / 17.0)
    HI = xr.where(adj1_mask, HI - adj1, HI)

    adj2_mask = (R > 85) & (T >= 80) & (T <= 87)
    adj2 = ((R - 85) / 10.0) * ((87 - T) / 5.0)
    HI = xr.where(adj2_mask, HI + adj2, HI)
    return HI

def F_to_C(F):
    return (F - 32.0) * 5.0 / 9.0
    
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python compute_HI.py <AT_DIR> <RH_DIR> <OUT_DIR_F> <OUT_DIR_C> <ARRAY_INDEX>")
        sys.exit(1)

    AT_DIR = sys.argv[1]
    RH_DIR = sys.argv[2]
    OUT_DIR_F = sys.argv[3]
    OUT_DIR_C = sys.argv[4]
    array_index = int(sys.argv[5])

    os.makedirs(OUT_DIR_F, exist_ok=True)
    os.makedirs(OUT_DIR_C, exist_ok=True)

    at_files = sorted([f for f in os.listdir(AT_DIR) if f.startswith("pred_abs_") and f.endswith(".tif")])

    if array_index < 0 or array_index >= len(at_files):
        raise IndexError(f"ARRAY_INDEX {array_index} out of range (0–{len(at_files)-1})")

    FNAME = at_files[array_index]
    AT_FILE = os.path.join(AT_DIR, FNAME)
    RH_FILE = os.path.join(RH_DIR, FNAME)
    OUT_FILE_F = os.path.join(OUT_DIR_F, FNAME)
    OUT_FILE_C = os.path.join(OUT_DIR_C, FNAME)

    if not os.path.exists(RH_FILE):
        print(f"RH file {RH_FILE} missing. Skipping.")
        sys.exit(0)

    T_da = rxr.open_rasterio(AT_FILE, masked=True).squeeze(drop=True)
    R_da = rxr.open_rasterio(RH_FILE, masked=True).squeeze(drop=True)

    # convert to degF for HI computation
    T_F = T_da * 9.0 / 5.0 + 32.0
    # compute the HI then save the degF files  
    HI_da_F = compute_heat_index(T_F, R_da)
    HI_da_F.name = "Heat_Index_F"
    HI_da_F.attrs["units"] = "degF"
    HI_da_F.rio.write_crs(T_da.rio.crs, inplace=True)
    HI_da_F.rio.to_raster(OUT_FILE_F)
    print(f"Saved {OUT_FILE_F}")

    # convert HI to degC
    HI_da_C = F_to_C(HI_da_F)
    HI_da_C.name = "Heat_Index_C"
    HI_da_C.attrs["units"] = "degC"
    HI_da_C.rio.write_crs(T_da.rio.crs, inplace=True)
    HI_da_C.rio.to_raster(OUT_FILE_C)
    print(f"Saved {OUT_FILE_C}")

