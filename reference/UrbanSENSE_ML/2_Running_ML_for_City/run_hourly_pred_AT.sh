#!/bin/bash
#SBATCH --job-name=hourly_pred_AT
#SBATCH --output=logs/hourly_pred_AT_%A_%a.out
#SBATCH --error=logs/hourly_pred_AT_%A_%a.err
#SBATCH --array=0-1487             # 62 days * 24 hours =1488 (0 to 1487)
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sberk@unc.edu  

module purge
# load environment
source ~/.bashrc
conda activate durham_ml_venv

# Paths 
DAY_MODEL_PATH="xgb_model_daytime.sav"
DAY_SCALER_PATH="day_scaler.sav"   
NIGHT_MODEL_PATH="xgb_model_nighttime.sav"
NIGHT_SCALER_PATH="night_scaler.sav" 
MET_CSV="Data/hourly_meteorology.csv"     # make sure there is a 'time' column in here (name must be 'time')
SVF_MOSAIC="Data/Aligned_Rasters/svf_aligned.tif"
ALBEDO="Data/Aligned_Rasters/albedo_aligned.tif"
LANDCOVER="Data/Aligned_Rasters/landcover_aligned.tif"
NDBI="Data/Aligned_Rasters/ndbi_aligned.tif"
NDVI="Data/Aligned_Rasters/ndvi_aligned.tif"
NDMI="Data/Aligned_Rasters/ndmi_aligned.tif"
ROAD_DIST="Data/Aligned_Rasters/road_dist_aligned.tif"
OUT_DIR="/work/users/s/b/sberk/Predictions_AT" #write to scratch to avoid running out of disk space

mkdir -p logs "$OUT_DIR"

python predict_hourly_AT.py \
  --start "2024-07-01T00:00" \
  --end   "2024-08-31T23:00" \
  --array_index "$SLURM_ARRAY_TASK_ID" \
  --day_model "$DAY_MODEL_PATH" \
  --day_scaler "$DAY_SCALER_PATH" \
  --night_model "$NIGHT_MODEL_PATH" \
  --night_scaler "$NIGHT_SCALER_PATH" \
  --met "$MET_CSV" \
  --svf "$SVF_MOSAIC" \
  --albedo "$ALBEDO" \
  --landcover "$LANDCOVER" \
  --ndbi "$NDBI" \
  --ndvi "$NDVI" \
  --road_dist "$ROAD_DIST" \
  --ndmi "$NDMI" \
  --outdir "$OUT_DIR"
