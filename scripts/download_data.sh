#!/usr/bin/env bash
# Downloads the UrbanSENSE dataset (Durham, NC) from UNC Dataverse.
# Dataset DOI: 10.15139/S3/5EHFTA (CC0). The dataset *landing page* blocks
# automated access (BotStopper/Anubis challenge), but the Dataverse REST API
# below is unaffected.
#
# Produces:
#   data/raw/Sensor_Data_Sensor_Info.csv   (raw 10-min sensor readings, ~67MB)
#   data/processed/ERA5_everyday_merged.csv (pre-built hourly feature table, ~26MB)
#   reference/UrbanSENSE_ML/                (authors' code, git history stripped)
set -euo pipefail
cd "$(dirname "$0")/.."

RAW_FILE_ID=7560526     # Sensor_Data_Sensor_Info.tab -> served as .csv
CODE_FILE_ID=7560527    # UrbanSENSE_ML_v1.0.zip

mkdir -p data/raw data/processed reference
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Downloading raw sensor data..."
curl -sL --fail "https://dataverse.unc.edu/api/access/datafile/${RAW_FILE_ID}" \
  -o "$TMP/Sensor_Data_Sensor_Info.csv"
cp "$TMP/Sensor_Data_Sensor_Info.csv" data/raw/Sensor_Data_Sensor_Info.csv

echo "Downloading authors' code + pre-built feature table..."
curl -sL --fail "https://dataverse.unc.edu/api/access/datafile/${CODE_FILE_ID}" \
  -o "$TMP/UrbanSENSE_ML_v1.0.zip"

echo "Extracting (excluding embedded .git history and duplicate data files)..."
unzip -oq "$TMP/UrbanSENSE_ML_v1.0.zip" -x "UrbanSENSE_ML/.git/*" -d "$TMP/extracted"

cp "$TMP/extracted/UrbanSENSE_ML/4_Data/ERA5_everyday_merged.csv" \
   data/processed/ERA5_everyday_merged.csv
cp "$TMP/extracted/UrbanSENSE_ML/4_Data/Raw_Data/README.txt" \
   data/raw/README.txt

rm -rf reference/UrbanSENSE_ML
mkdir -p reference/UrbanSENSE_ML
rsync -a --exclude='4_Data' --exclude='ERA5_everyday_merged.csv' \
  "$TMP/extracted/UrbanSENSE_ML/" reference/UrbanSENSE_ML/

echo "Done."
