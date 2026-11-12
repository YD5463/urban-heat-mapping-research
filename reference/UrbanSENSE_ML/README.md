# UrbanSENSE_ML

Code accompanying the manuscript:

**UrbanSENSE: A Stratified Sampling Approach for Balanced Hourly Machine Learning Models of Moist Heat Stress**

This repository contains scripts used to generate predictor datasets, train XGBoost models, create hourly citywide predictions of air temperature (Ta), relative humidity (RH), and heat index (HI), and reproduce analyses and figures presented in the manuscript. 

---

## Workflow Overview

The workflow consists of three folders:

1. Creating_Input_Data: Create predictor datasets
2. Running_ML_for_City: Train ML models and generate citywide predictions
3. Analysis_Plots: Analysis and create figures

---

## 1. Creating_Input_Data

### Google Earth Engine

Most spatial predictor variables are generated using:

* `Extract_inputs_GEE.js`

This script exports satellite-derived predictor datasets used in the machine learning framework.

### Sky View Factor (SVF)

1. Run:

   * `calculate_treecanopy_SVFs.ipynb`

   to generate SVF tiles from tree canopy data. **Note:** script is configured for execution on a system with a GPU.

2. Run:

   * `Mosaic_SVFs.ipynb`

   to mosaic individual tiles into a single raster.

### Distance to Roads

Run:

* `Generate_distance_to_roads.ipynb`

This creates distance-to-road rasters from OpenStreetMap road data using the NDVI raster as the reference grid.

### ERA5-Land and Meteorological Data

Run:

* `Met_input_data_processing.ipynb`

This script:

* Processes ERA5-Land data from GEE
* Calculates relative humidity from dew point temperature
* Formats meteorological inputs for input to ML model 

### Raster Alignment

Run:

* `Align_rasters_to_ndvi.ipynb`

All predictor layers are resampled and aligned to a common grid prior to model training.

---

## 2. Running_ML_for_City

### Model Training

Run:

* `Model_Fit_XGB.ipynb`

This script:

* Trains the XGBoost models
* Creates input scalers
* Uses data from `ERA5_everyday_merged.csv`

### Generating Hourly Predictions

**Note:** These scripts are configured for execution on a high-performance computing (HPC) system using parallel processing.

#### Air Temperature (Ta)

Run:

* `run_hourly_pred_AT.sh`

which calls:

* `predict_hourly_AT.py`

#### Relative Humidity (RH)

Run:

* `run_hourly_pred_RH.sh`

which calls:

* `predict_hourly_RH.py`

Outputs are hourly GeoTIFF predictions for the study period.

### Heat Index (HI)

Run:

* `run_compute_HI.sh`

which calls:

* `compute_HI.py`

This computes hourly HI rasters from the predicted Ta and RH rasters.

---

## 3. Analysis Plots

### Exploratory Analysis

* `Exploratory_analysis_plots.ipynb`

Exploration and visualization of the UrbanSENSE sensor network observations.

### Model Performance

* `AT_ML_performance.ipynb`
* `RH_ML_performance.ipynb`

Evaluation and visualization of model performance metrics.

### Spatial Outputs

* `Output_maps.ipynb`

Visualization of citywide Ta, RH, and HI predictions.

### Sampling Bias Analysis

* `Landcover_bias_analysis_plots.ipynb`

Analysis of spatial and temporal prediction biases associated with non-representative sampling.

* `AT_bias_landcover_hourly.ipynb`
* `RH_bias_landcover_hourly.ipynb`

These notebooks:

* Train models using biased land-cover or shading subsets
* Generate hourly performance statistics
* Export intermediate CSV files
* Produce the hourly bias figures used in the manuscript

Each notebook must be run separately for each land-cover and shading scenario.

---

## Outputs

Primary outputs include:

* Hourly Ta GeoTIFFs
* Hourly RH GeoTIFFs
* Hourly HI GeoTIFFs
* Model performance metrics
* Sampling bias analyses
* Figures used in the manuscript

---

## Contact

For questions regarding the code or manuscript, please contact:

Sarah Berk sberk@unc.edu
University of North Carolina at Chapel Hill
