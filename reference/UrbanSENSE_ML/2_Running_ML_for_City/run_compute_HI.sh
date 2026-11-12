#!/bin/bash
#SBATCH --job-name=compute_HI
#SBATCH --output=logs/compute_HI_%A_%a.out
#SBATCH --error=logs/compute_HI_%A_%a.err
#SBATCH --array=0-1487             # 62 days * 24 hours =1488 (0 to 1487)
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sberk@unc.edu

module purge
source ~/.bashrc
conda activate durham_ml_venv

AT_DIR="/work/users/s/b/sberk/Predictions_AT"
RH_DIR="/work/users/s/b/sberk/Predictions_RH"
OUT_DIR_F="/work/users/s/b/sberk/Predictions_HI_F"
OUT_DIR_C="/work/users/s/b/sberk/Predictions_HI_C"

mkdir -p logs "$OUT_DIR_F" "$OUT_DIR_C"

python compute_HI.py "$AT_DIR" "$RH_DIR" "$OUT_DIR_F" "$OUT_DIR_C" "$SLURM_ARRAY_TASK_ID"
