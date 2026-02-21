#!/bin/bash
#SBATCH --job-name=proj_svgd_search
#SBATCH --output=active_learning/logs/proj_svgd_search_%A_%a.out
#SBATCH --error=active_learning/logs/proj_svgd_search_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-2  # Default: 3 workers (override via launcher script)

# ==============================================================================
# Projected SVGD Hyperparameter Search - Distributed Optuna Worker
#
# Each array task is an independent Optuna worker. All workers share the same
# SQLite study database. Optuna coordinates trial distribution automatically.
#
# Environment variables (set by run_projected_svgd_search.sh or --export):
#   TRIALS_PER_WORKER  - Number of trials this worker should run (default: 5)
#   BUDGET             - Queries per active learning run (default: 100)
#   STUDY_NAME         - Optuna study name (default: projected_svgd_search)
#   STORAGE_DB         - Path to shared Optuna database (default: active_learning/optuna_studies.db)
#   MODEL_PATH         - Path to decoder model (default: models/best_model.pt)
#   USERS_PER_BUCKET   - Users per difficulty bucket (default: 10)
# ==============================================================================

# 1. Load Environment
source ~/miniforge3/bin/activate active_learning

# 2. Setup
cd $SLURM_SUBMIT_DIR
mkdir -p active_learning/logs

# 3. Print job info
STORAGE_DB=${STORAGE_DB:-active_learning/optuna_studies.db}

echo "=== Projected SVGD Search Worker ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Study: ${STUDY_NAME:-projected_svgd_search}"
echo "Storage: sqlite:///$STORAGE_DB"
echo "Trials: ${TRIALS_PER_WORKER:-5}, Budget: ${BUDGET:-100}, Users/bucket: ${USERS_PER_BUCKET:-10}"

# 4. Run search
python -m active_learning.test.param_search.projected_svgd_search \
    --n-trials ${TRIALS_PER_WORKER:-5} \
    --budget ${BUDGET:-100} \
    --users-per-bucket ${USERS_PER_BUCKET:-10} \
    --study-name "${STUDY_NAME:-projected_svgd_search}" \
    --model "${MODEL_PATH:-models/best_model.pt}" \
    --storage "sqlite:///$STORAGE_DB"

echo "Finished: $(date)"
