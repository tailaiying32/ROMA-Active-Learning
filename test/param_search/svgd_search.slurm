#!/bin/bash
#SBATCH --job-name=svgd_search
#SBATCH --output=active_learning/logs/svgd_search_%A_%a.out
#SBATCH --error=active_learning/logs/svgd_search_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-9

# ==============================================================================
# SVGD Hyperparameter Search - Distributed Optuna Worker
#
# Each array task is an independent Optuna worker. All workers share the same
# SQLite study database. Optuna coordinates trial distribution automatically.
#
# Environment variables (set by run_search.sh or --export):
#   TRIALS_PER_WORKER  - Number of trials this worker should run (default: 5)
#   BUDGET             - Queries per active learning run (default: 100)
#   SEEDS              - Seeds per trial (default: 5)
#   STUDY_NAME         - Optuna study name (default: svgd_param_search)
#   MODEL_PATH         - Path to decoder model (default: models/best_model.pt)
# ==============================================================================

# 1. Load Environment
source ~/miniforge3/bin/activate active_learning

# 2. Setup
cd $SLURM_SUBMIT_DIR
mkdir -p active_learning/logs

# 3. Print job info
echo "=== SVGD Search Worker ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Study: ${STUDY_NAME:-svgd_param_search}"
echo "Trials: ${TRIALS_PER_WORKER:-5}, Budget: ${BUDGET:-100}, Seeds: ${SEEDS:-5}"

# 4. Run search
python -m active_learning.test.param_search.svgd_search \
    --n-trials ${TRIALS_PER_WORKER:-5} \
    --budget ${BUDGET:-100} \
    --seeds ${SEEDS:-5} \
    --study-name "${STUDY_NAME:-svgd_param_search}" \
    --model "${MODEL_PATH:-models/best_model.pt}" \
    --storage "sqlite:///active_learning/${STUDY_NAME:-svgd_param_search}.db"

echo "Finished: $(date)"
