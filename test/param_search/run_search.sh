#!/bin/bash
# ==============================================================================
# SVGD Hyperparameter Search - SLURM Launcher
#
# Submits a distributed Optuna search as a SLURM array job.
# Each array task is an independent worker sharing the same SQLite study.
#
# Usage:
#   New search:     bash run_search.sh [TOTAL_TRIALS] [N_WORKERS] [BUDGET]
#   Resume search:  bash run_search.sh [TOTAL_TRIALS] [N_WORKERS] [BUDGET] --study-name <name>
#
# Examples:
#   bash active_learning/test/param_search/run_search.sh 50 10 100
#   bash active_learning/test/param_search/run_search.sh 20 5 100 --study-name svgd_search_20260127_143000
# ==============================================================================

TOTAL_TRIALS=${1:-50}
N_WORKERS=${2:-10}
BUDGET=${3:-100}

# Parse optional --study-name flag
STUDY_NAME=""
shift 3 2>/dev/null
while [[ $# -gt 0 ]]; do
    case "$1" in
        --study-name)
            STUDY_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

# Auto-generate timestamped name if not provided
if [ -z "$STUDY_NAME" ]; then
    STUDY_NAME="svgd_search_$(date +%Y%m%d_%H%M%S)"
    echo "New study: $STUDY_NAME"
else
    echo "Resuming study: $STUDY_NAME"
fi

TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + N_WORKERS - 1) / N_WORKERS ))

echo "Total trials: $TOTAL_TRIALS ($TRIALS_PER_WORKER per worker x $N_WORKERS workers)"
echo "Budget: $BUDGET queries per run"
echo "Storage: sqlite:///active_learning/${STUDY_NAME}.db"
echo ""

# Submit array job
JOB_ID=$(sbatch --parsable \
    --array=0-$((N_WORKERS - 1)) \
    --export=ALL,TRIALS_PER_WORKER=$TRIALS_PER_WORKER,BUDGET=$BUDGET,STUDY_NAME=$STUDY_NAME \
    active_learning/test/param_search/svgd_search.slurm)

echo "Submitted job: $JOB_ID"
echo "Monitor: squeue --me"
echo "Dashboard: optuna-dashboard sqlite:///active_learning/${STUDY_NAME}.db"
