#!/bin/bash
# ==============================================================================
# Projected SVGD Hyperparameter Search - SLURM Launcher
#
# Submits a distributed Optuna search as a SLURM array job.
# Each array task is an independent worker sharing the same SQLite study.
#
# Usage:
#   New search:     bash run_projected_svgd_search.sh [TOTAL_TRIALS] [N_WORKERS] [BUDGET]
#   Resume search:  bash run_projected_svgd_search.sh [TOTAL_TRIALS] [N_WORKERS] [BUDGET] --study-name <name>
#
# Examples:
#   bash active_learning/test/param_search/run_projected_svgd_search.sh 50 10 100
#   bash active_learning/test/param_search/run_projected_svgd_search.sh 20 5 100 --study-name proj_svgd_20260128
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
    STUDY_NAME="projected_svgd_search_$(date +%Y%m%d_%H%M%S)"
    echo "New study: $STUDY_NAME"
else
    echo "Resuming study: $STUDY_NAME"
fi

TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + N_WORKERS - 1) / N_WORKERS ))

STORAGE_DB="active_learning/optuna_studies.db"

echo "Total trials: $TOTAL_TRIALS ($TRIALS_PER_WORKER per worker x $N_WORKERS workers)"
echo "Budget: $BUDGET queries per run"
echo "Study: $STUDY_NAME"
echo "Storage: sqlite:///$STORAGE_DB"
echo ""

# Submit array job
JOB_ID=$(sbatch --parsable \
    --array=0-$((N_WORKERS - 1)) \
    --export=ALL,TRIALS_PER_WORKER=$TRIALS_PER_WORKER,BUDGET=$BUDGET,STUDY_NAME=$STUDY_NAME,STORAGE_DB=$STORAGE_DB \
    active_learning/test/param_search/projected_svgd_search.slurm)

echo "Submitted job: $JOB_ID"
echo "Monitor: squeue --me"
echo "Dashboard: optuna-dashboard sqlite:///$STORAGE_DB"
