#!/bin/bash
#SBATCH --job-name=roma_legacy_comp
#SBATCH --output=active_learning/logs/legacy_comp_%j.out
#SBATCH --error=active_learning/logs/legacy_comp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-4

# ==============================================================================
# ROMA Legacy Comparison SLURM Script
# Runs an extensive comparison trial across all baselines and metrics.
# ==============================================================================

# 1. Load Environment
source ~/miniforge3/bin/activate active_learning

# 2. Workflow Setup
cd $SLURM_SUBMIT_DIR
mkdir -p active_learning/logs

# Print job info
echo "Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Active Learning Baseline Comparison"

# 3. Run Comparison (Parallel)
# SLURM_ARRAY_TASK_ID will be 0..19
# We run a single trial per job, saving the result to disk.
python active_learning/test/legacy/compare_all.py \
    --baselines bald weighted-bald kl-annealed \
    --metrics all \
    --budget 100 \
    --trials 5 \
    --trial-idx $SLURM_ARRAY_TASK_ID \
    --results-dir active_learning/results/legacy_master_comparison_${TIMESTAMP}


# 4. Merge Results  
# Note: After all jobs finish, run the merge step manually (or via a dependency job):
# python active_learning/test/legacy/compare_all.py --merge --results-dir active_learning/results/legacy_master_comparison

echo "Job Finished: $(date)"
echo "Results located in: active_learning/results/legacy_master_comparison"
