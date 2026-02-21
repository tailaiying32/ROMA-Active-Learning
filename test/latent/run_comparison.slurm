#!/bin/bash
#SBATCH --job-name=roma_latent_comp
#SBATCH --output=active_learning/logs/latent_comp_%j.out
#SBATCH --error=active_learning/logs/latent_comp_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-4

# ==============================================================================
# ROMA Latent Comparison SLURM Script
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
echo "Active Learning Latent Comparison"

# 3. Run Comparison (Parallel)
# SLURM_ARRAY_TASK_ID will be 0..19
# We run a single trial per job, saving the result to disk.
python active_learning/test/latent/compare_all.py \
    --strategies bald gbald \
    --metrics all \
    --budget 100 \
    --trials 5 \
    --trial-idx $SLURM_ARRAY_TASK_ID \
    --results-dir active_learning/results/latent_master_comparison_${TIMESTAMP}

# Note: After all jobs finish, run the merge step manually (or via a dependency job):
# python active_learning/test/latent/compare_all.py --merge --results-dir active_learning/results/latent_master_comparison

echo "Job Finished: $(date)"
echo "Results located in: active_learning/results/latent_master_comparison"
