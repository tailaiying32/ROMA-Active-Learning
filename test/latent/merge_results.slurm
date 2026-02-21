#!/bin/bash
#SBATCH --job-name=roma_latent_merge
#SBATCH --output=active_learning/logs/latent_merge_%j.out
#SBATCH --error=active_learning/logs/latent_merge_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# ==============================================================================
# ROMA Latent Comparison - Merge Results
# Merges trial results and generates plots
# ==============================================================================

# Load Environment
source ~/miniforge3/bin/activate active_learning

# Workflow Setup
cd $SLURM_SUBMIT_DIR
mkdir -p active_learning/logs

# Print job info
echo "Merge Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"

# Run merge - save results to timestamped directory
save_dir="active_learning/images/latent_master_comparison_${TIMESTAMP}"
results_dir="active_learning/results/latent_master_comparison_${TIMESTAMP}"
echo "Loading results from: $results_dir"
echo "Saving images to: $save_dir"

python active_learning/test/latent/compare_all.py \
    --merge \
    --results-dir "$results_dir" \
    --output-dir "$save_dir"

echo "Merge Job Finished: $(date)"
echo "Results and images saved to timestamped directories"
echo "  Results: $results_dir"
echo "  Images: $save_dir"
