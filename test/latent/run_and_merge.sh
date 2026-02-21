#!/bin/bash
# ==============================================================================
# ROMA Latent Comparison - Submit and Auto-Merge
# Submits array job, then automatically runs merge after completion
# ==============================================================================

# Generate unique timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Run timestamp: $TIMESTAMP"

# Submit the array job and capture the job ID
echo "Submitting array job..."
JOB_ID=$(sbatch --parsable --requeue --export=ALL,TIMESTAMP=$TIMESTAMP active_learning/test/latent/run_comparison.slurm)
echo "Array job submitted: $JOB_ID"

# Submit merge job that depends on array job completion
echo "Submitting merge job (will run after array completes)..."
sbatch --dependency=afterok:$JOB_ID --export=ALL,TIMESTAMP=$TIMESTAMP active_learning/test/latent/merge_results.slurm
echo "Merge job submitted with dependency on $JOB_ID"
echo ""
echo "Monitor with: squeue --me"
