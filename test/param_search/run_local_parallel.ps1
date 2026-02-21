# PowerShell script to launch parallel Optuna workers
param (
    [int]$n_workers = 4,
    [int]$n_trials = 50,  # Total trials across all workers
    [int]$budget = 100,
    [int]$users_per_bucket = 10,
    [string]$study_name = "projected_svgd_search_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
)

# Calculate trials per worker (approximate)
$trials_per_worker = [math]::Ceiling($n_trials / $n_workers)

Write-Host "Starting $n_workers workers for study '$study_name'"
Write-Host "  Total Target Trials: $n_trials"
Write-Host "  Trials per worker:   $trials_per_worker"
Write-Host "  Budget per run:      $budget"
Write-Host "  Users per bucket:    $users_per_bucket"
Write-Host ""

$pids_list = @()

for ($i = 0; $i -lt $n_workers; $i++) {
    Write-Host "Launching Worker #$($i+1)..."

    # Start cmd /k so the window stays open (allowing you to read errors if it crashes)
    $cmd_args = "/k python -m active_learning.test.param_search.projected_svgd_search --n-trials $trials_per_worker --budget $budget --users-per-bucket $users_per_bucket --study-name $study_name --storage sqlite:///active_learning/optuna_studies.db"

    $job = Start-Process cmd.exe -ArgumentList $cmd_args -WorkingDirectory (Get-Location).Path -PassThru
    $pids_list += $job.Id
}

Write-Host ""
Write-Host "All workers started. PIDs: $($pids_list -join ', ')"
Write-Host "Monitor progress with: optuna-dashboard sqlite:///active_learning/optuna_studies.db"
