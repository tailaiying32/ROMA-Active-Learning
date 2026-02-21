import optuna
import sys
import pandas as pd

def check_progress(storage_url="sqlite:///active_learning/latent_optuna.db", study_name="latent_hyperparameter_optimization_study"):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        print(f"Study '{study_name}' not found in {storage_url}.")
        return
    except Exception as e:
        print(f"Error loading study: {e}")
        return

    print(f"Study: {study_name}")
    print(f"Storage: {storage_url}")
    print("-" * 40)
    
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
    running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    
    print(f"Total Trials: {len(trials)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Running:   {len(running)}")
    print(f"  Failed:    {len(failed)}")
    
    if completed:
        print("-" * 40)
        print(f"Best Value (Loss): {study.best_value:.4f}")
        print("Best Params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        print("-" * 40)
        print("Recent Trials:")
        df = study.trials_dataframe()
        if not df.empty:
            # Show last 5 complete trials
            cols = ['number', 'value', 'state', 'datetime_complete']
            print(df[cols].tail(5).to_string(index=False))

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///active_learning/latent_optuna.db"
    check_progress(storage_url=url)
