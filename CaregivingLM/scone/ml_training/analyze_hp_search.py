#!/usr/bin/env python3
"""
Analyze hyperparameter search results from log files and wandb.
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def parse_log_file(log_path: str) -> Dict:
    """Parse a single log file to extract key metrics."""
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract experiment name
    exp_name = Path(log_path).stem
    
    # Initialize result dict
    result = {
        'experiment': exp_name,
        'completed': False,
        'final_train_loss': None,
        'final_val_loss': None,
        'final_val_r2': None,
        'test_loss': None,
        'test_r2': None,
        'test_mae': None,
        'test_rmse': None,
        'best_val_loss': None,
        'total_epochs': None,
        'config': {}
    }
    
    # Check if completed
    if 'Training completed!' in content:
        result['completed'] = True
    
    # Extract final training metrics (last epoch shown)
    train_pattern = r'Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Val Loss:\s+([\d.]+)\s+\|\s+Val R²:\s+([-\d.]+)'
    train_matches = re.findall(train_pattern, content)
    
    if train_matches:
        last_match = train_matches[-1]
        result['total_epochs'] = int(last_match[0]) + 1
        result['final_train_loss'] = float(last_match[1])
        result['final_val_loss'] = float(last_match[2])
        result['final_val_r2'] = float(last_match[3])
    
    # Extract test results
    test_patterns = {
        'test_loss': r'Loss:\s+([\d.]+)',
        'test_r2': r'R²:\s+([-\d.]+)',
        'test_mae': r'MAE:\s+([\d.]+)',
        'test_rmse': r'RMSE:\s+([\d.]+)'
    }
    
    test_section = content.split('Test Results:')
    if len(test_section) > 1:
        test_content = test_section[1]
        for key, pattern in test_patterns.items():
            match = re.search(pattern, test_content)
            if match:
                result[key] = float(match.group(1))
    
    # Extract configuration parameters from experiment name
    result['config'] = parse_config_from_name(exp_name)
    
    # Find best validation loss during training
    val_loss_pattern = r'Val Loss:\s+([\d.]+)'
    val_losses = [float(x) for x in re.findall(val_loss_pattern, content)]
    if val_losses:
        result['best_val_loss'] = min(val_losses)
    
    return result


def parse_config_from_name(exp_name: str) -> Dict:
    """Extract configuration parameters from experiment name."""
    config = {}
    
    # Parse different experiment types
    if exp_name.startswith('capacity_'):
        config['category'] = 'capacity'
        config['size'] = exp_name.replace('capacity_', '')
    
    elif exp_name.startswith('lr_'):
        config['category'] = 'learning_rate'
        lr_str = exp_name.replace('lr_', '').replace('_', '.')
        config['learning_rate'] = float(lr_str)
    
    elif exp_name.startswith('optimizer_'):
        config['category'] = 'optimizer'
        config['optimizer'] = exp_name.replace('optimizer_', '')
    
    elif exp_name.startswith('loss_'):
        config['category'] = 'loss_function'
        config['loss_function'] = exp_name.replace('loss_', '')
    
    elif exp_name.startswith('weight_decay_'):
        config['category'] = 'weight_decay'
        wd_str = exp_name.replace('weight_decay_', '').replace('_', '.')
        config['weight_decay'] = float(wd_str)
    
    elif exp_name.startswith('dropout_'):
        config['category'] = 'dropout'
        dropout_str = exp_name.replace('dropout_', '').replace('_', '.')
        config['dropout'] = float(dropout_str)
    
    elif exp_name.startswith('activation_'):
        config['category'] = 'activation'
        config['activation'] = exp_name.replace('activation_', '')
    
    elif exp_name.startswith('batch_size_'):
        config['category'] = 'batch_size'
        config['batch_size'] = int(exp_name.replace('batch_size_', ''))
    
    elif exp_name.startswith('split_'):
        config['category'] = 'data_split'
        config['split'] = exp_name.replace('split_', '')
    
    elif 'best_' in exp_name:
        config['category'] = 'best_combined'
        config['variant'] = exp_name.replace('best_', '')
    
    elif 'long_' in exp_name:
        config['category'] = 'long_training'
        config['variant'] = exp_name.replace('long_', '')
    
    elif 'overfit' in exp_name:
        config['category'] = 'overfit'
        config['variant'] = exp_name
    
    else:
        config['category'] = 'other'
        config['variant'] = exp_name
    
    return config


def create_summary_plots(df: pd.DataFrame, output_dir: str):
    """Create summary plots for hyperparameter search results."""
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Search Results Summary', fontsize=16)
    
    # 1. Test R² by category
    df_completed = df[df['completed'] == True].copy()
    
    if len(df_completed) > 0:
        # Plot 1: Test R² by category
        category_r2 = df_completed.groupby('category')['test_r2'].agg(['mean', 'std', 'count']).reset_index()
        category_r2 = category_r2.sort_values('mean', ascending=False)
        
        axes[0,0].bar(range(len(category_r2)), category_r2['mean'], 
                     yerr=category_r2['std'], capsize=5)
        axes[0,0].set_xticks(range(len(category_r2)))
        axes[0,0].set_xticklabels(category_r2['category'], rotation=45, ha='right')
        axes[0,0].set_ylabel('Test R²')
        axes[0,0].set_title('Test R² by Hyperparameter Category')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Learning rate vs performance
        lr_data = df_completed[df_completed['category'] == 'learning_rate']
        if len(lr_data) > 0:
            axes[0,1].semilogx(lr_data['learning_rate'], lr_data['test_r2'], 'o-')
            axes[0,1].set_xlabel('Learning Rate')
            axes[0,1].set_ylabel('Test R²')
            axes[0,1].set_title('Learning Rate vs Test R²')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Model capacity comparison
        capacity_data = df_completed[df_completed['category'] == 'capacity']
        if len(capacity_data) > 0:
            capacity_order = ['small', 'medium', 'large', 'xlarge', 'deep_narrow']
            capacity_data = capacity_data[capacity_data['size'].isin(capacity_order)]
            capacity_data['size'] = pd.Categorical(capacity_data['size'], categories=capacity_order)
            capacity_data = capacity_data.sort_values('size')
            
            axes[0,2].bar(range(len(capacity_data)), capacity_data['test_r2'])
            axes[0,2].set_xticks(range(len(capacity_data)))
            axes[0,2].set_xticklabels(capacity_data['size'], rotation=45)
            axes[0,2].set_ylabel('Test R²')
            axes[0,2].set_title('Model Capacity vs Test R²')
            axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Training vs Validation Loss (best experiments)
        top_experiments = df_completed.nlargest(5, 'test_r2')
        axes[1,0].scatter(top_experiments['final_train_loss'], top_experiments['final_val_loss'])
        for i, exp in top_experiments.iterrows():
            axes[1,0].annotate(exp['experiment'][:15], 
                             (exp['final_train_loss'], exp['final_val_loss']),
                             fontsize=8, alpha=0.7)
        axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1,0].set_xlabel('Final Train Loss')
        axes[1,0].set_ylabel('Final Val Loss')
        axes[1,0].set_title('Train vs Val Loss (Top 5 Experiments)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Completion rate by category
        completion_rate = df.groupby('category')['completed'].agg(['mean', 'count']).reset_index()
        completion_rate = completion_rate.sort_values('mean', ascending=False)
        
        axes[1,1].bar(range(len(completion_rate)), completion_rate['mean'])
        axes[1,1].set_xticks(range(len(completion_rate)))
        axes[1,1].set_xticklabels(completion_rate['category'], rotation=45, ha='right')
        axes[1,1].set_ylabel('Completion Rate')
        axes[1,1].set_title('Experiment Completion Rate by Category')
        axes[1,1].set_ylim(0, 1)
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Best results distribution
        axes[1,2].hist(df_completed['test_r2'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,2].axvline(df_completed['test_r2'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df_completed["test_r2"].mean():.3f}')
        axes[1,2].axvline(df_completed['test_r2'].median(), color='orange', linestyle='--',
                         label=f'Median: {df_completed["test_r2"].median():.3f}')
        axes[1,2].set_xlabel('Test R²')
        axes[1,2].set_ylabel('Count')
        axes[1,2].set_title('Distribution of Test R² Scores')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hp_search_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter search results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing log files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save analysis results (defaults to results_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    # Find all log files
    log_files = list(Path(args.results_dir).glob('*.log'))
    
    if not log_files:
        print(f"No log files found in {args.results_dir}")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Parse all log files
    results = []
    for log_file in log_files:
        print(f"Parsing {log_file.name}...")
        try:
            result = parse_log_file(str(log_file))
            results.append(result)
        except Exception as e:
            print(f"Error parsing {log_file.name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Expand config column
    config_df = pd.json_normalize(df['config'])
    df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
    
    # Save results
    df.to_csv(f'{args.output_dir}/hp_search_results.csv', index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*60)
    
    print(f"Total experiments: {len(df)}")
    print(f"Completed experiments: {df['completed'].sum()}")
    print(f"Failed experiments: {(~df['completed']).sum()}")
    
    if df['completed'].any():
        completed_df = df[df['completed']]
        
        print(f"\nBest Test R²: {completed_df['test_r2'].max():.4f}")
        print(f"Mean Test R²: {completed_df['test_r2'].mean():.4f}")
        print(f"Std Test R²: {completed_df['test_r2'].std():.4f}")
        
        # Top 5 experiments
        print(f"\n🏆 TOP 5 EXPERIMENTS:")
        top_5 = completed_df.nlargest(5, 'test_r2')[['experiment', 'test_r2', 'test_rmse', 'category']]
        for i, (_, exp) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. {exp['experiment']:<25} | R²: {exp['test_r2']:.4f} | RMSE: {exp['test_rmse']:.4f} | {exp['category']}")
        
        # Category analysis
        print(f"\n📊 PERFORMANCE BY CATEGORY:")
        category_stats = completed_df.groupby('category')['test_r2'].agg(['count', 'mean', 'std', 'max']).round(4)
        category_stats = category_stats.sort_values('mean', ascending=False)
        print(category_stats)
        
        # Create plots
        print(f"\n📈 Creating summary plots...")
        try:
            create_summary_plots(df, args.output_dir)
            print(f"Plots saved to {args.output_dir}/hp_search_summary.png")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    print(f"\n💾 Results saved to {args.output_dir}/hp_search_results.csv")
    print(f"\n🎯 Next steps:")
    print("1. Review the top performing configurations")
    print("2. Run longer training with the best config")
    print("3. Consider implementing alternative architectures if needed")


if __name__ == "__main__":
    main()