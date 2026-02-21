#!/usr/bin/env python3
"""
Cleaning script for SCONE results directory.
- Removes all .par files except the one with the highest step number in each folder
- Also removes corresponding .sto files for deleted .par files
- Generates statistics report by model showing total folders and completed motions
"""

import os
import re
import glob
import json
import statistics
import argparse
from collections import defaultdict


def extract_step_from_par_filename(filename):
    """Extract step number from .par filename like '0123_45.678_90.123.par'"""
    match = re.match(r'^(\d+)_.*\.par$', filename)
    return int(match.group(1)) if match else -1


def extract_model_from_folder_name(folder_name):
    """Extract model name from folder like 'single_arm_c0_random_1_20250827...'"""
    # Look for the pattern single_arm_cX or single_arm_cX_Y
    match = re.match(r'^(condition_\d+(?:_\d+)?)', folder_name)
    return match.group(1) if match else "unknown"


def clean_folder(folder_path):
    """Clean a single folder by keeping only the .par file with highest step number"""
    folder_name = os.path.basename(folder_path)
    print(f"Processing folder: {folder_name}")
    
    # Find all .par files
    par_files = glob.glob(os.path.join(folder_path, "*.par"))
    if not par_files:
        print(f"  No .par files found")
        return 0, 0
    
    # Parse step numbers and find the highest
    par_info = []
    for par_file in par_files:
        filename = os.path.basename(par_file)
        step = extract_step_from_par_filename(filename)
        if step >= 0:
            par_info.append((step, par_file, filename))
    
    if not par_info:
        print(f"  No valid .par files found with step numbers")
        return 0, 0
    
    # Sort by step number and keep the highest
    par_info.sort(key=lambda x: x[0], reverse=True)
    highest_step, highest_par_file, highest_filename = par_info[0]
    
    print(f"  Found {len(par_info)} .par files")
    print(f"  Keeping: {highest_filename} (step {highest_step})")
    
    # Remove all other .par files and their corresponding .sto files
    files_removed = 0
    for step, par_file, filename in par_info[1:]:
        try:
            # Remove .par file
            os.remove(par_file)
            files_removed += 1
            print(f"    Removed: {filename}")
            
            # Remove corresponding .sto file if it exists
            sto_file = par_file + ".sto"
            if os.path.exists(sto_file):
                os.remove(sto_file)
                files_removed += 1
                print(f"    Removed: {filename}.sto")
                
        except OSError as e:
            print(f"    Error removing {filename}: {e}")
    
    return files_removed, len(par_info)


def analyze_fitness_convergence(fitness_scores, cv_threshold=10.0):
    """
    Analyze fitness scores for convergence patterns using lenient criteria
    Focus on whether the best runs (lowest fitness scores) are similar
    
    Args:
        fitness_scores: List of fitness values
        cv_threshold: Coefficient of variation threshold percentage (default 10%)
    
    Returns dict with analysis results
    """
    if len(fitness_scores) < 2:
        return {"status": "insufficient_data", "details": "Less than 2 fitness scores"}
    
    # Filter out invalid values (inf, -inf, nan)
    import math
    valid_scores = []
    for score in fitness_scores:
        if isinstance(score, (int, float)) and math.isfinite(score):
            valid_scores.append(float(score))
    
    if len(valid_scores) < 2:
        return {"status": "insufficient_data", "details": f"Only {len(valid_scores)} valid scores out of {len(fitness_scores)}"}
    
    if len(valid_scores) != len(fitness_scores):
        print(f"    Warning: Filtered out {len(fitness_scores) - len(valid_scores)} invalid fitness values")
    
    # Sort scores to identify best runs (lowest fitness values)
    sorted_scores = sorted(valid_scores)
    fitness_scores = valid_scores  # Use filtered scores for the rest of the analysis
    
    min_fitness = min(fitness_scores)
    max_fitness = max(fitness_scores)
    range_fitness = max_fitness - min_fitness
    
    # Handle extreme scale differences (e.g., 0.000000001 vs 12.0)
    # Use log-scale analysis when we have very small values mixed with larger ones
    min_log_ratio = 1e-6  # If min/max ratio is smaller than this, use log scale
    use_log_scale = False
    
    if min_fitness > 0 and (min_fitness / max_fitness) < min_log_ratio:
        use_log_scale = True
        print(f"    Info: Large scale difference detected ({min_fitness:.2e} to {max_fitness:.2e}), using log-scale analysis")
        
        # Convert to log scale for more stable analysis
        import math
        log_scores = [math.log10(max(score, 1e-15)) for score in fitness_scores]  # Clamp to avoid log(0)
        
        try:
            mean_fitness = statistics.mean(fitness_scores)  # Keep original mean for reporting
            log_mean = statistics.mean(log_scores)
            log_std = statistics.stdev(log_scores) if len(log_scores) > 1 else 0
            
            # CV on log scale is more stable for extreme ranges
            cv = (log_std / abs(log_mean)) * 100 if abs(log_mean) > 0 else 0
            std_dev = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0
        except (OverflowError, ValueError, ZeroDivisionError):
            # Even more extreme fallback
            mean_fitness = sum(fitness_scores) / len(fitness_scores)
            std_dev = 0
            cv = 1000  # Mark as highly diverse
            
    else:
        # Normal scale analysis
        try:
            mean_fitness = statistics.mean(fitness_scores)
            std_dev = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0
        except (OverflowError, ValueError) as e:
            # Handle extreme values by using manual calculation
            mean_fitness = sum(fitness_scores) / len(fitness_scores)
            if len(fitness_scores) > 1:
                try:
                    variance = sum((x - mean_fitness) ** 2 for x in fitness_scores) / (len(fitness_scores) - 1)
                    std_dev = variance ** 0.5 if variance >= 0 else 0
                except OverflowError:
                    std_dev = 0
                    print(f"    Warning: Cannot calculate std dev due to extreme values")
            else:
                std_dev = 0
            print(f"    Warning: Extreme fitness values detected, using fallback calculation")
        
        # Calculate coefficient of variation with safety checks
        try:
            if abs(mean_fitness) > 1e-15:  # Avoid division by very small numbers
                cv = (std_dev / abs(mean_fitness)) * 100
            else:
                cv = 0
        except (OverflowError, ZeroDivisionError):
            cv = 1000  # Mark as highly diverse
    
    # Alternative convergence metric: ratio of max to min (geometric spread)
    geometric_spread = max_fitness / max(min_fitness, 1e-15) if min_fitness > 0 else float('inf')
    
    # Calculate relative range as percentage of mean with safety checks
    try:
        if abs(mean_fitness) > 1e-15:
            relative_range = (range_fitness / abs(mean_fitness)) * 100
        else:
            relative_range = 0
    except (OverflowError, ZeroDivisionError):
        relative_range = 0
    
    # LENIENT ANALYSIS: Focus on best runs convergence
    # Take the best 2-3 runs (lowest scores) and check if they're similar
    num_best = min(3, len(sorted_scores))
    best_runs = sorted_scores[:num_best]
    
    if len(best_runs) >= 2:
        try:
            best_mean = statistics.mean(best_runs)
            best_std = statistics.stdev(best_runs) if len(best_runs) > 1 else 0
        except (OverflowError, ValueError):
            # Handle extreme values by using manual calculation
            best_mean = sum(best_runs) / len(best_runs)
            if len(best_runs) > 1:
                variance = sum((x - best_mean) ** 2 for x in best_runs) / (len(best_runs) - 1)
                best_std = variance ** 0.5 if variance >= 0 else 0
            else:
                best_std = 0
        
        # Calculate best runs CV with safety checks
        if abs(best_mean) > 0 and not (abs(best_mean) == float('inf') or abs(best_std) == float('inf')):
            best_cv = (best_std / abs(best_mean)) * 100
        else:
            best_cv = 0
            
        best_range = max(best_runs) - min(best_runs)
        
        # Calculate best relative range with safety checks
        if abs(best_mean) > 0 and not (abs(best_mean) == float('inf') or abs(best_range) == float('inf')):
            best_relative_range = (best_range / abs(best_mean)) * 100
        else:
            best_relative_range = 0
        
        # Use geometric spread as backup when CV is unreliable (too high)
        best_geometric_spread = max(best_runs) / max(min(best_runs), 1e-15)
        
        # Determine convergence status based on best runs similarity
        if best_cv >= 100 or cv >= 100:  # CV unreliable due to extreme scale differences
            # Fall back to geometric spread analysis
            if best_geometric_spread <= 1.5:  # Best runs within 50% of each other
                status = "converged"
                details = f"Best {num_best} runs converged (geometric spread: {best_geometric_spread:.2f}x)"
            elif best_geometric_spread <= 3.0:  # Within 3x of each other
                status = "likely_converged"
                details = f"Best {num_best} runs similar (geometric spread: {best_geometric_spread:.2f}x)"
            elif best_geometric_spread <= 10.0:  # Within 10x of each other
                status = "moderate_variation"
                details = f"Best {num_best} runs moderate spread (geometric spread: {best_geometric_spread:.2f}x)"
            else:
                status = "diverse"
                details = f"High variation in best runs (geometric spread: {best_geometric_spread:.2f}x)"
        else:
            # Use CV analysis as normal
            if best_cv <= cv_threshold * 0.5:  # Very tight convergence of best runs
                status = "converged"
                details = f"Best {num_best} runs converged (CV: {best_cv:.2f}% ≤ {cv_threshold*0.5:.1f}%)"
            elif best_cv <= cv_threshold:  # Good convergence of best runs
                status = "likely_converged"
                details = f"Best {num_best} runs similar (CV: {best_cv:.2f}% ≤ {cv_threshold:.1f}%)"
            elif cv <= cv_threshold:  # Overall convergence is still good
                status = "likely_converged"
                details = f"Overall convergence good (CV: {cv:.2f}% ≤ {cv_threshold:.1f}%)"
            elif best_cv <= cv_threshold * 2:  # Best runs have moderate variation
                status = "moderate_variation"
                details = f"Best {num_best} runs moderate variation (CV: {best_cv:.2f}%)"
            else:  # High variation even in best runs
                status = "diverse"
                details = f"High variation in best runs (CV: {best_cv:.2f}% > {cv_threshold*2:.1f}%)"
    else:
        # Fallback to overall analysis if we can't analyze best runs
        if cv <= cv_threshold * 0.5:
            status = "converged"
            details = f"Tight convergence (CV: {cv:.2f}% ≤ {cv_threshold*0.5:.1f}%)"
        elif cv <= cv_threshold:
            status = "likely_converged"
            details = f"Good convergence (CV: {cv:.2f}% ≤ {cv_threshold:.1f}%)"
        elif cv <= cv_threshold * 2:
            status = "moderate_variation"
            details = f"Moderate variation (CV: {cv:.2f}%)"
        else:
            status = "diverse"
            details = f"High variation (CV: {cv:.2f}% > {cv_threshold*2:.1f}%)"
    
    # Identify outliers using IQR method (more robust than z-score for small samples)
    outliers = []
    if len(fitness_scores) >= 4:  # Need at least 4 points for meaningful IQR
        n = len(sorted_scores)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_scores[q1_idx]
        q3 = sorted_scores[q3_idx]
        iqr = q3 - q1
        
        if iqr > 0:  # Avoid division by zero
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for i, score in enumerate(fitness_scores):
                if score < lower_bound or score > upper_bound:
                    # Calculate how many IQRs away from the nearest quartile
                    iqr_distance = min(abs(score - q1), abs(score - q3)) / iqr
                    outliers.append((i, score, iqr_distance))
    
    return {
        "status": status,
        "details": details,
        "mean": mean_fitness,
        "std_dev": std_dev,
        "range": range_fitness,
        "relative_range": relative_range,
        "cv": cv,
        "geometric_spread": geometric_spread,
        "best_runs": best_runs,
        "best_cv": best_cv if 'best_cv' in locals() else cv,
        "use_log_scale": use_log_scale,
        "outliers": outliers,
        "scores": fitness_scores
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Clean SCONE results and analyze convergence')
    parser.add_argument('--cv-threshold', type=float, default=10.0,
                       help='CV threshold %% for convergence detection (default: 10.0)')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed convergence analysis')
    parser.add_argument('--results-dir', type=str, 
                       default='/home/ziang/Workspace/CaregivingLM/scone/results',
                       help='Path to results directory')
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    cv_threshold = args.cv_threshold
    show_detailed = args.detailed
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Statistics tracking
    model_stats = defaultdict(lambda: {
        "total_folders": 0, 
        "completed_motions": 0,
        "convergence_analysis": {"converged": 0, "likely_converged": 0, "moderate_variation": 0, "diverse": 0, "no_data": 0}
    })
    total_files_removed = 0
    total_folders_processed = 0
    convergence_details = []  # Store detailed convergence info for reporting
    
    # Expected models
    # expected_models = [
    #     "single_arm_c0", "single_arm_c1", "single_arm_c2", "single_arm_c3", 
    #     "single_arm_c4", "single_arm_c5", "single_arm_c6_1", "single_arm_c6_2", "single_arm_c6_3"
    # ]

    expected_models = [
        "condition_1"
    ]
    
    print(f"Scanning results directory: {results_dir}")
    print("=" * 60)
    
    # Process each folder
    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)
        
        # Skip if not a directory
        if not os.path.isdir(folder_path):
            continue
            
        # Skip if doesn't match pattern
        if not folder_name.startswith("condition_"):
            continue
            
        total_folders_processed += 1
        
        # Extract model name
        model = extract_model_from_folder_name(folder_name)
        model_stats[model]["total_folders"] += 1
        
        # Check if motion is completed (has evaluation_results.json)
        eval_file = os.path.join(folder_path, "evaluation_results.json")
        fitness_analysis = None
        
        if os.path.exists(eval_file):
            model_stats[model]["completed_motions"] += 1
            
            # Analyze fitness convergence
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                    
                if "all_runs_fitness" in eval_data and eval_data["all_runs_fitness"]:
                    fitness_scores = eval_data["all_runs_fitness"]
                    fitness_analysis = analyze_fitness_convergence(fitness_scores, cv_threshold)
                    
                    # Update convergence statistics
                    status = fitness_analysis["status"]
                    if status in model_stats[model]["convergence_analysis"]:
                        model_stats[model]["convergence_analysis"][status] += 1
                    else:
                        model_stats[model]["convergence_analysis"]["no_data"] += 1
                    
                    # Store detailed info for later reporting
                    convergence_details.append({
                        "folder": folder_name,
                        "model": model,
                        "analysis": fitness_analysis
                    })
                    
                    print(f"  Fitness Analysis: {fitness_analysis['status']} - {fitness_analysis['details']}")
                    if fitness_analysis["outliers"]:
                        print(f"    Outliers detected: {len(fitness_analysis['outliers'])} runs")
                else:
                    model_stats[model]["convergence_analysis"]["no_data"] += 1
                    print(f"  No fitness data found in evaluation file")
                    
            except (json.JSONDecodeError, KeyError, IOError) as e:
                model_stats[model]["convergence_analysis"]["no_data"] += 1
                print(f"  Error reading evaluation file: {e}")
        else:
            model_stats[model]["convergence_analysis"]["no_data"] += 1
        
        # Clean the folder
        files_removed, total_par_files = clean_folder(folder_path)
        total_files_removed += files_removed
        
        print()  # Empty line for readability
    
    print("=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Total folders processed: {total_folders_processed}")
    print(f"Total files removed: {total_files_removed}")
    print()
    
    print("=" * 60)
    print("STATISTICS BY MODEL")
    print("=" * 60)
    print(f"{'Model':<20} {'Total Runs':<15} {'Completed':<12} {'Complete %':<12}")
    print("-" * 60)
    
    grand_total_folders = 0
    grand_total_completed = 0
    
    # Show stats for expected models first, then any others found
    all_models = set(expected_models + list(model_stats.keys()))
    
    for model in sorted(all_models):
        stats = model_stats[model]
        total = stats["total_folders"]
        completed = stats["completed_motions"]
        percentage = (completed / total * 100) if total > 0 else 0
        
        print(f"{model:<20} {total:<15} {completed:<12} {percentage:<12.1f}%")
        
        grand_total_folders += total
        grand_total_completed += completed
    
    print("-" * 60)
    grand_percentage = (grand_total_completed / grand_total_folders * 100) if grand_total_folders > 0 else 0
    print(f"{'TOTAL':<20} {grand_total_folders:<15} {grand_total_completed:<12} {grand_percentage:<12.1f}%")
    
    print()
    print("=" * 80)
    print("CONVERGENCE ANALYSIS BY MODEL")
    print(f"(CV Threshold: {cv_threshold}%)")
    print("=" * 80)
    print(f"{'Model':<20} {'Converged':<10} {'Likely':<10} {'Moderate':<10} {'Diverse':<10} {'No Data':<10}")
    print("-" * 80)
    
    grand_convergence = {"converged": 0, "likely_converged": 0, "moderate_variation": 0, "diverse": 0, "no_data": 0}
    
    for model in sorted(all_models):
        stats = model_stats[model]
        conv = stats["convergence_analysis"]
        
        print(f"{model:<20} {conv['converged']:<10} {conv['likely_converged']:<10} {conv['moderate_variation']:<10} {conv['diverse']:<10} {conv['no_data']:<10}")
        
        for key in grand_convergence:
            grand_convergence[key] += conv[key]
    
    print("-" * 80)
    print(f"{'TOTAL':<20} {grand_convergence['converged']:<10} {grand_convergence['likely_converged']:<10} {grand_convergence['moderate_variation']:<10} {grand_convergence['diverse']:<10} {grand_convergence['no_data']:<10}")
    
    # Show detailed convergence information if requested
    if show_detailed and convergence_details:
        print()
        print("=" * 60)
        print("DETAILED CONVERGENCE ANALYSIS")
        print("=" * 60)
        
        # Group by convergence status
        by_status = defaultdict(list)
        for detail in convergence_details:
            by_status[detail["analysis"]["status"]].append(detail)
        
        for status in ["converged", "likely_converged", "moderate_variation", "diverse"]:
            if status in by_status:
                print(f"\n{status.upper().replace('_', ' ')} RUNS ({len(by_status[status])}):")
                print("-" * 50)
                
                for detail in by_status[status][:10]:  # Show first 10 of each type
                    analysis = detail["analysis"]
                    print(f"  {detail['folder'][:60]}...")
                    print(f"    Overall CV: {analysis['cv']:.2f}%, Best runs CV: {analysis['best_cv']:.2f}%")
                    print(f"    All scores: {[f'{s:.2f}' for s in analysis['scores'][:5]]}")
                    if len(analysis['scores']) > 5:
                        print(f"               ... ({len(analysis['scores'])} total scores)")
                    if 'best_runs' in analysis:
                        print(f"    Best runs:  {[f'{s:.2f}' for s in analysis['best_runs']]}")
                    if analysis["outliers"]:
                        print(f"    Outliers: {[(i, f'{s:.2f}') for i, s, d in analysis['outliers']]}")
                    print()
                
                if len(by_status[status]) > 10:
                    print(f"    ... and {len(by_status[status]) - 10} more {status} runs")
                    print()
    
    print()
    print("=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()