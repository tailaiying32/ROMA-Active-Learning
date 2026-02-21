#!/usr/bin/env python3
"""
Parameter Coverage Visualization

This script creates comprehensive visualizations to analyze the coverage
and distribution of muscle parameters across all generated templates.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import json

from muscle_anatomy_config import ALL_MUSCLES, MUSCLE_GROUPS, SEVERITY_LEVELS

class ParameterCoverageAnalyzer:
    """Analyzes and visualizes parameter coverage across templates."""
    
    def __init__(self, template_dir: str = "muscle_params/v4"):
        """
        Initialize the analyzer.
        
        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load templates and metadata
        self.templates = {}
        self.metadata = None
        self.load_templates()
        self.load_metadata()
        
        # Set up plotting parameters
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_templates(self):
        """Load all template files into memory."""
        print("Loading templates...")
        
        template_files = list(self.template_dir.glob("template_*.txt"))
        template_files = [f for f in template_files if f.name != "template_summary.csv"]
        
        for template_file in sorted(template_files):
            template_id = template_file.stem
            
            # Parse template file
            muscle_values = {}
            try:
                with open(template_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and '{' in line:
                            # Parse: muscle_name {max_isometric_force.factor = value}
                            parts = line.split('{')
                            muscle_name = parts[0].strip()
                            
                            # Extract value
                            value_part = parts[1].split('=')[1].split('}')[0].strip()
                            value = float(value_part)
                            
                            muscle_values[muscle_name] = value
                
                self.templates[template_id] = muscle_values
                
            except Exception as e:
                print(f"Error loading {template_file}: {e}")
        
        print(f"Loaded {len(self.templates)} templates")
    
    def load_metadata(self):
        """Load template metadata from CSV file."""
        metadata_file = self.template_dir / "template_summary.csv"
        
        if metadata_file.exists():
            self.metadata = pd.read_csv(metadata_file)
            print(f"Loaded metadata for {len(self.metadata)} templates")
        else:
            print("Warning: No metadata file found")
    
    def create_parameter_distribution_plots(self):
        """Create plots showing parameter value distributions."""
        print("Creating parameter distribution plots...")
        
        # Collect all parameter values
        all_values = []
        for template_id, muscle_values in self.templates.items():
            all_values.extend(muscle_values.values())
        
        # Create overall distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Value Distributions', fontsize=16, fontweight='bold')
        
        # Overall histogram
        axes[0, 0].hist(all_values, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Parameter Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Parameter Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add severity level boundaries
        for severity, (min_val, max_val) in SEVERITY_LEVELS.items():
            axes[0, 0].axvspan(min_val, max_val, alpha=0.2, 
                             label=f'{severity.title()} ({min_val}-{max_val})')
        axes[0, 0].legend()
        
        # Distribution by muscle group
        group_values = {group: [] for group in MUSCLE_GROUPS.keys()}
        for template_id, muscle_values in self.templates.items():
            for group, muscles in MUSCLE_GROUPS.items():
                for muscle in muscles:
                    if muscle in muscle_values:
                        group_values[group].append(muscle_values[muscle])
        
        # Box plot by muscle group
        group_data = [group_values[group] for group in MUSCLE_GROUPS.keys()]
        group_labels = list(MUSCLE_GROUPS.keys())
        
        bp = axes[0, 1].boxplot(group_data, labels=group_labels, patch_artist=True)
        axes[0, 1].set_ylabel('Parameter Value')
        axes[0, 1].set_title('Distribution by Muscle Group')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Distribution by tier (if metadata available)
        if self.metadata is not None:
            tier_values = {tier: [] for tier in self.metadata['tier'].unique()}
            
            for _, row in self.metadata.iterrows():
                template_id = row['template_id']
                tier = row['tier']
                
                if template_id in self.templates:
                    tier_values[tier].extend(self.templates[template_id].values())
            
            # Box plot by tier
            tier_data = [tier_values[tier] for tier in sorted(tier_values.keys())]
            tier_labels = [f'Tier {tier}' for tier in sorted(tier_values.keys())]
            
            bp2 = axes[1, 0].boxplot(tier_data, labels=tier_labels, patch_artist=True)
            axes[1, 0].set_ylabel('Parameter Value')
            axes[1, 0].set_title('Distribution by Generation Tier')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Color the boxes
            colors2 = plt.cm.Set1(np.linspace(0, 1, len(bp2['boxes'])))
            for patch, color in zip(bp2['boxes'], colors2):
                patch.set_facecolor(color)
        
        # Severity level distribution using all 5 categories
        severity_counts = {level: 0 for level in SEVERITY_LEVELS.keys()}
        for value in all_values:
            for level, (min_val, max_val) in SEVERITY_LEVELS.items():
                if min_val <= value <= max_val:
                    severity_counts[level] += 1
                    break
        
        axes[1, 1].pie(severity_counts.values(), labels=severity_counts.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Severity Level Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_muscle_group_heatmap(self):
        """Create heatmap showing muscle group weakness patterns."""
        print("Creating muscle group heatmap...")
        
        # Calculate average weakness by muscle group for each template
        group_weakness = []
        template_ids = []
        
        for template_id, muscle_values in self.templates.items():
            template_ids.append(template_id)
            row = []
            
            for group, muscles in MUSCLE_GROUPS.items():
                group_values = [muscle_values.get(muscle, 1.0) for muscle in muscles]
                avg_weakness = 1.0 - np.mean(group_values)  # Convert to weakness (0=strong, 1=weak)
                row.append(avg_weakness)
            
            group_weakness.append(row)
        
        # Create DataFrame
        group_weakness_df = pd.DataFrame(group_weakness, 
                                       columns=list(MUSCLE_GROUPS.keys()),
                                       index=template_ids)
        
        # Create heatmap
        plt.figure(figsize=(12, 20))
        sns.heatmap(group_weakness_df, 
                   cmap='Reds', 
                   cbar_kws={'label': 'Weakness Level (0=Strong, 1=Weak)'},
                   yticklabels=True if len(template_ids) <= 50 else False)
        
        plt.title('Muscle Group Weakness Patterns Across Templates', fontweight='bold')
        plt.xlabel('Muscle Groups')
        plt.ylabel('Template ID')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'muscle_group_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics
        summary_stats = group_weakness_df.describe()
        summary_stats.to_csv(self.output_dir / 'muscle_group_summary_stats.csv')
    
    def create_coverage_matrix_plot(self):
        """Create matrix showing coverage of muscle group combinations."""
        print("Creating coverage matrix plot...")
        
        # Create pairwise coverage matrix
        groups = list(MUSCLE_GROUPS.keys())
        n_groups = len(groups)
        
        # Initialize coverage matrices
        severe_coverage = np.zeros((n_groups, n_groups))
        moderate_coverage = np.zeros((n_groups, n_groups))
        any_weakness_coverage = np.zeros((n_groups, n_groups))
        
        for template_id, muscle_values in self.templates.items():
            # Calculate group averages
            group_averages = {}
            for group, muscles in MUSCLE_GROUPS.items():
                group_values = [muscle_values.get(muscle, 1.0) for muscle in muscles]
                group_averages[group] = np.mean(group_values)
            
            # Check all pairs
            for i, group1 in enumerate(groups):
                for j, group2 in enumerate(groups):
                    val1 = group_averages[group1]
                    val2 = group_averages[group2]
                    
                    # Check if both groups have any weakness
                    if val1 < 0.8 or val2 < 0.8:
                        any_weakness_coverage[i, j] = 1
                    
                    # Check for moderate weakness in both (updated for new ranges)
                    if val1 < 0.4 and val2 < 0.4:
                        moderate_coverage[i, j] = 1
                    
                    # Check for severe weakness in both (updated for new ranges)
                    if val1 < 0.15 and val2 < 0.15:
                        severe_coverage[i, j] = 1
        
        # Create subplot figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Muscle Group Combination Coverage', fontsize=16, fontweight='bold')
        
        # Plot each coverage matrix
        matrices = [any_weakness_coverage, moderate_coverage, severe_coverage]
        titles = ['Any Weakness Coverage', 'Moderate Weakness Coverage', 'Severe Weakness Coverage']
        
        for idx, (matrix, title) in enumerate(zip(matrices, titles)):
            sns.heatmap(matrix, 
                       xticklabels=groups, 
                       yticklabels=groups,
                       annot=True, 
                       fmt='.0f',
                       cmap='Blues',
                       ax=axes[idx],
                       cbar_kws={'label': 'Coverage (0=No, 1=Yes)'})
            
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Muscle Group 2')
            axes[idx].set_ylabel('Muscle Group 1')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'coverage_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and save coverage statistics
        coverage_stats = {
            'any_weakness_coverage': float(np.sum(any_weakness_coverage)) / (n_groups * n_groups),
            'moderate_weakness_coverage': float(np.sum(moderate_coverage)) / (n_groups * n_groups),
            'severe_weakness_coverage': float(np.sum(severe_coverage)) / (n_groups * n_groups),
            'total_possible_combinations': n_groups * n_groups
        }
        
        with open(self.output_dir / 'coverage_statistics.json', 'w') as f:
            json.dump(coverage_stats, f, indent=2)
    
    def create_parameter_space_visualization(self):
        """Create PCA visualization of parameter space coverage."""
        print("Creating parameter space visualization...")
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("Warning: sklearn not available, skipping PCA visualization")
            return
        
        # Prepare data matrix
        template_matrix = []
        template_labels = []
        
        for template_id, muscle_values in self.templates.items():
            row = [muscle_values.get(muscle, 1.0) for muscle in ALL_MUSCLES]
            template_matrix.append(row)
            template_labels.append(template_id)
        
        template_matrix = np.array(template_matrix)
        
        # Standardize the data
        scaler = StandardScaler()
        template_matrix_scaled = scaler.fit_transform(template_matrix)
        
        # Apply PCA
        pca = PCA(n_components=2)
        template_pca = pca.fit_transform(template_matrix_scaled)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Color by tier if metadata available
        if self.metadata is not None:
            tier_colors = {}
            unique_tiers = sorted(self.metadata['tier'].unique())
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_tiers)))
            
            for tier, color in zip(unique_tiers, colors):
                tier_colors[tier] = color
            
            for _, row in self.metadata.iterrows():
                template_id = row['template_id']
                tier = row['tier']
                
                if template_id in template_labels:
                    idx = template_labels.index(template_id)
                    plt.scatter(template_pca[idx, 0], template_pca[idx, 1], 
                              c=[tier_colors[tier]], 
                              label=f'Tier {tier}' if f'Tier {tier}' not in plt.gca().get_legend_handles_labels()[1] else "",
                              alpha=0.7, s=50)
            
            plt.legend()
        else:
            plt.scatter(template_pca[:, 0], template_pca[:, 1], alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Parameter Space Coverage (PCA Visualization)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_space_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save PCA results
        pca_results = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
        }
        
        with open(self.output_dir / 'pca_results.json', 'w') as f:
            json.dump(pca_results, f, indent=2)
    
    def create_clinical_pattern_analysis(self):
        """Analyze and visualize clinical pattern coverage."""
        print("Creating clinical pattern analysis...")
        
        if self.metadata is None:
            print("Warning: No metadata available for clinical pattern analysis")
            return
        
        # Analyze pattern types
        pattern_counts = self.metadata['type'].value_counts()
        
        # Create pattern type distribution
        plt.figure(figsize=(12, 8))
        pattern_counts.plot(kind='bar')
        plt.title('Distribution of Template Types', fontweight='bold')
        plt.xlabel('Template Type')
        plt.ylabel('Number of Templates')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'template_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analyze validation results
        if 'valid' in self.metadata.columns:
            validation_summary = self.metadata['valid'].value_counts()
            
            plt.figure(figsize=(8, 6))
            plt.pie(validation_summary.values, 
                   labels=['Valid' if x else 'Invalid' for x in validation_summary.index],
                   autopct='%1.1f%%', startangle=90)
            plt.title('Template Validation Results', fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'validation_results.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Analyze severity distribution
        if 'severity' in self.metadata.columns:
            severity_counts = self.metadata['severity'].value_counts()
            
            plt.figure(figsize=(10, 6))
            severity_counts.plot(kind='bar')
            plt.title('Distribution of Severity Levels', fontweight='bold')
            plt.xlabel('Severity Level')
            plt.ylabel('Number of Templates')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'severity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive coverage report."""
        print("Generating comprehensive coverage report...")
        
        # Calculate comprehensive statistics
        stats = {
            'total_templates': len(self.templates),
            'total_muscles': len(ALL_MUSCLES),
            'muscle_groups': len(MUSCLE_GROUPS),
            'parameter_statistics': {},
            'coverage_analysis': {}
        }
        
        # Parameter statistics
        all_values = []
        for muscle_values in self.templates.values():
            all_values.extend(muscle_values.values())
        
        stats['parameter_statistics'] = {
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'median': float(np.median(all_values))
        }
        
        # Coverage analysis using all severity levels
        severity_counts = {}
        for level, (min_val, max_val) in SEVERITY_LEVELS.items():
            count = sum(1 for v in all_values if min_val <= v <= max_val)
            severity_counts[f'{level}_parameters'] = count
            severity_counts[f'{level}_percentage'] = count / len(all_values) * 100
        
        stats['coverage_analysis'] = severity_counts
        
        # Save comprehensive report
        with open(self.output_dir / 'comprehensive_report.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create summary text report
        with open(self.output_dir / 'coverage_summary.txt', 'w') as f:
            f.write("MUSCLE PARAMETER TEMPLATE COVERAGE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Templates Generated: {stats['total_templates']}\n")
            f.write(f"Total Parameters: {len(all_values)}\n")
            f.write(f"Muscle Groups Covered: {stats['muscle_groups']}\n\n")
            
            f.write("Parameter Value Statistics:\n")
            f.write(f"  Mean: {stats['parameter_statistics']['mean']:.3f}\n")
            f.write(f"  Std:  {stats['parameter_statistics']['std']:.3f}\n")
            f.write(f"  Min:  {stats['parameter_statistics']['min']:.3f}\n")
            f.write(f"  Max:  {stats['parameter_statistics']['max']:.3f}\n\n")
            
            f.write("Severity Level Distribution:\n")
            for level in SEVERITY_LEVELS.keys():
                param_key = f'{level}_parameters'
                percent_key = f'{level}_percentage'
                if param_key in stats['coverage_analysis']:
                    f.write(f"  {level.title():<13}: {stats['coverage_analysis'][param_key]:5d} ({stats['coverage_analysis'][percent_key]:.1f}%)\n")
            f.write("\n")
            
            if self.metadata is not None:
                f.write("Template Validation:\n")
                valid_count = self.metadata['valid'].sum() if 'valid' in self.metadata.columns else 0
                f.write(f"  Valid Templates: {valid_count}/{len(self.metadata)}\n")
                f.write(f"  Success Rate: {valid_count/len(self.metadata)*100:.1f}%\n")
    
    def run_all_analyses(self):
        """Run all coverage analyses and create visualizations."""
        print("=" * 60)
        print("PARAMETER COVERAGE ANALYSIS")
        print("=" * 60)
        print(f"Analyzing {len(self.templates)} templates")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Create all visualizations
        self.create_parameter_distribution_plots()
        self.create_muscle_group_heatmap()
        self.create_coverage_matrix_plot()
        self.create_parameter_space_visualization()
        self.create_clinical_pattern_analysis()
        self.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("Generated visualizations:")
        viz_files = list(self.output_dir.glob("*.png"))
        for viz_file in sorted(viz_files):
            print(f"  {viz_file}")
        
        print("\nGenerated reports:")
        report_files = list(self.output_dir.glob("*.json")) + list(self.output_dir.glob("*.txt")) + list(self.output_dir.glob("*.csv"))
        for report_file in sorted(report_files):
            print(f"  {report_file}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Analyze and visualize muscle parameter coverage')
    parser.add_argument('--template-dir', default='muscle_params/v4',
                       help='Directory containing template files (default: muscle_params/v4)')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Output directory for visualizations (default: visualizations)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ParameterCoverageAnalyzer(args.template_dir)
    if args.output_dir != 'visualizations':
        analyzer.output_dir = Path(args.output_dir)
        analyzer.output_dir.mkdir(exist_ok=True)
    
    # Run all analyses
    analyzer.run_all_analyses()

if __name__ == "__main__":
    main()