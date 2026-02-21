#!/usr/bin/env python3
"""
Muscle Parameter Template Generator

This script generates 250 muscle parameter templates following a systematic
4-tier approach to ensure comprehensive coverage of clinically plausible
muscle weakness patterns.
"""

import os
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from muscle_anatomy_config import (
    ALL_MUSCLES, MUSCLE_GROUPS, SEVERITY_LEVELS, 
    generate_random_value, generate_severe_population_value, validate_clinical_pattern, 
    apply_clinical_constraints, get_group_pairs, 
    get_functionally_important_triplets
)
from clinical_patterns import ClinicalPatternGenerator, get_all_clinical_patterns

class MusclTemplateGenerator:
    """Main class for generating comprehensive muscle parameter templates."""
    
    def __init__(self, output_dir: str = "muscle_params/v7", random_seed: int = 42, severe_population: bool = True):
        """
        Initialize the template generator.
        
        Args:
            output_dir: Directory to save generated templates
            random_seed: Random seed for reproducible generation
            severe_population: Whether to optimize for severe mobility limitations (default: True)
        """
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.severe_population = severe_population
        self.clinical_generator = ClinicalPatternGenerator(random_seed)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track generated templates
        self.templates = []
        self.template_metadata = []
        
        # Set random seed
        np.random.seed(random_seed)
    
    def _get_compensation_muscles(self, primary_group: str) -> List[str]:
        """Get muscles that typically compensate when primary group is weak."""
        compensation_map = {
            'deltoid': ['trap_scap_m_r', 'ser_ant_m_r'],  # Scapular compensation
            'rotator_cuff': ['delt_scap_m_r'],  # Deltoid compensation
            'elbow': ['lat_dors_m_r', 'pect_maj_thorax_m_r'],  # Shoulder compensation for elbow
            'scapular': ['delt_clav_a_r'],  # Some deltoid for scapular loss
            'large_movers': ['delt_scap_m_r', 'trap_scap_m_r'],  # Deltoid and scapular
            'wrist': ['brachiorad_r'],  # Brachioradialis for wrist
            'forearm': ['bic_long_r'],  # Biceps for forearm function
            'small_shoulder': []  # No major compensation
        }
        return compensation_map.get(primary_group, [])
    
    def generate_tier1_single_group_templates(self) -> List[Dict]:
        """
        Generate Tier 1: Single group primary weakness templates.
        8 muscle groups × 10 templates each = 80 total
        """
        print("Generating Tier 1: Single Group Primary Weakness Templates (80 templates)")
        
        templates = []
        template_id = 1
        
        for group_name, muscles in MUSCLE_GROUPS.items():
            print(f"  Generating {group_name} group templates...")
            
            # 4 severe templates
            for i in range(4):
                if self.severe_population:
                    # For severe population: background severe weakness, target extremely severe
                    template = {muscle: generate_severe_population_value('default', seed=template_id*100+i) 
                               for muscle in ALL_MUSCLES}
                    
                    # Make target group extremely severely weak
                    for muscle in muscles:
                        template[muscle] = generate_severe_population_value('affected', seed=template_id*100+i+50)
                        
                    # Identify and strengthen compensation muscles
                    compensation_muscles = self._get_compensation_muscles(group_name)
                    for muscle in compensation_muscles:
                        if muscle in template:
                            template[muscle] = generate_severe_population_value('compensation', seed=template_id*100+i+75)
                else:
                    # Original behavior for general population
                    template = {muscle: generate_random_value('normal', seed=template_id*100+i) 
                               for muscle in ALL_MUSCLES}
                    
                    # Make target group severely weak
                    for muscle in muscles:
                        template[muscle] = generate_random_value('severe', seed=template_id*100+i+50)
                
                # Apply clinical constraints
                template = apply_clinical_constraints(template, 'antagonist_balance')
                template = apply_clinical_constraints(template, 'minimal_function')
                
                # Validate
                is_valid, reason = validate_clinical_pattern(template)
                if not is_valid:
                    print(f"    Warning: Template {template_id:03d} failed validation: {reason}")
                
                templates.append(template)
                self.template_metadata.append({
                    'template_id': f'template_{template_id:03d}',
                    'tier': 1,
                    'type': 'single_group_severe',
                    'target_group': group_name,
                    'severity': 'severe',
                    'description': f'{group_name} severe weakness',
                    'valid': is_valid
                })
                template_id += 1
            
            # 3 moderate templates
            for i in range(3):
                if self.severe_population:
                    # For severe population: background severe weakness, target very severe
                    template = {muscle: generate_severe_population_value('default', seed=template_id*100+i) 
                               for muscle in ALL_MUSCLES}
                    
                    # Make target group very severely weak
                    for muscle in muscles:
                        template[muscle] = generate_severe_population_value('affected', seed=template_id*100+i+50)
                        
                    # Strengthen compensation muscles
                    compensation_muscles = self._get_compensation_muscles(group_name)
                    for muscle in compensation_muscles:
                        if muscle in template:
                            template[muscle] = generate_severe_population_value('compensation', seed=template_id*100+i+75)
                else:
                    template = {muscle: generate_random_value('normal', seed=template_id*100+i) 
                               for muscle in ALL_MUSCLES}
                    
                    # Make target group moderately weak
                    for muscle in muscles:
                        template[muscle] = generate_random_value('moderate', seed=template_id*100+i+50)
                
                # Apply clinical constraints
                template = apply_clinical_constraints(template, 'synergy_consistency')
                template = apply_clinical_constraints(template, 'minimal_function')
                
                # Validate
                is_valid, reason = validate_clinical_pattern(template)
                if not is_valid:
                    print(f"    Warning: Template {template_id:03d} failed validation: {reason}")
                
                templates.append(template)
                self.template_metadata.append({
                    'template_id': f'template_{template_id:03d}',
                    'tier': 1,
                    'type': 'single_group_moderate',
                    'target_group': group_name,
                    'severity': 'moderate',
                    'description': f'{group_name} moderate weakness',
                    'valid': is_valid
                })
                template_id += 1
            
            # 3 combination templates (mixed severities within group)
            for i in range(3):
                template = {muscle: generate_random_value('normal', seed=template_id*100+i) 
                           for muscle in ALL_MUSCLES}
                
                # Randomly assign different severities within the group
                for j, muscle in enumerate(muscles):
                    if j % 2 == 0:
                        template[muscle] = generate_random_value('severe', seed=template_id*100+i+j)
                    else:
                        template[muscle] = generate_random_value('moderate', seed=template_id*100+i+j+25)
                
                # Apply clinical constraints
                template = apply_clinical_constraints(template, 'synergy_consistency')
                
                # Validate
                is_valid, reason = validate_clinical_pattern(template)
                if not is_valid:
                    print(f"    Warning: Template {template_id:03d} failed validation: {reason}")
                
                templates.append(template)
                self.template_metadata.append({
                    'template_id': f'template_{template_id:03d}',
                    'tier': 1,
                    'type': 'single_group_mixed',
                    'target_group': group_name,
                    'severity': 'mixed',
                    'description': f'{group_name} mixed severity weakness',
                    'valid': is_valid
                })
                template_id += 1
        
        print(f"  Generated {len(templates)} Tier 1 templates")
        return templates
    
    def generate_tier2_pairwise_templates(self) -> List[Dict]:
        """
        Generate Tier 2: Pairwise muscle group combinations.
        28 pairs × 3-4 templates each = 96 total
        """
        print("Generating Tier 2: Pairwise Combinations (96 templates)")
        
        templates = []
        template_id = 81  # Continue from Tier 1
        
        pairs = get_group_pairs()
        templates_per_pair = [3, 4, 3, 4]  # Cycle through 3-4 templates per pair
        
        for pair_idx, (group1, group2) in enumerate(pairs):
            n_templates = templates_per_pair[pair_idx % len(templates_per_pair)]
            print(f"  Generating {group1} + {group2} pair templates ({n_templates} templates)...")
            
            severity_combos = [
                ('severe', 'severe'),
                ('severe', 'moderate'),
                ('moderate', 'moderate')
            ]
            if n_templates == 4:
                severity_combos.append(('severe', 'normal'))
            
            for i, (sev1, sev2) in enumerate(severity_combos[:n_templates]):
                template = {muscle: generate_random_value('normal', seed=template_id*100+i) 
                           for muscle in ALL_MUSCLES}
                
                # Apply severities to both groups
                for muscle in MUSCLE_GROUPS[group1]:
                    if sev1 != 'normal':
                        template[muscle] = generate_random_value(sev1, seed=template_id*100+i+10)
                
                for muscle in MUSCLE_GROUPS[group2]:
                    if sev2 != 'normal':
                        template[muscle] = generate_random_value(sev2, seed=template_id*100+i+20)
                
                # Apply clinical constraints
                template = apply_clinical_constraints(template, 'antagonist_balance')
                template = apply_clinical_constraints(template, 'minimal_function')
                
                # Validate
                is_valid, reason = validate_clinical_pattern(template)
                if not is_valid:
                    print(f"    Warning: Template {template_id:03d} failed validation: {reason}")
                
                templates.append(template)
                self.template_metadata.append({
                    'template_id': f'template_{template_id:03d}',
                    'tier': 2,
                    'type': 'pairwise_combination',
                    'target_groups': f'{group1}+{group2}',
                    'severity': f'{sev1}+{sev2}',
                    'description': f'{group1} {sev1}, {group2} {sev2}',
                    'valid': is_valid
                })
                template_id += 1
        
        print(f"  Generated {len(templates)} Tier 2 templates")
        return templates
    
    def generate_tier3_clinical_templates(self) -> List[Dict]:
        """
        Generate Tier 3: Clinical syndrome patterns.
        50 templates based on real clinical conditions
        """
        print("Generating Tier 3: Clinical Syndrome Patterns (50 templates)")
        
        templates = []
        template_id = 177  # Continue from Tier 2
        
        # Get all clinical patterns
        clinical_patterns = get_all_clinical_patterns()
        
        # Select 50 most important patterns
        selected_patterns = clinical_patterns[:50]  # Take first 50
        
        for pattern_name, params in selected_patterns:
            print(f"  Generating {pattern_name}...")
            
            # Generate pattern based on type
            pattern_type = params['type']
            
            if pattern_type == 'stroke_flexor_synergy':
                template = self.clinical_generator.generate_stroke_flexor_synergy(params['severity'])
            elif pattern_type == 'stroke_extensor_synergy':
                template = self.clinical_generator.generate_stroke_extensor_synergy(params['severity'])
            elif pattern_type == 'sci_pattern':
                template = self.clinical_generator.generate_sci_pattern(params['level'], params['completeness'])
            elif pattern_type == 'rotator_cuff_tear':
                template = self.clinical_generator.generate_rotator_cuff_tear(params['tear_type'])
            elif pattern_type == 'frozen_shoulder':
                template = self.clinical_generator.generate_frozen_shoulder(params['stage'])
            elif pattern_type == 'brachial_plexus_injury':
                template = self.clinical_generator.generate_brachial_plexus_injury(params['injury_type'])
            elif pattern_type == 'cerebral_palsy_spastic':
                template = self.clinical_generator.generate_cerebral_palsy_spastic(params['severity'])
            elif pattern_type == 'cerebral_palsy_hypotonic':
                template = self.clinical_generator.generate_cerebral_palsy_hypotonic(params['severity'])
            elif pattern_type == 'multiple_sclerosis':
                template = self.clinical_generator.generate_multiple_sclerosis(params['pattern_type'])
            elif pattern_type == 'parkinsons_disease':
                template = self.clinical_generator.generate_parkinsons_disease(params['severity'])
            elif pattern_type == 'nerve_injury_pattern':
                template = self.clinical_generator.generate_nerve_injury_pattern(params['nerve'])
            else:
                # Default to normal template
                template = {muscle: generate_random_value('normal') for muscle in ALL_MUSCLES}
            
            # Validate
            is_valid, reason = validate_clinical_pattern(template)
            if not is_valid:
                print(f"    Warning: Template {template_id:03d} failed validation: {reason}")
                # Try to fix the template
                template = apply_clinical_constraints(template, 'minimal_function')
                is_valid, reason = validate_clinical_pattern(template)
            
            templates.append(template)
            self.template_metadata.append({
                'template_id': f'template_{template_id:03d}',
                'tier': 3,
                'type': 'clinical_syndrome',
                'pattern_name': pattern_name,
                'severity': params.get('severity', 'variable'),
                'description': f'Clinical pattern: {pattern_name}',
                'valid': is_valid
            })
            template_id += 1
        
        print(f"  Generated {len(templates)} Tier 3 templates")
        return templates
    
    def generate_tier4_complex_templates(self) -> List[Dict]:
        """
        Generate Tier 4: Complex multi-group patterns.
        24 templates for 3+ group combinations and edge cases
        """
        print("Generating Tier 4: Complex Multi-Group Patterns (24 templates)")
        
        templates = []
        template_id = 227  # Continue from Tier 3
        
        # 12 three-group combinations
        triplets = get_functionally_important_triplets()
        
        for i, (group1, group2, group3) in enumerate(triplets):
            print(f"  Generating {group1}+{group2}+{group3} combination...")
            
            template = {muscle: generate_random_value('normal', seed=template_id*100) 
                       for muscle in ALL_MUSCLES}
            
            # Randomly assign severities to the three groups
            severities = ['severe', 'moderate', 'normal']
            np.random.shuffle(severities)
            
            for muscle in MUSCLE_GROUPS[group1]:
                if severities[0] != 'normal':
                    template[muscle] = generate_random_value(severities[0], seed=template_id*100+10)
            
            for muscle in MUSCLE_GROUPS[group2]:
                if severities[1] != 'normal':
                    template[muscle] = generate_random_value(severities[1], seed=template_id*100+20)
                    
            for muscle in MUSCLE_GROUPS[group3]:
                if severities[2] != 'normal':
                    template[muscle] = generate_random_value(severities[2], seed=template_id*100+30)
            
            # Apply clinical constraints
            template = apply_clinical_constraints(template, 'minimal_function')
            
            # Validate
            is_valid, reason = validate_clinical_pattern(template)
            if not is_valid:
                print(f"    Warning: Template {template_id:03d} failed validation: {reason}")
            
            templates.append(template)
            self.template_metadata.append({
                'template_id': f'template_{template_id:03d}',
                'tier': 4,
                'type': 'three_group_combination',
                'target_groups': f'{group1}+{group2}+{group3}',
                'severity': f'{severities[0]}+{severities[1]}+{severities[2]}',
                'description': f'Three-group: {group1} {severities[0]}, {group2} {severities[1]}, {group3} {severities[2]}',
                'valid': is_valid
            })
            template_id += 1
        
        # 6 global patterns
        global_patterns = [
            ('proximal_weakness', ['scapular', 'rotator_cuff', 'deltoid']),
            ('distal_weakness', ['elbow', 'forearm', 'wrist']),
            ('power_weakness', ['large_movers', 'deltoid']),
            ('stability_weakness', ['scapular', 'rotator_cuff']),
            ('fine_motor_weakness', ['forearm', 'wrist']),
            ('mixed_pattern', ['scapular', 'elbow', 'wrist'])
        ]
        
        for pattern_name, affected_groups in global_patterns:
            print(f"  Generating {pattern_name}...")
            
            template = {muscle: generate_random_value('normal', seed=template_id*100) 
                       for muscle in ALL_MUSCLES}
            
            # Apply weakness to affected groups
            for group in affected_groups:
                severity = np.random.choice(['moderate', 'severe'], p=[0.6, 0.4])
                for muscle in MUSCLE_GROUPS[group]:
                    template[muscle] = generate_random_value(severity, seed=template_id*100+hash(group)%1000)
            
            # Apply clinical constraints
            template = apply_clinical_constraints(template, 'minimal_function')
            
            # Validate
            is_valid, reason = validate_clinical_pattern(template)
            if not is_valid:
                print(f"    Warning: Template {template_id:03d} failed validation: {reason}")
            
            templates.append(template)
            self.template_metadata.append({
                'template_id': f'template_{template_id:03d}',
                'tier': 4,
                'type': 'global_pattern',
                'pattern_name': pattern_name,
                'target_groups': '+'.join(affected_groups),
                'severity': 'variable',
                'description': f'Global pattern: {pattern_name}',
                'valid': is_valid
            })
            template_id += 1
        
        # 6 edge case patterns
        edge_cases = [
            'single_muscle_deltoid_anterior',
            'single_muscle_biceps_long', 
            'single_muscle_triceps_long',
            'single_muscle_supraspinatus',
            'compensation_scapular_for_deltoid',
            'extreme_proximal_sparing_distal'
        ]
        
        for case_name in edge_cases:
            print(f"  Generating edge case: {case_name}...")
            
            template = {muscle: generate_random_value('normal', seed=template_id*100) 
                       for muscle in ALL_MUSCLES}
            
            if 'single_muscle' in case_name:
                # Single muscle severely affected
                target_muscle = case_name.split('_')[-2] + '_' + case_name.split('_')[-1] + '_r'
                if target_muscle in template:
                    template[target_muscle] = generate_random_value('severe')
                    
            elif 'compensation' in case_name:
                # Compensation pattern
                template = self.clinical_generator.generate_compensation_pattern('deltoid', 'proximal')
                
            elif 'extreme_proximal' in case_name:
                # Extreme proximal weakness, distal sparing
                proximal_groups = ['scapular', 'rotator_cuff', 'deltoid', 'large_movers']
                for group in proximal_groups:
                    for muscle in MUSCLE_GROUPS[group]:
                        template[muscle] = generate_random_value('severe')
            
            # Validate
            is_valid, reason = validate_clinical_pattern(template)
            if not is_valid:
                print(f"    Warning: Template {template_id:03d} failed validation: {reason}")
            
            templates.append(template)
            self.template_metadata.append({
                'template_id': f'template_{template_id:03d}',
                'tier': 4,
                'type': 'edge_case',
                'pattern_name': case_name,
                'severity': 'variable',
                'description': f'Edge case: {case_name}',
                'valid': is_valid
            })
            template_id += 1
        
        print(f"  Generated {len(templates)} Tier 4 templates")
        return templates
    
    def save_template_txt(self, template: Dict[str, float], template_id: int):
        """Save a template as a .txt file in SCONE format."""
        filename = self.output_dir / f"template_{template_id:03d}.txt"
        
        with open(filename, 'w') as f:
            for muscle in ALL_MUSCLES:
                value = template.get(muscle, 1.0)
                f.write(f"{muscle} {{max_isometric_force.factor = {value:.6f}}}\n")
    
    def save_summary_csv(self):
        """Save a summary CSV file with metadata for all templates."""
        filename = self.output_dir / "template_summary.csv"
        
        # Get all possible fieldnames from metadata
        all_fieldnames = set()
        for meta in self.template_metadata:
            all_fieldnames.update(meta.keys())
        
        fieldnames = ['template_id', 'tier', 'type', 'target_groups', 'target_group', 'pattern_name', 'severity', 'description', 'valid']
        # Add any other fields that might be present
        for field in all_fieldnames:
            if field not in fieldnames:
                fieldnames.append(field)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.template_metadata)
    
    def generate_all_templates(self):
        """Generate all 250 templates using the 4-tier approach."""
        print("=" * 60)
        print("MUSCLE PARAMETER TEMPLATE GENERATION")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Random seed: {self.random_seed}")
        print()
        
        # Generate all tiers
        all_templates = []
        
        # Tier 1: Single group patterns (80 templates)
        tier1_templates = self.generate_tier1_single_group_templates()
        all_templates.extend(tier1_templates)
        
        # Tier 2: Pairwise combinations (96 templates)  
        tier2_templates = self.generate_tier2_pairwise_templates()
        all_templates.extend(tier2_templates)
        
        # Tier 3: Clinical patterns (50 templates)
        tier3_templates = self.generate_tier3_clinical_templates()
        all_templates.extend(tier3_templates)
        
        # Tier 4: Complex patterns (24 templates)
        tier4_templates = self.generate_tier4_complex_templates()
        all_templates.extend(tier4_templates)
        
        # Shuffle templates for random sequence while preserving metadata alignment
        print("\nShuffling templates for random sequence...")
        template_pairs = list(zip(all_templates, self.template_metadata))
        np.random.shuffle(template_pairs)
        shuffled_templates, shuffled_metadata = zip(*template_pairs)
        
        # Update metadata with new template IDs
        self.template_metadata = []
        for i, meta in enumerate(shuffled_metadata, 1):
            updated_meta = meta.copy()
            updated_meta['template_id'] = f"template_{i:03d}"
            self.template_metadata.append(updated_meta)
        
        # Save all templates
        print("Saving templates...")
        for i, template in enumerate(shuffled_templates, 1):
            self.save_template_txt(template, i)
            self.templates.append(template)
        
        # Save summary
        self.save_summary_csv()
        
        # Final summary
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total templates generated: {len(all_templates)}")
        print(f"Tier 1 (Single group): {len(tier1_templates)}")
        print(f"Tier 2 (Pairwise): {len(tier2_templates)}")
        print(f"Tier 3 (Clinical): {len(tier3_templates)}")
        print(f"Tier 4 (Complex): {len(tier4_templates)}")
        
        # Validation summary
        valid_count = sum(1 for meta in self.template_metadata if meta['valid'])
        print(f"\nValidation: {valid_count}/{len(all_templates)} templates passed clinical validation")
        
        print(f"\nFiles created:")
        print(f"  Template files: {self.output_dir}/template_001.txt to template_{len(all_templates):03d}.txt")
        print(f"  Summary file: {self.output_dir}/template_summary.csv")
    
    def analyze_parameter_distribution(self):
        """Analyze the actual parameter distribution achieved."""
        if not self.templates:
            print("No templates to analyze")
            return
            
        all_values = []
        for template in self.templates:
            all_values.extend(template.values())
        
        # Count by severity levels
        severe_count = sum(1 for v in all_values if 0.1 <= v <= 0.3)
        moderate_count = sum(1 for v in all_values if 0.3 <= v <= 0.6)
        mild_count = sum(1 for v in all_values if 0.6 <= v <= 0.8)
        normal_count = sum(1 for v in all_values if 0.8 <= v <= 1.0)
        
        total_values = len(all_values)
        
        print(f"\n" + "=" * 60)
        print("PARAMETER DISTRIBUTION ANALYSIS")
        print("=" * 60)
        print(f"Population target: {'Severe mobility limitations' if self.severe_population else 'General population'}")
        print(f"Total parameters: {total_values}")
        print(f"Severe (0.1-0.3):   {severe_count:4d} ({severe_count/total_values*100:5.1f}%)")
        print(f"Moderate (0.3-0.6): {moderate_count:4d} ({moderate_count/total_values*100:5.1f}%)")
        print(f"Mild (0.6-0.8):     {mild_count:4d} ({mild_count/total_values*100:5.1f}%)")
        print(f"Normal (0.8-1.0):   {normal_count:4d} ({normal_count/total_values*100:5.1f}%)")
        
        if self.severe_population:
            print(f"\nTarget for severe mobility:")
            print(f"  Severe: 50%, Moderate: 35%, Mild: 15%, Normal: 5%")
        
        # Save distribution analysis
        distribution_data = {
            'total_parameters': total_values,
            'severe_count': severe_count,
            'moderate_count': moderate_count,
            'mild_count': mild_count,
            'normal_count': normal_count,
            'severe_percentage': severe_count/total_values*100,
            'moderate_percentage': moderate_count/total_values*100,
            'mild_percentage': mild_count/total_values*100,
            'normal_percentage': normal_count/total_values*100,
            'population_target': 'severe_mobility' if self.severe_population else 'general'
        }
        
        with open(self.output_dir / 'distribution_analysis.json', 'w') as f:
            json.dump(distribution_data, f, indent=2)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Generate comprehensive muscle parameter templates')
    parser.add_argument('--output-dir', default='muscle_params/v7', 
                       help='Output directory for templates (default: muscle_params/v7)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible generation (default: 42)')
    parser.add_argument('--general-population', action='store_true',
                       help='Generate for general population instead of severe mobility limitations')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing templates')
    
    args = parser.parse_args()
    
    if args.validate_only:
        # TODO: Implement validation-only mode
        print("Validation-only mode not yet implemented")
        return
    
    # Generate templates (default: severe population)
    severe_population = not args.general_population
    generator = MusclTemplateGenerator(args.output_dir, args.seed, severe_population)
    generator.generate_all_templates()
    
    # Analyze distribution
    generator.analyze_parameter_distribution()

if __name__ == "__main__":
    main()