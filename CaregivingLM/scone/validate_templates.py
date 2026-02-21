#!/usr/bin/env python3
"""
Template Validation Tool

This script provides comprehensive validation of muscle parameter templates
to ensure clinical plausibility and identify potential issues.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional

from muscle_anatomy_config import (
    ALL_MUSCLES, MUSCLE_GROUPS, SYNERGY_GROUPS, ANTAGONIST_PAIRS,
    NERVE_INNERVATION, SCI_LEVELS, validate_clinical_pattern,
    get_synergy_muscles, get_antagonist_muscles
)

class TemplateValidator:
    """Comprehensive validator for muscle parameter templates."""
    
    def __init__(self, template_dir: str = "muscle_params/v4"):
        """
        Initialize the validator.
        
        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = Path(template_dir)
        self.templates = {}
        self.metadata = None
        self.validation_results = []
        
        self.load_templates()
        self.load_metadata()
    
    def load_templates(self):
        """Load all template files."""
        print("Loading templates for validation...")
        
        template_files = list(self.template_dir.glob("template_*.txt"))
        template_files = [f for f in template_files if f.name != "template_summary.csv"]
        
        for template_file in sorted(template_files):
            template_id = template_file.stem
            
            muscle_values = {}
            try:
                with open(template_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and '{' in line:
                            parts = line.split('{')
                            muscle_name = parts[0].strip()
                            value_part = parts[1].split('=')[1].split('}')[0].strip()
                            value = float(value_part)
                            muscle_values[muscle_name] = value
                
                self.templates[template_id] = muscle_values
                
            except Exception as e:
                print(f"Error loading {template_file}: {e}")
        
        print(f"Loaded {len(self.templates)} templates")
    
    def load_metadata(self):
        """Load template metadata."""
        metadata_file = self.template_dir / "template_summary.csv"
        
        if metadata_file.exists():
            self.metadata = pd.read_csv(metadata_file)
            print(f"Loaded metadata for {len(self.metadata)} templates")
        else:
            print("Warning: No metadata file found")
    
    def validate_basic_constraints(self, template_id: str, muscle_values: Dict[str, float]) -> List[str]:
        """Validate basic constraints for a template."""
        issues = []
        
        # Check value ranges
        for muscle, value in muscle_values.items():
            if not (0.0 <= value <= 1.0):
                issues.append(f"Invalid value {value:.3f} for muscle {muscle}")
        
        # Check for missing muscles
        missing_muscles = set(ALL_MUSCLES) - set(muscle_values.keys())
        if missing_muscles:
            issues.append(f"Missing muscles: {list(missing_muscles)[:5]}...")  # Show first 5
        
        # Check for extra muscles
        extra_muscles = set(muscle_values.keys()) - set(ALL_MUSCLES)
        if extra_muscles:
            issues.append(f"Unknown muscles: {list(extra_muscles)[:5]}...")
        
        return issues
    
    def validate_synergy_consistency(self, template_id: str, muscle_values: Dict[str, float]) -> List[str]:
        """Validate synergy group consistency."""
        issues = []
        
        for group_name, muscles in SYNERGY_GROUPS.items():
            group_values = [muscle_values.get(m, 1.0) for m in muscles if m in muscle_values]
            
            if len(group_values) > 1:
                min_val = min(group_values)
                max_val = max(group_values)
                
                # Check for extreme differences within synergy groups
                if max_val - min_val > 0.7:
                    issues.append(f"Synergy group '{group_name}' has extreme value range: {min_val:.3f} to {max_val:.3f}")
                
                # Check for implausible patterns (all very weak)
                if all(v < 0.2 for v in group_values) and len(group_values) > 2:
                    issues.append(f"Synergy group '{group_name}' has all muscles severely weak (< 0.2)")
        
        return issues
    
    def validate_antagonist_relationships(self, template_id: str, muscle_values: Dict[str, float]) -> List[str]:
        """Validate antagonist muscle relationships."""
        issues = []
        
        for pair in ANTAGONIST_PAIRS:
            group1_name, group2_name = pair
            
            if group1_name in SYNERGY_GROUPS and group2_name in SYNERGY_GROUPS:
                group1_muscles = SYNERGY_GROUPS[group1_name]
                group2_muscles = SYNERGY_GROUPS[group2_name]
                
                # Calculate average strength for each group
                group1_values = [muscle_values.get(m, 1.0) for m in group1_muscles if m in muscle_values]
                group2_values = [muscle_values.get(m, 1.0) for m in group2_muscles if m in muscle_values]
                
                if group1_values and group2_values:
                    group1_avg = np.mean(group1_values)
                    group2_avg = np.mean(group2_values)
                    
                    # Both antagonist groups very weak is suspicious
                    if group1_avg < 0.3 and group2_avg < 0.3:
                        issues.append(f"Both antagonist groups '{group1_name}' and '{group2_name}' are very weak")
        
        return issues
    
    def validate_functional_requirements(self, template_id: str, muscle_values: Dict[str, float]) -> List[str]:
        """Validate functional requirements for basic movement."""
        issues = []
        
        # Check for minimal scapular stability
        scapular_muscles = MUSCLE_GROUPS['scapular']
        scapular_values = [muscle_values.get(m, 1.0) for m in scapular_muscles]
        
        if all(v < 0.3 for v in scapular_values):
            issues.append("Complete scapular stabilizer loss - functionally implausible")
        
        # Check for minimal elbow function
        elbow_flexors = ['bic_long_r', 'bic_brev_r', 'brach_r']
        flexor_values = [muscle_values.get(m, 1.0) for m in elbow_flexors if m in muscle_values]
        
        if all(v < 0.2 for v in flexor_values):
            issues.append("Complete elbow flexor loss - extremely rare clinically")
        
        # Check for balanced shoulder function
        deltoid_muscles = MUSCLE_GROUPS['deltoid']
        deltoid_values = [muscle_values.get(m, 1.0) for m in deltoid_muscles]
        
        rotator_cuff_muscles = MUSCLE_GROUPS['rotator_cuff']
        rc_values = [muscle_values.get(m, 1.0) for m in rotator_cuff_muscles]
        
        deltoid_avg = np.mean(deltoid_values)
        rc_avg = np.mean(rc_values)
        
        # Deltoid very strong but rotator cuff very weak is problematic
        if deltoid_avg > 0.8 and rc_avg < 0.2:
            issues.append("Strong deltoid with very weak rotator cuff - biomechanically unstable")
        
        return issues
    
    def validate_neurological_patterns(self, template_id: str, muscle_values: Dict[str, float]) -> List[str]:
        """Validate neurological innervation patterns."""
        issues = []
        
        # Check nerve-specific patterns
        for nerve, muscles in NERVE_INNERVATION.items():
            nerve_values = [muscle_values.get(m, 1.0) for m in muscles if m in muscle_values]
            
            if nerve_values:
                # If most muscles from one nerve are weak, check if pattern is consistent
                weak_count = sum(1 for v in nerve_values if v < 0.5)
                total_count = len(nerve_values)
                
                # Partial nerve injuries should be rare - mostly all or none
                if 0.2 < weak_count / total_count < 0.8 and total_count > 2:
                    issues.append(f"Partial {nerve} nerve pattern - check if clinically intended")
        
        return issues
    
    def validate_clinical_plausibility(self, template_id: str, muscle_values: Dict[str, float]) -> List[str]:
        """Overall clinical plausibility check."""
        issues = []
        
        # Count severely affected muscles
        severe_count = sum(1 for v in muscle_values.values() if v < 0.4)
        total_count = len(muscle_values)
        
        # Too many muscles severely affected
        if severe_count > 30:  # More than 75% severely weak
            issues.append(f"Too many muscles severely weak ({severe_count}/{total_count}) - check clinical realism")
        
        # Check for isolated single muscle deficits
        weak_muscles = [m for m, v in muscle_values.items() if v < 0.4]
        normal_muscles = [m for m, v in muscle_values.items() if v > 0.8]
        
        if len(weak_muscles) == 1 and len(normal_muscles) > 35:
            # Single muscle deficit - note for review
            issues.append(f"Single muscle deficit pattern: {weak_muscles[0]} - confirm if intended")
        
        return issues
    
    def validate_template(self, template_id: str) -> Dict:
        """Run comprehensive validation on a single template."""
        if template_id not in self.templates:
            return {
                'template_id': template_id,
                'valid': False,
                'issues': ['Template not found'],
                'severity': 'critical'
            }
        
        muscle_values = self.templates[template_id]
        all_issues = []
        
        # Run all validation checks
        all_issues.extend(self.validate_basic_constraints(template_id, muscle_values))
        all_issues.extend(self.validate_synergy_consistency(template_id, muscle_values))
        all_issues.extend(self.validate_antagonist_relationships(template_id, muscle_values))
        all_issues.extend(self.validate_functional_requirements(template_id, muscle_values))
        all_issues.extend(self.validate_neurological_patterns(template_id, muscle_values))
        all_issues.extend(self.validate_clinical_plausibility(template_id, muscle_values))
        
        # Use the original validation function as well
        is_valid, reason = validate_clinical_pattern(muscle_values)
        if not is_valid:
            all_issues.append(f"Clinical pattern validation: {reason}")
        
        # Determine severity
        severity = 'pass'
        if any('Invalid value' in issue or 'Missing muscles' in issue for issue in all_issues):
            severity = 'critical'
        elif any('severely weak' in issue or 'functionally implausible' in issue for issue in all_issues):
            severity = 'major'
        elif any('extreme' in issue or 'suspicious' in issue for issue in all_issues):
            severity = 'minor'
        elif all_issues:
            severity = 'warning'
        
        return {
            'template_id': template_id,
            'valid': len(all_issues) == 0,
            'issues': all_issues,
            'issue_count': len(all_issues),
            'severity': severity
        }
    
    def validate_all_templates(self) -> Dict:
        """Validate all templates and generate comprehensive report."""
        print("Running comprehensive validation on all templates...")
        
        results = []
        severity_counts = {'critical': 0, 'major': 0, 'minor': 0, 'warning': 0, 'pass': 0}
        
        for template_id in sorted(self.templates.keys()):
            result = self.validate_template(template_id)
            results.append(result)
            severity_counts[result['severity']] += 1
            
            # Print progress for critical issues
            if result['severity'] == 'critical':
                print(f"CRITICAL: {template_id} - {result['issues'][0]}")
        
        self.validation_results = results
        
        # Generate summary
        total_templates = len(results)
        valid_templates = sum(1 for r in results if r['valid'])
        
        summary = {
            'total_templates': total_templates,
            'valid_templates': valid_templates,
            'invalid_templates': total_templates - valid_templates,
            'validation_rate': valid_templates / total_templates * 100 if total_templates > 0 else 0,
            'severity_distribution': severity_counts,
            'results': results
        }
        
        return summary
    
    def generate_validation_report(self, output_file: str = "validation_report.json"):
        """Generate detailed validation report."""
        summary = self.validate_all_templates()
        
        # Save full report
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate text summary
        text_file = output_file.replace('.json', '_summary.txt')
        with open(text_file, 'w') as f:
            f.write("MUSCLE PARAMETER TEMPLATE VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Templates: {summary['total_templates']}\n")
            f.write(f"Valid Templates: {summary['valid_templates']}\n")
            f.write(f"Invalid Templates: {summary['invalid_templates']}\n")
            f.write(f"Validation Rate: {summary['validation_rate']:.1f}%\n\n")
            
            f.write("Issue Severity Distribution:\n")
            for severity, count in summary['severity_distribution'].items():
                f.write(f"  {severity.title()}: {count}\n")
            f.write("\n")
            
            # List templates by severity
            for severity in ['critical', 'major', 'minor', 'warning']:
                templates_with_severity = [r for r in summary['results'] if r['severity'] == severity]
                
                if templates_with_severity:
                    f.write(f"{severity.upper()} ISSUES:\n")
                    for result in templates_with_severity[:10]:  # Show first 10
                        f.write(f"  {result['template_id']}: {result['issue_count']} issues\n")
                        for issue in result['issues'][:3]:  # Show first 3 issues
                            f.write(f"    - {issue}\n")
                    if len(templates_with_severity) > 10:
                        f.write(f"    ... and {len(templates_with_severity) - 10} more\n")
                    f.write("\n")
        
        return summary
    
    def fix_templates(self, severity_threshold: str = 'critical', output_dir: str = None):
        """Attempt to fix templates with issues."""
        if not self.validation_results:
            print("No validation results available. Run validation first.")
            return
        
        if output_dir is None:
            output_dir = self.template_dir / "fixed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        severity_order = ['critical', 'major', 'minor', 'warning']
        threshold_idx = severity_order.index(severity_threshold)
        
        fixed_count = 0
        for result in self.validation_results:
            if severity_order.index(result['severity']) <= threshold_idx and not result['valid']:
                template_id = result['template_id']
                muscle_values = self.templates[template_id].copy()
                
                # Apply basic fixes
                fixed = False
                
                # Fix value ranges
                for muscle, value in muscle_values.items():
                    if value < 0.0:
                        muscle_values[muscle] = 0.1
                        fixed = True
                    elif value > 1.0:
                        muscle_values[muscle] = 1.0
                        fixed = True
                
                # Add missing muscles with default values
                for muscle in ALL_MUSCLES:
                    if muscle not in muscle_values:
                        muscle_values[muscle] = 0.9
                        fixed = True
                
                # Apply clinical constraints
                if fixed:
                    from muscle_anatomy_config import apply_clinical_constraints
                    muscle_values = apply_clinical_constraints(muscle_values, 'minimal_function')
                    
                    # Save fixed template
                    output_file = output_dir / f"{template_id}_fixed.txt"
                    with open(output_file, 'w') as f:
                        for muscle in ALL_MUSCLES:
                            value = muscle_values.get(muscle, 1.0)
                            f.write(f"{muscle} {{max_isometric_force.factor = {value:.6f}}}\n")
                    
                    fixed_count += 1
        
        print(f"Fixed {fixed_count} templates and saved to {output_dir}")
    
    def compare_with_existing(self, existing_dir: str = "muscle_params/v2"):
        """Compare validation results with existing templates."""
        existing_path = Path(existing_dir)
        
        if not existing_path.exists():
            print(f"Existing template directory {existing_dir} not found")
            return
        
        # Load existing templates
        existing_templates = {}
        for template_file in existing_path.glob("template_*.txt"):
            template_id = template_file.stem
            muscle_values = {}
            
            try:
                with open(template_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and '{' in line:
                            parts = line.split('{')
                            muscle_name = parts[0].strip()
                            value_part = parts[1].split('=')[1].split('}')[0].strip()
                            value = float(value_part)
                            muscle_values[muscle_name] = value
                
                existing_templates[template_id] = muscle_values
            except Exception as e:
                continue
        
        # Validate existing templates
        print(f"Comparing with {len(existing_templates)} existing templates...")
        
        existing_validation = []
        for template_id, muscle_values in existing_templates.items():
            result = self.validate_template(template_id)
            existing_validation.append(result)
        
        # Compare results
        new_valid_rate = sum(1 for r in self.validation_results if r['valid']) / len(self.validation_results) * 100
        existing_valid_rate = sum(1 for r in existing_validation if r['valid']) / len(existing_validation) * 100
        
        print(f"\nValidation Comparison:")
        print(f"New templates (v4): {new_valid_rate:.1f}% valid ({len(self.validation_results)} total)")
        print(f"Existing templates (v2): {existing_valid_rate:.1f}% valid ({len(existing_validation)} total)")
        
        return {
            'new_validation_rate': new_valid_rate,
            'existing_validation_rate': existing_valid_rate,
            'new_results': self.validation_results,
            'existing_results': existing_validation
        }

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Validate muscle parameter templates')
    parser.add_argument('--template-dir', default='muscle_params/v4',
                       help='Directory containing templates to validate')
    parser.add_argument('--output-file', default='validation_report.json',
                       help='Output file for validation report')
    parser.add_argument('--fix', action='store_true',
                       help='Attempt to fix templates with critical issues')
    parser.add_argument('--fix-threshold', default='critical',
                       choices=['critical', 'major', 'minor', 'warning'],
                       help='Severity threshold for fixing templates')
    parser.add_argument('--compare', default=None,
                       help='Compare with existing templates in specified directory')
    
    args = parser.parse_args()
    
    # Create validator
    validator = TemplateValidator(args.template_dir)
    
    # Generate validation report
    print("Generating validation report...")
    summary = validator.generate_validation_report(args.output_file)
    
    print(f"\nValidation completed:")
    print(f"  Total templates: {summary['total_templates']}")
    print(f"  Valid templates: {summary['valid_templates']}")
    print(f"  Validation rate: {summary['validation_rate']:.1f}%")
    print(f"  Report saved to: {args.output_file}")
    
    # Fix templates if requested
    if args.fix:
        validator.fix_templates(args.fix_threshold)
    
    # Compare with existing if requested
    if args.compare:
        validator.compare_with_existing(args.compare)

if __name__ == "__main__":
    main()