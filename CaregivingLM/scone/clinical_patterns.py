#!/usr/bin/env python3
"""
Clinical Patterns for Muscle Parameter Template Generation

This module defines specific clinical patterns and syndromes for generating
realistic muscle parameter templates based on real patient conditions.
"""

import numpy as np
from typing import Dict, List, Tuple
from muscle_anatomy_config import (
    ALL_MUSCLES, MUSCLE_GROUPS, SYNERGY_GROUPS, NERVE_INNERVATION, 
    SCI_LEVELS, generate_random_value, apply_clinical_constraints,
    generate_severe_population_value
)

class ClinicalPatternGenerator:
    """Generates muscle parameter patterns based on clinical conditions."""
    
    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_stroke_flexor_synergy(self, severity: str = 'moderate') -> Dict[str, float]:
        """
        Generate stroke pattern with flexor synergy dominance.
        
        Characteristics:
        - Biceps, wrist flexors strengthened/spastic
        - Triceps, wrist extensors weakened
        - Shoulder adductors/internal rotators strong
        - Shoulder abductors/external rotators weak
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        # "Strengthen" flexor synergy muscles (but still weak for severe population)
        strengthened = [
            'bic_long_r', 'bic_brev_r',  # Elbow flexors
            'flex_carpi_rad_r', 'flex_carpi_ulna_r',  # Wrist flexors
            'pect_maj_thorax_m_r', 'lat_dors_m_r',  # Adductors
            'subscap_s_r', 'subscap_i_r'  # Internal rotators
        ]
        
        for muscle in strengthened:
            if muscle in pattern:
                # Even "strong" muscles match reference template pattern
                pattern[muscle] = generate_severe_population_value('compensation')
        
        # Weaken extensor muscles based on severity
        weakened = [
            'tri_long_r', 'tri_lat_r',  # Elbow extensors
            'ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r', 'ext_carpi_ulna_r',  # Wrist extensors
            'delt_scap_m_r',  # Shoulder abductor
            'infraspi_i_r', 'infraspi_s_r', 'teres_min_r'  # External rotators
        ]
        
        for muscle in weakened:
            if muscle in pattern:
                pattern[muscle] = generate_random_value(severity)
        
        return pattern
    
    def generate_stroke_extensor_synergy(self, severity: str = 'moderate') -> Dict[str, float]:
        """
        Generate stroke pattern with extensor synergy dominance.
        
        Characteristics:
        - Triceps, wrist extensors strengthened
        - Biceps, wrist flexors weakened
        - Shoulder adductors still strong (common in stroke)
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        # "Strengthen" extensor synergy muscles (but still weak for severe population)
        strengthened = [
            'tri_long_r', 'tri_lat_r',  # Elbow extensors
            'ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r', 'ext_carpi_ulna_r',  # Wrist extensors
            'pect_maj_thorax_m_r', 'lat_dors_m_r',  # Adductors (still strong)
        ]
        
        for muscle in strengthened:
            if muscle in pattern:
                # Even "strong" muscles match reference template pattern
                pattern[muscle] = generate_severe_population_value('compensation')
        
        # Weaken flexor muscles
        weakened = [
            'bic_long_r', 'bic_brev_r',  # Elbow flexors
            'flex_carpi_rad_r', 'flex_carpi_ulna_r',  # Wrist flexors
            'delt_scap_m_r',  # Abductor (still weak)
        ]
        
        for muscle in weakened:
            if muscle in pattern:
                pattern[muscle] = generate_random_value(severity)
        
        return pattern
    
    def generate_sci_pattern(self, level: str, completeness: str = 'incomplete') -> Dict[str, float]:
        """
        Generate spinal cord injury pattern based on injury level.
        
        Args:
            level: C4, C5, C6, C7, or C8
            completeness: 'complete' or 'incomplete'
        """
        pattern = {}
        
        if level not in SCI_LEVELS:
            raise ValueError(f"Unknown SCI level: {level}")
        
        level_info = SCI_LEVELS[level]
        
        # Preserved muscles are normal
        for muscle in level_info['preserved']:
            pattern[muscle] = generate_random_value('normal')
        
        # Impaired muscles depend on completeness
        if completeness == 'complete':
            # Complete injury - severe weakness
            for muscle in level_info['impaired']:
                pattern[muscle] = generate_random_value('severe')
        else:
            # Incomplete injury - variable weakness
            for muscle in level_info['impaired']:
                # Random mix of moderate and severe
                severity = np.random.choice(['moderate', 'severe'], p=[0.6, 0.4])
                pattern[muscle] = generate_random_value(severity)
        
        # Fill in any missing muscles with normal values
        for muscle in ALL_MUSCLES:
            if muscle not in pattern:
                pattern[muscle] = generate_random_value('normal')
        
        return pattern
    
    def generate_rotator_cuff_tear(self, tear_type: str = 'massive') -> Dict[str, float]:
        """
        Generate rotator cuff tear pattern.
        
        Args:
            tear_type: 'isolated_supraspinatus', 'massive', 'subscapularis'
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        if tear_type == 'isolated_supraspinatus':
            # Only supraspinatus affected
            pattern['supraspi_r'] = generate_random_value('severe')
            # Slight compensation from deltoid
            pattern['delt_scap_m_r'] = np.random.uniform(0.9, 1.0)
            
        elif tear_type == 'massive':
            # Multiple rotator cuff muscles affected
            affected = ['supraspi_r', 'infraspi_i_r', 'infraspi_s_r', 'teres_min_r']
            for muscle in affected:
                pattern[muscle] = generate_random_value('severe')
            
            # Deltoid compensation
            for muscle in MUSCLE_GROUPS['deltoid']:
                pattern[muscle] = np.random.uniform(0.9, 1.0)
            
            # Some scapular instability
            unstable_scapular = ['ser_ant_m_r', 'ser_ant_i_r']
            for muscle in unstable_scapular:
                pattern[muscle] = generate_random_value('moderate')
                
        elif tear_type == 'subscapularis':
            # Subscapularis specific tear
            pattern['subscap_s_r'] = generate_random_value('severe')
            pattern['subscap_i_r'] = generate_random_value('severe')
            
            # Compensation from other internal rotators
            pattern['lat_dors_m_r'] = np.random.uniform(0.9, 1.0)
            pattern['teres_maj_r'] = np.random.uniform(0.9, 1.0)
        
        return pattern
    
    def generate_frozen_shoulder(self, stage: str = 'freezing') -> Dict[str, float]:
        """
        Generate frozen shoulder (adhesive capsulitis) pattern.
        
        Args:
            stage: 'freezing', 'frozen', 'thawing'
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        if stage == 'freezing':
            # Early stage - pain causes guarding
            # Deltoid and rotator cuff inhibited
            inhibited = MUSCLE_GROUPS['deltoid'] + MUSCLE_GROUPS['rotator_cuff']
            for muscle in inhibited:
                pattern[muscle] = generate_random_value('moderate')
                
        elif stage == 'frozen':
            # Peak restriction - severe limitation
            # Most shoulder muscles significantly affected
            affected = MUSCLE_GROUPS['deltoid'] + MUSCLE_GROUPS['rotator_cuff'] + ['pect_maj_clav_s_r']
            for muscle in affected:
                pattern[muscle] = generate_random_value('severe')
                
        elif stage == 'thawing':
            # Recovery stage - gradual improvement
            # Moderate weakness persists
            affected = MUSCLE_GROUPS['deltoid'] + MUSCLE_GROUPS['rotator_cuff']
            for muscle in affected:
                pattern[muscle] = generate_random_value('moderate')
        
        return pattern
    
    def generate_brachial_plexus_injury(self, injury_type: str = 'upper_trunk') -> Dict[str, float]:
        """
        Generate brachial plexus injury pattern.
        
        Args:
            injury_type: 'upper_trunk' (C5-C6), 'lower_trunk' (C8-T1), 'total'
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        if injury_type == 'upper_trunk':
            # Erb's palsy - C5-C6 roots affected
            # Shoulder and elbow flexion affected
            affected_nerves = ['suprascapular', 'axillary', 'musculocutaneous']
            for nerve in affected_nerves:
                if nerve in NERVE_INNERVATION:
                    for muscle in NERVE_INNERVATION[nerve]:
                        if muscle in pattern:
                            pattern[muscle] = generate_random_value('severe')
                            
        elif injury_type == 'lower_trunk':
            # Klumpke's palsy - C8-T1 roots affected
            # Hand and wrist function affected
            affected_nerves = ['median', 'ulnar']
            for nerve in affected_nerves:
                if nerve in NERVE_INNERVATION:
                    for muscle in NERVE_INNERVATION[nerve]:
                        if muscle in pattern:
                            pattern[muscle] = generate_random_value('severe')
                            
        elif injury_type == 'total':
            # Complete plexus injury - all affected
            for muscle in ALL_MUSCLES:
                pattern[muscle] = generate_random_value('severe')
        
        return pattern
    
    def generate_cerebral_palsy_spastic(self, severity: str = 'moderate') -> Dict[str, float]:
        """
        Generate cerebral palsy with spasticity pattern.
        
        Characteristics:
        - Flexor muscles often spastic/overactive
        - Extensor muscles often weak
        - Proximal muscles may be less affected
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        # Spastic flexors (appear "strong" but poorly controlled)
        spastic_muscles = [
            'bic_long_r', 'bic_brev_r',  # Elbow flexors
            'flex_carpi_rad_r', 'flex_carpi_ulna_r',  # Wrist flexors
            'pect_maj_thorax_m_r'  # Chest muscle
        ]
        
        for muscle in spastic_muscles:
            if muscle in pattern:
                pattern[muscle] = np.random.uniform(0.8, 1.0)
        
        # Weak extensors
        weak_muscles = [
            'tri_long_r', 'tri_lat_r',  # Elbow extensors
            'ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r', 'ext_carpi_ulna_r'  # Wrist extensors
        ]
        
        for muscle in weak_muscles:
            if muscle in pattern:
                pattern[muscle] = generate_random_value(severity)
        
        return pattern
    
    def generate_cerebral_palsy_hypotonic(self, severity: str = 'moderate') -> Dict[str, float]:
        """
        Generate cerebral palsy with hypotonia pattern.
        
        Characteristics:
        - Generalized weakness, especially proximal
        - Poor postural control
        """
        pattern = {}
        
        # Proximal muscles more affected
        proximal_groups = ['scapular', 'rotator_cuff', 'deltoid', 'large_movers']
        for group in proximal_groups:
            for muscle in MUSCLE_GROUPS[group]:
                pattern[muscle] = generate_random_value(severity)
        
        # Distal muscles less affected but still impaired
        distal_severity = 'moderate' if severity == 'severe' else 'normal'
        distal_groups = ['elbow', 'forearm', 'wrist']
        for group in distal_groups:
            for muscle in MUSCLE_GROUPS[group]:
                pattern[muscle] = generate_random_value(distal_severity)
        
        return pattern
    
    def generate_multiple_sclerosis(self, pattern_type: str = 'proximal') -> Dict[str, float]:
        """
        Generate multiple sclerosis weakness pattern.
        
        Args:
            pattern_type: 'proximal', 'asymmetric', 'fatigue'
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        if pattern_type == 'proximal':
            # Proximal weakness common in MS
            proximal_groups = ['scapular', 'rotator_cuff', 'deltoid', 'large_movers']
            for group in proximal_groups:
                for muscle in MUSCLE_GROUPS[group]:
                    # Variable weakness
                    severity = np.random.choice(['moderate', 'severe'], p=[0.7, 0.3])
                    pattern[muscle] = generate_random_value(severity)
                    
        elif pattern_type == 'asymmetric':
            # Asymmetric pattern - randomly affect some muscle groups
            affected_groups = np.random.choice(list(MUSCLE_GROUPS.keys()), 
                                             size=np.random.randint(2, 5), 
                                             replace=False)
            for group in affected_groups:
                for muscle in MUSCLE_GROUPS[group]:
                    pattern[muscle] = generate_random_value('moderate')
        
        return pattern
    
    def generate_parkinsons_disease(self, severity: str = 'moderate') -> Dict[str, float]:
        """
        Generate Parkinson's disease pattern.
        
        Characteristics:
        - Rigidity rather than weakness
        - Proximal muscles more affected
        - Relatively preserved strength but poor control
        """
        pattern = {}
        
        # Proximal muscles affected by rigidity (modeled as moderate weakness)
        proximal_groups = ['scapular', 'rotator_cuff', 'deltoid']
        for group in proximal_groups:
            for muscle in MUSCLE_GROUPS[group]:
                # Rigidity modeled as reduced effective strength
                if severity == 'mild':
                    pattern[muscle] = np.random.uniform(0.7, 0.9)
                else:
                    pattern[muscle] = generate_random_value('moderate')
        
        # Distal muscles relatively preserved
        distal_groups = ['elbow', 'forearm', 'wrist']
        for group in distal_groups:
            for muscle in MUSCLE_GROUPS[group]:
                pattern[muscle] = generate_random_value('normal')
        
        # Large movers variably affected
        for muscle in MUSCLE_GROUPS['large_movers']:
            pattern[muscle] = np.random.uniform(0.6, 0.9)
        
        return pattern
    
    def generate_nerve_injury_pattern(self, nerve: str) -> Dict[str, float]:
        """
        Generate pattern for specific peripheral nerve injury.
        
        Args:
            nerve: Name of affected nerve
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        if nerve in NERVE_INNERVATION:
            # Muscles innervated by this nerve are severely affected
            for muscle in NERVE_INNERVATION[nerve]:
                if muscle in pattern:
                    pattern[muscle] = generate_random_value('severe')
        
        return pattern
    
    def generate_single_muscle_deficit(self, target_muscle: str) -> Dict[str, float]:
        """
        Generate pattern with single muscle severely affected.
        
        Args:
            target_muscle: Name of muscle to affect
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        if target_muscle in pattern:
            pattern[target_muscle] = generate_random_value('severe')
        
        return pattern
    
    def generate_compensation_pattern(self, primary_weak_group: str, 
                                    compensation_strategy: str = 'antagonist') -> Dict[str, float]:
        """
        Generate pattern with specific compensation strategy.
        
        Args:
            primary_weak_group: Muscle group that is primarily weak
            compensation_strategy: 'antagonist', 'synergist', 'proximal'
        """
        pattern = {muscle: generate_severe_population_value() for muscle in ALL_MUSCLES}
        
        # Make primary group weak
        if primary_weak_group in MUSCLE_GROUPS:
            for muscle in MUSCLE_GROUPS[primary_weak_group]:
                pattern[muscle] = generate_random_value('severe')
        
        # Apply compensation strategy
        if compensation_strategy == 'antagonist':
            # Strengthen antagonist muscles
            from muscle_anatomy_config import get_antagonist_muscles
            for muscle in MUSCLE_GROUPS[primary_weak_group]:
                antagonists = get_antagonist_muscles(muscle)
                for ant_muscle in antagonists:
                    if ant_muscle in pattern:
                        pattern[ant_muscle] = np.random.uniform(0.9, 1.0)
                        
        elif compensation_strategy == 'proximal':
            # Strengthen proximal muscles for compensation
            proximal_groups = ['scapular', 'large_movers']
            for group in proximal_groups:
                if group != primary_weak_group:
                    for muscle in MUSCLE_GROUPS[group]:
                        pattern[muscle] = np.random.uniform(0.9, 1.0)
        
        return pattern

def get_all_clinical_patterns() -> List[Tuple[str, Dict]]:
    """
    Get all available clinical patterns with their parameters.
    
    Returns:
        List of (pattern_name, parameters) tuples
    """
    patterns = [
        # Stroke patterns
        ('stroke_flexor_mild', {'type': 'stroke_flexor_synergy', 'severity': 'moderate'}),
        ('stroke_flexor_severe', {'type': 'stroke_flexor_synergy', 'severity': 'severe'}),
        ('stroke_extensor_mild', {'type': 'stroke_extensor_synergy', 'severity': 'moderate'}),
        ('stroke_extensor_severe', {'type': 'stroke_extensor_synergy', 'severity': 'severe'}),
        
        # SCI patterns
        ('sci_c4_complete', {'type': 'sci_pattern', 'level': 'C4', 'completeness': 'complete'}),
        ('sci_c4_incomplete', {'type': 'sci_pattern', 'level': 'C4', 'completeness': 'incomplete'}),
        ('sci_c5_complete', {'type': 'sci_pattern', 'level': 'C5', 'completeness': 'complete'}),
        ('sci_c5_incomplete', {'type': 'sci_pattern', 'level': 'C5', 'completeness': 'incomplete'}),
        ('sci_c6_complete', {'type': 'sci_pattern', 'level': 'C6', 'completeness': 'complete'}),
        ('sci_c6_incomplete', {'type': 'sci_pattern', 'level': 'C6', 'completeness': 'incomplete'}),
        ('sci_c7_complete', {'type': 'sci_pattern', 'level': 'C7', 'completeness': 'complete'}),
        ('sci_c7_incomplete', {'type': 'sci_pattern', 'level': 'C7', 'completeness': 'incomplete'}),
        ('sci_c8_complete', {'type': 'sci_pattern', 'level': 'C8', 'completeness': 'complete'}),
        
        # Rotator cuff patterns
        ('rotator_cuff_isolated', {'type': 'rotator_cuff_tear', 'tear_type': 'isolated_supraspinatus'}),
        ('rotator_cuff_massive', {'type': 'rotator_cuff_tear', 'tear_type': 'massive'}),
        ('subscapularis_tear', {'type': 'rotator_cuff_tear', 'tear_type': 'subscapularis'}),
        
        # Frozen shoulder
        ('frozen_shoulder_early', {'type': 'frozen_shoulder', 'stage': 'freezing'}),
        ('frozen_shoulder_peak', {'type': 'frozen_shoulder', 'stage': 'frozen'}),
        ('frozen_shoulder_recovery', {'type': 'frozen_shoulder', 'stage': 'thawing'}),
        
        # Brachial plexus
        ('brachial_plexus_upper', {'type': 'brachial_plexus_injury', 'injury_type': 'upper_trunk'}),
        ('brachial_plexus_lower', {'type': 'brachial_plexus_injury', 'injury_type': 'lower_trunk'}),
        ('brachial_plexus_total', {'type': 'brachial_plexus_injury', 'injury_type': 'total'}),
        
        # Cerebral palsy
        ('cp_spastic_mild', {'type': 'cerebral_palsy_spastic', 'severity': 'moderate'}),
        ('cp_spastic_severe', {'type': 'cerebral_palsy_spastic', 'severity': 'severe'}),
        ('cp_hypotonic_mild', {'type': 'cerebral_palsy_hypotonic', 'severity': 'moderate'}),
        ('cp_hypotonic_severe', {'type': 'cerebral_palsy_hypotonic', 'severity': 'severe'}),
        
        # Neurological conditions
        ('ms_proximal', {'type': 'multiple_sclerosis', 'pattern_type': 'proximal'}),
        ('ms_asymmetric', {'type': 'multiple_sclerosis', 'pattern_type': 'asymmetric'}),
        ('parkinsons_mild', {'type': 'parkinsons_disease', 'severity': 'moderate'}),
        ('parkinsons_severe', {'type': 'parkinsons_disease', 'severity': 'severe'}),
        
        # Nerve injuries
        ('axillary_nerve', {'type': 'nerve_injury_pattern', 'nerve': 'axillary'}),
        ('radial_nerve', {'type': 'nerve_injury_pattern', 'nerve': 'radial'}),
        ('musculocutaneous_nerve', {'type': 'nerve_injury_pattern', 'nerve': 'musculocutaneous'}),
        ('suprascapular_nerve', {'type': 'nerve_injury_pattern', 'nerve': 'suprascapular'}),
    ]
    
    return patterns

if __name__ == "__main__":
    # Test pattern generation
    generator = ClinicalPatternGenerator(random_seed=42)
    
    # Test each pattern type
    patterns = get_all_clinical_patterns()
    print(f"Total clinical patterns available: {len(patterns)}")
    
    # Generate a sample pattern
    pattern = generator.generate_stroke_flexor_synergy('moderate')
    print(f"\nSample stroke flexor pattern (first 10 muscles):")
    for i, (muscle, value) in enumerate(pattern.items()):
        if i < 10:
            print(f"  {muscle}: {value:.3f}")
    
    print("\nPattern validation:")
    from muscle_anatomy_config import validate_clinical_pattern
    is_valid, reason = validate_clinical_pattern(pattern)
    print(f"  Valid: {is_valid} - {reason}")