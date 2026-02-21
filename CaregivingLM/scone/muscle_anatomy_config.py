#!/usr/bin/env python3
"""
Muscle Anatomy Configuration for Template Generation

This module defines the anatomical organization of muscles, their functional
relationships, and clinical constraints for generating muscle parameter templates.
"""

import numpy as np
from typing import Dict, List, Tuple, Set

# Muscle group definitions based on functional anatomy
MUSCLE_GROUPS = {
    'scapular': [
        'trap_scap_m_r', 'trap_scap_s_r', 'trap_scap_i_r', 'trap_clav_s_r',
        'ser_ant_i_r', 'ser_ant_m_r', 'ser_ant_s_r',
        'rhom_s_r', 'rhom_i_r', 'lev_scap_r'
    ],
    'rotator_cuff': [
        'supraspi_r', 'infraspi_i_r', 'infraspi_s_r', 'teres_min_r',
        'subscap_s_r', 'subscap_i_r'
    ],
    'deltoid': [
        'delt_clav_a_r', 'delt_scap_p_r', 'delt_scap_m_r'
    ],
    'large_movers': [
        'pect_maj_clav_s_r', 'pect_maj_thorax_i_r', 'pect_maj_thorax_m_r',
        'lat_dors_m_r', 'lat_dors_i_r', 'teres_maj_r'
    ],
    'small_shoulder': [
        'pect_min_r', 'coracobrach_r'
    ],
    'elbow': [
        'bic_long_r', 'bic_brev_r', 'brach_r',
        'tri_long_r', 'tri_lat_r'
    ],
    'forearm': [
        'brachiorad_r', 'supinator_r', 'pronator_teres_r'
    ],
    'wrist': [
        'ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r', 'ext_carpi_ulna_r',
        'flex_carpi_rad_r', 'flex_carpi_ulna_r'
    ]
}

# All muscle names in anatomical order
ALL_MUSCLES = [
    'trap_scap_m_r', 'trap_scap_s_r', 'trap_scap_i_r', 'trap_clav_s_r',
    'ser_ant_i_r', 'ser_ant_m_r', 'ser_ant_s_r', 'rhom_s_r', 'rhom_i_r', 'lev_scap_r',
    'pect_min_r', 'delt_clav_a_r', 'delt_scap_p_r', 'delt_scap_m_r',
    'lat_dors_m_r', 'lat_dors_i_r', 'pect_maj_clav_s_r', 'pect_maj_thorax_i_r', 'pect_maj_thorax_m_r',
    'teres_maj_r', 'infraspi_i_r', 'infraspi_s_r', 'teres_min_r',
    'subscap_s_r', 'subscap_i_r', 'supraspi_r', 'tri_long_r', 'tri_lat_r',
    'coracobrach_r', 'bic_long_r', 'bic_brev_r', 'brach_r', 'brachiorad_r',
    'supinator_r', 'pronator_teres_r', 'ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r',
    'ext_carpi_ulna_r', 'flex_carpi_rad_r', 'flex_carpi_ulna_r'
]

# Severity level definitions - EXTREME FOCUS TO MATCH REFERENCE TEMPLATE
SEVERITY_LEVELS = {
    'paralyzed': (0.01, 0.05),  # Near-complete paralysis (like reference template 0.01)
    'severe': (0.05, 0.15),     # Severe weakness with minimal function
    'moderate': (0.15, 0.4),    # Moderate weakness
    'weak_preserved': (0.4, 0.7), # Weak but preserved function
    'normal': (0.8, 1.0)        # Normal strength (compensation only)
}

# Target distribution for maximum functional diversity - SEVERE POPULATION
SEVERE_MOBILITY_TARGET_DISTRIBUTION = {
    'paralyzed': 0.02,       # 2% near-paralyzed (only critical affected muscles)
    'severe': 0.35,          # 35% severely weak (main weakness category)
    'moderate': 0.40,        # 40% moderately weak (maximum functional diversity)
    'weak_preserved': 0.20,  # 20% weakly preserved (compensation muscles)
    'normal': 0.03           # 3% normal (minimal critical compensation)
}

# Anatomical synergy relationships (muscles that typically work together)
SYNERGY_GROUPS = {
    'scapular_retractors': ['rhom_s_r', 'rhom_i_r', 'trap_scap_m_r'],
    'scapular_protractors': ['ser_ant_i_r', 'ser_ant_m_r', 'ser_ant_s_r'],
    'scapular_elevators': ['trap_scap_s_r', 'lev_scap_r'],
    'scapular_depressors': ['trap_scap_i_r', 'lat_dors_i_r'],
    'shoulder_flexors': ['delt_clav_a_r', 'pect_maj_clav_s_r', 'coracobrach_r'],
    'shoulder_extensors': ['delt_scap_p_r', 'lat_dors_m_r', 'lat_dors_i_r', 'teres_maj_r'],
    'shoulder_abductors': ['delt_scap_m_r', 'supraspi_r'],
    'shoulder_adductors': ['pect_maj_thorax_m_r', 'lat_dors_m_r', 'teres_maj_r'],
    'shoulder_internal_rotators': ['subscap_s_r', 'subscap_i_r', 'lat_dors_m_r', 'teres_maj_r'],
    'shoulder_external_rotators': ['infraspi_i_r', 'infraspi_s_r', 'teres_min_r'],
    'elbow_flexors': ['bic_long_r', 'bic_brev_r', 'brach_r', 'brachiorad_r'],
    'elbow_extensors': ['tri_long_r', 'tri_lat_r'],
    'forearm_supinators': ['supinator_r', 'bic_long_r', 'bic_brev_r'],
    'forearm_pronators': ['pronator_teres_r'],
    'wrist_extensors': ['ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r', 'ext_carpi_ulna_r'],
    'wrist_flexors': ['flex_carpi_rad_r', 'flex_carpi_ulna_r']
}

# Antagonist relationships (opposing muscle groups)
ANTAGONIST_PAIRS = [
    ('scapular_retractors', 'scapular_protractors'),
    ('scapular_elevators', 'scapular_depressors'), 
    ('shoulder_flexors', 'shoulder_extensors'),
    ('shoulder_abductors', 'shoulder_adductors'),
    ('shoulder_internal_rotators', 'shoulder_external_rotators'),
    ('elbow_flexors', 'elbow_extensors'),
    ('forearm_supinators', 'forearm_pronators'),
    ('wrist_extensors', 'wrist_flexors')
]

# Nerve innervation patterns for neurological constraints
NERVE_INNERVATION = {
    'spinal_accessory': ['trap_scap_m_r', 'trap_scap_s_r', 'trap_scap_i_r'],
    'long_thoracic': ['ser_ant_i_r', 'ser_ant_m_r', 'ser_ant_s_r'],
    'dorsal_scapular': ['rhom_s_r', 'rhom_i_r', 'lev_scap_r'],
    'suprascapular': ['supraspi_r', 'infraspi_s_r', 'infraspi_i_r'],
    'axillary': ['delt_clav_a_r', 'delt_scap_p_r', 'delt_scap_m_r', 'teres_min_r'],
    'subscapular': ['subscap_s_r', 'subscap_i_r', 'teres_maj_r'],
    'pectoral': ['pect_maj_clav_s_r', 'pect_maj_thorax_i_r', 'pect_maj_thorax_m_r', 'pect_min_r'],
    'thoracodorsal': ['lat_dors_m_r', 'lat_dors_i_r'],
    'musculocutaneous': ['bic_long_r', 'bic_brev_r', 'brach_r', 'coracobrach_r'],
    'radial': ['tri_long_r', 'tri_lat_r', 'brachiorad_r', 'supinator_r', 'ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r', 'ext_carpi_ulna_r'],
    'median': ['pronator_teres_r', 'flex_carpi_rad_r'],
    'ulnar': ['flex_carpi_ulna_r']
}

# Spinal level patterns for SCI constraints
SCI_LEVELS = {
    'C4': {
        'preserved': [],
        'impaired': ALL_MUSCLES
    },
    'C5': {
        'preserved': ['trap_scap_m_r', 'trap_scap_s_r', 'trap_scap_i_r', 'rhom_s_r', 'rhom_i_r', 'lev_scap_r'],
        'impaired': [m for m in ALL_MUSCLES if m not in ['trap_scap_m_r', 'trap_scap_s_r', 'trap_scap_i_r', 'rhom_s_r', 'rhom_i_r', 'lev_scap_r']]
    },
    'C6': {
        'preserved': ['trap_scap_m_r', 'trap_scap_s_r', 'trap_scap_i_r', 'rhom_s_r', 'rhom_i_r', 'lev_scap_r',
                     'ser_ant_i_r', 'ser_ant_m_r', 'ser_ant_s_r', 'delt_clav_a_r', 'delt_scap_p_r', 'delt_scap_m_r',
                     'pect_maj_clav_s_r', 'bic_long_r', 'bic_brev_r', 'brach_r'],
        'impaired': ['tri_long_r', 'tri_lat_r', 'ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r', 'ext_carpi_ulna_r',
                    'flex_carpi_rad_r', 'flex_carpi_ulna_r', 'supinator_r', 'pronator_teres_r']
    },
    'C7': {
        'preserved': ['trap_scap_m_r', 'trap_scap_s_r', 'trap_scap_i_r', 'rhom_s_r', 'rhom_i_r', 'lev_scap_r',
                     'ser_ant_i_r', 'ser_ant_m_r', 'ser_ant_s_r', 'delt_clav_a_r', 'delt_scap_p_r', 'delt_scap_m_r',
                     'pect_maj_clav_s_r', 'bic_long_r', 'bic_brev_r', 'brach_r', 'tri_long_r', 'tri_lat_r'],
        'impaired': ['ext_carpi_rad_long_r', 'ext_carpi_rad_brev_r', 'ext_carpi_ulna_r',
                    'flex_carpi_rad_r', 'flex_carpi_ulna_r', 'supinator_r', 'pronator_teres_r']
    },
    'C8': {
        'preserved': [m for m in ALL_MUSCLES if m not in ['flex_carpi_rad_r', 'flex_carpi_ulna_r']],
        'impaired': ['flex_carpi_rad_r', 'flex_carpi_ulna_r']
    }
}

def get_muscle_group(muscle_name: str) -> str:
    """Get the functional group for a given muscle."""
    for group, muscles in MUSCLE_GROUPS.items():
        if muscle_name in muscles:
            return group
    return 'unknown'

def get_synergy_muscles(muscle_name: str) -> List[str]:
    """Get muscles that are synergistic with the given muscle."""
    synergistic = []
    for group, muscles in SYNERGY_GROUPS.items():
        if muscle_name in muscles:
            synergistic.extend([m for m in muscles if m != muscle_name])
    return synergistic

def get_antagonist_muscles(muscle_name: str) -> List[str]:
    """Get muscles that are antagonistic to the given muscle."""
    antagonistic = []
    
    # Find which synergy group this muscle belongs to
    muscle_groups = []
    for group, muscles in SYNERGY_GROUPS.items():
        if muscle_name in muscles:
            muscle_groups.append(group)
    
    # Find antagonist groups
    for group in muscle_groups:
        for pair in ANTAGONIST_PAIRS:
            if group == pair[0]:
                if pair[1] in SYNERGY_GROUPS:
                    antagonistic.extend(SYNERGY_GROUPS[pair[1]])
            elif group == pair[1]:
                if pair[0] in SYNERGY_GROUPS:
                    antagonistic.extend(SYNERGY_GROUPS[pair[0]])
    
    return list(set(antagonistic))

def validate_clinical_pattern(muscle_values: Dict[str, float], pattern_type: str = None) -> Tuple[bool, str]:
    """
    Validate if a muscle parameter pattern is clinically plausible.
    
    Args:
        muscle_values: Dictionary mapping muscle names to parameter values
        pattern_type: Optional type of clinical pattern being validated
        
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    
    # Check for basic value ranges
    for muscle, value in muscle_values.items():
        if not (0.0 <= value <= 1.0):
            return False, f"Invalid value {value} for muscle {muscle}"
    
    # Check for implausible extreme patterns (very relaxed for extreme population)
    paralyzed_muscles = [m for m, v in muscle_values.items() if v < 0.1]
    if len(paralyzed_muscles) > 38:  # More than 95% of muscles paralyzed
        return False, "Complete paralysis of almost all muscles - not clinically plausible"
    
    # Check synergy group consistency
    for group, muscles in SYNERGY_GROUPS.items():
        group_values = [muscle_values.get(m, 1.0) for m in muscles if m in muscle_values]
        if len(group_values) > 1:
            # Synergistic muscles shouldn't have extreme differences
            if max(group_values) - min(group_values) > 0.7:
                return False, f"Synergy group {group} has implausible value range"
    
    # Check for minimal functional requirements (very relaxed for extreme population)
    # At least some scapular stabilizers should have minimal function
    scapular_values = [muscle_values.get(m, 1.0) for m in MUSCLE_GROUPS['scapular']]
    if all(v < 0.05 for v in scapular_values):
        return False, "Complete scapular stabilizer loss is extremely rare"
    
    # At least one elbow flexor should have minimal function  
    elbow_flexor_muscles = ['bic_long_r', 'bic_brev_r', 'brach_r', 'brachiorad_r']
    elbow_flexor_values = [muscle_values.get(m, 1.0) for m in elbow_flexor_muscles if m in muscle_values]
    if all(v < 0.05 for v in elbow_flexor_values):
        return False, "Complete elbow flexor loss is extremely rare"
    
    return True, "Pattern is clinically plausible"

def apply_clinical_constraints(muscle_values: Dict[str, float], constraint_type: str) -> Dict[str, float]:
    """
    Apply clinical constraints to muscle values based on constraint type.
    
    Args:
        muscle_values: Current muscle parameter values
        constraint_type: Type of constraint to apply
        
    Returns:
        Modified muscle parameter values
    """
    modified_values = muscle_values.copy()
    
    if constraint_type == 'synergy_consistency':
        # Make synergistic muscles more similar
        for group, muscles in SYNERGY_GROUPS.items():
            group_muscles = [m for m in muscles if m in modified_values]
            if len(group_muscles) > 1:
                mean_value = np.mean([modified_values[m] for m in group_muscles])
                # Add some variation but keep them reasonably close
                for muscle in group_muscles:
                    variation = np.random.uniform(-0.1, 0.1)
                    modified_values[muscle] = np.clip(mean_value + variation, 0.1, 1.0)
    
    elif constraint_type == 'antagonist_balance':
        # Ensure antagonist pairs don't both have extreme weakness
        for pair in ANTAGONIST_PAIRS:
            if pair[0] in SYNERGY_GROUPS and pair[1] in SYNERGY_GROUPS:
                group1_muscles = [m for m in SYNERGY_GROUPS[pair[0]] if m in modified_values]
                group2_muscles = [m for m in SYNERGY_GROUPS[pair[1]] if m in modified_values]
                
                if group1_muscles and group2_muscles:
                    group1_mean = np.mean([modified_values[m] for m in group1_muscles])
                    group2_mean = np.mean([modified_values[m] for m in group2_muscles])
                    
                    # If both groups are very weak, strengthen one
                    if group1_mean < 0.3 and group2_mean < 0.3:
                        # Strengthen the group with higher original values
                        if group1_mean >= group2_mean:
                            for muscle in group1_muscles:
                                modified_values[muscle] = np.clip(modified_values[muscle] + 0.3, 0.1, 1.0)
                        else:
                            for muscle in group2_muscles:
                                modified_values[muscle] = np.clip(modified_values[muscle] + 0.3, 0.1, 1.0)
    
    elif constraint_type == 'minimal_function':
        # Ensure minimal functional requirements are met
        
        # Ensure at least some scapular function
        scapular_muscles = [m for m in MUSCLE_GROUPS['scapular'] if m in modified_values]
        scapular_values = [modified_values[m] for m in scapular_muscles]
        if all(v < 0.3 for v in scapular_values):
            # Strengthen the strongest scapular muscles
            strongest_idx = np.argmax(scapular_values)
            modified_values[scapular_muscles[strongest_idx]] = 0.5
        
        # Ensure at least some elbow flexion
        elbow_flexors = ['bic_long_r', 'bic_brev_r', 'brach_r']
        elbow_flexor_muscles = [m for m in elbow_flexors if m in modified_values]
        if elbow_flexor_muscles:
            elbow_values = [modified_values[m] for m in elbow_flexor_muscles]
            if all(v < 0.2 for v in elbow_values):
                strongest_idx = np.argmax(elbow_values)
                modified_values[elbow_flexor_muscles[strongest_idx]] = 0.4
    
    return modified_values

def generate_random_value(severity: str, seed: int = None, severe_population: bool = True) -> float:
    """Generate a random value within the specified severity range."""
    if seed is not None:
        np.random.seed(seed)
    
    if severe_population and severity == 'normal':
        # For severe population, "normal" muscles are actually mildly weak
        # Only truly critical compensation muscles get normal strength
        return np.random.uniform(0.6, 0.8)
    
    min_val, max_val = SEVERITY_LEVELS[severity]
    return np.random.uniform(min_val, max_val)

def generate_severe_population_value(intended_role: str = 'affected', seed: int = None) -> float:
    """
    Generate parameter value optimized for severe mobility limitation population.
    Matches reference template_04.txt severity patterns.
    
    Args:
        intended_role: 'affected' (primary weakness), 'secondary' (moderate), 'compensation' (preserved)
        seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed)
    
    if intended_role == 'affected':
        # Primary affected muscles - diverse severe weakness for maximum coverage
        return np.random.choice([
            np.random.uniform(0.01, 0.05),  # 10% chance paralyzed
            np.random.uniform(0.05, 0.15),  # 70% chance severe
            np.random.uniform(0.15, 0.3)    # 20% chance moderate-severe
        ], p=[0.1, 0.7, 0.2])
    elif intended_role == 'secondary':
        # Secondary affected muscles - moderate weakness range for diversity
        return np.random.uniform(0.15, 0.4)  # Moderate range
    elif intended_role == 'compensation':
        # Critical compensation muscles - weak preservation most common
        return np.random.choice([
            np.random.uniform(0.4, 0.7),   # 85% chance weak preservation
            np.random.uniform(0.8, 1.0)    # 15% chance normal function
        ], p=[0.85, 0.15])
    else:
        # Default: sample from diversity-focused distribution
        severity_probs = [0.02, 0.35, 0.40, 0.20, 0.03]  # Diversity-focused distribution
        severity_ranges = [(0.01, 0.05), (0.05, 0.15), (0.15, 0.4), (0.4, 0.7), (0.8, 1.0)]
        
        selected_range = np.random.choice(len(severity_ranges), p=severity_probs)
        min_val, max_val = severity_ranges[selected_range]
        return np.random.uniform(min_val, max_val)

def get_group_pairs() -> List[Tuple[str, str]]:
    """Get all possible pairs of muscle groups."""
    groups = list(MUSCLE_GROUPS.keys())
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            pairs.append((groups[i], groups[j]))
    return pairs

def get_functionally_important_triplets() -> List[Tuple[str, str, str]]:
    """Get functionally important combinations of three muscle groups."""
    return [
        ('scapular', 'rotator_cuff', 'deltoid'),      # Shoulder complex
        ('deltoid', 'elbow', 'forearm'),               # Reaching chain
        ('large_movers', 'small_shoulder', 'elbow'),   # Power + precision
        ('scapular', 'deltoid', 'large_movers'),       # Proximal stability
        ('rotator_cuff', 'deltoid', 'elbow'),          # Shoulder-elbow coordination
        ('elbow', 'forearm', 'wrist'),                 # Distal control chain
        ('scapular', 'rotator_cuff', 'large_movers'),  # Complete shoulder girdle
        ('deltoid', 'large_movers', 'elbow'),          # Primary reaching muscles
        ('rotator_cuff', 'small_shoulder', 'forearm'), # Fine motor control
        ('scapular', 'elbow', 'wrist'),                # Proximal-distal coordination
        ('large_movers', 'forearm', 'wrist'),          # Power-precision transition
        ('rotator_cuff', 'elbow', 'wrist')             # Stability-mobility chain
    ]

# Clinical pattern metadata
CLINICAL_METADATA = {
    'stroke_flexor_synergy': {
        'description': 'Stroke with flexor synergy dominance',
        'strengthened_groups': ['elbow_flexors', 'wrist_flexors', 'shoulder_adductors'],
        'weakened_groups': ['elbow_extensors', 'wrist_extensors', 'shoulder_abductors']
    },
    'stroke_extensor_synergy': {
        'description': 'Stroke with extensor synergy dominance', 
        'strengthened_groups': ['elbow_extensors', 'wrist_extensors', 'shoulder_adductors'],
        'weakened_groups': ['elbow_flexors', 'wrist_flexors', 'shoulder_abductors']
    },
    'sci_c5': {
        'description': 'Spinal cord injury at C5 level',
        'preserved_groups': ['scapular'],
        'impaired_groups': ['elbow', 'forearm', 'wrist']
    },
    'sci_c6': {
        'description': 'Spinal cord injury at C6 level',
        'preserved_groups': ['scapular', 'deltoid'],
        'partially_preserved': ['elbow'],  # Only biceps
        'impaired_groups': ['forearm', 'wrist']
    },
    'rotator_cuff_massive': {
        'description': 'Massive rotator cuff tear',
        'severely_affected': ['rotator_cuff'],
        'compensatory_overuse': ['deltoid', 'large_movers']
    }
}

if __name__ == "__main__":
    # Test the configuration
    print("Muscle Groups:")
    for group, muscles in MUSCLE_GROUPS.items():
        print(f"  {group}: {len(muscles)} muscles")
    
    print(f"\nTotal muscles: {len(ALL_MUSCLES)}")
    print(f"Group pairs: {len(get_group_pairs())}")
    print(f"Important triplets: {len(get_functionally_important_triplets())}")
    
    # Test validation
    test_values = {muscle: 0.9 for muscle in ALL_MUSCLES}
    test_values['delt_clav_a_r'] = 0.2  # Make one muscle weak
    
    is_valid, reason = validate_clinical_pattern(test_values)
    print(f"\nTest validation: {is_valid} - {reason}")