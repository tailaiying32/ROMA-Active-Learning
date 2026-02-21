#!/usr/bin/env python3

import numpy as np
import time
from pathlib import Path
from sconetools import sconepy

def halton_sequence(index, base):
    """Generate the index-th number in the Halton sequence for given base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result

def generate_halton_samples(n_samples, n_dims):
    """Generate n_samples of n_dims using Halton sequence with different prime bases."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if n_dims > len(primes):
        raise ValueError(f"Too many dimensions ({n_dims}). Max supported: {len(primes)}")

    samples = np.zeros((n_samples, n_dims))
    for i in range(n_samples):
        for j in range(n_dims):
            samples[i, j] = halton_sequence(i + 1, primes[j])
    return samples

class JointCoverageSampler:
    """Coverage sampling for joint space exploration using HSA13T model."""

    def __init__(self, model_file="models/HSA13T_hfd.scone"):
        """Initialize with model file path."""
        self.model_file = model_file
        self.model = None
        self.joint_names = []
        self.joint_limits = {}

    def load_model(self):
        """Load the SCONE model and extract joint information."""
        print(f"Loading model: {self.model_file}")
        self.model = sconepy.load_model(self.model_file)

        # Extract joint names from DOFs
        self.joint_names = []
        for dof in self.model.dofs():
            self.joint_names.append(dof.name())

        print(f"Found {len(self.joint_names)} DOFs: {self.joint_names}")
        return self.joint_names

    def load_init_values(self, init_file="init/InitHSA13_bs.zml"):
        """Load initial joint values from init file."""
        init_values = {}
        with open(init_file, 'r') as f:
            in_values_section = False
            for line in f:
                line = line.strip()
                if line == "values {":
                    in_values_section = True
                    continue
                elif line == "}":
                    if in_values_section:
                        break
                    continue

                if in_values_section and "=" in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        joint_name = parts[0].strip()
                        joint_value = float(parts[1].strip())
                        init_values[joint_name] = joint_value

        return init_values

    def set_joint_limits(self, joint_limits=None):
        """
        Set joint limits for sampling.

        Args:
            joint_limits: Dict with joint names as keys and (min, max) tuples as values.
                         If None, uses default physiological limits.
        """
        if joint_limits is None:
            # Actual joint limits from HSA13T-2025-06-26.hfd (converted from degrees to radians)
            deg_to_rad = 3.14159265359 / 180.0
            self.joint_limits = {
                'clavicle_protraction_r': (-30*deg_to_rad, 10*deg_to_rad),
                'clavicle_elevation_r': (-15*deg_to_rad, 45*deg_to_rad),
                'clavicle_rotation_r': (-15*deg_to_rad, 60*deg_to_rad),
                'scapula_abduction_r': (-35*deg_to_rad, 45*deg_to_rad),
                'scapula_elevation_r': (-45*deg_to_rad, 45*deg_to_rad),
                'scapula_winging_r': (-45*deg_to_rad, 45*deg_to_rad),
                'shoulder_flexion_r': (-45*deg_to_rad, 150*deg_to_rad),
                'shoulder_abduction_r': (-60*deg_to_rad, 120*deg_to_rad),
                'shoulder_rotation_r': (-60*deg_to_rad, 60*deg_to_rad),
                'elbow_flexion_r': (0*deg_to_rad, 130*deg_to_rad),
                'forearm_pronation_r': (-90*deg_to_rad, 90*deg_to_rad),
                'wrist_flexion_r': (-70*deg_to_rad, 70*deg_to_rad),
                'radial_deviation_r': (-30*deg_to_rad, 15*deg_to_rad)
            }
        else:
            self.joint_limits = joint_limits

        # Filter to only include joints that exist in the model
        self.joint_limits = {name: limits for name, limits in self.joint_limits.items()
                           if name in self.joint_names}

        print(f"Using joint limits for {len(self.joint_limits)} joints")

    def _check_min_distance(self, new_position, existing_positions, min_distance):
        """
        Check if new position maintains minimum distance from all existing positions.

        Args:
            new_position: New hand position [x, y, z]
            existing_positions: List of existing hand positions
            min_distance: Minimum required distance threshold

        Returns:
            bool: True if new position is far enough from all existing positions
        """
        if len(existing_positions) == 0:
            return True

        new_pos = np.array(new_position)
        for existing_pos in existing_positions:
            existing_pos = np.array(existing_pos)
            distance = np.linalg.norm(new_pos - existing_pos)
            if distance < min_distance:
                return False
        return True

    def generate_samples(self, n_samples=200, output_dir="init", prefix="coverage_sample", max_attempts=None, check_collisions=True, strata_per_joint=1, stratified_joints=None, min_distance=0.05, main_dofs_only=False):
        """
        Generate coverage samples and save as init files, with collision checking and distance-based rejection.

        Args:
            n_samples: Number of valid samples to generate
            output_dir: Directory to save init files
            prefix: Prefix for output filenames
            max_attempts: Maximum attempts to find valid samples (default: 3*n_samples)
            check_collisions: Whether to check for collisions and reject invalid poses (default: True)
            strata_per_joint: Number of strata to divide each joint range into for stratified sampling (default: 1, no stratification)
            stratified_joints: List of joint names to apply stratification to (default: None, uses shoulder and elbow joints)
            min_distance: Minimum distance threshold between hand positions to maintain spatial distribution (default: 0.05)
            main_dofs_only: If True, sample only 3 shoulder DOFs + elbow, fix others to init values (default: False)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not self.joint_limits:
            raise RuntimeError("Joint limits not set. Call set_joint_limits() first.")

        # Load init values if using main DOFs only mode
        init_values = {}
        if main_dofs_only:
            init_values = self.load_init_values()
            print("Main DOFs only mode: sampling 3 shoulder DOFs + elbow, fixing others to init values")
            main_dofs = ['shoulder_flexion_r', 'shoulder_abduction_r', 'shoulder_rotation_r', 'elbow_flexion_r']
            # Filter joint limits to only include main DOFs
            filtered_joint_limits = {name: limits for name, limits in self.joint_limits.items() if name in main_dofs}
            self.joint_limits = filtered_joint_limits
            print(f"Sampling DOFs: {list(filtered_joint_limits.keys())}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Get ordered list of joints with limits
        joint_names_ordered = [name for name in self.joint_names if name in self.joint_limits]
        n_joints = len(joint_names_ordered)

        # Set up stratified sampling if requested
        use_stratified = strata_per_joint > 1
        if use_stratified:
            # Set default stratified joints if not specified
            if stratified_joints is None:
                stratified_joints = ['shoulder_flexion_r', 'shoulder_abduction_r', 'shoulder_rotation_r', 'elbow_flexion_r']

            # Filter to only include joints that exist and have limits
            stratified_joints = [name for name in stratified_joints if name in self.joint_limits]
            n_stratified_joints = len(stratified_joints)

            if n_stratified_joints == 0:
                print("Warning: No valid stratified joints found. Disabling stratification.")
                use_stratified = False
            else:
                total_strata = strata_per_joint ** n_stratified_joints
                print(f"Using stratified sampling on {n_stratified_joints} joints: {stratified_joints}")
                print(f"{strata_per_joint} strata per joint = {total_strata} total strata")

                # Calculate samples per stratum
                base_samples_per_stratum = n_samples // total_strata
                extra_samples = n_samples % total_strata

                if base_samples_per_stratum == 0:
                    print(f"Warning: {n_samples} samples across {total_strata} strata means some strata will be empty")
                    print(f"Consider reducing strata_per_joint or increasing n_samples")

        if not check_collisions:
            # If not checking collisions, just generate n_samples directly
            max_attempts = n_samples
            if use_stratified:
                print(f"Generating {n_samples} stratified coverage samples for {n_joints} joints (collision checking disabled)...")
            else:
                print(f"Generating {n_samples} coverage samples for {n_joints} joints (collision checking disabled)...")
        else:
            if max_attempts is None:
                max_attempts = 3 * n_samples
            if use_stratified:
                print(f"Generating {n_samples} valid stratified coverage samples for {n_joints} joints...")
            else:
                print(f"Generating {n_samples} valid coverage samples for {n_joints} joints...")
            print(f"Maximum attempts: {max_attempts}")

        # Generate samples based on sampling strategy
        if use_stratified:
            sample_targets = self._generate_stratified_targets(joint_names_ordered, strata_per_joint, n_samples, stratified_joints)

        # We'll generate Halton samples on-demand to avoid memory/bounds issues

        # Store hand positions and collision data for analysis
        hand_positions = []
        rejected_samples = []
        rejected_forces = []
        valid_count = 0
        attempt = 0

        while valid_count < n_samples:
            # Scale samples to joint limits
            joint_values = {}
            joint_velocities = {}

            if use_stratified and attempt < len(sample_targets):
                # Use stratified targets first
                target_values = sample_targets[attempt]
                for j, joint_name in enumerate(joint_names_ordered):
                    joint_values[joint_name] = target_values[j]
                    joint_velocities[joint_name] = 0.0  # Start with zero velocities
            else:
                # Use Halton samples (either no stratification or ran out of stratified targets)
                if use_stratified and attempt == len(sample_targets):
                    print(f"Exhausted stratified targets ({len(sample_targets)}), falling back to Halton sampling...")

                # Generate Halton sample on-demand for this attempt
                # We use attempt as the index in the Halton sequence
                halton_sample = []
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
                for j in range(n_joints):
                    halton_sample.append(halton_sequence(attempt + 1, primes[j]))

                for j, joint_name in enumerate(joint_names_ordered):
                    min_val, max_val = self.joint_limits[joint_name]
                    # Scale [0,1] Halton sample to [min_val, max_val]
                    joint_values[joint_name] = min_val + halton_sample[j] * (max_val - min_val)
                    joint_velocities[joint_name] = 0.0  # Start with zero velocities

            # Set all other joints to init values or zero if not in limits
            for joint_name in self.joint_names:
                if joint_name not in joint_values:
                    if main_dofs_only and joint_name in init_values:
                        joint_values[joint_name] = init_values[joint_name]
                    else:
                        joint_values[joint_name] = 0.0
                    joint_velocities[joint_name] = 0.0

            # Apply joint values to model
            self.model.reset()

            # Set DOF positions
            dof_positions = np.zeros(len(self.joint_names))
            for idx, joint_name in enumerate(self.joint_names):
                dof_positions[idx] = joint_values[joint_name]

            self.model.set_dof_positions(dof_positions)
            self.model.init_state_from_dofs()

            # Get hand position and examine forces
            hand_body = None
            arm_bodies = []

            # Collect all arm-related bodies for force analysis
            for body in self.model.bodies():
                body_name = body.name()
                if body_name == "hand_r":
                    hand_body = body
                if any(part in body_name for part in ['hand_r', 'radius_r', 'ulna_r', 'humerus_r', 'scapula_r', 'clavicle_r']):
                    arm_bodies.append(body)

            if hand_body is not None:
                hand_pos = hand_body.com_pos()
            else:
                print("Warning: hand_r body not found")
                hand_pos = type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0})()

            # Examine contact forces for collision detection
            total_contact_force = 0.0
            max_body_contact_force = 0.0
            collision_detected = False

            if check_collisions:
                # Check for collisions - examine contact forces
                print(f"\nAttempt {attempt + 1} - Force Analysis:")
                for body in arm_bodies:
                    try:
                        contact_force = body.contact_force()
                        force_magnitude = (contact_force.x**2 + contact_force.y**2 + contact_force.z**2)**0.5

                        if force_magnitude > 0.001:  # Threshold for meaningful contact force
                            print(f"  {body.name()}: contact_force = [{contact_force.x:.3f}, {contact_force.y:.3f}, {contact_force.z:.3f}], magnitude = {force_magnitude:.3f}")
                            collision_detected = True

                        total_contact_force += force_magnitude
                        max_body_contact_force = max(max_body_contact_force, force_magnitude)

                    except Exception as e:
                        print(f"  Error getting contact force for {body.name()}: {e}")

                # Also check model-level contact forces
                try:
                    model_contact_force = self.model.contact_force()
                    model_force_magnitude = (model_contact_force.x**2 + model_contact_force.y**2 + model_contact_force.z**2)**0.5
                    if model_force_magnitude > 0.001:
                        print(f"  Model total contact force: [{model_contact_force.x:.3f}, {model_contact_force.y:.3f}, {model_contact_force.z:.3f}], magnitude = {model_force_magnitude:.3f}")
                except Exception as e:
                    print(f"  Error getting model contact force: {e}")

                print(f"  Hand position: [{hand_pos.x:.3f}, {hand_pos.y:.3f}, {hand_pos.z:.3f}]")

                if collision_detected:
                    print(f"  *** COLLISION DETECTED: Total = {total_contact_force:.3f}, Max = {max_body_contact_force:.3f} - REJECTING")
                    rejected_samples.append(attempt)
                    rejected_forces.append(total_contact_force)
                else:
                    # Check minimum distance from existing samples
                    hand_position = [hand_pos.x, hand_pos.y, hand_pos.z]
                    if self._check_min_distance(hand_position, hand_positions, min_distance):
                        print(f"  ✓ VALID POSE - Accepting as sample {valid_count}")

                        # Store hand position for valid sample
                        hand_positions.append(hand_position)

                        # Write init file for valid sample
                        filename = f"{prefix}_{valid_count:03d}.zml"
                        filepath = output_path / filename
                        self._write_init_file(filepath, joint_values, joint_velocities)

                        valid_count += 1
                    else:
                        print(f"  *** DISTANCE TOO CLOSE: Hand position too close to existing samples (< {min_distance:.3f}m) - REJECTING")
                        rejected_samples.append(attempt)
                        rejected_forces.append(0.0)  # No collision force, rejected for distance
            else:
                # No collision checking - but still check distance
                print(f"\nSample {valid_count + 1} - Hand position: [{hand_pos.x:.3f}, {hand_pos.y:.3f}, {hand_pos.z:.3f}]")

                # Check minimum distance from existing samples
                hand_position = [hand_pos.x, hand_pos.y, hand_pos.z]
                if self._check_min_distance(hand_position, hand_positions, min_distance):
                    print(f"  ✓ VALID DISTANCE - Accepting sample")

                    # Store hand position for sample
                    hand_positions.append(hand_position)

                    # Write init file for sample
                    filename = f"{prefix}_{valid_count:03d}.zml"
                    filepath = output_path / filename
                    self._write_init_file(filepath, joint_values, joint_velocities)

                    valid_count += 1
                else:
                    print(f"  *** DISTANCE TOO CLOSE: Hand position too close to existing samples (< {min_distance:.3f}m) - REJECTING")

            attempt += 1

            if attempt % 50 == 0:
                print(f"Progress: {valid_count} valid samples from {attempt} attempts ({valid_count/attempt*100:.1f}% success rate)")

        # Check if we got all requested samples
        if valid_count < n_samples:
            print(f"\nWARNING: Only found {valid_count}/{n_samples} valid samples after {attempt} attempts!")

        # Save hand positions for analysis
        hand_positions = np.array(hand_positions)
        np.savetxt(output_path / f"{prefix}_hand_positions.txt", hand_positions,
                   header="x y z", fmt="%.6f")

        # Generate visualization HFD file
        self._create_visualization_hfd(hand_positions, prefix, valid_count)

        print(f"\n=== SAMPLING SUMMARY ===")
        if check_collisions:
            print(f"Generated {valid_count} valid samples in {output_dir}/")
            print(f"Total attempts: {attempt}")
            print(f"Success rate: {valid_count/attempt*100:.1f}%")
        else:
            print(f"Generated {valid_count} samples in {output_dir}/ (collision checking disabled)")

        if len(hand_positions) > 0:
            print(f"Hand position range: X[{hand_positions[:,0].min():.3f}, {hand_positions[:,0].max():.3f}], "
                  f"Y[{hand_positions[:,1].min():.3f}, {hand_positions[:,1].max():.3f}], "
                  f"Z[{hand_positions[:,2].min():.3f}, {hand_positions[:,2].max():.3f}]")

        # Collision analysis summary
        if check_collisions:
            n_rejections = len(rejected_samples)
            rejection_rate = n_rejections / attempt * 100
            print(f"\n=== COLLISION ANALYSIS ===")
            print(f"Rejected samples: {n_rejections}/{attempt} ({rejection_rate:.1f}%)")

            if n_rejections > 0:
                print(f"Average rejection force: {np.mean(rejected_forces):.3f}")
                print(f"Max rejection force: {np.max(rejected_forces):.3f}")

                # Save rejection data
                rejection_data = np.column_stack([rejected_samples, rejected_forces])
                np.savetxt(output_path / f"{prefix}_rejected_data.txt", rejection_data,
                           header="attempt_index collision_force", fmt="%d %.6f")
                print(f"Rejection data saved to {output_path / f'{prefix}_rejected_data.txt'}")
            else:
                print("No collisions detected - all samples valid!")
        else:
            print(f"\n=== COLLISION ANALYSIS ===")
            print("Collision checking was disabled - no collision analysis performed")

        return hand_positions

    def _generate_stratified_targets(self, joint_names_ordered, strata_per_joint, n_samples, stratified_joints):
        """
        Generate stratified sample targets to ensure coverage across specified joint space regions.

        Args:
            joint_names_ordered: List of all joint names in order
            strata_per_joint: Number of strata per joint dimension
            n_samples: Total number of samples desired
            stratified_joints: List of joint names to apply stratification to

        Returns:
            List of target joint value arrays
        """
        n_stratified_joints = len(stratified_joints)
        total_strata = strata_per_joint ** n_stratified_joints

        # Calculate samples per stratum
        base_samples_per_stratum = n_samples // total_strata
        extra_samples = n_samples % total_strata

        targets = []

        # Generate all possible stratum combinations for stratified joints only
        import itertools
        stratum_indices = list(itertools.product(range(strata_per_joint), repeat=n_stratified_joints))

        for stratum_idx, stratum_combo in enumerate(stratum_indices):
            # Determine how many samples for this stratum
            samples_for_stratum = base_samples_per_stratum
            if stratum_idx < extra_samples:
                samples_for_stratum += 1

            if samples_for_stratum == 0:
                continue

            # Generate samples within this stratum
            for sample_in_stratum in range(samples_for_stratum):
                target_values = []

                for j, joint_name in enumerate(joint_names_ordered):
                    min_val, max_val = self.joint_limits[joint_name]
                    joint_range = max_val - min_val

                    if joint_name in stratified_joints:
                        # This joint uses stratified sampling
                        stratified_idx = stratified_joints.index(joint_name)

                        # Calculate stratum boundaries for this joint
                        stratum_size = joint_range / strata_per_joint
                        stratum_min = min_val + stratum_combo[stratified_idx] * stratum_size
                        stratum_max = stratum_min + stratum_size

                        # Generate random sample within this stratum
                        # Use a different random seed for each sample to avoid correlation
                        np.random.seed(len(targets) + 42)  # Deterministic but varied
                        random_offset = np.random.random()

                        target_value = stratum_min + random_offset * stratum_size
                    else:
                        # This joint uses regular random sampling across its full range
                        np.random.seed(len(targets) + 100 + j)  # Different seed for non-stratified joints
                        random_offset = np.random.random()
                        target_value = min_val + random_offset * joint_range

                    target_values.append(target_value)

                targets.append(target_values)

        print(f"Generated {len(targets)} stratified targets across {total_strata} strata")
        if len(targets) != n_samples:
            print(f"Note: Generated {len(targets)} targets for {n_samples} requested samples")

        return targets

    def _create_visualization_hfd(self, hand_positions, prefix, n_samples):
        """
        Create an HFD visualization file with spheres at each sampled hand position.
        Copies the complete model structure from the original HSA13T model file.

        Args:
            hand_positions: Array of hand positions (n_samples x 3)
            prefix: Prefix for output filename
            n_samples: Number of samples generated
        """
        # Create visualizations directory if it doesn't exist
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)

        # Output filename
        viz_filename = viz_dir / f"{prefix}_{n_samples}_viz.hfd"

        print(f"Creating visualization file: {viz_filename}")

        with open(viz_filename, 'w') as f:
            # Copy the complete model structure from HSA13T model file
            try:
                with open("models/HSA13T-2025-06-26.hfd", 'r') as model_file:
                    model_content = model_file.read()

                    # Write the complete model content up to the closing brace
                    # Find the last muscle definition and copy everything up to and including the closing brace
                    lines = model_content.split('\n')

                    # Find the last line with content (before any potential target bodies)
                    last_content_idx = len(lines) - 1
                    for i in range(len(lines) - 1, -1, -1):
                        line = lines[i].strip()
                        if line == '}' and i > 100:  # The model closing brace should be near the end
                            last_content_idx = i
                            break

                    # Write all content up to and including the model closing brace
                    for i in range(last_content_idx + 1):
                        f.write(lines[i] + '\n')

                    # But we need to remove the final closing brace to add our content
                    # Go back and find the actual model structure end
                    f.seek(0)
                    f.truncate()

                    # Rewrite without the final closing brace
                    for i in range(last_content_idx):
                        f.write(lines[i] + '\n')

            except FileNotFoundError:
                print("Warning: Could not find models/HSA13T-2025-06-26.hfd, creating minimal model")
                # Fallback to minimal model structure
                f.write("""model {
	material {
		name = default_material
		static_friction = 0.9
		dynamic_friction = 0.6
		stiffness = 11006.4
		damping = 1
	}
	material {
		name = body_material
		static_friction = 0.9
		dynamic_friction = 0.6
		stiffness = 5000
		damping = 2
	}
	body {
		name = ground
		mass = 0
		inertia { x = 0 y = 0 z = 0 }
		geometry {
			name = platform
			type = plane
			normal { x = -1 y = 0 z = 0 }
			ori { x = 0 y = 0 z = -90 }
		}
	}
	body {
		name = target
		mass = 0
		inertia { x = 0 y = 0 z = 0 }
		mesh {
			shape { type = sphere radius = 0.025 }
			color { r = 0.8 g = 0.1 b = 0.1 a = 1 }
		}
	}
	model_options {
		joint_stiffness = 1e+06
		joint_limit_stiffness = 500
		joint_rotational_constraint_stiffness = 500
	}
""")

            # Add comment header for visualization
            f.write(f"\t# Visualization targets for {prefix}\n")
            f.write(f"\t# Total target positions: {len(hand_positions)}\n")

            # Calculate distance statistics for color coding
            if len(hand_positions) > 0:
                distances = np.sqrt(np.sum(hand_positions**2, axis=1))
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                f.write(f"\t# Distance range: {min_dist:.6f} to {max_dist:.6f}\n")

                # Count by quality zones
                threshold = 0.1
                excellent = np.sum(distances <= threshold/2)
                good = np.sum(distances <= threshold)
                poor = len(distances) - good
                f.write(f"\t# Color zones based on threshold {threshold:.6f}:\n")
                f.write(f"\t#   Green (excellent): distance <= {threshold/2:.6f} ({excellent} targets)\n")
                f.write(f"\t#   Yellow (good): distance <= {threshold:.6f} ({good} targets)\n")
                f.write(f"\t#   Red (poor): distance > {threshold:.6f} ({poor} targets)\n")

            f.write("\t# Color coding: Smooth Green -> Yellow -> Red transition\n")
            f.write("\t\n")

            # Add coordinate axes for reference
            f.write("\t# Coordinate axes for reference\n")
            f.write("\tbody {\n")
            f.write("\t\tname = axis_x\n")
            f.write("\t\tmass = 0\n")
            f.write("\t\tinertia { x = 0 y = 0 z = 0 }\n")
            f.write("\t\tmesh {\n")
            f.write("\t\t\tshape { type = capsule radius = 0.002 height = 0.2 }\n")
            f.write("\t\t\tcolor { r = 1 g = 0 b = 0 a = 1 }\n")
            f.write("\t\t\tpos { x = 0.1 y = 0 z = 0 }\n")
            f.write("\t\t\tori { x = 0 y = 0 z = 90 }\n")
            f.write("\t\t}\n")
            f.write("\t}\n")
            f.write("\tbody {\n")
            f.write("\t\tname = axis_y\n")
            f.write("\t\tmass = 0\n")
            f.write("\t\tinertia { x = 0 y = 0 z = 0 }\n")
            f.write("\t\tmesh {\n")
            f.write("\t\t\tshape { type = capsule radius = 0.002 height = 0.2 }\n")
            f.write("\t\t\tcolor { r = 0 g = 1 b = 0 a = 1 }\n")
            f.write("\t\t\tpos { x = 0 y = 0.1 z = 0 }\n")
            f.write("\t\t}\n")
            f.write("\t}\n")
            f.write("\tbody {\n")
            f.write("\t\tname = axis_z\n")
            f.write("\t\tmass = 0\n")
            f.write("\t\tinertia { x = 0 y = 0 z = 0 }\n")
            f.write("\t\tmesh {\n")
            f.write("\t\t\tshape { type = capsule radius = 0.002 height = 0.2 }\n")
            f.write("\t\t\tcolor { r = 0 g = 0 b = 1 a = 1 }\n")
            f.write("\t\t\tpos { x = 0 y = 0 z = 0.1 }\n")
            f.write("\t\t\tori { x = 90 y = 0 z = 0 }\n")
            f.write("\t\t}\n")
            f.write("\t}\n")
            f.write("\t\n")

            # Add sphere bodies for each hand position
            for i, pos in enumerate(hand_positions):
                x, y, z = pos
                f.write(f"""	body {{
		name = target_{i:03d}
		mass = 0
		inertia {{ x = 0 y = 0 z = 0 }}
		mesh {{
			shape {{ type = sphere radius = 0.03 }}
			color {{ r = 0.200 g = 0.800 b = 0.200 a = 0.8 }}
			pos {{ x = {x:.6f} y = {y:.6f} z = {z:.6f} }}
		}}
	}}
""")

            f.write("}\n")  # Close the model

        print(f"Visualization file created with {len(hand_positions)} spheres")

    def _write_init_file(self, filepath, joint_values, joint_velocities):
        """Write joint values and velocities to init file in SCONE format."""
        with open(filepath, 'w') as f:
            f.write("values {\n")
            for joint_name in self.joint_names:
                if joint_name in joint_values:
                    f.write(f"\t{joint_name} = {joint_values[joint_name]:.6f}\n")
            f.write("}\n")

            f.write("velocities {\n")
            for joint_name in self.joint_names:
                if joint_name in joint_velocities:
                    f.write(f"\t{joint_name} = {joint_velocities[joint_name]:.6f}\n")
            f.write("}\n")

def main():
    """Example usage of JointCoverageSampler."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate coverage samples for joint space exploration')
    parser.add_argument('--n-samples', type=int, default=200, help='Number of samples to generate (default: 200)')
    parser.add_argument('--output-dir', default='init/coverage_samples', help='Output directory for init files (default: init/coverage_samples)')
    parser.add_argument('--prefix', default='halton_sample', help='Prefix for output files (default: halton_sample)')
    parser.add_argument('--no-collision-check', action='store_true', help='Disable collision checking (generate all samples)')
    parser.add_argument('--max-attempts', type=int, help='Maximum attempts when collision checking is enabled (default: 3*n_samples)')
    parser.add_argument('--strata-per-joint', type=int, default=1, help='Number of strata per joint for stratified sampling (default: 1, no stratification)')
    parser.add_argument('--stratified-joints', nargs='+', help='Joint names to apply stratification to (default: shoulder and elbow joints)')
    parser.add_argument('--min-distance', type=float, default=0.05, help='Minimum distance threshold between hand positions (default: 0.05 meters)')
    parser.add_argument('--main-dofs-only', action='store_true', help='Sample only 3 shoulder DOFs + elbow, fix others to init values from InitHSA13_bs.zml')

    args = parser.parse_args()

    # Set SCONE log level
    sconepy.set_log_level(3)

    # Create sampler
    sampler = JointCoverageSampler()

    # Load model and get joint names
    joint_names = sampler.load_model()

    # Set default joint limits (can be customized)
    sampler.set_joint_limits()

    # Custom joint limits example:
    # custom_limits = {
    #     'clavicle_protraction_r': (-0.5, 0.5),
    #     'clavicle_elevation_r': (-0.5, 0.5),
    #     'clavicle_rotation_r': (-0.5, 0.5),
    #     'scapula_abduction_r': (-0.5, 0.5),
    #     'scapula_elevation_r': (-0.8, 0.2),
    #     'scapula_winging_r': (-0.2, 0.8),
    #     'shoulder_flexion_r': (-1.0, 2.0),
    #     'shoulder_abduction_r': (-0.5, 2.0),
    #     'shoulder_rotation_r': (-1.5, 1.5),
    #     'elbow_flexion_r': (0.0, 2.4),
    #     'forearm_pronation_r': (-1.5, 1.5),
    #     'wrist_flexion_r': (-1.0, 1.0),
    #     'radial_deviation_r': (-0.5, 0.5)
    # }
    # sampler.set_joint_limits(custom_limits)

    # Generate samples
    hand_positions = sampler.generate_samples(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        prefix=args.prefix,
        max_attempts=args.max_attempts,
        check_collisions=not args.no_collision_check,
        strata_per_joint=args.strata_per_joint,
        stratified_joints=args.stratified_joints,
        min_distance=args.min_distance,
        main_dofs_only=args.main_dofs_only
    )

    print("Coverage sampling completed!")

if __name__ == "__main__":
    main()