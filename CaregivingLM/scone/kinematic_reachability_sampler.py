import numpy as np
import json
import os
from scipy.stats import qmc
from sconetools import sconepy

def load_config(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def generate_halton_samples(n_samples, joint_limits):
    """Generate Halton sequence samples in joint space"""
    dof_names = list(joint_limits.keys())
    n_dims = len(dof_names)

    sampler = qmc.Halton(d=n_dims, scramble=False)  # No scrambling for consistency
    unit_samples = sampler.random(n_samples)

    # Map from [0,1]^n to joint limit ranges (in degrees)
    scaled_samples = np.zeros_like(unit_samples)
    for i, dof_name in enumerate(dof_names):
        min_angle, max_angle = joint_limits[dof_name]
        scaled_samples[:, i] = min_angle + unit_samples[:, i] * (max_angle - min_angle)

    return scaled_samples, dof_names

def generate_boundary_samples(joint_limits, mode="corners_plus_midpoints"):
    """Generate systematic samples at joint limit boundaries"""
    dof_names = list(joint_limits.keys())
    n_dims = len(dof_names)

    samples = []

    if mode == "corners_only":
        # Generate all corner combinations (2^n samples)
        for i in range(2**n_dims):
            sample = []
            for j in range(n_dims):
                min_angle, max_angle = joint_limits[dof_names[j]]
                # Use binary representation to choose min (0) or max (1)
                value = max_angle if (i >> j) & 1 else min_angle
                sample.append(value)
            samples.append(sample)

    elif mode == "corners_plus_midpoints":
        # Corners + samples with coordinates at min/mid/max
        for i in range(3**n_dims):
            sample = []
            temp_i = i
            for j in range(n_dims):
                min_angle, max_angle = joint_limits[dof_names[j]]
                mid_angle = (min_angle + max_angle) / 2
                choice = temp_i % 3
                temp_i //= 3

                if choice == 0:
                    value = min_angle
                elif choice == 1:
                    value = mid_angle
                else:
                    value = max_angle
                sample.append(value)
            samples.append(sample)

    return np.array(samples), dof_names

def find_dofs_and_hand(model, target_dofs, hand_body_name):
    """Find DOF objects and hand body"""
    dof_objects = {}

    for dof in model.dofs():
        if dof.name() in target_dofs:
            dof_objects[dof.name()] = dof

    hand_body = None
    for body in model.bodies():
        if body.name() == hand_body_name:
            hand_body = body
            break

    return dof_objects, hand_body

def sample_kinematic_reachability(config_file):
    """Sample kinematic reachability using direct joint positioning"""
    # Load configuration
    config = load_config(config_file)

    # Set up SCONE
    sconepy.set_log_level(3)
    print("SCONE Version", sconepy.version())

    # Load model
    model_file = config["model"]["file"]
    print(f"Loading model: {model_file}")
    model = sconepy.load_model(model_file)

    target_dofs = config["model"]["target_dofs"]
    hand_body_name = config["model"]["hand_body"]
    joint_limits = config["joint_limits"]

    print(f"Model loaded: {model.name()}")
    print(f"Target DOFs: {target_dofs}")

    # Find DOF objects and hand body
    dof_objects, hand_body = find_dofs_and_hand(model, target_dofs, hand_body_name)

    if len(dof_objects) != len(target_dofs):
        print("ERROR: Could not find all required DOFs")
        missing = set(target_dofs) - set(dof_objects.keys())
        print(f"Missing DOFs: {missing}")
        return

    if hand_body is None:
        print(f"ERROR: Could not find hand body '{hand_body_name}'")
        return

    print(f"Found hand body: {hand_body.name()}")

    # Generate samples
    print("\nGenerating samples...")

    # Halton samples
    n_halton = config["sampling"]["halton_samples"]
    halton_samples, dof_names = generate_halton_samples(n_halton, joint_limits)
    print(f"Generated {len(halton_samples)} Halton samples")

    # Boundary samples
    boundary_mode = config["sampling"]["boundary_mode"]
    boundary_samples, _ = generate_boundary_samples(joint_limits, boundary_mode)
    print(f"Generated {len(boundary_samples)} boundary samples")

    # Combine samples
    all_samples = np.vstack([halton_samples, boundary_samples])
    print(f"Total samples: {len(all_samples)}")

    # Sample reachability
    print("\nSampling kinematic reachability...")
    hand_positions = []
    joint_configurations = []

    for i, sample in enumerate(all_samples):
        # Reset model to neutral state
        model.reset()

        # Set joint positions (in radians)
        for j, dof_name in enumerate(dof_names):
            dof = dof_objects[dof_name]
            angle_degrees = sample[j]
            angle_radians = np.radians(angle_degrees)
            dof.set_pos(angle_radians)

        # Apply the joint positions
        model.init_state_from_dofs()

        # Get hand position
        hand_pos = hand_body.com_pos()
        hand_positions.append([hand_pos.x, hand_pos.y, hand_pos.z])
        joint_configurations.append(sample.copy())

        # Progress reporting
        if (i + 1) % 100 == 0 or i == len(all_samples) - 1:
            print(f"  Sampled {i + 1}/{len(all_samples)} configurations")

    hand_positions = np.array(hand_positions)
    joint_configurations = np.array(joint_configurations)

    print(f"\nReachability sampling complete!")
    print(f"Hand position range:")
    print(f"  X: [{hand_positions[:, 0].min():.3f}, {hand_positions[:, 0].max():.3f}]")
    print(f"  Y: [{hand_positions[:, 1].min():.3f}, {hand_positions[:, 1].max():.3f}]")
    print(f"  Z: [{hand_positions[:, 2].min():.3f}, {hand_positions[:, 2].max():.3f}]")

    # Save results
    output_dir = config["output"]["directory"]
    filename_prefix = config["output"]["filename_prefix"]

    os.makedirs(output_dir, exist_ok=True)

    if config["output"]["save_point_cloud"]:
        # Save hand positions
        hand_file = f"{output_dir}/{filename_prefix}_hand_positions.npy"
        np.save(hand_file, hand_positions)
        print(f"Hand positions saved to: {hand_file}")

        # Save joint configurations
        joint_file = f"{output_dir}/{filename_prefix}_joint_configs.npy"
        np.save(joint_file, joint_configurations)
        print(f"Joint configurations saved to: {joint_file}")

        # Save metadata
        metadata = {
            "dof_names": dof_names,
            "joint_limits": joint_limits,
            "n_halton_samples": n_halton,
            "n_boundary_samples": len(boundary_samples),
            "total_samples": len(all_samples),
            "hand_body": hand_body_name
        }

        metadata_file = f"{output_dir}/{filename_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_file}")

    return hand_positions, joint_configurations, dof_names

if __name__ == "__main__":
    hand_positions, joint_configs, dof_names = sample_kinematic_reachability("kinematic_reachability_config.json")
    print(f"\nKinematic reachability sampling completed!")
    print(f"Generated {len(hand_positions)} hand positions")
    print(f"Ready for 3D visualization!")