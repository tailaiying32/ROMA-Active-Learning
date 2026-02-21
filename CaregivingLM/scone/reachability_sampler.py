import numpy as np
import json
import time
from scipy.stats import qmc
from sconetools import sconepy

def load_config(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def generate_halton_samples(n_samples, n_dims, actuator_limits):
    """Generate Halton sequence samples in actuator space"""
    sampler = qmc.Halton(d=n_dims, scramble=False)  # No scrambling for consistency
    unit_samples = sampler.random(n_samples)

    # Map from [0,1]^n to actuator limit ranges
    scaled_samples = np.zeros_like(unit_samples)
    for i, dof_name in enumerate(actuator_limits.keys()):
        min_val, max_val = actuator_limits[dof_name]
        scaled_samples[:, i] = min_val + unit_samples[:, i] * (max_val - min_val)

    return scaled_samples

def generate_boundary_samples(actuator_limits, mode="corners_plus_midpoints"):
    """Generate systematic samples at boundaries"""
    dof_names = list(actuator_limits.keys())
    n_dims = len(dof_names)

    samples = []

    if mode == "corners_only":
        # Generate all corner combinations (2^n samples)
        for i in range(2**n_dims):
            sample = []
            for j in range(n_dims):
                min_val, max_val = actuator_limits[dof_names[j]]
                # Use binary representation to choose min (0) or max (1)
                value = max_val if (i >> j) & 1 else min_val
                sample.append(value)
            samples.append(sample)

    elif mode == "corners_plus_midpoints":
        # Corners + samples with one coordinate at midpoint
        for i in range(3**n_dims):  # 3 values per dim: min, mid, max
            sample = []
            temp_i = i
            for j in range(n_dims):
                min_val, max_val = actuator_limits[dof_names[j]]
                mid_val = (min_val + max_val) / 2
                choice = temp_i % 3
                temp_i //= 3

                if choice == 0:
                    value = min_val
                elif choice == 1:
                    value = mid_val
                else:
                    value = max_val
                sample.append(value)
            samples.append(sample)

    return np.array(samples)

def find_actuator_indices(model, target_dofs):
    """Find actuator indices corresponding to target DOFs"""
    # First, print all available actuators for debugging
    print("\nAvailable actuators:")
    for i, actuator in enumerate(model.actuators()):
        print(f"  {i}: {actuator.name()}")

    actuator_indices = {}

    for target_dof in target_dofs:
        found = False

        # First try exact name matching
        for i, actuator in enumerate(model.actuators()):
            if actuator.name() == target_dof:
                actuator_indices[target_dof] = i
                found = True
                print(f"Mapped {target_dof} -> actuator {i}: {actuator.name()}")
                break

        if not found:
            print(f"WARNING: Could not find exact actuator match for DOF {target_dof}")

    return actuator_indices

def interpolate_samples(samples, transition_time, hold_time, dt):
    """Create smooth interpolation between samples"""
    if len(samples) < 2:
        return samples, []

    interpolated_samples = []
    time_points = []
    current_time = 0

    for i in range(len(samples)):
        if i == 0:
            # First sample - just hold
            n_hold_steps = int(hold_time / dt)
            for _ in range(n_hold_steps):
                interpolated_samples.append(samples[i].copy())
                time_points.append(current_time)
                current_time += dt
        else:
            # Interpolate from previous to current sample
            n_transition_steps = int(transition_time / dt)
            prev_sample = samples[i-1]
            curr_sample = samples[i]

            for step in range(n_transition_steps):
                alpha = step / (n_transition_steps - 1)
                interpolated_sample = (1 - alpha) * prev_sample + alpha * curr_sample
                interpolated_samples.append(interpolated_sample)
                time_points.append(current_time)
                current_time += dt

            # Hold at current sample
            n_hold_steps = int(hold_time / dt)
            for _ in range(n_hold_steps):
                interpolated_samples.append(curr_sample.copy())
                time_points.append(current_time)
                current_time += dt

    return np.array(interpolated_samples), np.array(time_points)

def run_reachability_sampling(config_file):
    """Main reachability sampling function"""
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
    actuator_limits = config["actuator_limits"]

    print(f"Model loaded: {model.name()}")
    print(f"Target DOFs: {target_dofs}")
    print(f"Number of actuators: {len(model.actuators())}")

    # Find actuator indices
    actuator_indices = find_actuator_indices(model, target_dofs)
    if len(actuator_indices) != len(target_dofs):
        print("ERROR: Could not find all required actuators")
        return

    # Generate samples
    print("\nGenerating samples...")

    # Halton samples
    n_halton = config["sampling"]["halton_samples"]
    halton_samples = generate_halton_samples(n_halton, len(target_dofs), actuator_limits)
    print(f"Generated {len(halton_samples)} Halton samples")

    # Boundary samples
    boundary_mode = config["sampling"]["boundary_mode"]
    boundary_samples = generate_boundary_samples(actuator_limits, boundary_mode)
    print(f"Generated {len(boundary_samples)} boundary samples")

    # Combine samples
    all_samples = np.vstack([halton_samples, boundary_samples])
    print(f"Total samples: {len(all_samples)}")

    # Create smooth trajectory
    transition_time = config["sampling"]["transition_time"]
    hold_time = config["sampling"]["hold_time"]
    dt = config["simulation"]["dt"]

    trajectory_samples, time_points = interpolate_samples(
        all_samples, transition_time, hold_time, dt
    )
    print(f"Interpolated trajectory: {len(trajectory_samples)} points over {time_points[-1]:.1f}s")

    # Initialize simulation
    model.reset()
    model.set_store_data(config["simulation"]["store_data"])
    model.init_state_from_dofs()

    # Phase 1: Settling
    settle_time = config["simulation"]["settle_time"]
    print(f"\nPhase 1: Settling for {settle_time}s...")

    for t in np.arange(0, settle_time, dt):
        zero_inputs = np.zeros(len(model.actuators()))
        model.set_actuator_inputs(zero_inputs)
        model.advance_simulation_to(t)

    print(f"Settling complete at t={model.time():.3f}s")

    # Phase 2: Reachability sampling
    print("Phase 2: Running reachability sampling...")
    start_time = model.time()

    # Track progress
    n_trajectory_points = len(trajectory_samples)
    last_progress = 0

    for i, sample in enumerate(trajectory_samples):
        current_time = start_time + time_points[i]

        # Set actuator inputs for this sample
        motor_inputs = np.zeros(len(model.actuators()))
        for j, dof_name in enumerate(target_dofs):
            actuator_idx = actuator_indices[dof_name]
            motor_inputs[actuator_idx] = sample[j]

        model.set_actuator_inputs(motor_inputs)
        model.advance_simulation_to(current_time)

        # Progress reporting
        progress = int(100 * i / n_trajectory_points)
        if progress > last_progress and progress % 10 == 0:
            print(f"  Progress: {progress}% (t={current_time:.1f}s)")
            last_progress = progress

    print(f"Sampling complete at t={model.time():.3f}s")

    # Save results
    output_dir = config["output"]["directory"]
    filename_prefix = config["output"]["filename_prefix"]
    filename = f"{filename_prefix}_{n_halton}halton_{len(boundary_samples)}boundary_{model.time():.1f}s"

    model.write_results(output_dir, filename)
    print(f"Results saved to {output_dir}/{filename}.sto")

    # Print final statistics
    print("\nFinal joint angles:")
    for dof in model.dofs():
        if dof.name() in target_dofs:
            print(f"  {dof.name()}: {np.degrees(dof.pos()):.1f}°")

    # Try to get hand position if available
    try:
        hand_body = None
        for body in model.bodies():
            if "hand" in body.name().lower():
                hand_body = body
                break

        if hand_body:
            hand_pos = hand_body.com_pos()
            print(f"Final hand position: ({hand_pos.x:.3f}, {hand_pos.y:.3f}, {hand_pos.z:.3f})")
    except:
        print("Could not extract hand position")

    print(f"Open {output_dir}/{filename}.sto in SCONE Studio to visualize results")

    return f"{output_dir}/{filename}.sto"

if __name__ == "__main__":
    result_file = run_reachability_sampling("reachability_config.json")
    print(f"\nReachability sampling completed successfully!")
    print(f"Result file: {result_file}")