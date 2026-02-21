import numpy as np
import time
import sys
import os
import glob
import re
import argparse
import json
from datetime import datetime
from sconetools import sconepy

# Set the SCONE log level to 3
sconepy.set_log_level(3)

def halton_sequence(index, base):
    """Generate the index-th number in the Halton sequence with given base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result

def generate_halton_points_in_cube(n, bases=[2, 3, 5]):
    """Generate n points using Halton sequence in unit cube [0,1]³."""
    points = np.zeros((n, 3))
    for i in range(n):
        for dim in range(3):
            points[i, dim] = halton_sequence(i + 1, bases[dim])  # +1 to skip 0
    return points

def cube_to_sphere_equal_volume(cube_points):
    """
    Transform points from unit cube [0,1]³ to unit sphere using equal-volume mapping.
    This uses the inverse transform sampling method:
    1. u1 -> radius via r = u1^(1/3) (equal volume shells)
    2. u2 -> elevation angle via cos(θ) = 2*u2 - 1 (equal area latitude bands)
    3. u3 -> azimuth angle via φ = 2π*u3 (uniform around equator)
    """
    u1, u2, u3 = cube_points[:, 0], cube_points[:, 1], cube_points[:, 2]
    
    # Radius: r = u1^(1/3) for equal volume shells
    r = np.power(u1, 1.0/3.0)
    
    # Elevation angle: cos(theta) = 2*u2 - 1 for equal area latitude bands
    cos_theta = 2.0 * u2 - 1.0
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    
    # Azimuth angle: phi = 2*pi*u3 for uniform distribution around equator
    phi = 2.0 * np.pi * u3
    
    # Convert to Cartesian coordinates
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    
    return np.column_stack([x, y, z])

def generate_halton_sphere_samples(n, center, radius, filename=None):
    """
    Generate n points uniformly distributed in a sphere using Halton sequences.
    
    Args:
        n: Number of points to generate
        center: Sphere center as [x, y, z]
        radius: Sphere radius
        filename: Optional filename to save samples to
    
    Returns:
        Array of shape (n, 3) containing the sphere points
    """
    # Generate Halton points in unit cube
    cube_points = generate_halton_points_in_cube(n)
    
    # Transform to unit sphere
    unit_sphere_points = cube_to_sphere_equal_volume(cube_points)
    
    # Scale and translate to target sphere
    sphere_points = radius * unit_sphere_points + np.array(center)
    
    # Save to file if requested
    if filename:
        sample_data = {
            'n_samples': n,
            'sphere_center': center.tolist() if isinstance(center, np.ndarray) else center,
            'sphere_radius': radius,
            'samples': sphere_points.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Saved {n} Halton sphere samples to {filename}")
    
    return sphere_points

def load_halton_sphere_samples(filename):
    """Load pre-generated Halton sphere samples from file."""
    with open(filename, 'r') as f:
        data = json.load(f)

    samples = np.array(data['samples'])
    print(f"Loaded {len(samples)} Halton sphere samples from {filename}")
    print(f"Sphere center: {data['sphere_center']}")
    print(f"Sphere radius: {data['sphere_radius']}")

    return samples, data['sphere_center'], data['sphere_radius']

def load_hand_positions(filename):
    """Load hand positions from coverage sampling."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Hand positions file not found: {filename}")

    hand_positions = np.loadtxt(filename, skiprows=1)  # Skip header
    print(f"Loaded {len(hand_positions)} hand positions from {filename}")
    return hand_positions

def find_init_files(init_dir):
    """Find all init .zml files in the specified directory."""
    if not os.path.exists(init_dir):
        raise FileNotFoundError(f"Init files directory not found: {init_dir}")

    init_files = glob.glob(os.path.join(init_dir, "*.zml"))
    init_files.sort()  # Sort to match with hand positions by index
    print(f"Found {len(init_files)} init files in {init_dir}")
    return init_files

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run SCONE optimization with restart mechanism')
parser.add_argument('scenario_file', help='Path to the scenario file (e.g., scenes/single_arm_c1.scone)')
parser.add_argument('measure_file', help='Path to the measure file (e.g., measures/single_1.lua)')
parser.add_argument('--mode', choices=['regular', 'grav_comp'], default='regular', help='Optimization mode: regular (default) or grav_comp (gravity compensation)')
parser.add_argument('--samples-file', default='halton_samples_200.json', help='Path to pre-generated Halton samples JSON file (default: halton_samples_200.json)')
parser.add_argument('--generate-samples', type=int, help='Generate N Halton samples and save to samples file')
parser.add_argument('--sample-index', type=int, default=0, help='Index of sample to use from samples file (default: 0)')
parser.add_argument('--init-files-dir', default='init/coverage_samples', help='Directory containing init files for grav_comp mode (default: init/coverage_samples)')
parser.add_argument('--hand-positions-file', default='init/coverage_samples/halton_sample_hand_positions.txt', help='File containing hand positions for grav_comp mode')
parser.add_argument('--debug', '--no-cleanup', action='store_true', help='Disable file cleanup for debugging')
parser.add_argument('--no-restart', action='store_true', help='Disable restart optimization, run only initial optimizations')

args = parser.parse_args()
scenario_file = args.scenario_file
measure_file = args.measure_file
mode = args.mode
samples_file = args.samples_file
generate_samples = args.generate_samples
sample_index = args.sample_index
init_files_dir = args.init_files_dir
hand_positions_file = args.hand_positions_file
debug_mode = args.debug
use_restart = not args.no_restart

# Show which SCONE version we are using
print("SCONE Version", sconepy.version())
print(f"Running in {mode} mode")

# Load the specified scenario
scenario = sconepy.load_scenario(scenario_file)

# Handle different modes
if mode == "grav_comp":
    # Gravity compensation mode - use init files and hand positions
    try:
        hand_positions = load_hand_positions(hand_positions_file)
        init_files = find_init_files(init_files_dir)

        # Verify that we have matching number of init files and hand positions
        if len(init_files) != len(hand_positions):
            print(f"Error: Mismatch between init files ({len(init_files)}) and hand positions ({len(hand_positions)})")
            sys.exit(1)

        # Select init file and hand position using sample index
        if sample_index >= len(init_files):
            print(f"Error: Sample index {sample_index} is out of range (0-{len(init_files)-1})")
            sys.exit(1)

        selected_init_file = init_files[sample_index]
        target_x, target_y, target_z = hand_positions[sample_index]

        print(f"Using gravity compensation sample {sample_index}/{len(init_files)-1}")
        # print(f"Init file: {os.path.basename(selected_init_file)}")
        print(f"Target hand position: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

        # Set init file and duration for gravity compensation
        # scenario.set("CmaOptimizer.SimulationObjective.ModelHyfydyPrecise.state_init_file", f"../{selected_init_file}")
        # scenario.set("CmaOptimizer.SimulationObjective.max_duration", "1.0")

        # print("Modified scenario for gravity compensation:")
        # print(f"  state_init_file = ../{selected_init_file}")
        # print("  max_duration = 1.0")
        # print("  Note: Timing parameters (rest_dur=0, trg_dur=1.0, react_dur=0.0) should be set in .scone file")

    except FileNotFoundError as e:
        print(f"Error in gravity compensation mode: {e}")
        sys.exit(1)

else:
    # Regular mode - use sphere sampling
    # Calculate sphere parameters from previous box
    x_min, x_max = -0.85, 0.85
    y_min, y_max = 0.4, 2.1
    z_min, z_max = -0.7, 1.0

    # Sphere center (center of the box)
    center_x = (x_min + x_max) / 2  # 0.0
    center_y = (y_min + y_max) / 2  # 1.25
    center_z = (z_min + z_max) / 2  # 0.15

    # Sphere radius (largest box dimension)
    radius_x = (x_max - x_min) / 2  # 0.85
    radius_y = (y_max - y_min) / 2  # 0.85
    radius_z = (z_max - z_min) / 2  # 0.85
    radius = max(radius_x, radius_y, radius_z) + 0.2  # 0.85

    center = [center_x, center_y, center_z]

    print(f"Sphere center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})")
    print(f"Sphere radius: {radius:.3f}")

    # Handle sample generation or loading
    if generate_samples:
        if not samples_file:
            samples_file = f"halton_samples_{generate_samples}.json"

        print(f"Generating {generate_samples} Halton samples and saving to {samples_file}")
        samples = generate_halton_sphere_samples(generate_samples, center, radius, samples_file)
        print("Sample generation complete. Use this file with --samples-file for future runs.")
        sys.exit(0)

    else:
        # Load pre-generated samples (default behavior)
        if not os.path.exists(samples_file):
            print(f"Error: Default samples file {samples_file} not found!")
            print("Generate samples first using: --generate-samples 100")
            sys.exit(1)

        samples, loaded_center, loaded_radius = load_halton_sphere_samples(samples_file)

        # Verify sphere parameters match
        if not (np.allclose(loaded_center, center, rtol=1e-6) and np.isclose(loaded_radius, radius, rtol=1e-6)):
            print("Warning: Loaded samples have different sphere parameters!")
            print(f"Expected center: {center}, radius: {radius}")
            print(f"Loaded center: {loaded_center}, radius: {loaded_radius}")

        # Select target using sample index
        if sample_index >= len(samples):
            print(f"Error: Sample index {sample_index} is out of range (0-{len(samples)-1})")
            sys.exit(1)

        target_point = samples[sample_index]
        target_x, target_y, target_z = target_point
        print(f"Using Halton sample {sample_index}/{len(samples)-1}: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

print(f"Target for all runs: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

# Generate timestamp with milliseconds for unique folder names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# Start multiple optimizations with different settings
num_optimizers = 6
initial_opts = []
restart_opts = []

print("=== PHASE 1: Starting initial optimizations ===")
for i in range(42, 42+num_optimizers):
	# Change some settings based on i
	scenario.set("CmaOptimizer.random_seed", str(i+1))

	scenario.set("CmaOptimizer.min_progress", str("1e-5"))
	target_fitness = 0.05
	scenario.set("CmaOptimizer.target_fitness", str(target_fitness))

	scenario.set("CmaOptimizer.min_improvement_for_file_output", str(1.0))
	mode_suffix = f"_{mode}" if mode == "grav_comp" else ""
	scenario.set("CmaOptimizer.signature_prefix", f"{scenario_file.split('/')[-1].split('.')[0]}_{measure_file.split('/')[-1].split('.')[0]}{mode_suffix}_initial_{i+1}_{timestamp}")
	# print(scenario.get("CmaOptimizer.SimulationObjective.CompositeMeasure.ScriptMeasure.script_file"))
	


	# Set the script file for the measure
	scenario.set("CmaOptimizer.SimulationObjective.CompositeMeasure.ScriptMeasure.script_file", "../" + measure_file)

	# Set the same target position for all runs
	scenario.set("CmaOptimizer.SimulationObjective.CompositeMeasure.ScriptMeasure.target_x", str(target_x))
	scenario.set("CmaOptimizer.SimulationObjective.CompositeMeasure.ScriptMeasure.target_y", str(target_y))
	scenario.set("CmaOptimizer.SimulationObjective.CompositeMeasure.ScriptMeasure.target_z", str(target_z))
	


	# Set muscle parameters
	# scenario.set("CmaOptimizer.SimulationObjective.ModelHyfydyPrecise.Properties.bic_long_r.max_isometric_force.factor", str(0.1))
	
	# Debug print muscle parameters
	# print(scenario.get("CmaOptimizer.SimulationObjective.ModelHyfydyPrecise.model_options.use_legacy_pin_joints"))
	# print(scenario.get("CmaOptimizer.SimulationObjective.ModelHyfydyPrecise.Properties"))
	# print(scenario.get("CmaOptimizer.SimulationObjective.ModelHyfydyPrecise.Properties.bic_long_r.max_isometric_force"))
	# print(scenario.get("CmaOptimizer.SimulationObjective.ModelHyfydyPrecise.Properties.bic_long_r.max_isometric_force.factor"))

	# Start the initial optimization as background task
	opt = scenario.start_optimization()
	initial_opts.append(opt)
	print(f"Started initial optimization {i+1}")

# Wait for all initial optimizations to finish or early termination
print("\n=== Waiting for initial optimizations to complete ===")
print(f"Target fitness for early termination: {target_fitness}")
num_finished = 0
early_termination = False
while num_finished < num_optimizers and not early_termination:
	num_finished = 0
	time.sleep(1)
	# Iterate over all initial optimizers
	for i in range(0, num_optimizers):
		# Create a status string
		current_fitness = initial_opts[i].fitness()
		status_str = f"Initial optimization {i}: step={initial_opts[i].current_step()} fitness={current_fitness:.2f}";

		# Check for early termination condition FIRST (before checking finished status)
		if current_fitness <= target_fitness:
			status_str += f" TARGET_ACHIEVED"
			early_termination = True
			# If this run achieved target, we still count it as finished
			if initial_opts[i].finished():
				num_finished += 1
		elif initial_opts[i].finished():
			# This optimization is finished, add to status string
			status_str += " FINISHED"
			num_finished += 1
		elif initial_opts[i].current_step() >= 15000:
			# Optional: terminal optimizations after 15000 steps
			status_str += " TERMINATING"
			initial_opts[i].terminate()

		# Print status
		print(status_str, flush=True)

	# If early termination triggered, terminate all other running optimizations
	if early_termination:
		print(f"\n*** EARLY TERMINATION: Target fitness {target_fitness} achieved! ***")
		print("Terminating all other running optimizations...")
		for i in range(0, num_optimizers):
			if not initial_opts[i].finished():
				initial_opts[i].terminate()
				print(f"Terminated initial optimization {i}")

		# Wait a bit for terminations to complete
		time.sleep(2)
		break

if use_restart and not early_termination:
	print("\n=== PHASE 2: Starting restart optimizations ===")
	# Start restart optimizations using best .par files from initial runs
	for i in range(0, num_optimizers):
		# Find best .par file from initial run
		initial_output_folder = initial_opts[i].output_folder()
		par_files = glob.glob(os.path.join(initial_output_folder, "*.par"))
		
		best_par_file = None
		if par_files:
			# Parse step numbers from filenames and find the one with biggest step
			max_step = -1
			for par_file in par_files:
				filename = os.path.basename(par_file)
				# Extract step number from filename like "0007_88.459_9.665.par"
				match = re.match(r'^(\d+)_.*\.par$', filename)
				if match:
					step_num = int(match.group(1))
					if step_num > max_step:
						max_step = step_num
						best_par_file = par_file
		
		if best_par_file:
			print(f"Found best .par file for restart {i+1}: {os.path.basename(best_par_file)} (step {max_step})")
			
			# Configure restart optimization
			scenario.set("CmaOptimizer.random_seed", str(i+42))  # Original seed + 42
			mode_suffix = f"_{mode}" if mode == "grav_comp" else ""
			scenario.set("CmaOptimizer.signature_prefix", f"{scenario_file.split('/')[-1].split('.')[0]}_{measure_file.split('/')[-1].split('.')[0]}{mode_suffix}_restart_{i+1}_{timestamp}")
			
			# Set init parameters for restart
			scenario.set("CmaOptimizer.init.file", best_par_file)
			scenario.set("CmaOptimizer.init.std_factor", "20")
			
			# Start the restart optimization
			restart_opt = scenario.start_optimization()
			restart_opts.append(restart_opt)
			print(f"Started restart optimization {i+1} with seed {i+1+42}")
		else:
			print(f"Warning: No .par file found for initial run {i+1}, skipping restart")
			restart_opts.append(None)

	# Update main optimization list to only include restart runs
	opts = [opt for opt in restart_opts if opt is not None]
	num_optimizers = len(opts)
elif early_termination:
	print("\n=== Restart optimization skipped due to early termination ===")
	# Use initial optimizations as final results
	opts = initial_opts
	num_optimizers = len(opts)
else:
	print("\n=== Restart optimization disabled ===")
	# Use initial optimizations as final results
	opts = initial_opts
	num_optimizers = len(opts)



# Wait for all optimizations to finish or early termination
if not early_termination:  # Only run this loop if we didn't already terminate early
	print("\n=== Waiting for final optimizations to complete ===")
	print(f"Target fitness for early termination: {target_fitness}")
	num_finished = 0
	final_early_termination = False
	while num_finished < num_optimizers and not final_early_termination:
		num_finished = 0
		time.sleep(1)
		# Iterate over all optimizers
		for i in range(0, num_optimizers):
			# Create a status string
			current_fitness = opts[i].fitness()
			status_str = f"Optimization {i}: step={opts[i].current_step()} fitness={current_fitness:.2f}";

			# Check for early termination condition FIRST (before checking finished status)
			if current_fitness <= target_fitness:
				status_str += f" TARGET_ACHIEVED"
				final_early_termination = True
				# If this run achieved target, we still count it as finished
				if opts[i].finished():
					num_finished += 1
			elif opts[i].finished():
				# This optimization is finished, add to status string
				status_str += " FINISHED"
				num_finished += 1
			elif opts[i].current_step() >= 15000:
				# Optional: terminal optimizations after 15000 steps
				status_str += " TERMINATING"
				opts[i].terminate()

			# Print status
			print(status_str, flush=True)

		# If early termination triggered, terminate all other running optimizations
		if final_early_termination:
			print(f"\n*** EARLY TERMINATION: Target fitness {target_fitness} achieved! ***")
			print("Terminating all other running optimizations...")
			for i in range(0, num_optimizers):
				if not opts[i].finished():
					opts[i].terminate()
					print(f"Terminated optimization {i}")

			# Wait a bit for terminations to complete
			time.sleep(2)
			break
else:
	print("\n=== Using results from early-terminated initial runs ===")
	# Set final_early_termination for consistency
	final_early_termination = True

print("Output folder: ", opts[0].output_folder())
# At this point, all optimizations have finished or been terminated
termination_reason = "early termination due to target fitness achievement" if (early_termination or final_early_termination) else "completion"
print(f"All optimizations stopped due to: {termination_reason}")

for i in range(0, num_optimizers):
	print(f"Final optimization {i}: steps={opts[i].current_step()} fitness={opts[i].fitness():.2f}", flush=True)

# Find the run with the lowest (best) fitness at time of termination
best_fitness = float('inf')
best_run_index = -1
for i in range(0, num_optimizers):
	if opts[i].fitness() < best_fitness:
		best_fitness = opts[i].fitness()
		best_run_index = i

termination_status = " (achieved target)" if best_fitness <= target_fitness else ""
print(f"\nBest run: Optimization {best_run_index} with fitness {best_fitness:.2f}{termination_status}")

# Find the .par file with the biggest step number and evaluate it
best_output_folder = opts[best_run_index].output_folder()
import glob
import re

# Find all .par files in the best run's output folder
par_files = glob.glob(os.path.join(best_output_folder, "*.par"))

if par_files:
	# Parse step numbers from filenames and find the one with biggest step
	max_step = -1
	best_par_file = None
	
	for par_file in par_files:
		filename = os.path.basename(par_file)
		# Extract step number from filename like "0007_88.459_9.665.par"
		match = re.match(r'^(\d+)_.*\.par$', filename)
		if match:
			step_num = int(match.group(1))
			if step_num > max_step:
				max_step = step_num
				best_par_file = par_file
	
	if best_par_file:
		print(f"Found best .par file: {os.path.basename(best_par_file)} (step {max_step})")
		
		# Print the target used for evaluation
		print(f"Using target for evaluation: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")
		
		# Use evaluate_par_file to generate analysis .sto file
		print("Generating analysis .sto file...")
		sconepy.evaluate_par_file(best_par_file)
		print(f"Analysis completed.")
		
		# Extract distance from the generated .sto file
		sto_file_path = best_par_file + ".sto"
		distance_value = None
		
		if os.path.exists(sto_file_path):
			print(f"Reading .sto file: {sto_file_path}")
			try:
				with open(sto_file_path, 'r') as sto_file:
					lines = sto_file.readlines()
					
					# Find the header line to locate reach.distance column
					header_idx = -1
					reach_distance_col = -1
					
					for i, line in enumerate(lines):
						if line.startswith('time\t') or 'reach.distance' in line:
							header_idx = i
							columns = line.strip().split('\t')
							for j, col in enumerate(columns):
								if col == 'reach.distance':
									reach_distance_col = j
									break
							break
					
					# Read the last data line to get reach.distance value
					if header_idx >= 0 and reach_distance_col >= 0:
						# Find the last non-empty data line
						for line in reversed(lines[header_idx+1:]):
							line = line.strip()
							if line and not line.startswith('#'):
								data = line.split('\t')
								if len(data) > reach_distance_col:
									distance_value = float(data[reach_distance_col])
									break
						
						print(f"Extracted reach.distance from last timestep: {distance_value}")
					else:
						print("Could not find reach.distance column in .sto file")
			
			except Exception as e:
				print(f"Error reading .sto file: {e}")
		else:
			print(f".sto file not found: {sto_file_path}")
		
		# Extract and save script measure value and target pose as JSON
		import json
		
		# Collect fitness values for all runs
		fitness_values = []
		for i in range(num_optimizers):
			fitness_values.append(opts[i].fitness())
		
		result_data = {
			"distance": distance_value,
			"target_pose": {
				"x": target_x,
				"y": target_y,
				"z": target_z
			},
			"best_run_index": best_run_index,
			"best_par_file": os.path.basename(best_par_file),
			"max_step": max_step,
			"all_runs_fitness": fitness_values
		}
		
		# Save to JSON file in the output folder
		json_file_path = os.path.join(best_output_folder, "evaluation_results.json")
		with open(json_file_path, 'w') as json_file:
			json.dump(result_data, json_file, indent=2)
		
		print(f"Saved evaluation results to: {json_file_path}")
		print(f"Distance: {distance_value}")
	else:
		print("No valid .par files found with step number pattern")
else:
	print("No .par files found in output directory")

# Store output folder paths before freeing references
print("Storing output folder paths...")
output_folders = []
initial_output_folders = []

# Store restart output folders
for i in range(0, num_optimizers):
	output_folders.append(opts[i].output_folder())

# Store initial output folders  
for i in range(0, len(initial_opts)):
	initial_output_folders.append(initial_opts[i].output_folder())

# Free all held folders by explicitly releasing optimizer references
print("Freeing all optimization references to release folder handles...")
for i in range(0, num_optimizers):
	# Force garbage collection of optimizer objects to release file handles
	opts[i] = None

# Free initial optimizer references
for i in range(0, len(initial_opts)):
	initial_opts[i] = None

# Additional wait to ensure all file handles are released
print("Waiting for file handles to be released...")
time.sleep(5)

# Force garbage collection
import gc
gc.collect()

if debug_mode:
	print("Debug mode enabled - skipping file cleanup")
	print(f"All output folders preserved:")
	if use_restart:
		print(f"  Best restart run: {output_folders[best_run_index]}")
		for i in range(num_optimizers):
			if i != best_run_index:
				print(f"  Other restart run {i}: {output_folders[i]}")
		for i, initial_folder in enumerate(initial_output_folders):
			print(f"  Initial run {i}: {initial_folder}")
	else:
		print(f"  Best initial run: {output_folders[best_run_index]}")
		for i in range(num_optimizers):
			if i != best_run_index:
				print(f"  Other initial run {i}: {output_folders[i]}")
else:
	print("Cleaning up output folders...")

	# Remove output folders from all runs except the best one
	import os
	import shutil
	for i in range(0, num_optimizers):
		if i != best_run_index:
			output_folder = output_folders[i]
			
			if os.path.exists(output_folder):
				try:
					shutil.rmtree(output_folder)
					print(f"Removed output folder for run {i}: {output_folder}")
				except OSError as e:
					print(f"Warning: Could not remove folder {output_folder}: {e}")
					# Try to remove individual files if directory removal fails
					try:
						for root, dirs, files in os.walk(output_folder):
							for file in files:
								try:
									os.remove(os.path.join(root, file))
								except OSError:
									pass
						# Try to remove empty directories
						for root, dirs, files in os.walk(output_folder, topdown=False):
							for dir in dirs:
								try:
									os.rmdir(os.path.join(root, dir))
								except OSError:
									pass
						os.rmdir(output_folder)
						print(f"Successfully removed folder after retry: {output_folder}")
					except OSError:
						print(f"Failed to completely remove folder: {output_folder}")

	if use_restart:
		# Remove all initial run output folders
		print("Removing all initial run output folders...")
		for i, initial_folder in enumerate(initial_output_folders):
			if os.path.exists(initial_folder):
				try:
					shutil.rmtree(initial_folder)
					print(f"Removed initial output folder for run {i}: {initial_folder}")
				except OSError as e:
					print(f"Warning: Could not remove initial folder {initial_folder}: {e}")
					# Try to remove individual files if directory removal fails
					try:
						for root, dirs, files in os.walk(initial_folder):
							for file in files:
								try:
									os.remove(os.path.join(root, file))
								except OSError:
									pass
						# Try to remove empty directories
						for root, dirs, files in os.walk(initial_folder, topdown=False):
							for dir in dirs:
								try:
									os.rmdir(os.path.join(root, dir))
								except OSError:
									pass
						os.rmdir(initial_folder)
						print(f"Successfully removed initial folder after retry: {initial_folder}")
					except OSError:
						print(f"Failed to completely remove initial folder: {initial_folder}")

		print(f"Cleanup complete. Removed all initial runs and {len([i for i in range(num_optimizers) if i != best_run_index])} non-best restart runs.")
		print(f"Only results from the best restart run (index {best_run_index}) remain.")
	else:
		print(f"Cleanup complete. Removed {len([i for i in range(num_optimizers) if i != best_run_index])} non-best initial runs.")
		print(f"Only results from the best initial run (index {best_run_index}) remain.")