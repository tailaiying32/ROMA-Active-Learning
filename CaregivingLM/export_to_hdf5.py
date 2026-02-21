"""
Export SIRS-enhanced joint limit samples to HDF5 format.

Creates a hierarchical HDF5 file with the following structure:

/metadata/
  - n_samples (int)
  - joint_names (string array)
  - main_pairs (string array, flattened)
  - config (JSON string with all configuration parameters)

/samples/sample_000/
  /box_limits/
    - joint_names (string array)
    - lower_bounds (float array)
    - upper_bounds (float array)
  /sirs_bumps/
    /pair_0/
      - joint_names (string array, size 2)
      - n_bumps (int)
      - mu (float array, shape (n_bumps, 2))
      - ls (float array, shape (n_bumps, 2))
      - alpha (float array, shape (n_bumps,))
      - theta (float array, shape (n_bumps,), optional)
    /pair_1/
      ...
  /metadata/
    - sample_id (string)
    - target_feasibility (float array, per pair)
    - actual_feasibility (float array, per pair)
    - pair_names (string array, flattened)
"""

import numpy as np
import h5py
import json
from pathlib import Path

import sirs_sampling_config as config


def export_samples_to_hdf5(samples, output_path, compression='gzip', compression_opts=4):
    """
    Export SIRS-enhanced samples to HDF5 format.

    Args:
        samples: List of SIRS-enhanced sample dictionaries
        output_path: Path to output HDF5 file
        compression: Compression algorithm ('gzip', 'lzf', None)
        compression_opts: Compression level (0-9 for gzip)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(samples)} samples to HDF5...")
    print(f"  Output: {output_path}")
    print(f"  Compression: {compression} (level {compression_opts})")

    with h5py.File(output_path, 'w') as f:
        # ====================================================================
        # Global Metadata
        # ====================================================================
        meta_group = f.create_group('metadata')

        # Number of samples
        meta_group.attrs['n_samples'] = len(samples)

        # Joint names
        joint_names = config.ALL_JOINT_NAMES
        meta_group.create_dataset('joint_names', data=np.array(joint_names, dtype='S'))

        # Main joint pairs (flattened)
        pair_strings = [f"{j1}|{j2}" for j1, j2 in config.MAIN_JOINT_PAIRS]
        meta_group.create_dataset('main_pairs', data=np.array(pair_strings, dtype='S'))

        # Configuration (as JSON string)
        config_dict = config.get_config_dict()
        config_json = json.dumps(config_dict, indent=2)
        meta_group.create_dataset('config', data=config_json)

        # ====================================================================
        # Individual Samples
        # ====================================================================
        samples_group = f.create_group('samples')

        for i, sample in enumerate(samples):
            sample_group = samples_group.create_group(f'sample_{i:03d}')

            # ----------------------------------------------------------------
            # Box Limits
            # ----------------------------------------------------------------
            box_group = sample_group.create_group('box_limits')

            joint_names_ordered = list(sample['joint_limits'].keys())
            lower_bounds = np.array([sample['joint_limits'][j][0] for j in joint_names_ordered])
            upper_bounds = np.array([sample['joint_limits'][j][1] for j in joint_names_ordered])

            box_group.create_dataset('joint_names', data=np.array(joint_names_ordered, dtype='S'))
            box_group.create_dataset('lower_bounds', data=lower_bounds,
                                     compression=compression, compression_opts=compression_opts)
            box_group.create_dataset('upper_bounds', data=upper_bounds,
                                     compression=compression, compression_opts=compression_opts)

            # ----------------------------------------------------------------
            # SIRS Bumps
            # ----------------------------------------------------------------
            if 'sirs_bumps' in sample:
                bumps_group = sample_group.create_group('sirs_bumps')

                for pair_idx, (pair_key, bumps) in enumerate(sample['sirs_bumps'].items()):
                    pair_group = bumps_group.create_group(f'pair_{pair_idx}')

                    # Joint names for this pair
                    j1, j2 = pair_key
                    pair_group.create_dataset('joint_names', data=np.array([j1, j2], dtype='S'))

                    # Number of bumps
                    n_bumps = len(bumps)
                    pair_group.attrs['n_bumps'] = n_bumps

                    if n_bumps > 0:
                        # Extract bump parameters
                        mu_list = [b['mu'] for b in bumps]
                        ls_list = [b['ls'] for b in bumps]
                        alpha_list = [b['alpha'] for b in bumps]

                        pair_group.create_dataset('mu', data=np.array(mu_list),
                                                 compression=compression, compression_opts=compression_opts)
                        pair_group.create_dataset('ls', data=np.array(ls_list),
                                                 compression=compression, compression_opts=compression_opts)
                        pair_group.create_dataset('alpha', data=np.array(alpha_list),
                                                 compression=compression, compression_opts=compression_opts)

                        # Optional theta (rotation angle)
                        if 'theta' in bumps[0]:
                            theta_list = [b['theta'] for b in bumps]
                            pair_group.create_dataset('theta', data=np.array(theta_list),
                                                     compression=compression, compression_opts=compression_opts)

            # ----------------------------------------------------------------
            # Sample Metadata
            # ----------------------------------------------------------------
            if 'sirs_metadata' in sample:
                meta_group = sample_group.create_group('metadata')

                # Sample ID
                meta_group.attrs['sample_id'] = str(sample.get('id', f'sample_{i}'))

                # Per-pair metadata
                n_pairs = len(sample['sirs_metadata'])
                target_feas = np.zeros(n_pairs)
                actual_feas = np.zeros(n_pairs)
                pair_names_list = []

                for pair_idx, (pair_key, pair_meta) in enumerate(sample['sirs_metadata'].items()):
                    j1, j2 = pair_key
                    pair_names_list.append(f"{j1}|{j2}")

                    target_feas[pair_idx] = pair_meta['target_feasibility']
                    actual_feas[pair_idx] = pair_meta['actual_feasibility']

                meta_group.create_dataset('pair_names', data=np.array(pair_names_list, dtype='S'))
                meta_group.create_dataset('target_feasibility', data=target_feas,
                                         compression=compression, compression_opts=compression_opts)
                meta_group.create_dataset('actual_feasibility', data=actual_feas,
                                         compression=compression, compression_opts=compression_opts)

                # Store sampled SIRS parameters
                if 'sirs_params' in sample:
                    params = sample['sirs_params']
                    params_group = meta_group.create_group('sirs_params')

                    # Store each parameter
                    for key, value in params.items():
                        if value is not None:
                            params_group.attrs[key] = float(value) if isinstance(value, (int, float, np.number)) else value

            # Progress
            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                print(f"  Exported {i+1}/{len(samples)} samples...", end='\r')

    print(f"\n✓ Exported {len(samples)} samples to {output_path}")

    # Print file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")


def load_sample_from_hdf5(hdf5_path, sample_id):
    """
    Load a single sample from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        sample_id: Sample index (0-based)

    Returns:
        Sample dictionary in same format as original samples
    """
    with h5py.File(hdf5_path, 'r') as f:
        sample_key = f'sample_{sample_id:03d}'
        sample_group = f['samples'][sample_key]

        # Load box limits
        box_group = sample_group['box_limits']
        joint_names = [s.decode('utf-8') for s in box_group['joint_names'][:]]
        lower_bounds = box_group['lower_bounds'][:]
        upper_bounds = box_group['upper_bounds'][:]

        joint_limits = {
            name: (float(lower), float(upper))
            for name, lower, upper in zip(joint_names, lower_bounds, upper_bounds)
        }

        # Load SIRS bumps
        sirs_bumps = {}
        sirs_metadata = {}

        if 'sirs_bumps' in sample_group:
            bumps_group = sample_group['sirs_bumps']

            for pair_key in bumps_group.keys():
                pair_group = bumps_group[pair_key]

                # Joint names for this pair
                j1, j2 = [s.decode('utf-8') for s in pair_group['joint_names'][:]]
                pair_tuple = (j1, j2)

                # Load bumps
                n_bumps = pair_group.attrs['n_bumps']
                bumps = []

                if n_bumps > 0:
                    mu_array = pair_group['mu'][:]
                    ls_array = pair_group['ls'][:]
                    alpha_array = pair_group['alpha'][:]

                    has_theta = 'theta' in pair_group
                    theta_array = pair_group['theta'][:] if has_theta else None

                    for i in range(n_bumps):
                        bump = {
                            'mu': mu_array[i],
                            'ls': ls_array[i],
                            'alpha': float(alpha_array[i])
                        }

                        if has_theta:
                            bump['theta'] = float(theta_array[i])

                        bumps.append(bump)

                sirs_bumps[pair_tuple] = bumps

        # Load metadata
        sample_metadata = {}
        if 'metadata' in sample_group:
            meta_group = sample_group['metadata']
            sample_id_str = meta_group.attrs['sample_id']

            pair_names = [s.decode('utf-8') for s in meta_group['pair_names'][:]]
            target_feas = meta_group['target_feasibility'][:]
            actual_feas = meta_group['actual_feasibility'][:]

            for pair_name, target, actual in zip(pair_names, target_feas, actual_feas):
                j1, j2 = pair_name.split('|')
                pair_tuple = (j1, j2)

                sirs_metadata[pair_tuple] = {
                    'target_feasibility': float(target),
                    'actual_feasibility': float(actual),
                    'n_bumps': len(sirs_bumps.get(pair_tuple, [])),
                }

            sample_metadata['id'] = sample_id_str

        # Reconstruct sample dictionary
        sample = {
            'id': sample_metadata.get('id', f'sample_{sample_id}'),
            'joint_limits': joint_limits,
            'sirs_bumps': sirs_bumps,
            'sirs_metadata': sirs_metadata,
        }

        return sample


def print_hdf5_structure(hdf5_path, max_samples=3):
    """Print the structure of an HDF5 file."""
    print("\n" + "=" * 70)
    print(f"HDF5 Structure: {hdf5_path}")
    print("=" * 70)

    with h5py.File(hdf5_path, 'r') as f:
        # Global metadata
        print("\n[Global Metadata]")
        meta = f['metadata']
        print(f"  n_samples: {meta.attrs['n_samples']}")
        print(f"  joint_names: {len(meta['joint_names'])} joints")
        print(f"  main_pairs: {len(meta['main_pairs'])} pairs")

        # Sample structure
        print(f"\n[Sample Structure] (showing first {max_samples} samples)")
        samples_group = f['samples']

        for i, sample_key in enumerate(list(samples_group.keys())[:max_samples]):
            sample = samples_group[sample_key]

            print(f"\n  {sample_key}:")
            print(f"    /box_limits/")
            print(f"      - {len(sample['box_limits']['joint_names'])} joints")

            if 'sirs_bumps' in sample:
                print(f"    /sirs_bumps/")
                print(f"      - {len(sample['sirs_bumps'])} pairs")

                for pair_key in list(sample['sirs_bumps'].keys())[:2]:
                    pair = sample['sirs_bumps'][pair_key]
                    j1, j2 = [s.decode('utf-8') for s in pair['joint_names'][:]]
                    n = pair.attrs['n_bumps']
                    print(f"        /{pair_key}/ ({j1} × {j2}): {n} bumps")

            if 'metadata' in sample:
                print(f"    /metadata/")
                meta = sample['metadata']
                print(f"      - sample_id: {meta.attrs['sample_id']}")

    print("=" * 70)


if __name__ == '__main__':
    from sirs_batch_sampler import generate_sirs_enhanced_joint_limits

    print("=" * 70)
    print("Phase 8: HDF5 Export")
    print("=" * 70)

    # Generate samples (or load existing if available)
    n_total = config.N_SAMPLES_PER_BATCH * 10  # 10 batches hardcoded in batch_joint_limit_sampler
    print(f"\nGenerating {n_total} SIRS-enhanced samples...")
    samples = generate_sirs_enhanced_joint_limits(
        n_samples_per_batch=config.N_SAMPLES_PER_BATCH,
        reject_disconnected=config.REJECT_DISCONNECTED,
        max_rejection_attempts=config.MAX_REJECTION_ATTEMPTS,
        verbose=True
    )

    # Generate filename based on actual sample count
    n_actual = len(samples)
    hdf5_filename = f'sirs_samples_{n_actual}.h5'
    output_path = Path(config.OUTPUT_DIR) / hdf5_filename

    print(f"\n✓ Generated {n_actual} samples")
    print(f"  Saving to: {hdf5_filename}")

    # Export to HDF5
    export_samples_to_hdf5(
        samples,
        output_path,
        compression=config.HDF5_COMPRESSION,
        compression_opts=config.HDF5_COMPRESSION_LEVEL
    )

    # Print structure
    print_hdf5_structure(output_path, max_samples=3)

    # Test loading a sample
    print("\n[Testing Load]")
    sample_0 = load_sample_from_hdf5(output_path, sample_id=0)
    print(f"✓ Loaded sample 0: {len(sample_0['joint_limits'])} joints, {len(sample_0['sirs_bumps'])} pairs")

    print("\n✓ Phase 8 complete!")
    print(f"\nOutput file: {output_path}")
