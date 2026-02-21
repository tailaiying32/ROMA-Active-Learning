#!/usr/bin/env python3
"""
apply_best_hyperparams.py

Utility to read a flat JSON of best hyperparameters (from e.g. Optuna) and apply them
into the project's YAML configuration (`configs/default.yaml`).

Usage:
    python scripts/apply_best_hyperparams.py \
        --json best_hyperparameters.json \
        --yaml configs/default.yaml \
        [--backup]

The script uses a mapping from flattened json keys to YAML paths and will print a
summary of updated keys.
"""
import argparse
import json
from pathlib import Path
import yaml
import sys

# Mapping from flat json key -> yaml nested key path (as list)
KEY_MAP = {
    'init_std': ['prior', 'init_std'],

    'bald_tau': ['bald', 'tau'],
    'bald_n_mc_samples': ['bald', 'n_mc_samples'],

    'bald_n_restarts': ['bald_optimization', 'n_restarts'],
    'bald_n_iters_per_restart': ['bald_optimization', 'n_iters_per_restart'],
    'bald_lr_adam': ['bald_optimization', 'lr_adam'],
    'bald_lr_sgd': ['bald_optimization', 'lr_sgd'],
    'bald_switch_to_sgd_at': ['bald_optimization', 'switch_to_sgd_at'],
    'bald_plateau_patience': ['bald_optimization', 'plateau_patience'],
    'bald_plateau_threshold': ['bald_optimization', 'plateau_threshold'],

    'vi_n_mc_samples': ['vi', 'n_mc_samples'],
    'vi_lr': ['vi', 'learning_rate'],
    'vi_optimizer_type': ['vi', 'optimizer_type'],
    'vi_convergence_tol': ['vi', 'convergence_tol'],
    'vi_max_iters': ['vi', 'max_iters'],
    'vi_kl_weight': ['vi', 'kl_weight'],
    'vi_grad_clip': ['vi', 'grad_clip'],

    # stopping-related
    'stopping_elbo_plateau_patience': ['stopping', 'elbo_plateau_window'],
    'stopping_elbo_plateau_threshold': ['stopping', 'elbo_plateau_threshold'],
    'stopping_uncertainty_threshold': ['stopping', 'uncertainty_threshold'],
    'stopping_bald_threshold': ['stopping', 'bald_threshold'],
    'stopping_bald_patience': ['stopping', 'bald_patience'],
}


def set_in_dict(d, keys, value):
    """Set a nested key in dict `d` following list `keys` creating intermediate dicts if needed."""
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_from_dict(d, keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def apply_mappings(json_data, yaml_data):
    changes = {}
    for jkey, jval in json_data.items():
        if jkey not in KEY_MAP:
            # skip unknown keys but record them
            changes[jkey] = ('unknown_key', jval)
            continue
        path = KEY_MAP[jkey]
        old = get_from_dict(yaml_data, path)
        set_in_dict(yaml_data, path, jval)
        changes['.'.join(path)] = (old, jval)
    return changes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', type=Path, default=Path('best_hyperparameters.json'))
    parser.add_argument('--yaml', '-y', type=Path, default=Path('configs/default.yaml'))
    parser.add_argument('--backup', '-b', action='store_true', help='Create a backup of the YAML before overwriting')
    args = parser.parse_args(argv)

    jpath = args.json
    ypath = args.yaml

    if not jpath.exists():
        print(f"Error: JSON file not found: {jpath}")
        sys.exit(1)
    if not ypath.exists():
        print(f"Error: YAML file not found: {ypath}")
        sys.exit(1)

    with open(jpath, 'r') as f:
        jdata = json.load(f)

    # Read the YAML file as text so we can update scalars in-place and preserve comments/formatting
    with open(ypath, 'r') as f:
        lines = f.readlines()

    if args.backup:
        bak_path = ypath.with_suffix('.yaml.bak')
        with open(bak_path, 'w') as f:
            f.writelines(lines)
        print(f'Backup written to {bak_path}')

    # Build a mapping from parent section -> list of (leaf_key, json_key, new_value)
    insert_map = {}
    unknown_keys = {}
    for jkey, jval in jdata.items():
        if jkey not in KEY_MAP:
            unknown_keys[jkey] = jval
            continue
        path = KEY_MAP[jkey]
        if len(path) != 2:
            # For simplicity only support two-level mappings here
            # (top_level, leaf_key). Record as unknown otherwise.
            unknown_keys[jkey] = jval
            continue
        parent, leaf = path
        insert_map.setdefault(parent, []).append((leaf, jkey, jval))

    # Helper to format scalar into readable YAML inline scalar
    def fmt_scalar(val):
        if isinstance(val, bool):
            return 'true' if val else 'false'
        if isinstance(val, (int, float)):
            return str(val)
        s = str(val)
        # quote if contains spaces or special chars
        if any(c.isspace() for c in s) or ':' in s or s == '':
            return f"'{s}'"
        return s

    changes = {}

    # Simple stateful pass to modify leaf keys under parent sections
    cur_parent = None
    parent_line_idx = {}

    # First pass: record parent header line indices
    for idx, line in enumerate(lines):
        m = __import__('re').match(r'^(\s*)([A-Za-z0-9_]+):\s*(.*)$', line)
        if m:
            key = m.group(2)
            # Update current parent only if header (no inline nested content expected)
            cur_parent = key
            parent_line_idx[key] = idx

    # Second pass: update existing leaf keys
    cur_parent = None
    for idx, line in enumerate(lines):
        m = __import__('re').match(r'^(\s*)([A-Za-z0-9_]+):\s*(.*)$', line)
        if m:
            key = m.group(2)
            # This can be either parent header or a leaf; determine by indentation
            indent = len(m.group(1))
            if indent == 0:
                cur_parent = key
                continue
            # If we have a parent and leaf matches one we want to update
            if cur_parent in insert_map:
                for leaf, jkey, newval in insert_map[cur_parent]:
                    if leaf == key:
                        old_rhs = m.group(3).rstrip('\n')
                        # preserve inline comment if present
                        split_comment = old_rhs.split('#', 1)
                        old_val_str = split_comment[0].strip()
                        comment = (' #' + split_comment[1]) if len(split_comment) > 1 else ''
                        new_val_str = fmt_scalar(newval)
                        # Replace line preserving indent and comment
                        lines[idx] = f"{m.group(1)}{key}: {new_val_str}{comment}\n"
                        changes['.'.join(KEY_MAP[jkey])] = (old_val_str if old_val_str != '' else None, new_val_str)

    # Third: for any entries not updated because leaf was absent, insert them under the parent
    for parent, entries in insert_map.items():
        # find which entries were not updated
        missing = []
        for leaf, jkey, newval in entries:
            full = '.'.join(KEY_MAP[jkey])
            if full not in changes:
                missing.append((leaf, jkey, newval))
        if not missing:
            continue
        # find parent line index
        if parent not in parent_line_idx:
            # parent not found; append at end
            insert_idx = len(lines)
            parent_indent = ''
            # add parent header
            lines.append(f"{parent}:\n")
        else:
            insert_idx = parent_line_idx[parent] + 1
            # Determine indentation for children by inspecting next line under parent if possible
            parent_indent = '  '
            # move insert_idx to after any existing child lines (keep insertion at end of block)
            j = insert_idx
            while j < len(lines):
                lm = __import__('re').match(r'^(\s+)([A-Za-z0-9_]+):\s*(.*)$', lines[j])
                if lm and len(lm.group(1)) > 0:
                    j += 1
                    insert_idx = j
                else:
                    break
        # Insert missing lines
        for leaf, jkey, newval in missing:
            new_val_str = fmt_scalar(newval)
            lines.insert(insert_idx, f"{parent_indent}{leaf}: {new_val_str}\n")
            changes['.'.join(KEY_MAP[jkey])] = (None, new_val_str)
            insert_idx += 1

    # Record unknown keys as skipped
    for uk, v in unknown_keys.items():
        changes[uk] = ('unknown_key', v)

    # Write updated lines back to file
    with open(ypath, 'w') as f:
        f.writelines(lines)

    # Print summary
    print("Updated configuration:")
    for k, (old, new) in changes.items():
        if old == 'unknown_key':
            print(f"  JSON key '{k}' was unknown; value {new} was skipped")
        else:
            print(f"  {k}: {old} -> {new}")

    print('\nDone.')


if __name__ == '__main__':
    main()
