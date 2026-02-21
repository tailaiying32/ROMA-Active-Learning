#!/usr/bin/env python3
"""
apply_hyperparams_txt_to_yaml.py

Read a simple whitespace-delimited hyperparameter text file (one key value per line)
and apply those values into a YAML config (`configs/default.yaml`) by mapping
flat keys to YAML paths. Preserves comments and formatting in the YAML.

Usage:
    python scripts/apply_hyperparams_txt_to_yaml.py \
        --txt scripts/hyperparams.txt \
        --yaml configs/default.yaml \
        [--backup]

"""
from pathlib import Path
import argparse
import sys
import re

# Mapping from flat txt key -> yaml nested key path (as list)
KEY_MAP = {
    'init_std': ['prior', 'init_std'],

    'bald_tau': ['bald', 'tau'],
    'bald_n_mc_samples': ['bald', 'n_mc_samples'],
    'bald_sampling_temperature': ['bald', 'sampling_temperature'],

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
    'vi_patience': ['vi', 'patience'],

    # stopping / early-stopping
    'stopping_elbo_plateau_threshold': ['stopping', 'elbo_plateau_threshold'],
    'stopping_elbo_plateau_patience': ['stopping', 'elbo_plateau_window'],
    'stopping_uncertainty_threshold': ['stopping', 'uncertainty_threshold'],
    'stopping_bald_threshold': ['stopping', 'bald_threshold'],
    'stopping_bald_patience': ['stopping', 'bald_patience'],

    # other potential keys used in txt
    'bald_plateau_threshold': ['bald_optimization', 'plateau_threshold']
}


def parse_value(s: str):
    s = s.strip()
    # Booleans
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    # Try int
    try:
        i = int(s)
        return i
    except Exception:
        pass
    # Try float
    try:
        f = float(s)
        return f
    except Exception:
        pass
    # string
    # strip surrounding quotes if present
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def fmt_scalar(val):
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if isinstance(val, (int,)):
        return str(val)
    if isinstance(val, float):
        return repr(val)
    return str(val)


def read_txt(txt_path: Path):
    if not txt_path.exists():
        print(f"Error: TXT file not found: {txt_path}")
        sys.exit(1)
    data = {}
    with open(txt_path, 'r') as f:
        for line in f:
            s = line.strip()
            if s == '' or s.lstrip().startswith('#'):
                continue
            parts = s.split(None, 1)
            if len(parts) == 0:
                continue
            key = parts[0]
            val = parts[1] if len(parts) > 1 else ''
            data[key] = parse_value(val)
    return data


def apply_to_yaml(kv: dict, ypath: Path, backup: bool = False):
    with open(ypath, 'r') as f:
        lines = f.readlines()

    if backup:
        bak = ypath.with_suffix(ypath.suffix + '.bak')
        with open(bak, 'w') as bf:
            bf.writelines(lines)
        print(f'Backup written to {bak}')

    # Build insert map by parent
    insert_map = {}
    unknown = {}
    for k, v in kv.items():
        if k not in KEY_MAP:
            unknown[k] = v
            continue
        parent, leaf = KEY_MAP[k]
        insert_map.setdefault(parent, []).append((leaf, k, v))

    # Find parent header lines
    parent_line_idx = {}
    for idx, line in enumerate(lines):
        m = re.match(r'^(\s*)([A-Za-z0-9_]+):\s*(.*)$', line)
        if m and len(m.group(1)) == 0:
            parent_line_idx[m.group(2)] = idx

    changes = {}

    # Update existing leaves
    cur_parent = None
    for idx, line in enumerate(lines):
        m = re.match(r'^(\s*)([A-Za-z0-9_]+):\s*(.*)$', line)
        if not m:
            continue
        indent = len(m.group(1))
        key = m.group(2)
        if indent == 0:
            cur_parent = key
            continue
        if cur_parent in insert_map:
            for leaf, jkey, newval in insert_map[cur_parent]:
                if leaf == key:
                    old_rhs = m.group(3).rstrip('\n')
                    split_comment = old_rhs.split('#', 1)
                    old_val_str = split_comment[0].strip()
                    comment = (' #' + split_comment[1]) if len(split_comment) > 1 else ''
                    new_val_str = fmt_scalar(newval)
                    lines[idx] = f"{m.group(1)}{key}: {new_val_str}{comment}\n"
                    changes['.'.join(KEY_MAP[jkey])] = (old_val_str if old_val_str != '' else None, new_val_str)

    # Insert missing leaves
    for parent, entries in insert_map.items():
        missing = []
        for leaf, jkey, newval in entries:
            full = '.'.join(KEY_MAP[jkey])
            if full not in changes:
                missing.append((leaf, jkey, newval))
        if not missing:
            continue
        if parent not in parent_line_idx:
            # append parent at end
            lines.append(f"{parent}:\n")
            insert_idx = len(lines)
            parent_indent = ''
        else:
            insert_idx = parent_line_idx[parent] + 1
            parent_indent = '  '
            j = insert_idx
            while j < len(lines):
                lm = re.match(r'^(\s+)([A-Za-z0-9_]+):\s*(.*)$', lines[j])
                if lm and len(lm.group(1)) > 0:
                    j += 1
                    insert_idx = j
                else:
                    break
        for leaf, jkey, newval in missing:
            new_val_str = fmt_scalar(newval)
            lines.insert(insert_idx, f"{parent_indent}{leaf}: {new_val_str}\n")
            changes['.'.join(KEY_MAP[jkey])] = (None, new_val_str)
            insert_idx += 1

    for uk, v in unknown.items():
        changes[uk] = ('unknown_key', v)

    # Write back
    with open(ypath, 'w') as f:
        f.writelines(lines)

    # Print summary
    print('Updated YAML file:')
    for k, (old, new) in changes.items():
        if old == 'unknown_key':
            print(f"  TXT key '{k}' was unknown; value {new} was skipped")
        else:
            print(f"  {k}: {old} -> {new}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    # Resolve project root relative to this script
    script_root = Path(__file__).parent.parent
    parser.add_argument('--txt', '-t', type=Path, default=script_root / 'hyperparams.txt')
    parser.add_argument('--yaml', '-y', type=Path, default=script_root / 'configs' / 'default.yaml')
    parser.add_argument('--backup', '-b', action='store_true')
    args = parser.parse_args(argv)

    kv = read_txt(args.txt)
    apply_to_yaml(kv, args.yaml, backup=args.backup)


if __name__ == '__main__':
    main()
