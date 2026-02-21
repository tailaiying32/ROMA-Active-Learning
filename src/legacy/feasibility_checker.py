from pathlib import Path
import sys
from typing import Dict, Tuple, Optional, Union, List
import torch
import numpy as np

# Add CaregivingLM to path for imports
# We are in active_learning/src/legacy/feasibility_checker.py
# CaregivingLM is in active_learning/CaregivingLM
_current_dir = Path(__file__).parent
_caregiving_lm_path = _current_dir.parent.parent / 'CaregivingLM'

if _caregiving_lm_path.exists():
    if str(_caregiving_lm_path) not in sys.path:
        sys.path.insert(0, str(_caregiving_lm_path))
else:
    print(f"Warning: CaregivingLM not found at {_caregiving_lm_path}")

try:
    from sirs2d.sirs import h_value, compute_auto_smoothing_k
except ImportError:
    print("Warning: Could not import sirs2d.sirs. h_value computation will fail.")
    h_value = None
    compute_auto_smoothing_k = None

from active_learning.src.config import DEVICE

class FeasibilityChecker:
    '''
    Differentiable h-value computation for feasibility checking.
    If h < 0, the test is not feasible. Otherwise, it is feasible.
    '''
    
    @staticmethod
    def _smooth_min_cubic_batched(a, b, k):
        """Batched version of smooth_min_cubic."""
        if isinstance(k, (float, int)) and k == 0.0:
            return torch.min(a, b)

        # Handle tensor k or scalar non-zero
        h = torch.clamp(k - torch.abs(a - b), min=0.0) / k
        return torch.min(a, b) - h * h * h * k / 6.0

    @staticmethod
    def compute_h_batched(q, joint_limits, pairwise_constraints, joint_names, config=None):
        """
        Compute h-values for a batch of test points AND a batch of user parameters simultaneously.

        Args:
            q: Test points tensor (n_joints,) or (B, n_joints)
            joint_limits: Batched limits (N, n_joints, 2)
            pairwise_constraints: Dict of batched bump parameters per pair
            joint_names: List of joint names (to map pair strings to indices)
            config: Config dict
        
        Returns:
            min_h: Tensor of shape (N, B) representing h-value for each (user, test_point) pair.
                   (or (N,) if q was 1D)
        """
        config = config or {}
        
        # Dimensions
        if q.dim() == 1:
            q = q.unsqueeze(0) # (1, n_joints)
        
        B = q.shape[0]
        N = joint_limits.shape[0]
        n_joints = q.shape[1]
        device = q.device
        
        # 1. Check Global Joint Limits
        # q: (1, B, n_joints)
        # limits: (N, 1, n_joints)
        q_exp = q.unsqueeze(0) # (1, B, J)
        lower = joint_limits[..., 0].unsqueeze(1) # (N, 1, J)
        upper = joint_limits[..., 1].unsqueeze(1) # (N, 1, J)
        
        d_lower = q_exp - lower # (N, B, J)
        d_upper = upper - q_exp # (N, B, J)
        
        # Min h across joints: min(d_lower, d_upper) -> min across J
        h_per_joint = torch.min(d_lower, d_upper)
        min_h, _ = torch.min(h_per_joint, dim=-1) # (N, B)
        
        # 2. Check Pairwise Constraints
        # Pre-calculate indices
        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        
        use_smooth = config.get('bald', {}).get('use_smooth', True)
        
        # Logic for smoothing k (simplified for batch: use fixed or derive from box if easy, 
        # but box logic varies per sample. For performance, fixed k is safer in batch mode unless needed)
        # We will use a default k=0.1 if auto, or the configured float.
        # Vectorized auto-k is complex because box size varies per sample.
        smoothing_k = config.get('bald', {}).get('smoothing_k', 'auto')
        if smoothing_k == 'auto':
            k_val = 0.1 # Fallback/Default for batched mode to avoid complex per-sample K calc
        else:
            k_val = float(smoothing_k)

        for pair, bump_params in pairwise_constraints.items():
            if pair[0] not in name_to_idx or pair[1] not in name_to_idx:
                continue
                
            idx1 = name_to_idx[pair[0]]
            idx2 = name_to_idx[pair[1]]
            
            # Extract q for pair: (1, B, 2)
            q_pair = q_exp[..., [idx1, idx2]] 
            
            # --- Box Margin (Batched) ---
            # Limits for this pair: (N, 1, 2) - from joint_limits
            # L1, L2 from joint_limits
            # lower is (N, 1, J). lower[..., idx1] is (N, 1)
            l1 = lower[..., idx1] # (N, 1)
            u1 = upper[..., idx1]
            l2 = lower[..., idx2]
            u2 = upper[..., idx2]
            
            # q_pair is (1, B, 2)
            # q1 is q_pair[..., 0] -> (1, B)
            val1 = q_pair[..., 0]
            val2 = q_pair[..., 1]
            
            d1_min = val1 - l1 # (N, B)
            d1_max = u1 - val1
            d2_min = val2 - l2
            d2_max = u2 - val2
            
            # Smooth Min of these 4 distances
            dists = [d1_min, d1_max, d2_min, d2_max]
            
            margin = dists[0]
            if use_smooth:
                for d in dists[1:]:
                    margin = FeasibilityChecker._smooth_min_cubic_batched(margin, d, k_val)
            else:
                for d in dists[1:]:
                    margin = torch.min(margin, d)
            
            # --- Penalty (Batched) ---
            # bump_params contains 'mu', 'ls', 'alpha', 'R' (optional)
            # mu: (N, K, 2)
            # ls: (N, K, 2)
            # alpha: (N, K)
            # R: (N, K, 2, 2)
            
            # Check if bump parameters exist (might be empty if no bumps for this pair)
            if not bump_params or 'mu' not in bump_params:
                # No bumps, mean penalty is 0, continue loop (margin is already set)
                # But wait, we need to combine with overall min_h.
                # If no penalty, h_pair = margin.
                # So just proceed to update min_h with margin.
                min_h = torch.min(min_h, margin)
                continue

            mu = bump_params['mu'] # (N, K, 2)
            ls = bump_params['ls']
            alpha = bump_params['alpha']
            R = bump_params.get('R', None)
            
            # Expand for broadcasting against B test points
            # q_pair: (1, B, 2) -> (1, B, 1, 2)  (Sample, Test, Bump, Dim)
            q_b = q_pair.unsqueeze(2) 
            
            # mu: (N, K, 2) -> (N, 1, K, 2)
            mu_b = mu.unsqueeze(1)
            ls_b = ls.unsqueeze(1)
            
            dq = q_b - mu_b # (N, B, K, 2)
            
            if R is not None:
                # R: (N, K, 2, 2) -> (N, 1, K, 2, 2)
                R_b = R.unsqueeze(1)
                # Einsum: ...d (dq), ...ij (R) -> ...i (rotated)
                # Transpose R logic: dq_rot = R.T @ dq
                # R is (row, col). R.T is (col, row). 
                # Here R is batch of matrices. 
                # dq is vector at end. 
                # dq_rot = einsum('...ji, ...j -> ...i', R_b, dq)
                # Indices: n(N), b(B), k(K), j(dim_in), i(dim_out)
                dq_rot = torch.einsum('nbkji, nbkj -> nbki', R_b, dq)
                
                # Check dims: R_b is (N, 1, K, 2, 2). dq is (N, B, K, 2)
                # 'nbkji' matches R_b
                # 'nbkj' matches dq
                
                sq_dist = (dq_rot[..., 0] / ls_b[..., 0])**2 + (dq_rot[..., 1] / ls_b[..., 1])**2
            else:
                sq_dist = (dq[..., 0] / ls_b[..., 0])**2 + (dq[..., 1] / ls_b[..., 1])**2
            
            rbf = torch.exp(-0.5 * sq_dist) # (N, B, K)
            
            # Weighted sum over bumps
            # alpha: (N, K) -> (N, 1, K)
            alpha_b = alpha.unsqueeze(1)
            penalty = (alpha_b * rbf).sum(dim=-1) # (N, B)
            
            h_pair = margin - penalty
            
            # Combine with overall min
            min_h = torch.min(min_h, h_pair)
            
        return min_h

    def __init__(
            self,
            joint_limits: Dict[str, Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]],
            pairwise_constraints: Dict[Tuple[str, str], Dict] = None,
            config: Dict = None,
            # Legacy/Override
            use_smooth: bool = None,
            smoothing_k: Union[float, str] = None
            ):
        '''
        Args:
            joint_limits: dictionary of joint limits
            pairwise_constraints: dictionary of pairwise constraints
            config: configuration dictionary (expects 'bald' section for smoothing params)
        '''

        self.joint_limits = joint_limits
        self.joint_names = list(joint_limits.keys())
        self.device = DEVICE
        config = config or {}

        # Load smoothing params
        # Check deprecated first for legacy compat (as per new legacy.yaml structure), then main bald section
        deprecated_bald = config.get('deprecated', {}).get('bald', {})
        bald_config = config.get('bald', {})

        self.use_smooth = deprecated_bald.get('use_smooth', bald_config.get('use_smooth', True))
        self.smoothing_k = deprecated_bald.get('smoothing_k', bald_config.get('smoothing_k', 'auto'))

        # Overrides
        if use_smooth is not None: self.use_smooth = use_smooth
        if smoothing_k is not None: self.smoothing_k = smoothing_k

        # Build pairwise constraint index
        self.pairwise_constraints = {}
        if pairwise_constraints:
            for (j1, j2), constraint_info in pairwise_constraints.items():
                if j1 not in self.joint_names or j2 not in self.joint_names:
                    # In some legacy configs, pair names might use different strings or be ignored
                    # Here we raise error to be safe, or just skip
                    continue
                    # raise ValueError(f"Unknown joint(s) in pair ({j1}, {j2})")

                i1 = self.joint_names.index(j1)
                i2 = self.joint_names.index(j2)

                # Build box from joint limits if not provided
                if 'box' in constraint_info:
                    box = constraint_info['box']
                else:
                    box = {
                        'q1_range': self.joint_limits[j1],
                        'q2_range': self.joint_limits[j2]
                    }

                bumps = constraint_info.get('bumps', [])

                self.pairwise_constraints[(i1, i2)] = {
                    'box': box,
                    'bumps': bumps,
                    'joint_names': (j1, j2)
                }

    def _smooth_min_cubic_torch(self, a, b, k):
        """PyTorch implementation of smooth_min_cubic."""
        if k == 0.0:
            return torch.min(a, b)

        # h = max(k - |a - b|, 0) / k
        h = torch.clamp(k - torch.abs(a - b), min=0.0) / k
        return torch.min(a, b) - h * h * h * k / 6.0

    def _rbf_value_torch(self, q, mu, ls, R=None):
        """PyTorch implementation of rbf_value."""
        device = q.device
        dtype = q.dtype

        mu = torch.as_tensor(mu, device=device, dtype=dtype)
        ls = torch.as_tensor(ls, device=device, dtype=dtype)

        if R is None:
            sq_dist = ((q[..., 0] - mu[0]) / ls[0])**2 + ((q[..., 1] - mu[1]) / ls[1])**2
        else:
            R = torch.as_tensor(R, device=device, dtype=dtype)
            dq = q - mu
            # dq_rot = R.T @ dq
            # Batched matrix multiplication: ...j, ij -> ...i
            dq_rot = torch.einsum('ij,...j->...i', R.T, dq)
            sq_dist = (dq_rot[..., 0] / ls[0])**2 + (dq_rot[..., 1] / ls[1])**2

        return torch.exp(-0.5 * sq_dist)

    def _h_value_torch(self, q):
        """PyTorch implementation of h_value computation."""
        device = q.device

        # 1. Check global joint limits (sharp)
        # Handle both float and tensor limits (for differentiability)
        l_vals = [self.joint_limits[name][0] for name in self.joint_names]
        u_vals = [self.joint_limits[name][1] for name in self.joint_names]

        if len(l_vals) > 0 and isinstance(l_vals[0], torch.Tensor):
             lower_limits = torch.stack(l_vals).to(device=device, dtype=q.dtype).view(-1)
        else:
             lower_limits = torch.tensor(l_vals, device=device, dtype=q.dtype)

        if len(u_vals) > 0 and isinstance(u_vals[0], torch.Tensor):
             upper_limits = torch.stack(u_vals).to(device=device, dtype=q.dtype).view(-1)
        else:
             upper_limits = torch.tensor(u_vals, device=device, dtype=q.dtype)

        d_lower = q - lower_limits
        d_upper = upper_limits - q

        # Min distance to any joint limit
        h_limits, _ = torch.min(torch.min(d_lower, d_upper), dim=-1)
        min_h = h_limits

        # 2. Check pairwise constraints
        for (i1, i2), constraint in self.pairwise_constraints.items():
            box = constraint['box']
            bumps = constraint['bumps']

            # Extract pair: (..., 2)
            q_pair = q[..., [i1, i2]]

            # --- Compute h_pair (margin - penalty) ---

            # A. Box Margin
            # Extract ranges for this pair
            q1_min, q1_max = box['q1_range']
            q2_min, q2_max = box['q2_range']

            # Ensure ranges are tensors
            if not isinstance(q1_min, torch.Tensor): q1_min = torch.tensor(q1_min, device=device)
            if not isinstance(q1_max, torch.Tensor): q1_max = torch.tensor(q1_max, device=device)
            if not isinstance(q2_min, torch.Tensor): q2_min = torch.tensor(q2_min, device=device)
            if not isinstance(q2_max, torch.Tensor): q2_max = torch.tensor(q2_max, device=device)

            d1_min = q_pair[..., 0] - q1_min
            d1_max = q1_max - q_pair[..., 0]
            d2_min = q_pair[..., 1] - q2_min
            d2_max = q2_max - q_pair[..., 1]

            distances = [d1_min, d1_max, d2_min, d2_max]

            # Determine smoothing k
            k = 0.0
            if self.use_smooth:
                if self.smoothing_k == 'auto' or self.smoothing_k is None:
                    # compute_auto_smoothing_k expects values, not tensors usually, but let's see
                    # If we are in torch mode, we probably want a fixed k or robust auto k
                    # For now, if auto, default to small value or try to compute
                    try:
                        # Extract float values for auto computation if possible
                        box_float = {
                            'q1_range': (float(q1_min), float(q1_max)),
                            'q2_range': (float(q2_min), float(q2_max))
                        }
                        k = float(compute_auto_smoothing_k(box_float))
                    except:
                        k = 0.1 # Fallback
                else:
                    k = float(self.smoothing_k)

            # Combine distances
            if k == 0.0:
                margin = distances[0]
                for d in distances[1:]:
                    margin = torch.min(margin, d)
            else:
                margin = distances[0]
                for d in distances[1:]:
                    margin = self._smooth_min_cubic_torch(margin, d, k)

            # B. Penalty
            penalty = torch.zeros_like(margin)
            if bumps:
                for bump in bumps:
                    rbf = self._rbf_value_torch(q_pair, bump['mu'], bump['ls'], bump.get('R'))
                    penalty = penalty + bump['alpha'] * rbf

            h_pair = margin - penalty

            # Combine with overall min_h (sharp min or smooth min if we wanted)
            # For H-value aggregation, sharp min is usually standard
            min_h = torch.min(min_h, h_pair)

        return min_h

    def h_value(self, q):
        '''
        Compute the h-value for the joint configuration(s).

        Args:
            q: Joint configuration (dict, list, numpy array, or torch tensor)
               Shape (n_joints,) or (batch_size, n_joints)

        Returns:
            h: Minimum h-value across all constraints.
               Scalar or array of shape (batch_size,)
        '''
        # Handle Torch Tensor input (Differentiable)
        if isinstance(q, torch.Tensor):
            return self._h_value_torch(q)

        # Convert input to numpy array
        if isinstance(q, dict):
            q_arr = np.array([q[name] for name in self.joint_names])
        else:
            q_arr = np.array(q)

        # Initialize min_h with joint limits check
        lower_limits = np.array([self.joint_limits[name][0] for name in self.joint_names])
        upper_limits = np.array([self.joint_limits[name][1] for name in self.joint_names])

        d_lower = q_arr - lower_limits
        d_upper = upper_limits - q_arr

        # Min distance to any joint limit
        h_limits = np.minimum(d_lower, d_upper).min(axis=-1)

        min_h = h_limits

        # Check pairwise constraints
        for (i1, i2), constraint in self.pairwise_constraints.items():
            box = constraint['box']
            bumps = constraint['bumps']

            # Extract pair: (..., 2)
            q_pair = q_arr[..., [i1, i2]]

            h_pair = h_value(q_pair, box, bumps, use_smooth=self.use_smooth, smoothing_k=self.smoothing_k)

            min_h = np.minimum(min_h, h_pair)

        return min_h

    def is_feasible(self, q):
        '''
        Check if the configuration is feasible (h >= 0).
        '''
        h = self.h_value(q)
        # Check if tensor
        if isinstance(h, torch.Tensor):
             is_feas = (h >= 0).item()
        else:
             is_feas = (h >= 0)
        
        result = {
            'h_value': h,
            'is_feasible': is_feas
        }
        return result
