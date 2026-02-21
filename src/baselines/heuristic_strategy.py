
import torch
import numpy as np
from active_learning.src.config import DEVICE
from active_learning.src.utils import binary_entropy

class HeuristicStrategy:
    """
    Heuristic Baseline Strategy ("Dense Banking").
    
    Selects optimal queries by evaluating information gain on a fixed, dense set of 
    candidate hypotheses (Candidate Bank) and potential queries (Query Bank).
    
    This serves as an upper bound / ideal approximation of BALD without optimization issues.
    """
    
    def __init__(
        self,
        candidates: torch.Tensor, # (N_candidates, ParamDim) or specialized
        queries: torch.Tensor,    # (N_queries, JointDim)
        checker,                  # FeasibilityChecker (Latent or Legacy)
        device: str = DEVICE,
        batch_size: int = 1000,
        joint_names: list = None  # Required for Legacy FeasibilityChecker
    ):
        """
        Args:
            candidates: Pre-sampled hypotheses. 
            queries: Pre-sampled pool of test queries to select from.
            checker: Object with `batched_logit_values(candidates, queries)` or equivalent.
            joint_names: List of joint names (required for legacy FeasibilityChecker.compute_h_batched)
        """
        self.candidates = candidates
        self.queries = queries
        self.checker = checker
        self.device = device
        self.batch_size = batch_size
        self.joint_names = joint_names
        self.selected_indices = []

    def select_test_point(self) -> torch.Tensor:
        """
        Select the best query from the query bank maximizing Entropy(Mean) - Mean(Entropy).
        """
        # We need to compute BALD score for ALL queries in the bank against ALL candidates.
        # This is expensive: O(N_candidates * N_queries).
        # We process queries in batches to save memory.
        
        n_queries = len(self.queries)
        n_candidates = len(self.candidates) # OR batch dimension of candidates
        
        best_score = -float('inf')
        best_query_idx = -1
        
        # 1. Iterate over query batches
        for i_start in range(0, n_queries, self.batch_size):
            i_end = min(i_start + self.batch_size, n_queries)
            q_batch = self.queries[i_start:i_end] # (B_q, Joints)
            
            # 2. Compute Feasibility Matrix H for this batch
            # Shape: (N_candidates, B_q)
            
            # Dispatch based on checker type or assumed interface
            # We assume candidates are already in format expected by checker?
            # Latent: candidates is tensor Z. Checker takes (prob_model, Z, x).
            # Legacy: candidates is dict of params. Checker takes (x, limits, bumps).
            
            # We need a unified interface or a helper.
            # Let's assume the caller wrapped the checker or we handle it here.
            
            with torch.no_grad():
                if hasattr(self.checker, 'batched_logit_values'):
                    # Latent style
                    # batched_logit_values(decoder, z, x)
                    # We need access to decoder? It's usually in checker or passed.
                    # LatentFeasibilityChecker methods usually static or need instance.
                    # If passed as instance of LatentFeasibilityChecker:
                    # It has self.decoder (maybe).
                    h_batch = self.checker.batched_logit_values(
                         self.checker.decoder, # Assuming checker has decoder attached or passed manually
                         self.candidates, 
                         q_batch
                    ) 
                    # Note: batched_logit_values might expect (samples, queries) or (queries, samples)?
                    # LatentFeasibilityChecker.batched_logit_values signature: (decoder, z, test_points)
                    # Returns (N_z, N_test)
                
                elif hasattr(self.checker, 'compute_h_batched'):
                    # Legacy style (static method usually)
                    # self.checker might be the class itself or an instance
                    # compute_h_batched(q, joint_limits, pairwise_constraints, ...)
                    # Candidates must be formatted as dict: {'joint_limits': ..., 'bumps': ...}
                    h_batch = self.checker.compute_h_batched(
                        q=q_batch,
                        joint_limits=self.candidates['joint_limits'],
                        pairwise_constraints=self.candidates['bumps'],
                        joint_names=self.joint_names if self.joint_names else (self.checker.joint_names if hasattr(self.checker, 'joint_names') else None),
                        config={} # Default config
                    )
                else:
                    raise ValueError("Unknown checker interface")

            # 3. Compute BALD Score
            # h_batch: (N_samples, B_q)
            # Logits -> Probs
            probs = torch.sigmoid(h_batch)
            
            # P_mean: (B_q,)
            p_mean = probs.mean(dim=0)
            
            # Entropy of mean
            ent_mean = binary_entropy(p_mean)
            
            # Mean of entropy
            mean_ent = binary_entropy(probs).mean(dim=0)
            
            scores = ent_mean - mean_ent # (B_q,)
            
            # 4. Find max in batch
            batch_max_score, batch_argmax = scores.max(dim=0)
            
            if batch_max_score > best_score:
                best_score = batch_max_score.item()
                best_query_idx = i_start + batch_argmax.item()
                
        if best_query_idx == -1:
            # Fallback (random?)
            return self.queries[0]
            
        return self.queries[best_query_idx]

    def select_test(self, bounds: torch.Tensor, **kwargs) -> tuple:
        """Adapter to conform to AcquisitionStrategy interface."""
        test_point = self.select_test_point()
        return test_point, 0.0

