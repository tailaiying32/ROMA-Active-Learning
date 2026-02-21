
import torch
import numpy as np
from active_learning.src.config import DEVICE
from active_learning.src.latent_feasibility_checker import LatentFeasibilityChecker

class VersionSpaceStrategy:
    """
    Explicit Version Space (Greedy Maximin) Strategy.
    
    1. Hypothesis Space: Fixed set of latent embeddings from the trained decoder.
    2. Query Space: Fixed grid of test points.
    3. Selection: Greedy Maximin. Select query q that maximizes min(N_pos, N_neg),
       where N_pos/N_neg are the number of currently valid versions consistent with that outcome.
       This guarantees worst-case pruning.
    
    4. Update: Hard pruning of hypotheses that disagree with observed ground truth.
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor, # (N_candidates, LatentDim)
        queries: torch.Tensor,    # (N_queries, JointDim)
        decoder,                  # Decoder model
        device: str = DEVICE,
        batch_size: int = 1000
    ):
        """
        Args:
            embeddings: Tensor of all user embeddings (the version space candidates)
            queries: Tensor of candidate queries/tests
            decoder: The decoder model used for feasibility checking
        """
        self.candidates = embeddings.to(device)
        self.queries = queries.to(device)
        self.decoder = decoder
        self.device = device
        self.batch_size = batch_size
        
        # Mask of currently valid candidates (initially all True)
        self.valid_mask = torch.ones(len(self.candidates), dtype=torch.bool, device=device)
        
        print(f"[VersionSpace] Initialized with {len(self.candidates)} candidates and {len(self.queries)} queries.")

    def select_test_point(self) -> torch.Tensor:
        """
        Select the best query maximizing min(N_pos, N_neg) of the VALID candidates.
        """
        # 1. Get current valid candidates
        valid_z = self.candidates[self.valid_mask]
        n_valid = len(valid_z)
        
        if n_valid == 0:
            print("[VersionSpace] Warning: No valid candidates remaining! Selecting random.")
            return self.queries[torch.randint(0, len(self.queries), (1,)).item()]
        
        # If very few candidates, random choice might be better or just pick first
        if n_valid <= 1:
            # Game over effectively, just explore
            return self.queries[torch.randint(0, len(self.queries), (1,)).item()]

        best_score = -1
        best_query_idx = -1
        
        n_queries = len(self.queries)
        
        # 2. Iterate over query batches
        for i_start in range(0, n_queries, self.batch_size):
            i_end = min(i_start + self.batch_size, n_queries)
            q_batch = self.queries[i_start:i_end] # (B_q, Joints)
            
            # 3. Compute Feasibility Matrix H for this batch
            # Shape: (N_valid, B_q)
            with torch.no_grad():
                # raw logits: >0 feasible, <0 infeasible
                logits = LatentFeasibilityChecker.batched_logit_values(
                     self.decoder,
                     valid_z, 
                     q_batch
                ) 
                
                # Binarize: True if > 0 (Feasible)
                predictions = (logits > 0)
                
                # 4. Count Pos/Neg for each query
                # shape (B_q,)
                n_pos = predictions.sum(dim=0).float()
                n_neg = n_valid - n_pos
                
                # 5. Greedy Score = min(N_pos, N_neg)
                # We want to maximize the WORST case elimination
                scores = torch.min(n_pos, n_neg)
                
                # 6. Find max in batch
                batch_max_score, batch_argmax = scores.max(dim=0)
                
                if batch_max_score > best_score:
                    best_score = batch_max_score.item()
                    best_query_idx = i_start + batch_argmax.item()
                    
        if best_query_idx == -1:
            return self.queries[0]
            
        return self.queries[best_query_idx]

    def update(self, test_point: torch.Tensor, outcome: bool):
        """
        Prune candidates that are inconsistent with the observed outcome.
        """
        # 1. Evaluate all CURRENTLY VALID candidates on this single point
        valid_indices = torch.nonzero(self.valid_mask).squeeze()
        if valid_indices.numel() == 0:
            return

        # Ensure valid_indices is at least 1D (handle single candidate case)
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)
        
        valid_z = self.candidates[valid_indices]
        
        with torch.no_grad():
            # Check feasibility (1 point, N candidates)
            # batched_logit_values handles (N_z, N_points)
            logits = LatentFeasibilityChecker.batched_logit_values(
                self.decoder,
                valid_z,
                test_point.unsqueeze(0) # (1, D)
            ) # Returns (N_valid, 1)
            
            logits = logits.squeeze() # (N_valid,)
            
            # Predict
            predictions = (logits > 0)
            
            # Match ground truth
            # If outcome is True (Feasible), keep predictions=True
            # If outcome is False (Infeasible), keep predictions=False
            matches = (predictions == outcome)
            
            # 2. Identify indices to keep (relative to valid_z)
            keep_relative = matches
            
            # 3. Update global mask
            # We only touch indices that were already True
            # valid_indices[~keep_relative] are the ones to flip to False
            if keep_relative.dim() == 0: # Handle scalar case if only 1 candidate
                if not keep_relative:
                     self.valid_mask[valid_indices] = False
            else:
                raw_indices_to_drop = valid_indices[~keep_relative]
                self.valid_mask[raw_indices_to_drop] = False
                
        print(f"[VersionSpace] Pruned {len(valid_z) - matches.sum().item()} candidates. Remaining: {self.valid_mask.sum().item()}")

    def get_posterior_mean(self):
        """Return mean of valid embeddings (approximate posterior mean)."""
        if self.valid_mask.sum() == 0:
            return self.candidates.mean(dim=0) # Fallback to prior
        return self.candidates[self.valid_mask].mean(dim=0)

    def get_posterior_std(self):
        """Return std of valid embeddings (approximate uncertainty)."""
        if self.valid_mask.sum() < 2:
            return torch.zeros(self.candidates.shape[1], device=self.device)
        return self.candidates[self.valid_mask].std(dim=0)

    def select_test(self, bounds: torch.Tensor, **kwargs) -> tuple:
        """Adapter to conform to AcquisitionStrategy interface."""
        test_point = self.select_test_point()
        return test_point, 0.0

    def post_query_update(self, test_point, outcome, history) -> None:
        """Prune version space based on observed outcome."""
        self.update(test_point, outcome > 0)
