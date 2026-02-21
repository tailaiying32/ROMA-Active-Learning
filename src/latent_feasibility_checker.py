import torch

from active_learning.src.config import DEVICE
from infer_params.training.model import LevelSetDecoder
from infer_params.training.level_set_torch import evaluate_level_set_batched


class LatentFeasibilityChecker:
    '''
    Latent feasibility checker using the decoder model to evaluate feasibility of latent codes.
    '''
    def __init__(
            self,
            decoder,
            z,
            device=DEVICE
    ):
        '''
        Initialize the latent feasibility checker.

        Args:
            decoder: decoder model to map latent codes to outputs
            z: latent code tensor of shape (latent_dim,)
            device: torch device
        '''
        self.decoder = decoder
        self.z = z
        self.device = device

    @staticmethod
    def decode_latent_params(decoder, zs) -> tuple:
        """
        Decode latent parameters into RBF parameters.
        
        Args:
            decoder: LevelSetDecoder model
            zs: Latent codes tensor of shape (B, latent_dim)
            
        Returns:
            Tuple of (lower, upper, weights, pres_logits, blob_params)
        """
        return decoder.decode_from_embedding(zs)

    @staticmethod
    def evaluate_from_decoded(test_points, decoded_params) -> torch.Tensor:
        """
        Compute feasibility logits using pre-decoded RBF parameters.
        
        Args:
            test_points: Test points tensor of shape (n_points, n_joints) or (n_joints,)
            decoded_params: Tuple of (lower, upper, weights, pres_logits, blob_params)
            
        Returns:
            Logits tensor of shape (B, n_points) or (B,) if single test point
        """
        lower, upper, weights, pres_logits, blob_params = decoded_params
        
        # Handle single test point
        if test_points.dim() == 1:
            test_points = test_points.unsqueeze(0)  # (1, n_joints)
            
        # Convert logits to sigmoid probabilities
        pres = torch.sigmoid(pres_logits)

        # Calculate level set function
        logits = evaluate_level_set_batched(test_points, lower, upper, weights, pres, blob_params)
        
        return logits

    @staticmethod
    def batched_logit_values(decoder, zs, test_points) -> torch.Tensor:
        """
        Compute feasibility logits using RBF level set decoder, in batches, at the given test points.
        f(x) = d_box(x) - Penalty(x)

        Args:
            decoder: LevelSetDecoder model
            zs: Latent codes tensor of shape (B, latent_dim)
            test_points: Test points tensor of shape (n_points, n_joints) or (n_joints,)

        Returns:
            Logits tensor of shape (B, n_points) or (B,) if single test point
        """
        decoded = LatentFeasibilityChecker.decode_latent_params(decoder, zs)
        return LatentFeasibilityChecker.evaluate_from_decoded(test_points, decoded)

    def logit_value(self, test_point: torch.Tensor) -> torch.Tensor:
        """
        Compute feasibility logit for a single test point using this checker's latent code.

        Args:
            test_point: Test point tensor of shape (n_joints,)

        Returns:
            Logit value (positive = feasible, negative = infeasible)
        """
        z_batch = self.z.unsqueeze(0)  # (1, latent_dim)
        logits = self.batched_logit_values(self.decoder, z_batch, test_point)
        return logits.squeeze()  # Scalar

    def is_feasible(self, test_point):
        """
        Check if the test point is feasible (logit >= 0).
        """
        return self.logit_value(test_point) >= 0

    def h_value(self, test_points):
        """
        Alias for batched_logit_values or logit_value to remain compatible 
        with legacy Reachability metrics.
        """
        if not isinstance(test_points, torch.Tensor):
            test_points = torch.tensor(test_points, dtype=torch.float32, device=self.device)
            
        z_batch = self.z.unsqueeze(0)
        logits = self.batched_logit_values(self.decoder, z_batch, test_points)
        
        val = logits.squeeze(0).cpu().detach().numpy()
        if val.ndim == 1 and val.shape[0] == 1:
            return val.item()
        return val

    def joint_limits(self):
        """Extract limits directly from the decoder output."""
        z_batch = self.z.unsqueeze(0)
        lower, upper, _, _, _ = self.decoder.decode_from_embedding(z_batch)
        return lower.squeeze(0), upper.squeeze(0)