import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
import warnings

# Suppress sklearn warnings about convergence if necessary
warnings.filterwarnings("ignore")

class GPStrategy:
    """
    Gaussian Process Baseline Strategy.
    
    Models the feasibility boundary directly in joint space using a GP.
    Uses 'Straddle' (Uncertainty Sampling near boundary) to select points.
    
    Since we don't have gradients for the sklearn GP easily in Torch, 
    we maximize acquisition via random sampling or grid search.
    """
    
    def __init__(self, joint_limits, n_candidates=5000):
        """
        Args:
            joint_limits: Dict of limits
            n_candidates: Number of random candidates to evaluate for acquisition
        """
        self.joint_limits = joint_limits
        self.joint_names = list(joint_limits.keys())
        self.n_candidates = n_candidates
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Kernel: Matern kernel is often better for physical processes than RBF
        # 1.0 * Matern(length_scale=1.0, nu=2.5)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        
        # We model the SIGNED DISTANCE (logit) directly if possible, or just binary 0/1.
        # But regressing on outcomes (0/1) with GP is technically classification.
        # However, for "Straddle", GP Regression on 0/1 targets is a common heuristic 
        # (approximates p(feas) approx mean).
        # Better: Regress on the LOGIT if available? 
        # The LegacyOracle returns 'outcome' (bool) usually, but the LatentOracle can return logits.
        # For a fair baseline, we assume we only get BINARY labels (0/1).
        
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        
        self.X_train = []
        self.y_train = []
        
        # Prepare bounds for candidate generation
        self.bounds = []
        for name in self.joint_names:
            l, h = self.joint_limits[name]
            if isinstance(l, torch.Tensor): l = l.item()
            if isinstance(h, torch.Tensor): h = h.item()
            self.bounds.append((l, h))
        self.bounds = np.array(self.bounds)

    def update(self, test_point: torch.Tensor, outcome: bool):
        """Update GP with new observation."""
        # Convert to numpy
        if isinstance(test_point, torch.Tensor):
            tp = test_point.cpu().numpy()
        else:
            tp = test_point
            
        y = 1.0 if outcome else 0.0 # -1 to 1 or 0 to 1? 0/1 is fine for prob interpretation
        
        self.X_train.append(tp)
        self.y_train.append(y)
        
        # Refit GP
        # If too slow, can do every N steps, but for baseline correctness we do every step
        X = np.array(self.X_train)
        Y = np.array(self.y_train)
        
        self.gp.fit(X, Y)

    def select_test_point(self) -> torch.Tensor:
        """Select test point using Straddle (1.96*std - |mean - 0.5|)."""
        if len(self.X_train) == 0:
            # Random first point
            return self._random_point()
            
        # Generate candidates
        candidates = self._generate_candidates()
        
        # Predict
        # y_pred, sigma = self.gp.predict(candidates, return_std=True)
        mu, sigma = self.gp.predict(candidates, return_std=True)
        
        # Straddle Heuristic:
        # We want points near the boundary (mu approx 0.5) AND high uncertainty (sigma high)
        # Score = Uncertainty - DistanceToBoundary
        # boundary_dist = abs(mu - 0.5)
        # score = 1.96 * sigma - boundary_dist
        
        # Alternatively: UCB-like exploration
        # But for Active Level Set, we specifically want the boundary.
        # "Efficient Active Learning of Curves and Surfaces" (Gotovos et al.) -> Straddle
        # score(x) = beta * sigma(x) - |mu(x)| (assuming 0 is boundary, if 0/1 then 0.5)
        
        beta = 1.96
        score = beta * sigma - np.abs(mu - 0.5)
        
        best_idx = np.argmax(score)
        best_pt = candidates[best_idx]
        
        return torch.tensor(best_pt, dtype=torch.float32, device=self.device)

    def _generate_candidates(self) -> np.ndarray:
        """Generate random candidates within bounds."""
        # Uniform sampling
        # (N, D)
        candidates = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(self.n_candidates, len(self.bounds))
        )
        return candidates

    def _random_point(self) -> torch.Tensor:
        pt = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        return torch.tensor(pt, dtype=torch.float32, device=self.device)

    def select_test(self, bounds: torch.Tensor, **kwargs) -> tuple:
        """Adapter to conform to AcquisitionStrategy interface."""
        test_point = self.select_test_point()
        return test_point, 0.0

    def post_query_update(self, test_point, outcome, history) -> None:
        """Update GP model with new observation."""
        # outcome from oracle is a signed distance float; convert to bool
        self.update(test_point, outcome > 0)
