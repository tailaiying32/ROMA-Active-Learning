import torch
import numpy as np
from typing import Tuple, Optional

from infer_params.training.level_set_torch import evaluate_level_set_batched
from infer_params.core.metrics import soft_iou, compute_occupancy


def precompute_gt_metrics(
    ground_truth_params: dict,
    test_grid: torch.Tensor,
    device: torch.device,
    batch_size: int = 10000,
    temperature: float = 5.0
) -> dict:
    """
    Precompute ground truth evaluation on the test grid.

    Call this once at the start of a run, then pass the result as
    cached_gt to compute_reachability_metrics for each iteration.

    Args:
        ground_truth_params: Dictionary of ground truth parameters
        test_grid: Tensor of grid points (N, n_joints)
        device: Device for computation
        batch_size: Batch size for evaluation
        temperature: Temperature for occupancy computation

    Returns:
        dict with 'gt_f' and 'gt_probs' tensors
    """
    num_points = test_grid.shape[0]

    def to_tensor(x, dev):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().to(dev).unsqueeze(0)
        return torch.tensor(x, device=dev).unsqueeze(0)

    gt_lower = to_tensor(ground_truth_params['box_lower'], device)
    gt_upper = to_tensor(ground_truth_params['box_upper'], device)
    gt_weights = to_tensor(ground_truth_params['box_weights'], device)
    gt_pres = to_tensor(ground_truth_params['presence'], device)
    gt_blob_params = to_tensor(ground_truth_params['blob_params'], device)

    gt_feasible_f = []
    for i in range(0, num_points, batch_size):
        batch_points = test_grid[i:i+batch_size].to(device)
        val_gt = evaluate_level_set_batched(
            batch_points, gt_lower, gt_upper, gt_weights, gt_pres, gt_blob_params
        )
        gt_feasible_f.append(val_gt.cpu())

    gt_f = torch.cat(gt_feasible_f, dim=1)
    gt_probs = compute_occupancy(gt_f, temperature)

    return {
        'gt_f': gt_f,
        'gt_probs': gt_probs,
    }


def compute_reachability_metrics(
    decoder,
    ground_truth_params: dict,
    posterior_mean: torch.Tensor,
    test_grid: torch.Tensor,
    batch_size: int = 10000,
    cached_gt: dict = None
) -> Tuple[float, float, float, float]:
    """
    Compute IoU, Accuracy, F1, and Boundary-Local Accuracy of the predicted
    feasible region vs ground truth.

    Args:
        decoder: LevelSetDecoder model
        ground_truth_params: Dictionary of ground truth parameters
        posterior_mean: Latent mean of posterior (1, latent_dim)
        test_grid: Tensor of grid points (N, n_joints)
        batch_size: Batch size for evaluation
        cached_gt: Optional pre-computed GT dict with 'gt_f' and 'gt_probs' keys
                   (use precompute_gt_metrics to generate this)

    Returns:
        (iou, accuracy, f1, boundary_accuracy)
    """
    device = posterior_mean.device
    num_points = test_grid.shape[0]

    # 1. Decode Predicted Parameters
    # output: lower, upper, weights, pres_logits, blob_params
    with torch.no_grad():
        p_lower, p_upper, p_weights, p_pres_logits, p_blob_params = decoder.decode_from_embedding(posterior_mean)
        p_pres = torch.sigmoid(p_pres_logits)

    # 2. Get Ground Truth (from cache or compute)
    if cached_gt is not None:
        gt_f = cached_gt['gt_f']
        gt_probs = cached_gt['gt_probs']
    else:
        # Prepare Ground Truth Parameters (convert to batch size 1 tensors)
        def to_tensor(x, device):
            if isinstance(x, torch.Tensor):
                return x.detach().clone().to(device).unsqueeze(0)
            return torch.tensor(x, device=device).unsqueeze(0)

        gt_lower = to_tensor(ground_truth_params['box_lower'], device)
        gt_upper = to_tensor(ground_truth_params['box_upper'], device)
        gt_weights = to_tensor(ground_truth_params['box_weights'], device)
        gt_pres = to_tensor(ground_truth_params['presence'], device)
        gt_blob_params = to_tensor(ground_truth_params['blob_params'], device)

        gt_feasible_f = []
        for i in range(0, num_points, batch_size):
            batch_points = test_grid[i:i+batch_size].to(device)
            val_gt = evaluate_level_set_batched(
                batch_points, gt_lower, gt_upper, gt_weights, gt_pres, gt_blob_params
            )
            gt_feasible_f.append(val_gt.cpu())
        gt_f = torch.cat(gt_feasible_f, dim=1)
        temperature = 5.0
        gt_probs = compute_occupancy(gt_f, temperature)

    # 3. Evaluate predictions in batches
    pred_feasible_f = []
    for i in range(0, num_points, batch_size):
        batch_points = test_grid[i:i+batch_size].to(device)
        val_pred = evaluate_level_set_batched(
            batch_points, p_lower, p_upper, p_weights, p_pres, p_blob_params
        )
        pred_feasible_f.append(val_pred.cpu())

    pred_f = torch.cat(pred_feasible_f, dim=1)

    # 4. Compute Metrics (Soft IoU)
    temperature = 5.0
    pred_probs = compute_occupancy(pred_f, temperature)
    
    # Compute Soft IoU
    iou = soft_iou(pred_probs, gt_probs, reduce_batch=True)
    
    # Compute Accuracy (Thresholded at 0.5)
    gt_binary = (gt_probs >= 0.5).float()
    pred_binary = (pred_probs >= 0.5).float()
    accuracy = (gt_binary == pred_binary).float().mean()

    # Compute F1 Score
    tp = (gt_binary * pred_binary).sum()
    fp = ((1 - gt_binary) * pred_binary).sum()
    fn = (gt_binary * (1 - pred_binary)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Compute Boundary-Local Accuracy (closest 20% of points to GT boundary)
    gt_f_flat = gt_f.squeeze(0)  # (N,)
    boundary_distances = gt_f_flat.abs()
    threshold = torch.quantile(boundary_distances, 0.2)
    boundary_mask = boundary_distances <= threshold
    if boundary_mask.sum() > 0:
        boundary_accuracy = (gt_binary.squeeze(0)[boundary_mask] == pred_binary.squeeze(0)[boundary_mask]).float().mean()
    else:
        boundary_accuracy = 0.0

    return float(iou), float(accuracy), float(f1), float(boundary_accuracy)


def compute_ensemble_reachability_metrics(
    decoder,
    ground_truth_params: dict,
    posteriors: list,
    test_grid: torch.Tensor,
    batch_size: int = 10000,
    cached_gt: dict = None
) -> Tuple[float, float, float, float]:
    """
    Compute IoU, Accuracy, F1, and Boundary-Local Accuracy using mean prediction
    across ensemble members.

    Each member's posterior mean is decoded to level-set parameters, occupancy
    probabilities are computed, and then averaged across members before comparing
    to the ground truth.

    Args:
        decoder: LevelSetDecoder model
        ground_truth_params: Dictionary of ground truth parameters
        posteriors: List of K LatentUserDistribution (ensemble members)
        test_grid: Tensor of grid points (N, n_joints)
        batch_size: Batch size for evaluation
        cached_gt: Optional pre-computed GT dict with 'gt_f' and 'gt_probs' keys

    Returns:
        (iou, accuracy, f1, boundary_accuracy)
    """
    device = posteriors[0].device
    num_points = test_grid.shape[0]
    temperature = 5.0

    # 1. Ground Truth (from cache or compute)
    if cached_gt is not None:
        gt_f = cached_gt['gt_f']
        gt_probs = cached_gt['gt_probs']
    else:
        def to_tensor(x, dev):
            if isinstance(x, torch.Tensor):
                return x.detach().clone().to(dev).unsqueeze(0)
            return torch.tensor(x, device=dev).unsqueeze(0)

        gt_lower = to_tensor(ground_truth_params['box_lower'], device)
        gt_upper = to_tensor(ground_truth_params['box_upper'], device)
        gt_weights = to_tensor(ground_truth_params['box_weights'], device)
        gt_pres = to_tensor(ground_truth_params['presence'], device)
        gt_blob_params = to_tensor(ground_truth_params['blob_params'], device)

        gt_feasible_f = []
        for i in range(0, num_points, batch_size):
            batch_points = test_grid[i:i+batch_size].to(device)
            val_gt = evaluate_level_set_batched(
                batch_points, gt_lower, gt_upper, gt_weights, gt_pres, gt_blob_params
            )
            gt_feasible_f.append(val_gt.cpu())
        gt_f = torch.cat(gt_feasible_f, dim=1)
        gt_probs = compute_occupancy(gt_f, temperature)

    # 2. Ensemble Mean Prediction
    member_probs_list = []
    for posterior in posteriors:
        z = posterior.mean.unsqueeze(0)  # (1, latent_dim)
        with torch.no_grad():
            p_lower, p_upper, p_weights, p_pres_logits, p_blob_params = decoder.decode_from_embedding(z)
            p_pres = torch.sigmoid(p_pres_logits)

        member_f = []
        for i in range(0, num_points, batch_size):
            batch_points = test_grid[i:i+batch_size].to(device)
            val_pred = evaluate_level_set_batched(
                batch_points, p_lower, p_upper, p_weights, p_pres, p_blob_params
            )
            member_f.append(val_pred.cpu())
        member_f = torch.cat(member_f, dim=1)
        member_probs = compute_occupancy(member_f, temperature)
        member_probs_list.append(member_probs)

    # Average occupancy probabilities across members
    pred_probs = torch.stack(member_probs_list).mean(dim=0)

    # 3. Compute Metrics
    iou = soft_iou(pred_probs, gt_probs, reduce_batch=True)

    gt_binary = (gt_probs >= 0.5).float()
    pred_binary = (pred_probs >= 0.5).float()
    accuracy = (gt_binary == pred_binary).float().mean()

    tp = (gt_binary * pred_binary).sum()
    fp = ((1 - gt_binary) * pred_binary).sum()
    fn = (gt_binary * (1 - pred_binary)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Compute Boundary-Local Accuracy (closest 20% of points to GT boundary)
    gt_f_flat = gt_f.squeeze(0)  # (N,)
    boundary_distances = gt_f_flat.abs()
    threshold = torch.quantile(boundary_distances, 0.2)
    boundary_mask = boundary_distances <= threshold
    if boundary_mask.sum() > 0:
        boundary_accuracy = (gt_binary.squeeze(0)[boundary_mask] == pred_binary.squeeze(0)[boundary_mask]).float().mean()
    else:
        boundary_accuracy = 0.0

    return float(iou), float(accuracy), float(f1), float(boundary_accuracy)


def compute_legacy_reachability_metrics(
    true_checker,
    posterior,
    test_grid: np.ndarray,
    n_samples: int = 1
) -> Tuple[float, float, float]:
    """
    Compute IoU, Accuracy, and F1 of the predicted feasible region vs ground truth for legacy pipeline.
    
    Args:
        true_checker: Ground truth FeasibilityChecker
        posterior: UserDistribution (posterior)
        test_grid: Grid points (N, n_joints)
        n_samples: Number of posterior samples to average (if 1, use mean parameters)
        
    Returns:
        (iou, accuracy, f1)
    """
    from active_learning.src.legacy.feasibility_checker import FeasibilityChecker
    
    # 1. Ground Truth Occupancy
    h_true = true_checker.h_value(test_grid)
    gt_feasible = (h_true >= 0).astype(float)
    
    # 2. Predicted Occupancy
    if n_samples == 1:
        # Use mean parameters
        theta_mean = {
            'joint_limits': {},
            'pairwise_constraints': {}
        }
        for joint in posterior.joint_names:
            p = posterior.params['joint_limits'][joint]
            theta_mean['joint_limits'][joint] = (p['lower_mean'].item(), p['upper_mean'].item())
        
        for pair in posterior.pairs:
            # For bumps, we could use means, but legacy sampling is easier.
            # If we only have 1 sample and it's mean, we'll just sample once with temp=0 if possible? 
            # Or just use the sample method with small temperature?
            # Actually UserDistribution.sample doesn't support temp=0 easily for bumps.
            # Let's just sample N times and average.
            pass

    # Sample and average
    # We use a temperature of 0.1 to get "near-mean" behavior or just use regular samples.
    # The latent version uses just the mean latent code.
    # Sample and average
    # We use a temperature of 0.1 to get "near-mean" behavior or just use regular samples.
    # The latent version uses just the mean latent code.
    samples = posterior.sample(n_samples, return_list=True)
    pred_occupancy = np.zeros(len(test_grid))
    
    for theta in samples:
        # Convert tensors to numpy if necessary
        cpu_theta = {'joint_limits': {}, 'pairwise_constraints': {}}
        for j, (l, u) in theta['joint_limits'].items():
            cpu_theta['joint_limits'][j] = (l.item(), u.item())
        
        for pair, constraint in theta['pairwise_constraints'].items():
            cpu_theta['pairwise_constraints'][pair] = constraint
            
        pred_checker = FeasibilityChecker(
            joint_limits=cpu_theta['joint_limits'],
            pairwise_constraints=cpu_theta['pairwise_constraints']
        )
        h_pred = pred_checker.h_value(test_grid)
        pred_occupancy += (h_pred >= 0).astype(float)
    
    pred_occupancy /= n_samples
    
    # 3. Compute Metrics
    # IoU
    intersection = np.minimum(gt_feasible, pred_occupancy).sum()
    union = np.maximum(gt_feasible, pred_occupancy).sum()
    iou = intersection / union if union > 0 else 1.0
    
    # Accuracy
    gt_binary = (gt_feasible >= 0.5).astype(float)
    pred_binary = (pred_occupancy >= 0.5).astype(float)
    accuracy = (gt_binary == pred_binary).mean()
    
    # Compute F1 Score
    tp = (gt_binary * pred_binary).sum()
    fp = ((1 - gt_binary) * pred_binary).sum()
    fn = (gt_binary * (1 - pred_binary)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Compute Boundary-Local Accuracy (closest 20% of points to GT boundary)
    # Note: h_true is the signed distance to boundary (approx) for legacy checker
    boundary_distances = np.abs(h_true)
    threshold = np.quantile(boundary_distances, 0.2)
    boundary_mask = boundary_distances <= threshold
    
    if boundary_mask.sum() > 0:
        boundary_accuracy = (gt_binary[boundary_mask] == pred_binary[boundary_mask]).mean()
    else:
        boundary_accuracy = 0.0
    
    return float(iou), float(accuracy), float(f1), float(boundary_accuracy)
