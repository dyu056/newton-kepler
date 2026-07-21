import torch
import torch.nn.functional as F

def compute_loss_with_mask(predictions, targets, loss_mask='all'):
    """
    Compute MSE loss with optional masking.
    
    Args:
        predictions: tensor of shape (batch, seq_len, 2)
        targets: tensor of shape (batch, target_len, 2)
        loss_mask: 'all' to compute loss on all positions, 'last' to compute only on last position
    
    Returns:
        loss: scalar tensor
    """
    # Handle target shape: if 2D, add dimension
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)  # (b, t) -> (b, t, 1)
    
    # Handle case where targets might have different sequence length than predictions
    if targets.size(1) < predictions.size(1):
        # Only use the last predictions that match target length
        predictions = predictions[:, -targets.size(1):, :]  # Take last t_target positions
    elif targets.size(1) > predictions.size(1):
        raise ValueError(f"Target sequence length {targets.size(1)} > prediction length {predictions.size(1)}")
    
    if loss_mask == 'all':
        # Compute loss on all positions
        predictions_flat = predictions.reshape(-1)  # shape (b*t*d,)
        targets_flat = targets.reshape(-1)  # shape (b*t*d,)
        loss = F.mse_loss(predictions_flat, targets_flat)
    elif loss_mask == 'last':
        # Compute loss only on the last position
        predictions_last = predictions[:, -1, :].reshape(-1)  # shape (b*d,)
        targets_last = targets[:, -1, :].reshape(-1)  # shape (b*d,)
        loss = F.mse_loss(predictions_last, targets_last)
    else:
        raise ValueError(f"loss_mask must be 'all' or 'last', got {loss_mask}")
    
    return loss


def compute_varcov_penalty(
    representation: torch.Tensor,
    target_std: float = 0.1,
    eps: float = 1e-4,
):
    """Compute variance/covariance penalties for token representations.

    Args:
        representation: Tensor shaped ``[batch, tokens, hidden]``.
        target_std: Minimum desired standard deviation for each feature.
        eps: Numerical stability constant used by the standard deviation.

    Returns:
        ``(variance_loss, covariance_loss, mean_std)``. All three values remain
        connected to the computation graph and can therefore be used either as
        training losses or detached monitoring statistics.
    """
    if representation.dim() != 3:
        raise ValueError(
            "representation must have shape [batch, tokens, hidden], "
            f"got {tuple(representation.shape)}"
        )

    z = representation.reshape(-1, representation.size(-1)).float()
    z = z - z.mean(dim=0, keepdim=True)

    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    variance_loss = F.relu(float(target_std) - std).mean()

    if z.size(0) > 1 and z.size(1) > 1:
        covariance = z.T @ z / (z.size(0) - 1)
        off_diagonal_mask = ~torch.eye(
            covariance.size(0),
            dtype=torch.bool,
            device=covariance.device,
        )
        covariance_loss = covariance[off_diagonal_mask].square().mean()
    else:
        covariance_loss = z.new_zeros(())

    return variance_loss, covariance_loss, std.mean()
