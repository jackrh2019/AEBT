import torch
import torch.nn.functional as F
from typing import Tuple

class BarlowTwinsLossThreeHead(torch.nn.Module):
    def __init__(self, lambda_param: float = 5e-3):
        super(BarlowTwinsLossThreeHead, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        # normalize representations along the batch dimension
        z_a_norm, z_b_norm, z_c_norm = _normalize(z_a, z_b, z_c)

        N = z_a.size(0)

        # cross-correlation matrices
        c_ab = z_a_norm.T @ z_b_norm / N
        c_ac = z_a_norm.T @ z_c_norm / N
        c_bc = z_b_norm.T @ z_c_norm / N

        # loss calculation for each cross-correlation matrix
        loss_ab = _calculate_loss(c_ab, self.lambda_param)
        loss_ac = _calculate_loss(c_ac, self.lambda_param)
        loss_bc = _calculate_loss(c_bc, self.lambda_param)

        # sum the losses
        total_loss = (loss_ab + loss_ac + loss_bc) / 3

        return total_loss

def _normalize(*args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Helper function to normalize tensors along the batch dimension."""
    combined = torch.stack(args, dim=0)  # Shape: k x N x D (k = number of tensors)
    normalized = F.batch_norm(
        combined.flatten(0, 1),
        running_mean=None,
        running_var=None,
        weight=None,
        bias=None,
        training=True,
    ).view_as(combined)
    return tuple(normalized)

def _off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def _calculate_loss(c_matrix, lambda_param):
    """Calculate the loss for a given cross-correlation matrix."""
    invariance_loss = torch.diagonal(c_matrix).add_(-1).pow_(2).sum()
    redundancy_reduction_loss = _off_diagonal(c_matrix).pow_(2).sum()
    return invariance_loss + lambda_param * redundancy_reduction_loss
