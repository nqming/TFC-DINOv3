"""Auxiliary relational time-frequency consistency, complementary to paired NT-Xent."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RelationalConsistencyLoss(nn.Module):
    """Mean squared difference of off-diagonal cosine Gram entries."""

    def forward(self, z_time, z_freq):
        if z_time.ndim != 2 or z_time.shape != z_freq.shape or z_time.shape[0] < 2:
            raise ValueError("Expected matching [B,D] embeddings with B>=2")
        if not torch.isfinite(z_time).all() or not torch.isfinite(z_freq).all():
            raise ValueError("Loss inputs contain NaN/Inf")
        t = F.normalize(z_time.float(), dim=1, eps=1e-8)
        f = F.normalize(z_freq.float(), dim=1, eps=1e-8)
        delta = t @ t.T - f @ f.T
        mask = ~torch.eye(t.shape[0], dtype=torch.bool, device=t.device)
        return delta[mask].square().mean()


def add_custom_loss(original, z_t, z_f, z_t_aug, z_f_aug, config):
    weight = float(getattr(config, "custom_loss_weight", 0.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("custom_loss_weight must be finite and nonnegative")
    if not getattr(config, "custom_loss_enabled", False) or weight == 0:
        return original, original.new_zeros(())
    criterion = RelationalConsistencyLoss()
    auxiliary = .5 * (criterion(z_t, z_f) + criterion(z_t_aug, z_f_aug))
    return original + weight * auxiliary, auxiliary
