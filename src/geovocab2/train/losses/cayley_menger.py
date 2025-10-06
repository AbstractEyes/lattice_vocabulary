import torch
from torch import Tensor, nn
from typing import Dict

from geovocab2.shapes.formula import SimplexQuality, SimplexVolumeExtended
from geovocab2.train.config.pentachoron_flow import PentachoronFlowConfig


class CayleyMengerValidator(nn.Module):
    """Pure geometric validation loss with Rose margin."""

    def __init__(self, config: PentachoronFlowConfig):
        super().__init__()
        self.config = config
        self.quality_check = SimplexQuality()
        self.volume_calc = SimplexVolumeExtended(mode="auto", check_degeneracy=True)

    def compute_loss(self, predicted_simplices: Tensor) -> Dict[str, Tensor]:
        """Memory-efficient geometric loss with sampling."""
        batch_size = predicted_simplices.shape[0]
        num_origins = predicted_simplices.shape[1]
        k_plus_1 = predicted_simplices.shape[2]
        dim = predicted_simplices.shape[3]

        # CRITICAL: Sample only a subset of simplices to reduce memory
        n_samples = max(1, int(num_origins * self.config.sample_size))  # 15% sampling (tune as needed)

        # Random sampling per batch element
        sample_indices = torch.stack([
            torch.randperm(num_origins, device=predicted_simplices.device)[:n_samples]
            for _ in range(batch_size)
        ])  # [batch_size, n_samples]

        # Gather sampled simplices
        sampled = torch.stack([
            predicted_simplices[b, sample_indices[b]]
            for b in range(batch_size)
        ])  # [batch_size, n_samples, k+1, dim]

        pred_flat = sampled.reshape(batch_size * n_samples, k_plus_1, dim)

        # 1. Rose score (directional coherence) - optimized version
        rose_score = self._compute_rose_score_efficient(pred_flat)
        rose_loss = (1.0 - rose_score).mean()

        # 2. Quality check - only compute regularity (skip expensive volume_quality)
        quality_result = self.quality_check.forward(pred_flat)
        quality_loss = (1.0 - quality_result['regularity']).mean()

        # 3. Volume degeneracy check only (skip full volume computation)
        volume_result = self.volume_calc.forward(pred_flat)
        volume_loss = volume_result['is_degenerate'].float().mean()

        w = self.config.validation_weights
        total_loss = (
                w.get('rose', 0.5) * rose_loss +
                w.get('quality', 0.3) * quality_loss +
                w.get('volume', 0.2) * volume_loss
        )

        return {
            'loss': total_loss,
            'rose_loss': rose_loss.detach(),
            'quality_loss': quality_loss.detach(),
            'volume_loss': volume_loss.detach(),
            'mean_regularity': quality_result['regularity'].mean().detach(),
            'mean_volume': volume_result['volume'].mean().detach(),
            'mean_rose': rose_score.mean().detach()
        }

    def _compute_rose_score_efficient(self, vertices: Tensor) -> Tensor:
        """
        Memory-efficient rose score using centroid-based radial vectors.
        Avoids full edge matrix construction.
        """
        n = vertices.shape[-2]
        if n < 3:
            return torch.zeros(vertices.shape[:-2], device=vertices.device, dtype=vertices.dtype)

        # Use centroid-relative vectors instead of all pairwise edges
        centroid = vertices.mean(dim=-2, keepdim=True)
        radial = vertices - centroid  # [B, k+1, dim]

        # Normalize
        norms = torch.norm(radial, dim=-1, keepdim=True).clamp(min=1e-8)
        unit_radial = radial / norms

        # Efficient pairwise cosine via einsum (no intermediate [B, E, E] matrix)
        cos_matrix = torch.einsum('...id,...jd->...ij', unit_radial, unit_radial)

        # Mask diagonal and take mean
        mask = ~torch.eye(n, dtype=torch.bool, device=vertices.device)
        cos_mean = cos_matrix[..., mask].reshape(*vertices.shape[:-2], -1).mean(dim=-1)

        # Map [-1, 1] -> [0, 1]
        return (cos_mean + 1.0) * 0.5
