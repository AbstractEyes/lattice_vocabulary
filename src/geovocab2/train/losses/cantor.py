# geovocab2/train/model/losses/cantor_relational_loss.py

"""
VAE-style Loss for Cantor Relational Model

Treats the relational transformation as a variational process:
- Reconstruction: Output CLIP should preserve input CLIP structure
- KL Regularization: Transformations should stay bounded
- Cross-modal consistency: CLIP and T5 should converge geometrically
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CantorRelationalVAELoss(nn.Module):
    """
    VAE-style loss for Cantor Relational Model.

    Loss = Reconstruction + β₁·KL + β₂·CrossModal + β₃·Sparsity

    Args:
        beta_kl: Weight for KL divergence term (default: 0.1)
        beta_cross: Weight for cross-modal consistency (default: 0.05)
        beta_sparse: Weight for sparsity regularization (default: 0.001)
        recon_type: 'mse' or 'cosine' for reconstruction loss
    """

    def __init__(
            self,
            beta_kl: float = 0.1,
            beta_cross: float = 0.05,
            beta_sparse: float = 0.001,
            recon_type: str = 'mse'
    ):
        super().__init__()
        self.beta_kl = beta_kl
        self.beta_cross = beta_cross
        self.beta_sparse = beta_sparse
        self.recon_type = recon_type

        assert recon_type in ['mse', 'cosine'], \
            f"recon_type must be 'mse' or 'cosine', got {recon_type}"

    def reconstruction_loss(
            self,
            clip_out: torch.Tensor,
            clip_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruction loss: Output should preserve input structure.

        Args:
            clip_out: Output CLIP embeddings (batch, seq, dim)
            clip_in: Input CLIP embeddings (batch, seq, dim)

        Returns:
            Reconstruction loss scalar
        """
        if self.recon_type == 'mse':
            # L2 reconstruction
            return F.mse_loss(clip_out, clip_in)

        elif self.recon_type == 'cosine':
            # Cosine similarity loss (better for normalized embeddings)
            # Flatten to (batch*seq, dim)
            clip_out_flat = clip_out.reshape(-1, clip_out.shape[-1])
            clip_in_flat = clip_in.reshape(-1, clip_in.shape[-1])

            # Normalize
            clip_out_norm = F.normalize(clip_out_flat, dim=-1)
            clip_in_norm = F.normalize(clip_in_flat, dim=-1)

            # Cosine similarity (1 = identical, -1 = opposite)
            cos_sim = (clip_out_norm * clip_in_norm).sum(dim=-1)

            # Convert to loss (0 = identical, 2 = opposite)
            return (1 - cos_sim).mean()

    def kl_divergence_loss(
            self,
            clip_out: torch.Tensor,
            clip_in: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence: Penalize large deviations from input.

        Treats input as μ₁ with unit variance, output as μ₂ with unit variance.
        KL(N(μ₂,I) || N(μ₁,I)) = 0.5 * ||μ₂ - μ₁||²

        This regularizes the transformation to not change embeddings too drastically.

        Args:
            clip_out: Output CLIP embeddings (batch, seq, dim)
            clip_in: Input CLIP embeddings (batch, seq, dim)

        Returns:
            KL divergence scalar
        """
        # Simplified KL for Gaussians with unit variance
        delta = clip_out - clip_in
        kl = 0.5 * (delta ** 2).mean()

        return kl

    def cross_modal_consistency_loss(
            self,
            clip_out: torch.Tensor,
            t5_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal consistency: CLIP and T5 should agree after processing.

        Encourages the two modalities to converge to a shared understanding.

        Args:
            clip_out: Output CLIP embeddings (batch, seq, dim)
            t5_out: Output T5 embeddings (batch, seq, dim)

        Returns:
            Cross-modal consistency loss
        """
        # Simple L2 distance between the two modalities
        return F.mse_loss(clip_out, t5_out)

    def sparsity_loss(
            self,
            clip_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Sparsity regularization: Encourage efficient use of embedding dimensions.

        Prevents all dimensions from being used equally (promotes specialization).

        Args:
            clip_out: Output CLIP embeddings (batch, seq, dim)

        Returns:
            Sparsity loss scalar
        """
        # L1 penalty on activations
        return torch.abs(clip_out).mean()

    def forward(
            self,
            clip_in: torch.Tensor,
            clip_out: torch.Tensor,
            t5_in: Optional[torch.Tensor] = None,
            t5_out: Optional[torch.Tensor] = None,
            return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute total VAE-style loss.

        Args:
            clip_in: Input CLIP embeddings (batch, seq, dim)
            clip_out: Output CLIP embeddings (batch, seq, dim)
            t5_in: Input T5 embeddings (batch, seq, dim) [optional]
            t5_out: Output T5 embeddings (batch, seq, dim) [optional]
            return_components: If True, return dict of loss components

        Returns:
            Total loss (scalar)
            Optional dict of loss components
        """
        # 1. Reconstruction loss (primary objective)
        recon = self.reconstruction_loss(clip_out, clip_in)

        # 2. KL divergence (regularization)
        kl = self.kl_divergence_loss(clip_out, clip_in)

        # 3. Cross-modal consistency (if T5 outputs provided)
        cross_modal = torch.tensor(0.0, device=clip_out.device)
        if t5_out is not None:
            cross_modal = self.cross_modal_consistency_loss(clip_out, t5_out)

        # 4. Sparsity regularization
        sparsity = self.sparsity_loss(clip_out)

        # Total weighted loss
        total_loss = (
                recon +
                self.beta_kl * kl +
                self.beta_cross * cross_modal +
                self.beta_sparse * sparsity
        )

        if return_components:
            components = {
                'total': total_loss,
                'reconstruction': recon,
                'kl_divergence': kl,
                'cross_modal': cross_modal,
                'sparsity': sparsity
            }
            return total_loss, components

        return total_loss, None


# Convenience functions for different configurations
def create_vae_loss(
        beta_kl: float = 0.1,
        beta_cross: float = 0.05,
        beta_sparse: float = 0.001,  # ADD THIS
        recon_type: str = 'mse'
) -> CantorRelationalVAELoss:
    """
    Create VAE loss with standard hyperparameters.

    #Example:
    #    >>> loss_fn = create_vae_loss(beta_kl=0.1, recon_type='cosine')
    #    >>> loss, components = loss_fn(
    #    ...     clip_in, clip_out, t5_in, t5_out,
    #    ...     return_components=True
    #    ... )
    #"""
    return CantorRelationalVAELoss(
        beta_kl=beta_kl,
        beta_cross=beta_cross,
        beta_sparse=beta_sparse,  # ADD THIS
        recon_type=recon_type
    )


def create_strict_vae_loss() -> CantorRelationalVAELoss:
    """High KL weight - forces conservative transformations."""
    return CantorRelationalVAELoss(beta_kl=1.0, beta_cross=0.1)


def create_loose_vae_loss() -> CantorRelationalVAELoss:
    """Low KL weight - allows more aggressive transformations."""
    return CantorRelationalVAELoss(beta_kl=0.01, beta_cross=0.01)


if __name__ == "__main__":
    print("Testing Cantor Relational VAE Loss...")

    # Create loss function
    loss_fn = create_vae_loss(beta_kl=0.1, beta_cross=0.05, recon_type='mse')

    # Test data
    batch_size, seq_len, dim = 4, 77, 512
    clip_in = torch.randn(batch_size, seq_len, dim)
    clip_out = clip_in + 0.1 * torch.randn_like(clip_in)  # Small perturbation
    t5_in = torch.randn(batch_size, seq_len, dim)
    t5_out = torch.randn(batch_size, seq_len, dim)

    # Compute loss
    total_loss, components = loss_fn(
        clip_in, clip_out, t5_in, t5_out,
        return_components=True
    )

    print(f"\nLoss components:")
    for name, value in components.items():
        print(f"  {name}: {value.item():.6f}")

    # Test different reconstruction types
    print("\nComparing reconstruction types:")
    for recon_type in ['mse', 'cosine']:
        loss_fn = create_vae_loss(recon_type=recon_type)
        loss, _ = loss_fn(clip_in, clip_out)
        print(f"  {recon_type}: {loss.item():.6f}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    clip_out_with_grad = clip_out.clone().requires_grad_(True)
    loss, _ = loss_fn(clip_in, clip_out_with_grad, t5_in, t5_out)
    loss.backward()
    print(f"  ✓ Gradient shape: {clip_out_with_grad.grad.shape}")
    print(f"  ✓ Gradient mean: {clip_out_with_grad.grad.mean().item():.6f}")

    print("\n✓ All tests passed!")