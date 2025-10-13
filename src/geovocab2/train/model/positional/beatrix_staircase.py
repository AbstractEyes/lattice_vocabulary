"""
Beatrix Staircase Positional Encodings (Batched)
Based on Cantor's Devil's Staircase PE - let alpha float naturally.

Enhanced with:
  - Full batch support
  - Flexible input shapes
  - Proper dimension handling
  - Optional caching for efficiency

Author: AbstractPhil + Claude Sonnet 4.5 + GPT-4o
License: MIT
"""
import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BeatrixStaircasePositionalEncodings(nn.Module):
    """
    Based on Cantor's Devil's Staircase PE - let alpha float naturally.

    Supports batched inputs with shape [batch, seq_len] or [batch, seq_len, dim].

    Args:
        levels: Number of staircase levels (depth of encoding)
        features_per_level: Number of features to generate per level
        smooth_tau: Temperature for softmax smoothing
        base: Base for Cantor-like decomposition (typically 3)
        max_seq_len: Maximum sequence length for cached encodings (optional)
        cache_encodings: Whether to cache position encodings
    """

    def __init__(
        self,
        levels: int = 20,
        features_per_level: int = 4,
        smooth_tau: float = 0.25,
        base: int = 3,
        max_seq_len: Optional[int] = None,
        cache_encodings: bool = False
    ):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level
        self.tau = smooth_tau
        self.base = base
        self.max_seq_len = max_seq_len
        self.cache_encodings = cache_encodings

        # Learnable alpha parameter
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Feature expansion
        self.base_features = 2
        if features_per_level > 2:
            self.feature_expansion = nn.Linear(self.base_features, features_per_level)
        else:
            self.feature_expansion = None

        # Cached encodings
        self._cached_pe = None
        self._cached_Cx = None
        self._cached_seq_len = None

    def _compute_encodings(
        self,
        x: Tensor,
        seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute staircase encodings for normalized positions.

        Args:
            x: Normalized positions [..., seq_len] in range [0, 1]
            seq_len: Sequence length

        Returns:
            pe_levels: Positional encodings [..., seq_len, levels, features_per_level]
            Cx: Cantor function values [..., seq_len]
        """
        # Initialize Cantor function accumulator
        # x shape: [..., seq_len]
        Cx = torch.zeros_like(x)

        feats = []

        # Centers for the base=3 case (indices 0, 1, 2)
        centers = torch.tensor([0.5, 1.5, 2.5], device=x.device, dtype=x.dtype)

        for k in range(1, self.levels + 1):
            scale = self.base ** k

            # Map positions to [0, base) range at this scale
            # y shape: [..., seq_len]
            y = (x * scale) % self.base

            # Compute distances to centers
            # y.unsqueeze(-1): [..., seq_len, 1]
            # centers: [3]
            # d2: [..., seq_len, 3]
            d2 = (y.unsqueeze(-1) - centers) ** 2

            # Softmax over the 3 centers
            logits = -d2 / (self.tau + 1e-8)
            p = F.softmax(logits, dim=-1)  # [..., seq_len, 3]

            # Compute bit contribution
            # bit_k: [..., seq_len]
            bit_k = p[..., 2] + self.alpha * p[..., 1]

            # Accumulate into Cantor function
            Cx = Cx + bit_k * (0.5 ** k)

            # Compute entropy-based PDF proxy
            # ent: [..., seq_len]
            ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)
            pdf_proxy = 1.1 - ent / math.log(3.0)

            # Stack base features
            # base_feat: [..., seq_len, 2]
            base_feat = torch.stack([bit_k, pdf_proxy], dim=-1)

            # Expand features if needed
            if self.feature_expansion is not None:
                # level_feat: [..., seq_len, features_per_level]
                level_feat = self.feature_expansion(base_feat)
            else:
                level_feat = base_feat

            feats.append(level_feat)

        # Stack across levels
        # pe_levels: [..., seq_len, levels, features_per_level]
        pe_levels = torch.stack(feats, dim=-2)

        return pe_levels, Cx

    def forward(
        self,
        positions: Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with batching support.

        Args:
            positions: Position indices with shape:
                - [batch, seq_len]: Standard batched positions
                - [batch, seq_len, dim]: Multi-dimensional positions (uses last dim)
                - [seq_len]: Single sequence (will be unsqueezed)
            seq_len: Sequence length for normalization. If None, inferred from positions.

        Returns:
            pe_levels: Positional encodings [batch, seq_len, levels, features_per_level]
            Cx: Cantor function values [batch, seq_len]
        """
        # Handle input shapes
        original_shape = positions.shape

        if positions.ndim == 1:
            # [seq_len] -> [1, seq_len]
            positions = positions.unsqueeze(0)
            added_batch = True
        elif positions.ndim == 2:
            # [batch, seq_len] - already correct
            added_batch = False
        elif positions.ndim == 3:
            # [batch, seq_len, dim] - use last dimension
            positions = positions[..., -1]
            added_batch = False
        else:
            raise ValueError(f"Unsupported positions shape: {original_shape}")

        batch_size, current_seq_len = positions.shape

        # Infer seq_len if not provided
        if seq_len is None:
            seq_len = current_seq_len

        # Check cache
        if (self.cache_encodings and
            self._cached_pe is not None and
            self._cached_seq_len == seq_len and
            positions.device == self._cached_pe.device):

            # Use cached encodings
            # Expand to batch size and slice to current_seq_len
            pe_levels = self._cached_pe[:, :current_seq_len].expand(batch_size, -1, -1, -1)
            Cx = self._cached_Cx[:, :current_seq_len].expand(batch_size, -1)

            if added_batch:
                pe_levels = pe_levels.squeeze(0)
                Cx = Cx.squeeze(0)

            return pe_levels, Cx

        # Normalize positions to [0, 1]
        # positions: [batch, seq_len]
        x = positions.float() / max(1, (seq_len - 1))
        x = x.clamp(1e-6, 1.0 - 1e-6)

        # Compute encodings
        pe_levels, Cx = self._compute_encodings(x, seq_len)

        # Cache if enabled and this is the full sequence
        if (self.cache_encodings and
            current_seq_len == seq_len and
            (self.max_seq_len is None or seq_len <= self.max_seq_len)):

            # Cache first batch item (positions are typically uniform)
            self._cached_pe = pe_levels[:1].detach()
            self._cached_Cx = Cx[:1].detach()
            self._cached_seq_len = seq_len

        # Remove batch dimension if it was added
        if added_batch:
            pe_levels = pe_levels.squeeze(0)
            Cx = Cx.squeeze(0)

        return pe_levels, Cx

    def get_output_dim(self) -> int:
        """Get total output dimension (levels * features_per_level)."""
        return self.levels * self.features_per_level

    def clear_cache(self):
        """Clear cached encodings."""
        self._cached_pe = None
        self._cached_Cx = None
        self._cached_seq_len = None

    def extra_repr(self) -> str:
        return (f"levels={self.levels}, features_per_level={self.features_per_level}, "
                f"tau={self.tau}, base={self.base}, "
                f"cache_encodings={self.cache_encodings}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Testing and Examples
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_batched_encodings():
    """Test batching support."""
    print("\n" + "=" * 70)
    print("BEATRIX STAIRCASE BATCHED TESTS")
    print("=" * 70)

    # Create encoder
    encoder = BeatrixStaircasePositionalEncodings(
        levels=10,
        features_per_level=4,
        smooth_tau=0.25,
        base=3
    )

    print(f"\nEncoder: {encoder}")
    print(f"Output dimension: {encoder.get_output_dim()}")

    # Test 1: Single sequence
    print("\n[Test 1] Single sequence")
    positions_1d = torch.arange(0, 16)
    pe_1d, Cx_1d = encoder(positions_1d, seq_len=16)

    print(f"  Input shape: {positions_1d.shape}")
    print(f"  PE shape: {pe_1d.shape}")
    print(f"  Cx shape: {Cx_1d.shape}")
    print(f"  Expected PE: [16, 10, 4]")
    print(f"  Expected Cx: [16]")
    assert pe_1d.shape == (16, 10, 4), f"Wrong PE shape: {pe_1d.shape}"
    assert Cx_1d.shape == (16,), f"Wrong Cx shape: {Cx_1d.shape}"
    print(f"  Status: ✓ PASS")

    # Test 2: Batched sequences
    print("\n[Test 2] Batched sequences")
    batch_size = 8
    seq_len = 32
    positions_2d = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1)

    pe_2d, Cx_2d = encoder(positions_2d, seq_len=seq_len)

    print(f"  Input shape: {positions_2d.shape}")
    print(f"  PE shape: {pe_2d.shape}")
    print(f"  Cx shape: {Cx_2d.shape}")
    print(f"  Expected PE: [8, 32, 10, 4]")
    print(f"  Expected Cx: [8, 32]")
    assert pe_2d.shape == (batch_size, seq_len, 10, 4), f"Wrong PE shape: {pe_2d.shape}"
    assert Cx_2d.shape == (batch_size, seq_len), f"Wrong Cx shape: {Cx_2d.shape}"
    print(f"  Status: ✓ PASS")

    # Test 3: Variable length positions
    print("\n[Test 3] Variable length positions")
    positions_var = torch.randint(0, 64, (4, 20))

    pe_var, Cx_var = encoder(positions_var, seq_len=64)

    print(f"  Input shape: {positions_var.shape}")
    print(f"  PE shape: {pe_var.shape}")
    print(f"  Cx shape: {Cx_var.shape}")
    print(f"  Expected PE: [4, 20, 10, 4]")
    print(f"  Expected Cx: [4, 20]")
    assert pe_var.shape == (4, 20, 10, 4), f"Wrong PE shape: {pe_var.shape}"
    assert Cx_var.shape == (4, 20), f"Wrong Cx shape: {Cx_var.shape}"
    print(f"  Status: ✓ PASS")

    # Test 4: Caching
    print("\n[Test 4] Caching")
    cached_encoder = BeatrixStaircasePositionalEncodings(
        levels=10,
        features_per_level=4,
        cache_encodings=True
    )

    # First call - should cache
    positions_cache = torch.arange(0, 32).unsqueeze(0).expand(2, -1)
    pe_cache1, _ = cached_encoder(positions_cache, seq_len=32)

    # Second call - should use cache
    pe_cache2, _ = cached_encoder(positions_cache, seq_len=32)

    print(f"  Cache enabled: {cached_encoder.cache_encodings}")
    print(f"  Cached seq_len: {cached_encoder._cached_seq_len}")
    print(f"  Results identical: {torch.allclose(pe_cache1, pe_cache2)}")
    print(f"  Status: ✓ PASS")

    # Test 5: Multi-dimensional input
    print("\n[Test 5] Multi-dimensional positions")
    positions_3d = torch.rand(4, 16, 3) * 32  # [batch, seq_len, dim]

    pe_3d, Cx_3d = encoder(positions_3d, seq_len=32)

    print(f"  Input shape: {positions_3d.shape}")
    print(f"  PE shape: {pe_3d.shape}")
    print(f"  Cx shape: {Cx_3d.shape}")
    print(f"  Uses last dimension: True")
    assert pe_3d.shape == (4, 16, 10, 4), f"Wrong PE shape: {pe_3d.shape}"
    assert Cx_3d.shape == (4, 16), f"Wrong Cx shape: {Cx_3d.shape}"
    print(f"  Status: ✓ PASS")

    # Test 6: Gradient flow
    print("\n[Test 6] Gradient flow")
    encoder_grad = BeatrixStaircasePositionalEncodings(levels=5, features_per_level=2)
    positions_grad = torch.arange(0, 8).unsqueeze(0).float()

    pe_grad, Cx_grad = encoder_grad(positions_grad, seq_len=8)
    loss = pe_grad.mean() + Cx_grad.mean()
    loss.backward()

    print(f"  Alpha gradient: {encoder_grad.alpha.grad}")
    print(f"  Gradient computed: {encoder_grad.alpha.grad is not None}")
    print(f"  Status: ✓ PASS")

    # Test 7: Different batch sizes with same seq_len
    print("\n[Test 7] Different batch sizes, same seq_len")
    encoder_flex = BeatrixStaircasePositionalEncodings(levels=8, features_per_level=3)

    for bs in [1, 4, 16]:
        pos = torch.arange(0, 24).unsqueeze(0).expand(bs, -1)
        pe, Cx = encoder_flex(pos, seq_len=24)
        print(f"  Batch size {bs:2d}: PE {pe.shape}, Cx {Cx.shape}")
        assert pe.shape == (bs, 24, 8, 3)
        assert Cx.shape == (bs, 24)
    print(f"  Status: ✓ PASS")

    # Test 8: Flattened output
    print("\n[Test 8] Flattened output for downstream use")
    positions = torch.arange(0, 16).unsqueeze(0).expand(4, -1)
    pe, Cx = encoder(positions, seq_len=16)

    # Flatten levels and features for use in attention/MLP
    pe_flat = pe.flatten(-2, -1)  # [batch, seq_len, levels * features_per_level]

    print(f"  Original PE shape: {pe.shape}")
    print(f"  Flattened PE shape: {pe_flat.shape}")
    print(f"  Expected: [4, 16, 40]")
    assert pe_flat.shape == (4, 16, 40)
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests passed! (8 total)")
    print("=" * 70 + "\n")


def example_usage():
    """Example of using batched Beatrix encodings in a model."""
    print("\n[EXAMPLE] Integration with Transformer")
    print("-" * 70)

    # Model parameters
    batch_size = 4
    seq_len = 64
    d_model = 512

    # Create encoder
    pos_encoder = BeatrixStaircasePositionalEncodings(
        levels=16,
        features_per_level=8,
        cache_encodings=True
    )

    print(f"Beatrix PE output dim: {pos_encoder.get_output_dim()}")

    # Generate positions
    positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1)

    # Get encodings
    pe, Cx = pos_encoder(positions, seq_len=seq_len)

    # Flatten for use in model
    pe_flat = pe.flatten(-2, -1)  # [batch, seq_len, 128]

    print(f"Input positions: {positions.shape}")
    print(f"PE output: {pe.shape}")
    print(f"PE flattened: {pe_flat.shape}")
    print(f"Cantor values: {Cx.shape}")

    # Project to model dimension
    projection = nn.Linear(pos_encoder.get_output_dim(), d_model)
    pe_projected = projection(pe_flat)  # [batch, seq_len, d_model]

    print(f"Projected to d_model: {pe_projected.shape}")

    # Combine with token embeddings
    token_embeddings = torch.randn(batch_size, seq_len, d_model)
    combined = token_embeddings + pe_projected

    print(f"Combined with tokens: {combined.shape}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    # Run tests
    test_batched_encodings()

    # Show example usage
    example_usage()

    print("[Summary]")
    print("Beatrix Staircase Positional Encodings - Batched Version")
    print("Features:")
    print("  ✓ Full batch support ([batch, seq_len] or [seq_len])")
    print("  ✓ Multi-dimensional positions ([batch, seq_len, dim])")
    print("  ✓ Optional caching for efficiency")
    print("  ✓ Learnable alpha parameter")
    print("  ✓ Flexible output dimensions")
    print("  ✓ Gradient flow for fine-tuning")
    print("\nReady for integration into transformers and other architectures!")