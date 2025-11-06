# geovocab2/train/model/cantor_relational.py

"""
Cantor Relational Model - Cross-Modal Geometric Understanding

Learns relationships between two latent spaces (CLIP-B and T5-base) using
Cantor fractal routing. T5 maps and catalogues patterns in CLIP space,
then both modalities exchange information through geometric cross-attention.

Architecture:
    1. Parallel self-attention for each modality (Cantor routing)
    2. Bidirectional cross-attention to exchange relational knowledge
    3. Multiple blocks for iterative refinement
    4. Final output: modified CLIP-B latent (the "truth")

Key Properties:
    - O(n) complexity per attention operation via Cantor routing
    - Geometric pattern matching across embedding spaces
    - Bidirectional information flow between modalities
    - CLIP-B as ground truth output (T5 augments understanding)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

from geovocab2.train.model.layers.attention.cantor_global import CantorAttention, CantorAttentionConfig


@dataclass
class CantorRelationalConfig:
    """
    Configuration for Cantor Relational Model.

    Args:
        dim: Embedding dimension (512 for both CLIP-B and T5-base)
        num_heads: Number of attention heads
        num_blocks: Number of relational blocks
        seq_len: Sequence length (77 for CLIP)
        cantor_depth: Cantor fractal depth
        local_window: Neighbors per token in Cantor space
        dropout: Dropout probability
        use_adaptive_window: Enable adaptive k based on sequence length
    """
    dim: int = 512
    num_heads: int = 8
    num_blocks: int = 6
    seq_len: int = 77
    cantor_depth: int = 8
    local_window: int = 64
    dropout: float = 0.1
    use_adaptive_window: bool = False

    # Cross-attention specific
    cross_window: Optional[int] = None  # Use different k for cross-attention

    def __post_init__(self):
        if self.cross_window is None:
            self.cross_window = self.local_window
        assert self.dim % self.num_heads == 0


class CantorRelationalBlock(nn.Module):
    """
    Single block of cross-modal relational learning.

    Flow:
        1. CLIP self-attention (understand own structure)
        2. T5 self-attention (understand own structure)
        3. T5 → CLIP cross-attention (T5 queries CLIP patterns)
        4. CLIP → T5 cross-attention (CLIP gets T5 context)
        5. Feed-forward networks for both
    """

    def __init__(self, config: CantorRelationalConfig):
        super().__init__()
        self.config = config

        # Self-attention for each modality
        self.clip_self_attn = self._create_attention(config.local_window)
        self.t5_self_attn = self._create_attention(config.local_window)

        # Cross-attention: T5 queries CLIP (T5 learns from CLIP)
        self.t5_cross_attn = self._create_cross_attention()

        # Cross-attention: CLIP queries T5 (CLIP gets T5 context)
        self.clip_cross_attn = self._create_cross_attention()

        # Feed-forward networks
        self.clip_ffn = self._create_ffn()
        self.t5_ffn = self._create_ffn()

        # Layer norms
        self.clip_norm1 = nn.LayerNorm(config.dim)
        self.clip_norm2 = nn.LayerNorm(config.dim)
        self.clip_norm3 = nn.LayerNorm(config.dim)

        self.t5_norm1 = nn.LayerNorm(config.dim)
        self.t5_norm2 = nn.LayerNorm(config.dim)
        self.t5_norm3 = nn.LayerNorm(config.dim)

    def _create_attention(self, window: int) -> CantorAttention:
        """Create self-attention with Cantor routing."""
        attn_config = CantorAttentionConfig(
            dim=self.config.dim,
            num_heads=self.config.num_heads,
            depth=self.config.cantor_depth,
            max_seq_len=self.config.seq_len,
            local_window=window,
            adaptive_window=self.config.use_adaptive_window,
            dropout=self.config.dropout,
            causal=False  # Not causal for this use case
        )
        return CantorAttention(attn_config)

    def _create_cross_attention(self) -> 'CantorCrossAttention':
        """Create cross-attention with Cantor routing."""
        return CantorCrossAttention(
            dim=self.config.dim,
            num_heads=self.config.num_heads,
            depth=self.config.cantor_depth,
            seq_len=self.config.seq_len,
            local_window=self.config.cross_window,
            dropout=self.config.dropout
        )

    def _create_ffn(self) -> nn.Module:
        """Create feed-forward network."""
        hidden_dim = 4 * self.config.dim
        return nn.Sequential(
            nn.Linear(self.config.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, self.config.dim),
            nn.Dropout(self.config.dropout)
        )

    def forward(
            self,
            clip_embed: torch.Tensor,
            t5_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through relational block.

        Args:
            clip_embed: CLIP-B embeddings (batch, 77, 512)
            t5_embed: T5-base embeddings (batch, 77, 512)

        Returns:
            Tuple of (modified CLIP, modified T5)
        """
        # 1. Self-attention for each modality
        clip_self = clip_embed + self.clip_self_attn(self.clip_norm1(clip_embed))
        t5_self = t5_embed + self.t5_self_attn(self.t5_norm1(t5_embed))

        # 2. Cross-attention: T5 queries CLIP (T5 learns CLIP patterns)
        t5_cross = t5_self + self.t5_cross_attn(
            query=self.t5_norm2(t5_self),
            key_value=clip_self  # T5 attends to CLIP structure
        )

        # 3. Cross-attention: CLIP queries T5 (CLIP gets T5 context)
        clip_cross = clip_self + self.clip_cross_attn(
            query=self.clip_norm2(clip_self),
            key_value=t5_self  # CLIP attends to T5 understanding
        )

        # 4. Feed-forward networks
        clip_out = clip_cross + self.clip_ffn(self.clip_norm3(clip_cross))
        t5_out = t5_cross + self.t5_ffn(self.t5_norm3(t5_cross))

        return clip_out, t5_out


class CantorCrossAttention(nn.Module):
    """
    Cross-attention using Cantor routing.

    Query comes from one modality, Key/Value from another.
    Uses Cantor geometry to find relevant cross-modal relationships.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            depth: int,
            seq_len: int,
            local_window: int,
            dropout: float
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

        # Use Cantor routing for cross-modal attention
        self.cantor_attn = CantorAttention(CantorAttentionConfig(
            dim=dim,
            num_heads=num_heads,
            depth=depth,
            max_seq_len=seq_len,
            local_window=local_window,
            dropout=dropout,
            causal=False
        ))

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.

        Args:
            query: Query embeddings (batch, seq, dim)
            key_value: Key/Value embeddings (batch, seq, dim)

        Returns:
            Cross-attended output (batch, seq, dim)
        """
        batch_size, seq_len, _ = query.shape

        # Project query from one modality
        q = self.q_proj(query)

        # Project key and value from other modality
        kv = self.kv_proj(key_value)
        kv = kv.reshape(batch_size, seq_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, batch, heads, seq, head_dim)
        k, v = kv[0], kv[1]

        # Reshape query
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

        # Use Cantor routing for sparse cross-attention
        # This finds geometrically relevant cross-modal relationships
        attn_out = self.cantor_attn._sparse_attention(q, k, v, seq_len)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(attn_out)
        output = self.dropout(output)

        return output


class CantorRelationalModel(nn.Module):
    """
    Full Cantor Relational Model.

    Takes CLIP-B and T5-base embeddings, learns their geometric relationships,
    outputs modified CLIP-B as the "truth".
    """

    def __init__(self, config: CantorRelationalConfig):
        super().__init__()
        self.config = config

        # Stack of relational blocks
        self.blocks = nn.ModuleList([
            CantorRelationalBlock(config)
            for _ in range(config.num_blocks)
        ])

        # Final layer norms
        self.clip_final_norm = nn.LayerNorm(config.dim)
        self.t5_final_norm = nn.LayerNorm(config.dim)

        # Optional: projection head for CLIP output
        self.clip_out_proj = nn.Linear(config.dim, config.dim)

    def forward(
            self,
            clip_embed: torch.Tensor,
            t5_embed: torch.Tensor,
            return_both: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through relational model.

        Args:
            clip_embed: CLIP-B embeddings (batch, 77, 512)
            t5_embed: T5-base embeddings (batch, 77, 512)
            return_both: If True, return both modalities; else just CLIP

        Returns:
            Modified CLIP-B latent (the "truth")
            Or tuple of (CLIP, T5) if return_both=True
        """
        # Process through all relational blocks
        for block in self.blocks:
            clip_embed, t5_embed = block(clip_embed, t5_embed)

        # Final normalization
        clip_out = self.clip_final_norm(clip_embed)
        t5_out = self.t5_final_norm(t5_embed)

        # Project CLIP output (the "truth")
        clip_out = self.clip_out_proj(clip_out)

        if return_both:
            return clip_out, t5_out
        return clip_out


def create_cantor_relational(
        dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 6,
        seq_len: int = 77,
        cantor_depth: int = 8,
        local_window: int = 64,
        **kwargs
) -> CantorRelationalModel:
    """
    Convenience function to create Cantor Relational Model.

    #Example:
    #    >>> model = create_cantor_relational()
    #    >>> clip_embed = torch.randn(4, 77, 512)  # CLIP-B
    #    >>> t5_embed = torch.randn(4, 77, 512)    # T5-base
    #    >>> output = model(clip_embed, t5_embed)  # Modified CLIP
    #    >>> print(output.shape)  # (4, 77, 512)
    """
    config = CantorRelationalConfig(
        dim=dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        seq_len=seq_len,
        cantor_depth=cantor_depth,
        local_window=local_window,
        **kwargs
    )
    return CantorRelationalModel(config)


if __name__ == "__main__":
    print("Testing Cantor Relational Model...")

    # Create model
    model = create_cantor_relational(
        dim=512,
        num_heads=8,
        num_blocks=6,
        seq_len=77,
        cantor_depth=8,
        local_window=64
    )

    # Test inputs
    batch_size = 4
    clip_embed = torch.randn(batch_size, 77, 512)
    t5_embed = torch.randn(batch_size, 77, 512)

    print(f"\nInput shapes:")
    print(f"  CLIP-B: {clip_embed.shape}")
    print(f"  T5-base: {t5_embed.shape}")

    # Forward pass
    output = model(clip_embed, t5_embed)
    print(f"\nOutput shape: {output.shape}")

    # Test return_both mode
    clip_out, t5_out = model(clip_embed, t5_embed, return_both=True)
    print(f"\nReturn both:")
    print(f"  CLIP: {clip_out.shape}")
    print(f"  T5: {t5_out.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✓ All tests passed!")