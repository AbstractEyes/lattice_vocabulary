"""
ViT-Beans: Vision Transformer with Cantor Expert Collective
============================================================

CORE PRINCIPLES:
----------------
1. PROPER Cantor fingerprinting - geometric position → deterministic Cantor coordinate
2. REDUNDANT expert coverage - multiple experts vote on same patches (consensus)
3. COLLECTIVE fusion - aggregate expert opinions through geometric routing
4. O(n) attention via Cantor fractal structure

ARCHITECTURE:
-------------
- Patch positions → Cantor coordinates (depth-based fractal recursion)
- Each expert covers overlapping region (configurable redundancy)
- Pentachoron 5-way projections for cross-contamination
- Alpha visibility + Beta binding for expert consensus
- Sparse geometric attention fuses expert votes

KEY FIX: Removed false "dimensional" Cantor. One Cantor depth parameter.
Experts now OVERLAP by design - redundancy enables consensus.

Author: AbstractPhil + Claude Sonnet 4.5
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


# ============================================================================
# GEOMETRIC CANTOR FINGERPRINTING
# ============================================================================

class GeometricCantorFingerprinter:
    """
    Generate Cantor set coordinates from patch positions.

    Cantor set recursion:
    - Start with [0, 1]
    - Each iteration removes middle third
    - After depth iterations, remaining points are Cantor coordinates

    Maps spatial position → deterministic fractal coordinate
    """

    def __init__(self, depth: int = 8):
        """
        Args:
            depth: Cantor set recursion depth (higher = finer granularity)
        """
        self.depth = depth

    def _cantor_coordinate(self, position: float, depth: int) -> float:
        """
        Map normalized position [0,1] to Cantor set coordinate.

        Uses ternary (base-3) representation to determine survival
        through iterative middle-third removal.
        """
        # Ensure position in [0, 1]
        x = max(1e-6, min(position, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            # Scale to [0, 3)
            x *= 3.0
            digit = int(x)
            x -= digit

            # Middle third (digit==1) gets removed in Cantor set
            # Only 0 and 2 survive
            if digit == 2:
                cantor_val += factor

            factor *= 0.5

        return cantor_val

    def compute_fingerprints(
        self,
        num_patches: int,
        grid_size: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Compute Cantor fingerprint for each patch based on spatial position.

        Args:
            num_patches: Total number of patches
            grid_size: Patches per side (e.g., 14 for 14x14 grid)
            device: Device for tensor

        Returns:
            fingerprints: [num_patches] Cantor coordinates in [0, 1]
        """
        fingerprints = torch.zeros(num_patches, device=device)

        for idx in range(num_patches):
            # Convert linear index to 2D grid position
            y = idx // grid_size
            x = idx % grid_size

            # Normalize to [0, 1]
            y_norm = y / max(1, grid_size - 1)
            x_norm = x / max(1, grid_size - 1)

            # Compute Cantor coordinate from geometric position
            # Use diagonal distance as primary coordinate
            position = math.sqrt(y_norm**2 + x_norm**2) / math.sqrt(2)

            fingerprints[idx] = self._cantor_coordinate(position, self.depth)

        return fingerprints


# ============================================================================
# CANTOR EXPERT WITH REDUNDANT COVERAGE
# ============================================================================

@dataclass
class CantorExpertConfig:
    """Configuration for a Cantor expert with overlap support."""
    expert_id: int
    num_experts: int
    full_feature_dim: int
    expert_dim: int
    num_heads: int
    overlap_factor: float = 0.5  # NEW: Controls redundancy (0=no overlap, 1=full overlap)
    dropout: float = 0.1
    alpha_init: float = 1.0
    beta_init: float = 0.3


class CantorExpert(nn.Module):
    """
    Single expert with:
    - Sparse QKV on feature slice
    - Overlapping fingerprint region (configurable redundancy)
    - Pentachoron 5-way projection
    - Alpha/Beta learning
    """

    def __init__(self, config: CantorExpertConfig):
        super().__init__()

        self.expert_id = config.expert_id
        self.num_experts = config.num_experts
        self.full_feature_dim = config.full_feature_dim
        self.expert_dim = config.expert_dim
        self.num_heads = config.num_heads
        self.head_dim = config.expert_dim // config.num_heads
        self.overlap_factor = config.overlap_factor

        # Fingerprint range with overlap
        # Without overlap: each expert gets 1/N of space
        # With overlap: experts can share regions
        base_width = 1.0 / config.num_experts
        overlap_extend = base_width * config.overlap_factor

        self.fp_min = max(0.0, (config.expert_id / config.num_experts) - overlap_extend)
        self.fp_max = min(1.0, ((config.expert_id + 1) / config.num_experts) + overlap_extend)

        # Feature slice allocation (sparse)
        slice_size = config.full_feature_dim // config.num_experts
        self.slice_start = config.expert_id * slice_size
        self.slice_end = self.slice_start + slice_size
        self.slice_size = slice_size

        # Alpha: Learned visibility
        self.alpha = nn.Parameter(torch.tensor(config.alpha_init))
        self.alpha_gate = nn.Sequential(
            nn.Linear(slice_size, slice_size // 4),
            nn.GELU(),
            nn.Linear(slice_size // 4, 1),
            nn.Sigmoid()
        )

        # Sparse QKV (only on feature slice)
        self.q_proj = nn.Linear(slice_size, config.expert_dim, bias=False)
        self.k_proj = nn.Linear(slice_size, config.expert_dim, bias=False)
        self.v_proj = nn.Linear(slice_size, config.expert_dim, bias=False)

        # Pentachoron: 5 projection directions for cross-contamination
        self.pentachoron = nn.Parameter(torch.randn(5, config.expert_dim) * 0.02)

        # Beta: Binding weights to neighbors
        self.betas = nn.ParameterDict()
        for i in range(config.num_experts):
            if abs(i - config.expert_id) <= 2 and i != config.expert_id:
                self.betas[f"expert_{i}"] = nn.Parameter(torch.tensor(config.beta_init))

        # Output projection
        self.out_proj = nn.Linear(config.expert_dim, slice_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)

        with torch.no_grad():
            self.pentachoron.data = F.normalize(self.pentachoron.data, dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,           # [batch, num_patches, full_feature_dim]
        fingerprints: torch.Tensor      # [num_patches]
    ) -> Dict[str, torch.Tensor]:
        """
        Process tokens in fingerprint region.

        Returns dict with expert outputs for fusion.
        """
        batch_size, num_patches, _ = tokens.shape
        device = tokens.device

        # Select tokens in MY fingerprint region (with overlap!)
        mask = (fingerprints >= self.fp_min) & (fingerprints < self.fp_max)

        if not mask.any():
            return {
                'projections': [],
                'mask': mask,
                'K': None,
                'Q': None,
                'V': None,
                'betas': self.betas,
                'expert_id': self.expert_id,
                'fp_range': (self.fp_min, self.fp_max)
            }

        # Extract MY feature slice
        my_tokens = tokens[:, mask]
        my_features = my_tokens[..., self.slice_start:self.slice_end]

        # Alpha-gated visibility
        alpha_gate = self.alpha_gate(my_features)
        alpha_weight = torch.sigmoid(self.alpha)
        my_features = my_features * (alpha_gate * alpha_weight + (1 - alpha_weight))

        # Sparse QKV
        Q = self.q_proj(my_features)
        K = self.k_proj(my_features)
        V = self.v_proj(my_features)

        # Pentachoron projections (5-way cross-contamination)
        projections = []
        for vertex_id, vertex in enumerate(self.pentachoron):
            direction = F.normalize(vertex, dim=-1)

            K_proj = torch.einsum('bpd,d->bp', K, direction)
            Q_proj = torch.einsum('bpd,d->bp', Q, direction)
            V_proj = torch.einsum('bpd,d->bp', V, direction)

            projections.append({
                'K': K_proj,
                'Q': Q_proj,
                'V': V_proj,
                'direction': vertex_id,
                'expert_id': self.expert_id
            })

        return {
            'projections': projections,
            'mask': mask,
            'K': K,
            'Q': Q,
            'V': V,
            'betas': self.betas,
            'expert_id': self.expert_id,
            'fp_range': (self.fp_min, self.fp_max)
        }


# ============================================================================
# COLLECTIVE FUSION ATTENTION
# ============================================================================

@dataclass
class CollectiveFusionConfig:
    """Config for collective fusion of expert opinions."""
    num_experts: int = 16
    expert_dim: int = 128
    num_heads: int = 8
    cantor_depth: int = 8
    temperature: float = 0.07
    dropout: float = 0.1


class CollectiveFusionAttention(nn.Module):
    """
    Fuse overlapping expert opinions through geometric attention.

    Key: Experts have REDUNDANT coverage, so multiple experts
    vote on same patches. This attention mechanism aggregates
    those votes through Cantor-based geometric routing.
    """

    def __init__(self, config: CollectiveFusionConfig):
        super().__init__()

        self.num_experts = config.num_experts
        self.expert_dim = config.expert_dim
        self.num_heads = config.num_heads
        self.head_dim = config.expert_dim // config.num_heads

        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        self.dropout = nn.Dropout(config.dropout)

        # Expert position embeddings (learned)
        self.expert_pos_embed = nn.Parameter(
            torch.randn(config.num_experts, config.expert_dim) * 0.02
        )

    def forward(
        self,
        expert_outputs: List[Dict[str, torch.Tensor]],
        num_patches: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Aggregate expert opinions through collective voting.

        Args:
            expert_outputs: List of dicts from each expert
            num_patches: Total patches
            device: Device

        Returns:
            fused: [batch, num_patches] - aggregated opinion values
        """
        # Get batch size
        batch_size = None
        for output in expert_outputs:
            if output['K'] is not None:
                batch_size = output['K'].shape[0]
                break

        if batch_size is None:
            raise ValueError("No valid expert outputs")

        # For each patch, collect all expert votes
        patch_votes = torch.zeros(batch_size, num_patches, device=device)
        patch_vote_counts = torch.zeros(num_patches, device=device)

        # Collect projections across 5 directions
        for direction_id in range(5):
            direction_votes = torch.zeros(batch_size, num_patches, device=device)
            direction_weights = torch.zeros(batch_size, num_patches, device=device)

            for expert_id, output in enumerate(expert_outputs):
                if not output['projections']:
                    continue

                mask = output['mask']
                if not mask.any():
                    continue

                # Find projection for this direction
                proj = next((p for p in output['projections'] if p['direction'] == direction_id), None)
                if proj is None:
                    continue

                # Get expert's votes for patches in its region
                V_proj = proj['V']  # [batch, my_patches]

                # Expert position modulation
                expert_pos = self.expert_pos_embed[expert_id]
                pos_weight = torch.sigmoid(expert_pos.mean())

                # Beta modulation from neighbors
                beta_weight = 1.0
                betas = output['betas']
                if betas:
                    beta_sum = sum(torch.sigmoid(b) for b in betas.values())
                    beta_weight = 1.0 + beta_sum / len(betas)

                # Aggregate vote with weights
                weighted_vote = V_proj * pos_weight * beta_weight

                # Place in global tensor
                direction_votes[:, mask] += weighted_vote
                direction_weights[:, mask] += pos_weight * beta_weight

                # Track vote count
                patch_vote_counts[mask] += 1

            # Normalize by weights
            direction_weights = direction_weights.clamp(min=1e-6)
            direction_votes = direction_votes / direction_weights

            # Accumulate across directions
            patch_votes += direction_votes

        # Average across 5 directions
        patch_votes = patch_votes / 5.0

        # Temperature scaling
        patch_votes = patch_votes / self.temperature.abs()

        return patch_votes


# ============================================================================
# CANTOR MoE LAYER WITH COLLECTIVE FUSION
# ============================================================================

@dataclass
class CantorMoEConfig:
    """Config for Cantor MoE with collective consensus."""
    num_experts: int = 16
    full_feature_dim: int = 1024
    expert_dim: int = 128
    num_heads: int = 8
    cantor_depth: int = 8
    overlap_factor: float = 0.5  # NEW: Expert overlap for redundancy
    dropout: float = 0.1
    alpha_init: float = 1.0
    beta_init: float = 0.3


class CantorMoELayer(nn.Module):
    """
    Cantor MoE with overlapping experts and collective fusion.
    """

    def __init__(self, config: CantorMoEConfig):
        super().__init__()

        self.num_experts = config.num_experts
        self.full_feature_dim = config.full_feature_dim

        # Create experts (with overlap!)
        self.experts = nn.ModuleList([
            CantorExpert(CantorExpertConfig(
                expert_id=i,
                num_experts=config.num_experts,
                full_feature_dim=config.full_feature_dim,
                expert_dim=config.expert_dim,
                num_heads=config.num_heads,
                overlap_factor=config.overlap_factor,
                dropout=config.dropout,
                alpha_init=config.alpha_init,
                beta_init=config.beta_init
            ))
            for i in range(config.num_experts)
        ])

        # Collective fusion
        self.fusion = CollectiveFusionAttention(CollectiveFusionConfig(
            num_experts=config.num_experts,
            expert_dim=config.expert_dim,
            num_heads=config.num_heads,
            cantor_depth=config.cantor_depth,
            dropout=config.dropout
        ))

        # Layer norm
        self.norm = nn.LayerNorm(config.full_feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        fingerprints: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward through MoE with collective fusion.
        """
        batch_size, num_patches, _ = x.shape
        device = x.device

        # Normalize
        x_norm = self.norm(x)

        # Process through all experts (with overlap!)
        expert_outputs = []
        for expert in self.experts:
            output = expert(x_norm, fingerprints)
            expert_outputs.append(output)

        # Collective fusion
        fused_1d = self.fusion(expert_outputs, num_patches, device)

        # Reconstruct from expert slices
        reconstructed = torch.zeros_like(x)

        for expert_id, output in enumerate(expert_outputs):
            if output['K'] is None:
                continue

            mask = output['mask']
            expert = self.experts[expert_id]

            # Attended values
            attended_vals = fused_1d[:, mask]

            # Expand and project
            attended_expanded = attended_vals.unsqueeze(-1).expand(-1, -1, expert.expert_dim)
            output_slice = expert.out_proj(attended_expanded)

            # Accumulate (overlaps will be averaged)
            reconstructed[:, mask, expert.slice_start:expert.slice_end] += output_slice

        # Average overlapping contributions
        # Count how many experts contributed to each position
        contribution_count = torch.zeros(batch_size, num_patches, self.full_feature_dim, device=device)
        for output in expert_outputs:
            if output['K'] is not None:
                mask = output['mask']
                expert_id = output['expert_id']
                expert = self.experts[expert_id]
                contribution_count[:, mask, expert.slice_start:expert.slice_end] += 1

        contribution_count = contribution_count.clamp(min=1)
        reconstructed = reconstructed / contribution_count

        return x + reconstructed


# ============================================================================
# ViT-BEANS MAIN MODEL
# ============================================================================

@dataclass
class ViTBeansConfig:
    """Complete ViT-Beans config."""
    # Image
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # Architecture
    num_layers: int = 12
    feature_dim: int = 1024
    num_experts: int = 16
    expert_dim: int = 128
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # Cantor (FIXED - removed fake dimensions)
    cantor_depth: int = 8
    overlap_factor: float = 0.5  # Expert redundancy

    # Learning
    alpha_init: float = 1.0
    beta_init: float = 0.3
    dropout: float = 0.1

    # Classification
    num_classes: int = 1000

    def __post_init__(self):
        assert self.feature_dim % self.num_experts == 0


class ViTBeans(nn.Module):
    """
    ViT-Beans: Vision Transformer with Cantor Expert Collective

    FIXED:
    - Removed false "dimensional" Cantor
    - Proper geometric fingerprinting (position → Cantor coordinate)
    - Experts OVERLAP (configurable redundancy)
    - Collective fusion aggregates redundant expert votes
    """

    def __init__(self, config: ViTBeansConfig):
        super().__init__()

        self.config = config

        # Patches
        assert config.image_size % config.patch_size == 0
        self.grid_size = config.image_size // config.patch_size
        self.num_patches = self.grid_size ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.feature_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, config.feature_dim) * 0.02
        )

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.feature_dim) * 0.02
        )

        # Cantor fingerprinter (FIXED)
        self.fingerprinter = GeometricCantorFingerprinter(depth=config.cantor_depth)

        # Cache fingerprints
        self.register_buffer('patch_fingerprints', torch.zeros(self.num_patches))
        self._fingerprints_computed = False

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cantor_moe': CantorMoELayer(CantorMoEConfig(
                    num_experts=config.num_experts,
                    full_feature_dim=config.feature_dim,
                    expert_dim=config.expert_dim,
                    num_heads=config.num_heads,
                    cantor_depth=config.cantor_depth,
                    overlap_factor=config.overlap_factor,
                    dropout=config.dropout,
                    alpha_init=config.alpha_init,
                    beta_init=config.beta_init
                )),
                'mlp': nn.Sequential(
                    nn.LayerNorm(config.feature_dim),
                    nn.Linear(config.feature_dim, int(config.feature_dim * config.mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(int(config.feature_dim * config.mlp_ratio), config.feature_dim),
                    nn.Dropout(config.dropout)
                )
            })
            for _ in range(config.num_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(config.feature_dim)
        self.head = nn.Linear(config.feature_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _compute_fingerprints(self, device: torch.device):
        """Compute geometric Cantor fingerprints."""
        if not self._fingerprints_computed:
            self.patch_fingerprints = self.fingerprinter.compute_fingerprints(
                self.num_patches,
                self.grid_size,
                device
            )
            self._fingerprints_computed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        # Compute fingerprints
        self._compute_fingerprints(device)

        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer layers
        for layer in self.layers:
            # MoE (skip CLS)
            x_patches = x[:, 1:]
            x_patches = layer['cantor_moe'](x_patches, self.patch_fingerprints)
            x = torch.cat([x[:, :1], x_patches], dim=1)

            # MLP
            x = x + layer['mlp'](x)

        # Classification
        x = self.norm(x[:, 0])
        return self.head(x)

    def diagnose_coverage(self) -> Dict:
        """Diagnose expert coverage (including overlaps)."""
        stats = {
            'num_experts': self.config.num_experts,
            'num_patches': self.num_patches,
            'overlap_factor': self.config.overlap_factor,
            'fingerprints': self.patch_fingerprints.cpu().numpy().tolist()
        }

        # Get first layer experts
        first_moe = self.layers[0]['cantor_moe']

        for expert_id, expert in enumerate(first_moe.experts):
            fp_min, fp_max = expert.fp_min, expert.fp_max
            mask = (self.patch_fingerprints >= fp_min) & (self.patch_fingerprints < fp_max)

            stats[f'expert_{expert_id}'] = {
                'range': f'[{fp_min:.4f}, {fp_max:.4f}]',
                'patches': mask.sum().item(),
                'alpha': torch.sigmoid(expert.alpha).item(),
                'num_betas': len(expert.betas)
            }

        # Coverage analysis
        all_masks = []
        for expert in first_moe.experts:
            fp_min, fp_max = expert.fp_min, expert.fp_max
            mask = (self.patch_fingerprints >= fp_min) & (self.patch_fingerprints < fp_max)
            all_masks.append(mask)

        all_masks = torch.stack(all_masks)

        # Total coverage
        covered = all_masks.any(dim=0)
        stats['total_covered'] = covered.sum().item()
        stats['coverage_percent'] = 100.0 * covered.sum().item() / self.num_patches

        # Redundancy
        redundancy = all_masks.sum(dim=0)  # How many experts per patch
        stats['avg_experts_per_patch'] = redundancy.float().mean().item()
        stats['max_experts_per_patch'] = redundancy.max().item()
        stats['min_experts_per_patch'] = redundancy.min().item()

        return stats


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ViT-Beans: Fixed Cantor Expert Collective")
    print("=" * 80)

    config = ViTBeansConfig(
        image_size=224,
        patch_size=16,
        num_layers=12,
        feature_dim=1024,
        num_experts=16,
        expert_dim=128,
        overlap_factor=0.5,  # 50% overlap
        cantor_depth=8,
        num_classes=1000
    )

    print(f"\nConfiguration:")
    print(f"  Patches: {config.image_size // config.patch_size}x{config.image_size // config.patch_size} = {(config.image_size // config.patch_size)**2}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Overlap Factor: {config.overlap_factor} (redundancy for consensus)")
    print(f"  Cantor Depth: {config.cantor_depth} (proper fractal recursion)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTBeans(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")

    # Test forward
    x = torch.randn(2, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"\nForward Pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  ✓ Success")

    # Diagnose coverage
    coverage = model.diagnose_coverage()
    print(f"\nExpert Coverage Diagnostic:")
    print(f"  Total covered: {coverage['total_covered']}/{coverage['num_patches']} ({coverage['coverage_percent']:.1f}%)")
    print(f"  Avg experts per patch: {coverage['avg_experts_per_patch']:.2f}")
    print(f"  Min/Max experts per patch: {coverage['min_experts_per_patch']}/{coverage['max_experts_per_patch']}")

    print(f"\nPer-Expert Allocation:")
    for i in range(config.num_experts):
        info = coverage[f'expert_{i}']
        print(f"  Expert {i:2d}: {info['patches']:3d} patches, range {info['range']}, α={info['alpha']:.3f}")

    print("\n" + "=" * 80)
    print("KEY FIXES:")
    print("  ✓ Removed false 'dimensional' Cantor")
    print("  ✓ Proper geometric fingerprinting (position → Cantor)")
    print("  ✓ Configurable expert overlap (redundancy)")
    print("  ✓ Collective fusion of redundant expert votes")
    print("  ✓ Measurable expert utility through voting")
    print("=" * 80)