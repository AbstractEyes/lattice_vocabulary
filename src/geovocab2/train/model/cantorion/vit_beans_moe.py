"""
ViT-Beans: Vision Transformer with Geometric Cantor Expert Collective
======================================================================

COALESCED ARCHITECTURE:
-----------------------
1. Cantor Fingerprinting: Deterministic spatial routing (WHERE)
2. Pentachoron Geometry: Learned 5-vertex simplex opinion structure (HOW)
3. Query as K-Simplex Navigation: Query learns geometric paths
4. Role-Weighted Projections: anchor/need/relation/purpose/observer
5. Cayley-Menger Constraints: Enforce valid geometric structure
6. Collective Fusion: Expert consensus through alpha/beta modulation

Key Insight: Cantor provides spatial addresses, Pentachoron provides
geometric opinion formation mechanism. Both work together.

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
# GEOMETRIC LOSS FUNCTIONS
# ============================================================================

class PentachoronGeometricLoss(nn.Module):
    """
    Cayley-Menger loss for expert pentachora.

    Ensures each expert's 5-vertex simplex maintains valid geometry:
    - Volume > threshold (prevents collapse)
    - Edge uniformity (prevents distortion)
    - Gram matrix condition (ensures embedding quality)
    """

    def __init__(
        self,
        volume_floor: float = 0.1,
        edge_uniformity_weight: float = 0.1,
        gram_weight: float = 0.05
    ):
        super().__init__()
        self.volume_floor = volume_floor
        self.edge_weight = edge_uniformity_weight
        self.gram_weight = gram_weight

    def compute_cayley_menger_volume(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute 4-simplex volume via Cayley-Menger determinant.

        Args:
            vertices: [5, dim] pentachoron vertices

        Returns:
            volume: scalar volume
        """
        # Pairwise squared distances
        diff = vertices.unsqueeze(1) - vertices.unsqueeze(0)  # [5, 5, dim]
        distsq = (diff * diff).sum(dim=-1)  # [5, 5]

        # Build Cayley-Menger matrix
        M = torch.zeros(6, 6, device=vertices.device, dtype=vertices.dtype)
        M[0, 1:] = 1.0
        M[1:, 0] = 1.0
        M[1:, 1:] = distsq

        # Volume = sqrt(-det(M) / 9216)
        det = torch.linalg.det(M)
        volume_sq = (-det / 9216.0).clamp(min=0.0)
        volume = volume_sq.sqrt()

        return volume

    def compute_edge_uniformity(self, vertices: torch.Tensor) -> torch.Tensor:
        """Measure edge length variation."""
        diff = vertices.unsqueeze(1) - vertices.unsqueeze(0)
        distsq = (diff * diff).sum(dim=-1)

        # Extract upper triangle (10 edges)
        triu_indices = torch.triu_indices(5, 5, offset=1)
        edge_lengths = distsq[triu_indices[0], triu_indices[1]]

        # Coefficient of variation
        edge_std = edge_lengths.std()
        edge_mean = edge_lengths.mean()
        uniformity = edge_std / edge_mean.clamp(min=1e-6)

        return uniformity

    def compute_gram_condition(self, vertices: torch.Tensor) -> torch.Tensor:
        """Gram matrix conditioning."""
        centered = vertices - vertices.mean(dim=0, keepdim=True)
        gram = torch.mm(centered, centered.T)

        gram_trace = torch.diagonal(gram).sum()
        gram_det = torch.linalg.det(gram)

        condition = gram_det / gram_trace.clamp(min=1e-6)
        penalty = F.relu(1.0 - condition)

        return penalty

    def forward(self, expert_pentachora: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute geometric loss across all expert pentachora.

        Args:
            expert_pentachora: List of [5, expert_dim] tensors

        Returns:
            total_loss: scalar loss
        """
        total_loss = 0.0

        for vertices in expert_pentachora:
            # Volume loss
            volume = self.compute_cayley_menger_volume(vertices)
            volume_loss = F.relu(self.volume_floor - volume)

            # Edge uniformity loss
            edge_loss = self.compute_edge_uniformity(vertices)

            # Gram condition loss
            gram_loss = 0.0
            if self.gram_weight > 0:
                gram_loss = self.compute_gram_condition(vertices)

            total_loss += volume_loss + self.edge_weight * edge_loss + self.gram_weight * gram_loss

        return total_loss / len(expert_pentachora)


# ============================================================================
# CANTOR FINGERPRINTING
# ============================================================================

class GeometricCantorFingerprinter:
    """Cantor pairing for deterministic spatial routing."""

    def __init__(self, depth: int = 8):
        self.depth = depth
        self.num_buckets = 2 ** depth

    def _cantor_pairing(self, x: int, y: int) -> int:
        return (x + y) * (x + y + 1) // 2 + y

    def compute_fingerprints(
        self,
        num_patches: int,
        grid_size: int,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        fingerprints = torch.zeros(num_patches, device=device, dtype=torch.float32)

        max_coord = grid_size - 1
        max_cantor = self._cantor_pairing(max_coord, max_coord)

        for idx in range(num_patches):
            y = idx // grid_size
            x = idx % grid_size

            cantor_value = self._cantor_pairing(x, y)
            normalized = cantor_value / max(1, max_cantor)

            bucket = int(normalized * (self.num_buckets - 1))
            bucketed_value = bucket / (self.num_buckets - 1)

            fingerprints[idx] = bucketed_value

        return fingerprints


# ============================================================================
# CANTOR EXPERT WITH PENTACHORON GEOMETRY
# ============================================================================

@dataclass
class CantorExpertConfig:
    """Configuration for single Cantor expert."""
    expert_id: int
    num_experts: int
    full_feature_dim: int
    expert_dim: int
    num_heads: int
    alpha_init: float = 0.5
    beta_init: float = 0.3
    dropout: float = 0.1


class CantorExpert(nn.Module):
    """
    Cantor Expert with Pentachoron Geometric Opinion Structure.

    Architecture:
    1. Receives patches via Cantor fingerprint routing (deterministic)
    2. Forms opinion via pentachoron geometry (learned 5-vertex simplex)
    3. Query navigates k-simplex structure
    4. Role-weighted projections (anchor/need/relation/purpose/observer)
    5. Alpha visibility + Beta binding for collective consensus
    """

    def __init__(
        self,
        config: CantorExpertConfig,
        fp_min: float,
        fp_max: float,
        neighbor_ids: List[int]
    ):
        super().__init__()

        self.expert_id = config.expert_id
        self.num_experts = config.num_experts
        self.fp_min = fp_min
        self.fp_max = fp_max
        self.neighbor_ids = neighbor_ids

        # Feature slice for this expert (each expert handles 1/num_heads of features)
        self.slice_size = config.full_feature_dim // config.num_heads
        self.slice_start = config.expert_id % config.num_heads * self.slice_size
        self.slice_end = self.slice_start + self.slice_size

        # ====================================================================
        # PENTACHORON GEOMETRY (5-vertex learned simplex)
        # ====================================================================
        # Like David's crystals, each vertex has semantic role
        self.pentachoron_vertices = nn.Parameter(
            torch.randn(5, config.expert_dim) * 0.02
        )

        # Role weights (anchor, need, relation, purpose, observer)
        role_weights = torch.tensor([1.0, -0.75, 0.75, 0.75, -0.75])
        self.register_buffer("role_weights", role_weights)

        # ====================================================================
        # QUERY AS K-SIMPLEX NAVIGATION
        # ====================================================================
        # Query learns to navigate through pentachoron geometry
        self.q_proj = nn.Linear(self.slice_size, config.expert_dim)

        # Query refinement via vertex attention (learns geometric paths)
        self.q_vertex_attention = nn.MultiheadAttention(
            config.expert_dim,
            num_heads=1,
            batch_first=True,
            dropout=config.dropout
        )

        # ====================================================================
        # KEY/VALUE GEOMETRIC PROJECTIONS
        # ====================================================================
        self.k_proj = nn.Linear(self.slice_size, config.expert_dim)
        self.v_proj = nn.Linear(self.slice_size, config.expert_dim)

        # Output projection
        self.out_proj = nn.Linear(config.expert_dim, self.slice_size)
        self.dropout = nn.Dropout(config.dropout)

        # ====================================================================
        # ALPHA VISIBILITY (per-expert gating)
        # ====================================================================
        self.alpha = nn.Parameter(torch.tensor(config.alpha_init))
        self.alpha_gate = nn.Sequential(
            nn.Linear(config.expert_dim, config.expert_dim // 2),
            nn.LayerNorm(config.expert_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.expert_dim // 2, 1),
            nn.Sigmoid()
        )

        # ====================================================================
        # BETA BINDING (inter-expert connections)
        # ====================================================================
        self.betas = nn.ParameterDict()
        for neighbor_id in neighbor_ids:
            if neighbor_id != config.expert_id:
                self.betas[f"expert_{neighbor_id}"] = nn.Parameter(
                    torch.tensor(config.beta_init)
                )

    def get_pentachoron_vertices(self) -> torch.Tensor:
        """Get vertices for geometric loss."""
        return self.pentachoron_vertices

    def project_through_pentachoron(
        self,
        features: torch.Tensor,
        use_role_weights: bool = True
    ) -> torch.Tensor:
        """
        Project features through pentachoron with role weighting.

        Like David's Rose loss mechanism.

        Args:
            features: [B, P, expert_dim]
            use_role_weights: Apply semantic role weights

        Returns:
            projection: [B, P, 1] aggregated geometric projection
        """
        B, P, D = features.shape

        # Normalize
        vertices_norm = F.normalize(self.pentachoron_vertices, dim=-1)
        features_norm = F.normalize(features, dim=-1)

        # Similarity to each vertex [B, P, 5]
        similarities = torch.einsum('bpd,vd->bpv', features_norm, vertices_norm)

        if use_role_weights:
            # Apply role-based weighting
            role_weighted = similarities * self.role_weights.view(1, 1, 5)
            projection = role_weighted.sum(dim=-1, keepdim=True)  # [B, P, 1]
        else:
            # Simple average
            projection = similarities.mean(dim=-1, keepdim=True)

        return projection

    def forward(
        self,
        tokens: torch.Tensor,
        fingerprints: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Process tokens via pentachoron geometric opinion formation.

        CRITICAL: Only processes patches assigned to this expert.
        No wasted computation, no contamination.

        Args:
            tokens: [B, P, full_feature_dim] patch features
            fingerprints: [P] Cantor coordinates

        Returns:
            Dict with expert's geometric opinion
        """
        B, P, D = tokens.shape
        device = tokens.device

        # ====================================================================
        # EARLY SELECTION: Determine MY patches FIRST
        # ====================================================================
        if self.expert_id == self.num_experts - 1:
            # Last expert catches boundary
            mask = (fingerprints >= self.fp_min) & (fingerprints <= self.fp_max)
        else:
            mask = (fingerprints >= self.fp_min) & (fingerprints < self.fp_max)

        if not mask.any():
            # No patches assigned - return empty
            return {
                'output': torch.zeros(B, P, self.slice_size, device=device),
                'mask': mask,
                'geometric_scores': torch.zeros(B, P, device=device),
                'expert_id': self.expert_id,
                'betas': self.betas
            }

        # ====================================================================
        # SELECT ONLY MY PATCHES (prevent contamination)
        # ====================================================================
        my_patch_indices = mask.nonzero(as_tuple=False).squeeze(-1)  # [num_my_patches]
        num_my_patches = my_patch_indices.shape[0]

        # Extract only MY patches
        my_tokens = tokens[:, my_patch_indices, :]  # [B, num_my_patches, D]
        my_tokens_slice = my_tokens[:, :, self.slice_start:self.slice_end]  # [B, num_my_patches, slice_size]

        # ====================================================================
        # QUERY: K-SIMPLEX NAVIGATION (only over MY patches)
        # ====================================================================
        Q = self.q_proj(my_tokens_slice)  # [B, num_my_patches, expert_dim]

        # Refine query by attending to pentachoron vertices
        vertices_expanded = self.pentachoron_vertices.unsqueeze(0).expand(B, -1, -1)
        Q_refined, _ = self.q_vertex_attention(
            Q,  # query from MY patches
            vertices_expanded,  # key = pentachoron vertices
            vertices_expanded   # value = pentachoron vertices
        )
        Q = Q + Q_refined  # Residual

        # ====================================================================
        # KEY/VALUE: GEOMETRIC PROJECTIONS (only MY patches)
        # ====================================================================
        K = self.k_proj(my_tokens_slice)  # [B, num_my_patches, expert_dim]
        V = self.v_proj(my_tokens_slice)  # [B, num_my_patches, expert_dim]

        # Project through pentachoron geometry
        K_geometric = self.project_through_pentachoron(K, use_role_weights=True)
        V_geometric = self.project_through_pentachoron(V, use_role_weights=True)

        # ====================================================================
        # GEOMETRIC ATTENTION (ONLY over MY patches - no contamination)
        # ====================================================================
        # Attention scores modulated by geometric projections
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(K.shape[-1])

        # Modulate by K's geometric projection
        geometric_modulation = K_geometric.squeeze(-1)  # [B, num_my_patches]
        scores = scores * geometric_modulation.unsqueeze(1)

        # No masking needed - all patches are valid (we pre-selected)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output = torch.bmm(attn, V)  # [B, num_my_patches, expert_dim]

        # Further modulate by V's geometric projection
        output = output * V_geometric

        # ====================================================================
        # ALPHA VISIBILITY GATING
        # ====================================================================
        alpha_weight = torch.sigmoid(self.alpha)
        alpha_context = self.alpha_gate(output.mean(dim=1, keepdim=True))
        visibility = alpha_weight * alpha_context

        output = output * visibility

        # ====================================================================
        # PROJECT BACK TO FEATURE SLICE
        # ====================================================================
        output = self.out_proj(output)  # [B, num_my_patches, slice_size]
        output = self.dropout(output)

        # ====================================================================
        # SCATTER BACK to full patch dimension
        # ====================================================================
        output_full = torch.zeros(B, P, self.slice_size, device=device)

        # Use advanced indexing to scatter
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_my_patches)
        output_full[batch_indices, my_patch_indices.unsqueeze(0).expand(B, -1), :] = output

        # Prepare geometric scores (full dimension for consistency)
        geometric_scores_full = torch.zeros(B, P, device=device)
        geometric_scores_full[batch_indices, my_patch_indices.unsqueeze(0).expand(B, -1)] = geometric_modulation

        return {
            'output': output_full,  # Only MY patches are non-zero
            'mask': mask,
            'geometric_scores': geometric_scores_full,  # [B, P] with scores only for MY patches
            'expert_id': self.expert_id,
            'betas': self.betas
        }


# ============================================================================
# CANTOR MOE LAYER
# ============================================================================

@dataclass
class CantorMoEConfig:
    """Configuration for Cantor MoE layer."""
    num_experts: int = 16
    full_feature_dim: int = 1024
    expert_dim: int = 64
    num_heads: int = 8
    cantor_depth: int = 8
    overlap_factor: float = 0.5
    alpha_init: float = 0.5
    beta_init: float = 0.3
    dropout: float = 0.1


class CantorMoELayer(nn.Module):
    """
    Cantor Mixture-of-Experts with Geometric Pentachoron Structure.

    Multiple experts with overlapping regions form collective consensus.
    """

    def __init__(self, config: CantorMoEConfig):
        super().__init__()

        self.config = config
        self.num_experts = config.num_experts

        # Expert fingerprint ranges (with overlap)
        expert_ranges = self._compute_expert_ranges(
            config.num_experts,
            config.overlap_factor
        )

        # Create experts
        self.experts = nn.ModuleList()
        for expert_id in range(config.num_experts):
            fp_min, fp_max = expert_ranges[expert_id]

            # Neighbors (for beta binding)
            neighbor_ids = list(range(max(0, expert_id - 2), min(config.num_experts, expert_id + 3)))

            expert_config = CantorExpertConfig(
                expert_id=expert_id,
                num_experts=config.num_experts,
                full_feature_dim=config.full_feature_dim,
                expert_dim=config.expert_dim,
                num_heads=config.num_heads,
                alpha_init=config.alpha_init,
                beta_init=config.beta_init,
                dropout=config.dropout
            )

            expert = CantorExpert(expert_config, fp_min, fp_max, neighbor_ids)
            self.experts.append(expert)

    def _compute_expert_ranges(
        self,
        num_experts: int,
        overlap_factor: float
    ) -> List[Tuple[float, float]]:
        """Compute overlapping fingerprint ranges."""
        base_width = 1.0 / num_experts
        overlap_width = base_width * overlap_factor

        ranges = []
        for i in range(num_experts):
            fp_min = max(0.0, i * base_width - overlap_width / 2)
            fp_max = min(1.0, (i + 1) * base_width + overlap_width / 2)
            ranges.append((fp_min, fp_max))

        return ranges

    def get_all_pentachora(self) -> List[torch.Tensor]:
        """Get all expert pentachora for geometric loss."""
        return [expert.get_pentachoron_vertices() for expert in self.experts]

    def forward(
        self,
        tokens: torch.Tensor,
        fingerprints: torch.Tensor
    ) -> torch.Tensor:
        """
        Process tokens through collective expert voting.

        Args:
            tokens: [B, P, full_feature_dim]
            fingerprints: [P] Cantor coordinates

        Returns:
            output: [B, P, full_feature_dim] fused expert opinions
        """
        B, P, D = tokens.shape
        device = tokens.device

        # Collect expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(tokens, fingerprints)
            expert_outputs.append(expert_out)

        # ====================================================================
        # COLLECTIVE FUSION via Beta-Weighted Consensus
        # ====================================================================
        # Each expert contributes to patches in its region
        # Overlap means multiple experts vote on same patches

        fused_output = torch.zeros(B, P, D, device=device)
        vote_counts = torch.zeros(B, P, 1, device=device)

        for expert_out in expert_outputs:
            output = expert_out['output']
            mask = expert_out['mask']
            expert_id = expert_out['expert_id']

            # Reconstruct full feature vector (expert only outputs its slice)
            full_output = torch.zeros(B, P, D, device=device)

            # Determine slice position
            slice_size = D // self.config.num_heads
            head_idx = expert_id % self.config.num_heads
            start_idx = head_idx * slice_size
            end_idx = start_idx + slice_size

            full_output[:, :, start_idx:end_idx] = output

            # Weight by mask (only contribute to assigned patches)
            mask_weight = mask.float().unsqueeze(0).unsqueeze(-1)
            fused_output += full_output * mask_weight
            vote_counts += mask_weight

        # Average across voting experts
        vote_counts = vote_counts.clamp(min=1)
        fused_output = fused_output / vote_counts

        return fused_output


# ============================================================================
# VIT-BEANS MODEL
# ============================================================================

@dataclass
class ViTBeansConfig:
    """Configuration for ViT-Beans."""
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_layers: int = 8
    feature_dim: int = 1024
    num_experts: int = 16
    expert_dim: int = 64
    num_heads: int = 8
    mlp_ratio: float = 4.0
    cantor_depth: int = 8
    overlap_factor: float = 0.5
    alpha_init: float = 0.5
    beta_init: float = 0.3
    dropout: float = 0.1
    num_classes: int = 100


class ViTBeans(nn.Module):
    """
    ViT-Beans: Vision Transformer with Geometric Cantor Expert Collective.

    Combines:
    - Cantor fingerprinting for spatial routing
    - Pentachoron geometry for opinion formation
    - Collective consensus through redundant expert coverage
    """

    def __init__(self, config: ViTBeansConfig):
        super().__init__()

        self.config = config

        # Image dimensions
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.grid_size = config.image_size // config.patch_size

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

        # Cantor fingerprinter
        self.fingerprinter = GeometricCantorFingerprinter(depth=config.cantor_depth)

        # Cache fingerprints
        self.register_buffer('patch_fingerprints', torch.zeros(self.num_patches))
        self._fingerprints_computed = False

        # Transformer layers with Cantor MoE
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

        # Geometric loss
        self.geometric_loss = PentachoronGeometricLoss(
            volume_floor=0.1,
            edge_uniformity_weight=0.1,
            gram_weight=0.05
        )

    def _compute_fingerprints(self, device: torch.device):
        """Compute and cache Cantor fingerprints."""
        if not self._fingerprints_computed:
            fingerprints = self.fingerprinter.compute_fingerprints(
                self.num_patches,
                self.grid_size,
                device
            )
            self.patch_fingerprints.copy_(fingerprints)
            self._fingerprints_computed = True

    def get_geometric_loss(self) -> torch.Tensor:
        """Compute geometric loss across all expert pentachora."""
        all_pentachora = []
        for layer in self.layers:
            layer_pentachora = layer['cantor_moe'].get_all_pentachora()
            all_pentachora.extend(layer_pentachora)

        return self.geometric_loss(all_pentachora)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, C, H, W] images

        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]
        device = x.device

        # Compute fingerprints if needed
        self._compute_fingerprints(device)

        # Patch embedding
        x = self.patch_embed(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, P, D]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer layers
        for layer in self.layers:
            # Cantor MoE (skip CLS)
            x_patches = x[:, 1:]
            x_patches = layer['cantor_moe'](x_patches, self.patch_fingerprints)
            x = torch.cat([x[:, :1], x_patches], dim=1)

            # MLP
            x = x + layer['mlp'](x)

        # Classification
        x = self.norm(x[:, 0])
        return self.head(x)

    def diagnose_coverage(self) -> Dict:
        """Diagnose expert coverage."""
        stats = {
            'num_experts': self.config.num_experts,
            'num_patches': self.num_patches,
            'overlap_factor': self.config.overlap_factor,
            'fingerprints': self.patch_fingerprints.cpu().numpy().tolist()
        }

        first_moe = self.layers[0]['cantor_moe']

        # Per-expert stats
        for expert_id, expert in enumerate(first_moe.experts):
            fp_min, fp_max = expert.fp_min, expert.fp_max

            if expert_id == self.config.num_experts - 1:
                mask = (self.patch_fingerprints >= fp_min) & (self.patch_fingerprints <= fp_max)
            else:
                mask = (self.patch_fingerprints >= fp_min) & (self.patch_fingerprints < fp_max)

            stats[f'expert_{expert_id}'] = {
                'range': f'[{fp_min:.4f}, {fp_max:.4f}]',
                'patches': mask.sum().item(),
                'alpha': torch.sigmoid(expert.alpha).item(),
                'num_betas': len(expert.betas)
            }

        # Coverage analysis
        all_masks = []
        for expert_id, expert in enumerate(first_moe.experts):
            fp_min, fp_max = expert.fp_min, expert.fp_max

            if expert_id == self.config.num_experts - 1:
                mask = (self.patch_fingerprints >= fp_min) & (self.patch_fingerprints <= fp_max)
            else:
                mask = (self.patch_fingerprints >= fp_min) & (self.patch_fingerprints < fp_max)

            all_masks.append(mask)

        all_masks = torch.stack(all_masks)
        patches_covered = all_masks.any(dim=0).sum().item()
        experts_per_patch = all_masks.sum(dim=0)

        stats['total_covered'] = patches_covered
        stats['coverage_percent'] = 100.0 * patches_covered / self.num_patches
        stats['avg_experts_per_patch'] = experts_per_patch.float().mean().item()
        stats['min_experts_per_patch'] = experts_per_patch.min().item()
        stats['max_experts_per_patch'] = experts_per_patch.max().item()

        return stats


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ViT-Beans: Geometric Cantor Expert Collective")
    print("=" * 80)

    config = ViTBeansConfig(
        image_size=32,
        patch_size=4,
        num_layers=4,
        feature_dim=512,
        num_experts=16,
        expert_dim=64,
        overlap_factor=0.5,
        num_classes=100
    )

    model = ViTBeans(config)

    # Force fingerprint computation
    model._compute_fingerprints(torch.device('cpu'))

    # Test forward
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print(f"\nForward pass: {x.shape} → {logits.shape}")

    # Geometric loss
    geo_loss = model.get_geometric_loss()
    print(f"Geometric loss: {geo_loss.item():.4f}")

    # Coverage
    coverage = model.diagnose_coverage()
    print(f"\nCoverage: {coverage['total_covered']}/{coverage['num_patches']} ({coverage['coverage_percent']:.1f}%)")
    print(f"Avg experts/patch: {coverage['avg_experts_per_patch']:.2f}")
    print(f"Pentachoron geometry maintained via Cayley-Menger constraints")

    print("\n" + "=" * 80)
    print("🏴‍☠️⚓ Cantor routes WHERE, Pentachoron defines HOW")
    print("=" * 80)