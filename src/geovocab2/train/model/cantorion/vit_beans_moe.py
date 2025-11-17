"""
ViT-Beans: Vision Transformer with Cantor Expert Attention Network System
==========================================================================

A sparse, geometric vision transformer using:
- Multi-dimensional Cantor fingerprinting for token routing
- Sparse expert QKV (each expert processes feature slice)
- Pentachoron multi-directional projections (5-way cross-contamination)
- Alpha (visibility) and Beta (binding) learning from VAE-Lyra
- Cantor global attention with O(n) complexity
- Geometric routing to fill Cantor dead space

Architecture Philosophy:
- Small experts with large representational capacity
- Fingerprint-determined feature slice allocation
- Geometric coherence through fractal routing
- Cross-contamination via pentachoron projections

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
# CANTOR FINGERPRINTING
# ============================================================================

class MultiDimensionalCantorFingerprinter:
    """
    Generate multi-dimensional Cantor fingerprints for token routing.

    Different dimensions capture different behavioral scales:
    - 2D: Coarse regions (expert groups)
    - 3D: Medium regions (specific experts)
    - 4D: Fine regions (feature slices within expert)
    - 5D: Ultra-fine (neuron-level routing)
    """

    def __init__(self, dimensions: List[int] = [2, 3, 4, 5], depth: int = 8):
        self.dimensions = dimensions
        self.depth = depth

    def _cantor_pair(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Classic 2D Cantor pairing."""
        s = x + y
        return (s * (s + 1)) // 2 + y

    def _cantor_pair_nd(self, coords: torch.Tensor) -> torch.Tensor:
        """Recursive N-dimensional Cantor pairing."""
        result = self._cantor_pair(coords[..., 0], coords[..., 1])
        for i in range(2, coords.shape[-1]):
            result = self._cantor_pair(result, coords[..., i])
        return result

    def _normalize_fingerprint(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Normalize to [0, 1]."""
        fp_min = fingerprint.min()
        fp_max = fingerprint.max()
        return (fingerprint - fp_min) / (fp_max - fp_min + 1e-10)

    def compute_fingerprints(
        self,
        num_patches: int,
        device: torch.device = torch.device('cpu')
    ) -> Dict[int, torch.Tensor]:
        """
        Compute fingerprints at all dimensions.

        Returns:
            Dict[dimension, fingerprints] where fingerprints: [num_patches]
        """
        fingerprints = {}

        for dim in self.dimensions:
            # Create coordinate grid for this dimension
            # Use patch positions as base coordinates
            coords = torch.zeros(num_patches, dim, dtype=torch.long, device=device)

            # Fill coordinates based on patch index
            for i in range(num_patches):
                for d in range(dim):
                    coords[i, d] = (i * (d + 1)) % (num_patches)

            # Compute Cantor pairing
            raw_fp = self._cantor_pair_nd(coords).float()

            # Normalize to [0, 1]
            fp_normalized = self._normalize_fingerprint(raw_fp)

            fingerprints[dim] = fp_normalized

        return fingerprints


# ============================================================================
# CANTOR EXPERT (SPARSE QKV + PENTACHORON PROJECTION)
# ============================================================================

@dataclass
class CantorExpertConfig:
    """Configuration for a single Cantor expert."""
    expert_id: int
    num_experts: int
    full_feature_dim: int
    expert_dim: int
    num_heads: int
    dropout: float = 0.1
    alpha_init: float = 1.0
    beta_init: float = 0.3
    alpha_lr_scale: float = 0.1
    beta_lr_scale: float = 1.0


class CantorExpert(nn.Module):
    """
    Single Cantor expert with:
    - Sparse QKV (only on feature slice)
    - Pentachoron multi-directional projection (5 directions)
    - Alpha visibility gating
    - Beta binding weights to neighbors
    """

    def __init__(self, config: CantorExpertConfig):
        super().__init__()

        self.expert_id = config.expert_id
        self.num_experts = config.num_experts
        self.full_feature_dim = config.full_feature_dim
        self.expert_dim = config.expert_dim
        self.num_heads = config.num_heads
        self.head_dim = config.expert_dim // config.num_heads

        # Fingerprint range allocation
        self.fp_min = config.expert_id / config.num_experts
        self.fp_max = (config.expert_id + 1) / config.num_experts

        # Feature slice allocation (SPARSE!)
        slice_size = config.full_feature_dim // config.num_experts
        self.slice_start = config.expert_id * slice_size
        self.slice_end = self.slice_start + slice_size
        self.slice_size = slice_size

        # Alpha: Learned visibility (VAE-Lyra mechanism)
        self.alpha = nn.Parameter(torch.tensor(config.alpha_init))

        # Alpha gating network
        self.alpha_gate = nn.Sequential(
            nn.Linear(slice_size, slice_size // 4),
            nn.GELU(),
            nn.Linear(slice_size // 4, 1),
            nn.Sigmoid()
        )

        # Sparse QKV projections (ONLY on feature slice)
        self.q_proj = nn.Linear(slice_size, config.expert_dim, bias=False)
        self.k_proj = nn.Linear(slice_size, config.expert_dim, bias=False)
        self.v_proj = nn.Linear(slice_size, config.expert_dim, bias=False)

        # Pentachoron: 5 vertices = 5 projection directions
        # This enables cross-contamination to fill Cantor dead space
        self.pentachoron = nn.Parameter(
            torch.randn(5, config.expert_dim) * 0.02
        )

        # Beta: Learned binding weights to neighbor experts
        # Local binding (only to nearby experts in Cantor space)
        self.betas = nn.ParameterDict()
        for i in range(config.num_experts):
            if abs(i - config.expert_id) <= 2 and i != config.expert_id:
                self.betas[f"expert_{i}"] = nn.Parameter(
                    torch.tensor(config.beta_init)
                )

        # Output projection
        self.out_proj = nn.Linear(config.expert_dim, slice_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)

        # Normalize pentachoron vertices
        with torch.no_grad():
            self.pentachoron.data = F.normalize(self.pentachoron.data, dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,           # [batch, num_patches, full_feature_dim]
        fingerprints: torch.Tensor      # [num_patches] - Cantor coordinates
    ) -> Dict[str, torch.Tensor]:
        """
        Process tokens in my fingerprint region.

        Returns:
            Dict containing:
                - projections: List of 5 directional projections
                - mask: Boolean mask of my tokens
                - K, Q, V: Full expert-space representations
                - betas: Binding weights
        """
        batch_size, num_patches, _ = tokens.shape
        device = tokens.device

        # 1. Select tokens in MY fingerprint region
        mask = (fingerprints >= self.fp_min) & (fingerprints < self.fp_max)

        # If no tokens in my region, return empty
        if not mask.any():
            return {
                'projections': [],
                'mask': mask,
                'K': None,
                'Q': None,
                'V': None,
                'betas': self.betas,
                'attended_output': torch.zeros(
                    batch_size, num_patches, self.slice_size,
                    device=device, dtype=tokens.dtype
                )
            }

        # 2. Extract MY feature slice (SPARSE!)
        my_tokens = tokens[:, mask]  # [batch, my_patches, full_dim]
        my_features = my_tokens[..., self.slice_start:self.slice_end]
        # [batch, my_patches, slice_size]

        # 3. Alpha-gated visibility (VAE-Lyra mechanism)
        alpha_gate = self.alpha_gate(my_features)  # [batch, my_patches, 1]
        alpha_weight = torch.sigmoid(self.alpha)
        my_features = my_features * (alpha_gate * alpha_weight + (1 - alpha_weight))

        # 4. Sparse QKV computation (ONLY on my slice!)
        Q = self.q_proj(my_features)  # [batch, my_patches, expert_dim]
        K = self.k_proj(my_features)
        V = self.v_proj(my_features)

        # 5. Pentachoron multi-directional projection
        # Each vertex projects into different latent region
        # This creates 5-way cross-contamination to fill dead space
        projections = []

        for vertex_id, vertex in enumerate(self.pentachoron):
            direction = F.normalize(vertex, dim=-1)  # [expert_dim]

            # Project Q, K, V along this direction
            # Einsum for batched projection: [batch, patches, dim] @ [dim] -> [batch, patches]
            K_proj = torch.einsum('bpd,d->bp', K, direction)
            Q_proj = torch.einsum('bpd,d->bp', Q, direction)
            V_proj = torch.einsum('bpd,d->bp', V, direction)

            projections.append({
                'K': K_proj,      # [batch, my_patches]
                'Q': Q_proj,      # [batch, my_patches]
                'V': V_proj,      # [batch, my_patches]
                'direction': vertex_id,
                'vertex': direction
            })

        return {
            'projections': projections,
            'mask': mask,
            'K': K,  # [batch, my_patches, expert_dim]
            'Q': Q,
            'V': V,
            'betas': self.betas,
            'attended_output': None  # Filled by attention layer
        }


# ============================================================================
# CANTOR GLOBAL ATTENTION
# ============================================================================

@dataclass
class CantorAttentionConfig:
    """Configuration for Cantor global attention."""
    num_experts: int = 16
    expert_dim: int = 128
    num_heads: int = 8
    cantor_depth: int = 8
    local_window: int = 3
    temperature: float = 0.07
    dropout: float = 0.1


# ============================================================================
# OPTIMIZED CANTOR GLOBAL ATTENTION (BATCHED)
# ============================================================================

# ============================================================================
# FIXED OPTIMIZED CANTOR GLOBAL ATTENTION
# ============================================================================

class CantorGlobalAttention(nn.Module):
    """
    OPTIMIZED: Batched sparse attention with correct indexing.
    """

    def __init__(self, config: CantorAttentionConfig):
        super().__init__()

        self.num_experts = config.num_experts
        self.expert_dim = config.expert_dim
        self.num_heads = config.num_heads
        self.head_dim = config.expert_dim // config.num_heads
        self.local_window = min(config.local_window, config.num_experts)
        self.cantor_depth = config.cantor_depth

        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer('expert_coords', self._compute_expert_cantor_coordinates())
        self.register_buffer('expert_routes', self._build_expert_routes())

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        x = position / max(1, max_len - 1)
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x *= 3.0
            digit = int(x)
            x -= digit
            if digit == 2:
                cantor_val += factor
            factor *= 0.5

        return cantor_val

    def _compute_expert_cantor_coordinates(self) -> torch.Tensor:
        coords = torch.tensor([
            self._cantor_coordinate(i, self.num_experts, self.cantor_depth)
            for i in range(self.num_experts)
        ], dtype=torch.float32)
        return coords

    def _build_expert_routes(self) -> torch.Tensor:
        routes = torch.zeros(self.num_experts, self.local_window, dtype=torch.long)
        for i in range(self.num_experts):
            distances = torch.abs(self.expert_coords - self.expert_coords[i])
            _, nearest = torch.topk(distances, self.local_window, largest=False)
            routes[i] = nearest
        return routes

    def forward(
            self,
            expert_outputs: List[Dict[str, torch.Tensor]],
            num_patches: int
    ) -> torch.Tensor:
        """
        OPTIMIZED & FIXED: Batched fusion with correct tensor dimensions.
        """
        device = self.expert_coords.device

        # Get batch info
        batch_size = None
        dtype = torch.float32
        for output in expert_outputs:
            if output['K'] is not None:
                batch_size = output['K'].shape[0]
                dtype = output['K'].dtype
                break

        if batch_size is None:
            return torch.zeros(1, num_patches, device=device, dtype=dtype)

        # Collect all valid projections
        valid_experts = []
        expert_Qs = []
        expert_Ks = []
        expert_Vs = []
        expert_masks = []
        expert_betas_list = []

        for expert_id, output in enumerate(expert_outputs):
            if output['K'] is not None and len(output['projections']) > 0:
                valid_experts.append(expert_id)

                # Average across 5 directions
                proj_Qs = torch.stack([p['Q'] for p in output['projections']], dim=0)
                proj_Ks = torch.stack([p['K'] for p in output['projections']], dim=0)
                proj_Vs = torch.stack([p['V'] for p in output['projections']], dim=0)

                expert_Qs.append(proj_Qs.mean(dim=0))  # [batch, patches]
                expert_Ks.append(proj_Ks.mean(dim=0))
                expert_Vs.append(proj_Vs.mean(dim=0))
                expert_masks.append(output['mask'])
                expert_betas_list.append(output['betas'])

        if len(valid_experts) == 0:
            return torch.zeros(batch_size, num_patches, device=device, dtype=dtype)

        # Pad all to same length
        max_expert_patches = max(q.shape[1] for q in expert_Qs)

        expert_Qs_padded = []
        expert_Ks_padded = []
        expert_Vs_padded = []

        for Q, K, V in zip(expert_Qs, expert_Ks, expert_Vs):
            if Q.shape[1] < max_expert_patches:
                pad_size = max_expert_patches - Q.shape[1]
                Q = F.pad(Q, (0, pad_size))
                K = F.pad(K, (0, pad_size))
                V = F.pad(V, (0, pad_size))
            expert_Qs_padded.append(Q)
            expert_Ks_padded.append(K)
            expert_Vs_padded.append(V)

        # Stack: [num_valid_experts, batch, max_patches]
        Q_all = torch.stack(expert_Qs_padded, dim=0)
        K_all = torch.stack(expert_Ks_padded, dim=0)
        V_all = torch.stack(expert_Vs_padded, dim=0)

        # Build routing
        routes = self.expert_routes[valid_experts]
        expert_id_to_idx = {eid: idx for idx, eid in enumerate(valid_experts)}

        # Gather neighbors - SIMPLIFIED for speed
        # Instead of complex neighbor gathering, use mean pooling
        output = torch.zeros(batch_size, num_patches, device=device, dtype=dtype)

        for i, (expert_id, mask) in enumerate(zip(valid_experts, expert_masks)):
            # Get this expert's Q, V
            Q_i = Q_all[i]  # [batch, max_patches]
            V_i = V_all[i]  # [batch, max_patches]

            # Get neighbor K, V (simplified - just use global mean)
            neighbor_ids = routes[i]

            # Collect neighbor keys/values
            K_neighbors = []
            V_neighbors = []
            for neighbor_id in neighbor_ids:
                neighbor_id = neighbor_id.item()
                if neighbor_id in expert_id_to_idx:
                    idx = expert_id_to_idx[neighbor_id]
                    K_neighbors.append(K_all[idx])
                    V_neighbors.append(V_all[idx])

            if len(K_neighbors) == 0:
                # No valid neighbors, use self
                valid_patches = mask.sum().item()
                output[:, mask] = V_i[:, :valid_patches]
                continue

            # Stack: [num_neighbors, batch, max_patches]
            K_stack = torch.stack(K_neighbors, dim=0)
            V_stack = torch.stack(V_neighbors, dim=0)

            # Compute attention: [batch, max_patches] x [num_neighbors, batch, max_patches]
            # -> [batch, max_patches, num_neighbors]
            scores = torch.einsum('bp,nbp->bpn', Q_i, K_stack)
            scores = scores / math.sqrt(self.expert_dim)
            scores = scores / self.temperature.abs()

            # Apply beta modulation
            betas = expert_betas_list[i]
            for j, neighbor_id in enumerate(neighbor_ids):
                neighbor_id = neighbor_id.item()
                if neighbor_id != expert_id and neighbor_id in expert_id_to_idx:
                    beta_key = f"expert_{neighbor_id}"
                    if beta_key in betas:
                        beta = torch.sigmoid(betas[beta_key])
                        scores[:, :, j] = scores[:, :, j] * beta

            # Softmax
            attn = F.softmax(scores, dim=-1)  # [batch, max_patches, num_neighbors]
            attn = self.dropout(attn)

            # Apply attention: [batch, max_patches, num_neighbors] x [num_neighbors, batch, max_patches]
            # -> [batch, max_patches]
            attended = torch.einsum('bpn,nbp->bp', attn, V_stack)

            # Place in output
            valid_patches = mask.sum().item()
            output[:, mask] = attended[:, :valid_patches]

        return output

# ============================================================================
# CANTOR MoE LAYER
# ============================================================================

@dataclass
class CantorMoEConfig:
    """Configuration for Cantor MoE layer."""
    num_experts: int = 16
    full_feature_dim: int = 1024
    expert_dim: int = 128
    num_heads: int = 8
    cantor_depth: int = 8
    local_window: int = 3
    dropout: float = 0.1
    alpha_init: float = 1.0
    beta_init: float = 0.3


class CantorMoELayer(nn.Module):
    """
    Complete Cantor MoE layer combining:
    - Multiple sparse experts
    - Cantor global attention
    - Geometric routing and fusion
    """

    def __init__(self, config: CantorMoEConfig):
        super().__init__()

        self.num_experts = config.num_experts
        self.full_feature_dim = config.full_feature_dim

        # Create experts
        self.experts = nn.ModuleList([
            CantorExpert(CantorExpertConfig(
                expert_id=i,
                num_experts=config.num_experts,
                full_feature_dim=config.full_feature_dim,
                expert_dim=config.expert_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                alpha_init=config.alpha_init,
                beta_init=config.beta_init
            ))
            for i in range(config.num_experts)
        ])

        # Cantor attention
        self.attention = CantorGlobalAttention(CantorAttentionConfig(
            num_experts=config.num_experts,
            expert_dim=config.expert_dim,
            num_heads=config.num_heads,
            cantor_depth=config.cantor_depth,
            local_window=config.local_window,
            dropout=config.dropout
        ))

        # Layer norm
        self.norm = nn.LayerNorm(config.full_feature_dim)

    def forward(
        self,
        x: torch.Tensor,                # [batch, num_patches, full_feature_dim]
        fingerprints: torch.Tensor      # [num_patches]
    ) -> torch.Tensor:
        """
        Forward pass through Cantor MoE.

        Returns:
            output: [batch, num_patches, full_feature_dim]
        """
        batch_size, num_patches, _ = x.shape

        # Normalize input
        x_norm = self.norm(x)

        # Process through all experts (sparse, each handles its region)
        expert_outputs = []
        for expert in self.experts:
            output = expert(x_norm, fingerprints)
            expert_outputs.append(output)

        # Fuse via Cantor global attention
        fused_1d = self.attention(expert_outputs, num_patches)
        # [batch, num_patches] - single attended value per patch

        # Reconstruct full feature space from expert outputs
        reconstructed = torch.zeros_like(x)

        for expert_id, output in enumerate(expert_outputs):
            if output['attended_output'] is None and output['K'] is not None:
                # Project attended value back through expert
                mask = output['mask']
                expert = self.experts[expert_id]

                # Get attended values for this expert's patches
                attended_vals = fused_1d[:, mask]  # [batch, my_patches]

                # Expand to expert_dim for projection
                attended_expanded = attended_vals.unsqueeze(-1).expand(
                    -1, -1, expert.expert_dim
                )

                # Project back to feature slice
                output_slice = expert.out_proj(attended_expanded)
                # [batch, my_patches, slice_size]

                # Place in full feature tensor
                reconstructed[:, mask, expert.slice_start:expert.slice_end] = output_slice

        # Residual connection
        return x + reconstructed


# ============================================================================
# ViT-BEANS COMPLETE ARCHITECTURE
# ============================================================================

@dataclass
class ViTBeansConfig:
    """Complete ViT-Beans configuration."""
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

    # Cantor
    cantor_dimensions: List[int] = None
    cantor_depth: int = 8
    local_window: int = 3

    # Learning
    alpha_init: float = 1.0
    beta_init: float = 0.3
    dropout: float = 0.1

    # Classification
    num_classes: int = 1000

    def __post_init__(self):
        if self.cantor_dimensions is None:
            self.cantor_dimensions = [2, 3, 4, 5]

        assert self.feature_dim % self.num_experts == 0, \
            "feature_dim must be divisible by num_experts"


class ViTBeans(nn.Module):
    """
    ViT-Beans: Vision Transformer with Cantor Expert Attention Network System

    A sparse, geometric ViT using:
    - Multi-dimensional Cantor fingerprinting
    - Sparse expert QKV with feature slice allocation
    - Pentachoron multi-directional projections
    - Alpha/Beta learning from VAE-Lyra
    - O(n) Cantor global attention
    """

    def __init__(self, config: ViTBeansConfig):
        super().__init__()

        self.config = config

        # Calculate number of patches
        assert config.image_size % config.patch_size == 0
        self.num_patches = (config.image_size // config.patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.feature_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Position embedding (traditional, for baseline)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, config.feature_dim) * 0.02
        )

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.feature_dim) * 0.02
        )

        # Cantor fingerprinter
        self.fingerprinter = MultiDimensionalCantorFingerprinter(
            dimensions=config.cantor_dimensions,
            depth=config.cantor_depth
        )

        # Pre-compute fingerprints (cached)
        self.register_buffer(
            'patch_fingerprints',
            torch.zeros(self.num_patches)  # Will be computed on first forward
        )
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
                    local_window=config.local_window,
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
        """Initialize weights."""
        # Patch embedding
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        # Classification head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _compute_fingerprints(self, device: torch.device):
        """Compute and cache Cantor fingerprints."""
        if not self._fingerprints_computed:
            fingerprints_dict = self.fingerprinter.compute_fingerprints(
                self.num_patches, device
            )
            # Use primary dimension (3D) for routing
            self.patch_fingerprints = fingerprints_dict[3]
            self._fingerprints_computed = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features through ViT-Beans layers.

        Args:
            x: [batch, channels, height, width]

        Returns:
            features: [batch, num_patches+1, feature_dim]
        """
        batch_size = x.shape[0]
        device = x.device

        # Compute fingerprints if needed
        self._compute_fingerprints(device)

        # Patch embedding
        x = self.patch_embed(x)  # [batch, feature_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, feature_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Process through layers
        for layer in self.layers:
            # Cantor MoE attention (skip CLS token)
            x_patches = x[:, 1:]
            x_patches = layer['cantor_moe'](x_patches, self.patch_fingerprints)
            x = torch.cat([x[:, :1], x_patches], dim=1)

            # MLP
            x = x + layer['mlp'](x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            x: [batch, channels, height, width]

        Returns:
            logits: [batch, num_classes]
        """
        x = self.forward_features(x)

        # Classification from CLS token
        x = self.norm(x[:, 0])
        logits = self.head(x)

        return logits

    def get_expert_stats(self) -> Dict:
        """Get statistics about expert utilization."""
        stats = {
            'num_experts': self.config.num_experts,
            'expert_params': [],
            'alpha_values': [],
            'beta_values': []
        }

        for layer_idx, layer in enumerate(self.layers):
            moe = layer['cantor_moe']

            for expert_idx, expert in enumerate(moe.experts):
                # Count parameters
                params = sum(p.numel() for p in expert.parameters())
                stats['expert_params'].append(params)

                # Get alpha
                alpha = torch.sigmoid(expert.alpha).item()
                stats['alpha_values'].append(alpha)

                # Get betas
                for key, beta in expert.betas.items():
                    beta_val = torch.sigmoid(beta).item()
                    stats['beta_values'].append(beta_val)

        stats['total_expert_params'] = sum(stats['expert_params'])
        stats['mean_alpha'] = sum(stats['alpha_values']) / len(stats['alpha_values'])
        stats['mean_beta'] = sum(stats['beta_values']) / len(stats['beta_values']) if stats['beta_values'] else 0.0

        return stats


# ============================================================================
# DEMO & TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ViT-Beans: Cantor Expert Attention Network System")
    print("=" * 80)

    # Configuration
    config = ViTBeansConfig(
        image_size=224,
        patch_size=16,
        num_layers=12,
        feature_dim=1024,
        num_experts=16,
        expert_dim=128,
        num_heads=8,
        num_classes=1000
    )

    print(f"\nConfiguration:")
    print(f"  Image size: {config.image_size}x{config.image_size}")
    print(f"  Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  Patches: {(config.image_size // config.patch_size) ** 2}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Feature dim: {config.feature_dim}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Expert dim: {config.expert_dim}")
    print(f"  Classes: {config.num_classes}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = ViTBeans(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Get expert stats
    expert_stats = model.get_expert_stats()
    print(f"\nExpert Statistics:")
    print(f"  Total expert params: {expert_stats['total_expert_params']:,}")
    print(f"  Params per expert: {expert_stats['expert_params'][0]:,}")
    print(f"  Mean alpha: {expert_stats['mean_alpha']:.4f}")
    print(f"  Mean beta: {expert_stats['mean_beta']:.4f}")

    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  ✓ Forward pass successful")

    # Compute feature slicing efficiency
    slice_size = config.feature_dim // config.num_experts
    dense_qkv_params = config.feature_dim * config.feature_dim * 3
    sparse_qkv_params = slice_size * config.expert_dim * 3 * config.num_experts

    print(f"\nParameter Efficiency:")
    print(f"  Dense QKV params (baseline): {dense_qkv_params:,}")
    print(f"  Sparse QKV params (Beans): {sparse_qkv_params:,}")
    print(f"  Reduction: {(1 - sparse_qkv_params/dense_qkv_params) * 100:.1f}%")

    print("\n" + "=" * 80)
    print("Key Innovations:")
    print("  ✓ Multi-dimensional Cantor fingerprinting")
    print("  ✓ Sparse expert QKV with feature slice allocation")
    print("  ✓ Pentachoron 5-way projection for cross-contamination")
    print("  ✓ Alpha visibility and Beta binding learning")
    print("  ✓ O(n) Cantor global attention")
    print("  ✓ Geometric routing fills Cantor dead space")
    print("=" * 80)