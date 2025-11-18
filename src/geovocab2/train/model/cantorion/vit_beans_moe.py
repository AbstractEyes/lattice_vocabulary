"""
ViT-Beans: Vision Transformer with Cantor Expert Attention Network System (FIXED)
==================================================================================

Critical fixes applied:
1. Fixed fp_max boundary bug (patches with fingerprint=1.0 were dropped)
2. Proper attention masking (no zombie padding tokens)
3. Adjusted temperature scaling
4. Learnable direction fusion weights
5. Better initialization
6. Expert utilization tracking

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
# CANTOR FINGERPRINTING (unchanged)
# ============================================================================

class MultiDimensionalCantorFingerprinter:
    """Generate multi-dimensional Cantor fingerprints for token routing."""

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
        """Compute fingerprints at all dimensions."""
        fingerprints = {}

        for dim in self.dimensions:
            coords = torch.zeros(num_patches, dim, dtype=torch.long, device=device)

            for i in range(num_patches):
                for d in range(dim):
                    coords[i, d] = (i * (d + 1)) % (num_patches)

            raw_fp = self._cantor_pair_nd(coords).float()
            fp_normalized = self._normalize_fingerprint(raw_fp)

            fingerprints[dim] = fp_normalized

        return fingerprints


# ============================================================================
# CANTOR EXPERT (FIXED)
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

    FIXES:
    - Proper boundary handling for last expert (<=)
    - Better weight initialization
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
        # FIX: Track if we're the last expert
        self.is_last_expert = (config.expert_id == config.num_experts - 1)

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
        self.pentachoron = nn.Parameter(
            torch.randn(5, config.expert_dim) * 0.02
        )

        # Beta: Learned binding weights to neighbor experts
        self.betas = nn.ParameterDict()
        for i in range(config.num_experts):
            if abs(i - config.expert_id) <= 2 and i != config.expert_id:
                self.betas[f"expert_{i}"] = nn.Parameter(
                    torch.tensor(config.beta_init)
                )

        self.dropout = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights - FIXED: Better gain."""
        for module in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)  # Changed from 0.5

        # Normalize pentachoron vertices
        with torch.no_grad():
            self.pentachoron.data = F.normalize(self.pentachoron.data, dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        fingerprints: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Process tokens in my fingerprint region.

        FIX: Last expert now includes fingerprint=1.0
        """
        batch_size, num_patches, _ = tokens.shape
        device = tokens.device

        # FIX: Proper boundary handling for last expert
        if self.is_last_expert:
            mask = (fingerprints >= self.fp_min) & (fingerprints <= self.fp_max)
        else:
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
                'attended_output': None,
                'num_patches_processed': 0  # NEW: Track utilization
            }

        # Extract MY feature slice (SPARSE!)
        my_tokens = tokens[:, mask]
        my_features = my_tokens[..., self.slice_start:self.slice_end]

        # Alpha-gated visibility
        alpha_gate = self.alpha_gate(my_features)
        alpha_weight = torch.sigmoid(self.alpha)
        my_features = my_features * (alpha_gate * alpha_weight + (1 - alpha_weight))

        # Sparse QKV computation
        Q = self.q_proj(my_features)
        K = self.k_proj(my_features)
        V = self.v_proj(my_features)

        # Pentachoron multi-directional projection
        projections = []

        for vertex_id, vertex in enumerate(self.pentachoron):
            direction = F.normalize(vertex, dim=-1)

            K_affinity = torch.einsum('bpd,d->bp', K, direction)
            Q_affinity = torch.einsum('bpd,d->bp', Q, direction)

            projections.append({
                'K_affinity': K_affinity,
                'Q_affinity': Q_affinity,
                'V': V,
                'direction': vertex_id,
            })

        return {
            'projections': projections,
            'mask': mask,
            'K': K,
            'Q': Q,
            'V': V,
            'betas': self.betas,
            'attended_output': None,
            'num_patches_processed': mask.sum().item()  # NEW: Track utilization
        }


# ============================================================================
# CANTOR GLOBAL ATTENTION (FIXED)
# ============================================================================

@dataclass
class CantorAttentionConfig:
    """Configuration for Cantor global attention."""
    num_experts: int = 16
    expert_dim: int = 128
    num_heads: int = 8
    cantor_depth: int = 8
    local_window: int = 3
    temperature: float = 0.5  # FIX: Increased from 0.07
    dropout: float = 0.1


class CantorGlobalAttention(nn.Module):
    """
    Sparse O(n) attention using Cantor fingerprint routing.

    FIXES:
    - Proper attention masking (no padding artifacts)
    - Learnable direction fusion
    - Better temperature scaling
    """

    def __init__(self, config: CantorAttentionConfig):
        super().__init__()

        self.num_experts = config.num_experts
        self.expert_dim = config.expert_dim
        self.num_heads = config.num_heads
        self.head_dim = config.expert_dim // config.num_heads
        self.local_window = min(config.local_window, config.num_experts)
        self.cantor_depth = config.cantor_depth

        # FIX: Better temperature initialization
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        self.dropout = nn.Dropout(config.dropout)

        # NEW: Learnable direction fusion weights
        self.direction_fusion_weights = nn.Parameter(torch.ones(5) / 5.0)

        # Pre-compute Cantor coordinates for experts
        self.register_buffer(
            'expert_coords',
            self._compute_expert_cantor_coordinates()
        )

        # Pre-compute routing table
        self.register_buffer(
            'expert_routes',
            self._build_expert_routes()
        )

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor set coordinate for expert."""
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
        """Map each expert to Cantor coordinate."""
        coords = torch.tensor([
            self._cantor_coordinate(i, self.num_experts, self.cantor_depth)
            for i in range(self.num_experts)
        ], dtype=torch.float32)
        return coords

    def _build_expert_routes(self) -> torch.Tensor:
        """Build routing table: which experts attend to which."""
        routes = torch.zeros(self.num_experts, self.local_window, dtype=torch.long)

        for i in range(self.num_experts):
            distances = torch.abs(
                self.expert_coords - self.expert_coords[i]
            )
            _, nearest = torch.topk(distances, self.local_window, largest=False)
            routes[i] = nearest

        return routes

    def forward(
        self,
        expert_outputs: List[Dict[str, torch.Tensor]],
        num_patches: int
    ) -> torch.Tensor:
        """
        Fuse expert outputs through sparse geometric attention.

        FIXES:
        - Proper padding masking
        - Learnable direction fusion
        """
        device = self.expert_coords.device

        # Get batch size
        batch_size = None
        for output in expert_outputs:
            if output['K'] is not None:
                batch_size = output['K'].shape[0]
                break

        if batch_size is None:
            raise ValueError("No valid expert outputs")

        # Collect projections across all 5 directions
        all_projections = {dir_id: [] for dir_id in range(5)}

        for expert_id, output in enumerate(expert_outputs):
            if output['projections']:
                for proj in output['projections']:
                    dir_id = proj['direction']
                    all_projections[dir_id].append({
                        'expert_id': expert_id,
                        'K_affinity': proj['K_affinity'],
                        'Q_affinity': proj['Q_affinity'],
                        'V': proj['V'],
                        'mask': output['mask']
                    })

        # Process each direction independently
        direction_outputs = []

        for dir_id in range(5):
            dir_projs = all_projections[dir_id]
            if not dir_projs:
                continue

            attended_patches = torch.zeros(
                batch_size, num_patches, self.expert_dim,
                device=device, dtype=torch.float32
            )

            for expert_id, output in enumerate(expert_outputs):
                if not output['projections']:
                    continue

                neighbors = self.expert_routes[expert_id]

                my_proj = None
                for p in dir_projs:
                    if p['expert_id'] == expert_id:
                        my_proj = p
                        break

                if my_proj is None:
                    continue

                Q_affinity = my_proj['Q_affinity']
                mask_i = my_proj['mask']

                neighbor_projs = [p for p in dir_projs if p['expert_id'] in neighbors]
                if not neighbor_projs:
                    continue

                # Gather neighbor data
                K_affinities = []
                V_values = []
                neighbor_indices = []
                neighbor_actual_patches = []  # NEW: Track actual lengths

                for n_proj in neighbor_projs:
                    K_affinities.append(n_proj['K_affinity'])
                    V_values.append(n_proj['V'])
                    neighbor_indices.append(n_proj['expert_id'])
                    neighbor_actual_patches.append(n_proj['K_affinity'].shape[1])

                max_neighbor_patches = max(neighbor_actual_patches)

                # Pad to same length
                K_affinities_padded = []
                V_values_padded = []
                for k_aff, v_val in zip(K_affinities, V_values):
                    if k_aff.shape[1] < max_neighbor_patches:
                        pad_size = max_neighbor_patches - k_aff.shape[1]
                        k_aff = F.pad(k_aff, (0, pad_size))
                        v_val = F.pad(v_val, (0, 0, 0, pad_size))
                    K_affinities_padded.append(k_aff)
                    V_values_padded.append(v_val)

                K_affinity_stack = torch.stack(K_affinities_padded, dim=1)
                V_stack = torch.stack(V_values_padded, dim=1)

                # Compute attention scores
                scores = Q_affinity.unsqueeze(-1).unsqueeze(-1) * K_affinity_stack.unsqueeze(1)
                B, my_patches, num_neighbors, neighbor_patches = scores.shape
                scores = scores.reshape(B, my_patches, -1)

                # FIX: Only scale by temperature (remove sqrt scaling for scalar affinities)
                scores = scores / (self.temperature.abs() + 1e-6)

                # FIX: Create attention mask for padding
                attention_mask = torch.zeros_like(scores, dtype=torch.bool)
                patch_offset = 0
                for neighbor_idx, actual_patches in enumerate(neighbor_actual_patches):
                    # Mark padding positions as invalid
                    if actual_patches < max_neighbor_patches:
                        start_invalid = patch_offset + actual_patches
                        end_invalid = patch_offset + max_neighbor_patches
                        attention_mask[:, :, start_invalid:end_invalid] = True
                    patch_offset += max_neighbor_patches

                # Apply beta modulation
                betas = output['betas']
                patch_offset = 0
                for neighbor_idx, neighbor_id in enumerate(neighbor_indices):
                    actual_patches = neighbor_actual_patches[neighbor_idx]

                    if neighbor_id != expert_id:
                        beta_key = f"expert_{neighbor_id}"
                        if beta_key in betas:
                            beta = torch.sigmoid(betas[beta_key])
                            scores[:, :, patch_offset:patch_offset + actual_patches] *= beta

                    patch_offset += max_neighbor_patches

                # FIX: Mask padding before softmax
                scores = scores.masked_fill(attention_mask, float('-inf'))

                # Softmax attention
                attn = F.softmax(scores, dim=-1)
                attn = self.dropout(attn)

                # Apply to V
                V_flat = V_stack.reshape(B, -1, self.expert_dim)
                attended_V = torch.bmm(attn, V_flat)

                # Place in global tensor
                attended_patches[:, mask_i, :] = attended_V

            direction_outputs.append(attended_patches)

        # FIX: Learnable fusion instead of simple mean
        if direction_outputs:
            fusion_weights = F.softmax(self.direction_fusion_weights, dim=0)
            direction_stack = torch.stack(direction_outputs)  # [5, B, N, D]
            fused = torch.einsum('d,dbnd->bnd', fusion_weights, direction_stack)
        else:
            fused = torch.zeros(batch_size, num_patches, self.expert_dim, device=device)

        return fused


# ============================================================================
# CANTOR MoE LAYER (FIXED)
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
    temperature: float = 0.5  # FIX: Better default


class CantorMoELayer(nn.Module):
    """
    Complete Cantor MoE layer with all fixes applied.
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
            dropout=config.dropout,
            temperature=config.temperature
        ))

        # Dense projection
        self.fusion_proj = nn.Linear(config.expert_dim, config.full_feature_dim)

        # Layer norm
        self.norm = nn.LayerNorm(config.full_feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        fingerprints: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Forward pass through Cantor MoE.

        NEW: Returns expert utilization stats
        """
        batch_size, num_patches, _ = x.shape

        # Normalize input
        x_norm = self.norm(x)

        # Process through all experts
        expert_outputs = []
        expert_utilization = {}

        for expert in self.experts:
            output = expert(x_norm, fingerprints)
            expert_outputs.append(output)
            expert_utilization[f'expert_{expert.expert_id}'] = output['num_patches_processed']

        # Fuse via Cantor global attention
        fused = self.attention(expert_outputs, num_patches)

        # Dense projection to full feature space
        output = self.fusion_proj(fused)

        # Residual connection
        return x + output, expert_utilization

    def get_expert_utilization(self) -> Dict[str, float]:
        """Get expert utilization statistics."""
        stats = {}
        for expert in self.experts:
            stats[f'expert_{expert.expert_id}_alpha'] = torch.sigmoid(expert.alpha).item()
            for beta_key, beta_val in expert.betas.items():
                stats[f'expert_{expert.expert_id}_beta_{beta_key}'] = torch.sigmoid(beta_val).item()

        # Direction fusion weights
        weights = F.softmax(self.attention.direction_fusion_weights, dim=0)
        for i, w in enumerate(weights):
            stats[f'direction_{i}_weight'] = w.item()

        return stats


# ============================================================================
# ViT-BEANS (UPDATED)
# ============================================================================

@dataclass
class ViTBeansConfig:
    """Complete ViT-Beans configuration."""
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_layers: int = 12
    feature_dim: int = 1024
    num_experts: int = 16
    expert_dim: int = 128
    num_heads: int = 8
    mlp_ratio: float = 4.0
    cantor_dimensions: List[int] = None
    cantor_depth: int = 8
    local_window: int = 3
    alpha_init: float = 1.0
    beta_init: float = 0.3
    temperature: float = 0.5  # FIX: Better default
    dropout: float = 0.1
    num_classes: int = 1000

    def __post_init__(self):
        if self.cantor_dimensions is None:
            self.cantor_dimensions = [2, 3, 4, 5]

        assert self.feature_dim % self.num_experts == 0, \
            "feature_dim must be divisible by num_experts"


class ViTBeans(nn.Module):
    """
    ViT-Beans with all critical fixes applied.
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

        # Position embedding
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

        # Pre-compute fingerprints
        self.register_buffer(
            'patch_fingerprints',
            torch.zeros(self.num_patches)
        )
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
                    local_window=config.local_window,
                    dropout=config.dropout,
                    alpha_init=config.alpha_init,
                    beta_init=config.beta_init,
                    temperature=config.temperature
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
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _compute_fingerprints(self, device: torch.device):
        """Compute and cache Cantor fingerprints."""
        if not self._fingerprints_computed:
            fingerprints_dict = self.fingerprinter.compute_fingerprints(
                self.num_patches, device
            )
            self.patch_fingerprints = fingerprints_dict[3]
            self._fingerprints_computed = True

    def forward_features(self, x: torch.Tensor, return_utilization: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Extract features through ViT-Beans layers.

        NEW: Optionally return expert utilization stats
        """
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

        # Track utilization if requested
        all_utilization = {} if return_utilization else None

        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            # Cantor MoE (skip CLS token)
            x_patches = x[:, 1:]
            x_patches, utilization = layer['cantor_moe'](x_patches, self.patch_fingerprints)
            x = torch.cat([x[:, :1], x_patches], dim=1)

            if return_utilization:
                all_utilization[f'layer_{layer_idx}'] = utilization

            # MLP
            x = x + layer['mlp'](x)

        if return_utilization:
            return x, all_utilization
        return x, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        x, _ = self.forward_features(x)

        # Classification from CLS token
        x = self.norm(x[:, 0])
        logits = self.head(x)

        return logits

    def get_expert_stats(self) -> Dict:
        """Get comprehensive expert statistics."""
        stats = {
            'num_experts': self.config.num_experts,
            'expert_params': [],
            'alpha_values': [],
            'beta_values': [],
            'direction_weights': []
        }

        for layer_idx, layer in enumerate(self.layers):
            moe = layer['cantor_moe']

            # Per-expert stats
            for expert_idx, expert in enumerate(moe.experts):
                params = sum(p.numel() for p in expert.parameters())
                stats['expert_params'].append(params)

                alpha = torch.sigmoid(expert.alpha).item()
                stats['alpha_values'].append(alpha)

                for key, beta in expert.betas.items():
                    beta_val = torch.sigmoid(beta).item()
                    stats['beta_values'].append(beta_val)

            # Direction fusion weights
            weights = F.softmax(moe.attention.direction_fusion_weights, dim=0)
            stats['direction_weights'].append(weights.cpu().tolist())

        stats['total_expert_params'] = sum(stats['expert_params'])
        stats['mean_alpha'] = sum(stats['alpha_values']) / len(stats['alpha_values'])
        stats['mean_beta'] = sum(stats['beta_values']) / len(stats['beta_values']) if stats['beta_values'] else 0.0

        return stats

    def diagnose_expert_coverage(self) -> Dict:
        """
        NEW: Diagnose expert coverage to detect dropped patches.
        """
        device = self.patch_fingerprints.device
        fingerprints = self.patch_fingerprints

        coverage = {}
        total_covered = 0

        # Check first layer's experts
        moe = self.layers[0]['cantor_moe']
        for expert in moe.experts:
            if expert.is_last_expert:
                mask = (fingerprints >= expert.fp_min) & (fingerprints <= expert.fp_max)
            else:
                mask = (fingerprints >= expert.fp_min) & (fingerprints < expert.fp_max)

            num_patches = mask.sum().item()
            coverage[f'expert_{expert.expert_id}'] = {
                'patches': num_patches,
                'range': f'[{expert.fp_min:.4f}, {expert.fp_max:.4f}{")" if not expert.is_last_expert else "]"}',
                'is_last': expert.is_last_expert
            }
            total_covered += num_patches

        coverage['total_patches'] = self.num_patches
        coverage['total_covered'] = total_covered
        coverage['max_fingerprint'] = fingerprints.max().item()
        coverage['min_fingerprint'] = fingerprints.min().item()
        coverage['patches_at_max'] = (fingerprints == fingerprints.max()).sum().item()

        return coverage


# ============================================================================
# DEMO & TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ViT-Beans: FIXED VERSION")
    print("=" * 80)

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
    print(f"  Patches: {(config.image_size // config.patch_size) ** 2}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Temperature: {config.temperature}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViTBeans(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # DIAGNOSTIC: Check expert coverage
    print("\n" + "=" * 80)
    print("EXPERT COVERAGE DIAGNOSTIC")
    print("=" * 80)
    coverage = model.diagnose_expert_coverage()
    print(f"Total patches: {coverage['total_patches']}")
    print(f"Total covered: {coverage['total_covered']}")
    print(f"Fingerprint range: [{coverage['min_fingerprint']:.4f}, {coverage['max_fingerprint']:.4f}]")
    print(f"Patches at max: {coverage['patches_at_max']}")
    print("\nPer-expert coverage:")
    for expert_id in range(config.num_experts):
        info = coverage[f'expert_{expert_id}']
        print(f"  Expert {expert_id:2d}: {info['patches']:3d} patches in {info['range']} {'(LAST)' if info['is_last'] else ''}")

    # Test forward pass
    print("\n" + "=" * 80)
    print("TESTING FORWARD PASS")
    print("=" * 80)
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"✓ Forward pass successful")

    print("\n" + "=" * 80)
    print("FIXES APPLIED:")
    print("  1. ✓ Last expert now includes fingerprint=1.0")
    print("  2. ✓ Proper attention masking (no zombie padding)")
    print("  3. ✓ Better temperature scaling (0.5 vs 0.07)")
    print("  4. ✓ Learnable direction fusion weights")
    print("  5. ✓ Better initialization (gain=1.0)")
    print("  6. ✓ Expert utilization tracking")
    print("=" * 80)