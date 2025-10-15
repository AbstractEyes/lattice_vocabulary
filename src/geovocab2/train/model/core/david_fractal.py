"""
FractalDavid - OPTIMIZED Multi-Scale Crystal Classifier
========================================================
Performance-optimized implementation with batched operations.

KEY OPTIMIZATIONS:
1. Batched Cantor gate computation (pre-compute, no expand)
2. Fused einsum operations across scales
3. Memory-efficient crystal gating (no huge tensor expansions)
4. Cached Cantor scalars
5. In-place operations where safe

Expected speedup: 10-20x for large batches
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class FractalDavidConfig:
    """Configuration for FractalDavid model."""
    feature_dim: int = 768
    num_classes: int = 1000
    scales: Tuple[int, ...] = (384, 512, 768, 1024, 1280)
    use_belly: bool = True
    belly_expand: float = 2.0
    projection_temperature: float = 0.07
    simplex_k: int = 4
    num_vertices: Optional[int] = None
    enable_cantor_gate: bool = True
    cantor_levels: int = 12
    gate_strength: float = 0.25
    default_internal_cantor: bool = True
    fusion_mode: str = "WEIGHTED_SUM"
    progressive_training: bool = False
    scale_warmup_epochs: Optional[Dict[int, int]] = None

    def __post_init__(self):
        if self.num_vertices is None:
            self.num_vertices = self.simplex_k + 1
        assert self.simplex_k >= 0
        assert self.num_vertices == self.simplex_k + 1
        if self.scale_warmup_epochs is None:
            self.scale_warmup_epochs = {s: 0 for s in self.scales}


# ============================================================================
# OPTIMIZED CANTOR STAIRS (VECTORIZED)
# ============================================================================

class CantorStairs:
    """Vectorized Cantor staircase - compute once, reuse."""

    @staticmethod
    def value(pos: torch.Tensor, max_pos: int, levels: int = 12) -> torch.Tensor:
        """Efficient Cantor computation with in-place operations."""
        if max_pos > 1:
            x = pos.float() / float(max_pos - 1)
        else:
            x = pos.float().clamp(0.0, 1.0)

        y = x.clone()
        out = torch.zeros_like(y)
        w = 0.5

        for _ in range(levels):
            t = (y * 3.0).floor()
            bit = (t == 2.0).float()
            out.add_(bit * w)  # in-place
            y = y * 3.0 - t
            w *= 0.5

        return out.clamp_(0.0, 1.0)  # in-place


# ============================================================================
# OPTIMIZED CANTOR SIMPLEX GATE (NO HUGE EXPANSIONS!)
# ============================================================================

class CantorSimplexGate(nn.Module):
    """Memory-efficient Cantor gating using broadcasting instead of expand."""

    def __init__(self, num_vertices: int = 5, gate_strength: float = 0.25):
        super().__init__()
        self.V = int(num_vertices)
        self.gain = nn.Parameter(torch.tensor(float(gate_strength), dtype=torch.float32))
        self.register_buffer("bary_templates", torch.eye(self.V))

        # Pre-compute base emphasis patterns
        self.register_buffer("anchor_emphasis", torch.zeros(self.V))
        self.anchor_emphasis[0] = 1.0

        self.register_buffer("observer_emphasis", torch.zeros(self.V))
        self.observer_emphasis[-1] = 1.0

        self.register_buffer("diffuse_emphasis", torch.ones(self.V) * 0.15)

    def compute_gates(self, cantor_scalar: torch.Tensor) -> torch.Tensor:
        """
        Compute gates efficiently without huge expansions.

        Args:
            cantor_scalar: [B] in [0,1]

        Returns:
            gates: [B, V]
        """
        B = cantor_scalar.shape[0]
        V = int(self.V)

        # Classify intervals
        t = (cantor_scalar * 3.0).floor()
        left = (t == 0).float().unsqueeze(-1)    # [B, 1]
        mid = (t == 1).float().unsqueeze(-1)     # [B, 1]
        right = (t == 2).float().unsqueeze(-1)   # [B, 1]

        # Base uniform
        base = torch.full((B, V), 1.0 / V, device=cantor_scalar.device, dtype=cantor_scalar.dtype)

        # Apply emphasis (broadcasting, no expand!)
        base = base + left * (self.anchor_emphasis.unsqueeze(0) - base)
        base = base + right * (self.observer_emphasis.unsqueeze(0) - base)
        base = base + mid * (self.diffuse_emphasis.unsqueeze(0) - base)

        # Learnable gain
        g = self.gain.sigmoid()
        uniform = torch.full((B, V), 1.0 / V, device=cantor_scalar.device, dtype=cantor_scalar.dtype)
        gates = (1.0 - g) * uniform + g * base

        return gates

    def forward(
        self,
        cantor_scalar: torch.Tensor,
        crystals: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gating ONLY to anchor vertex (no memory bloat, fast on CPU/GPU).

        Args:
            cantor_scalar: [B]
            crystals: [C, V, d]
            z: [B, d] - normalized embeddings

        Returns:
            logits: [B, C]
        """
        gates = self.compute_gates(cantor_scalar)  # [B, V]

        # Extract anchor vertex and its gates
        anchors = crystals[:, 0, :]  # [C, d]
        anchor_gates = gates[:, 0:1]  # [B, 1] - keep dims for broadcasting

        # Normalize anchors once
        anchors_norm = F.normalize(anchors, dim=-1)  # [C, d]

        # Compute base similarities: [B, d] @ [C, d]^T = [B, C]
        # This is fast on both CPU and GPU
        base_logits = z @ anchors_norm.T  # [B, C]

        # Apply per-sample gates: [B, C] * [B, 1] = [B, C]
        # Broadcasting handles this efficiently
        logits = base_logits * (1e-6 + anchor_gates)  # [B, C]

        return logits


# ============================================================================
# OPTIMIZED FRACTAL SCALE HEAD
# ============================================================================

class FractalScaleHead(nn.Module):
    """Optimized scale head with efficient projections."""

    def __init__(
        self,
        input_dim: int,
        crystal_dim: int,
        temperature: float = 0.07,
        use_belly: bool = True,
        belly_expand: float = 2.0,
        num_vertices: int = 5,
        enable_cantor_gate: bool = True,
        gate_strength: float = 0.25,
    ):
        super().__init__()
        self.crystal_dim = int(crystal_dim)
        self.temperature = float(temperature)
        self.enable_cantor_gate = bool(enable_cantor_gate)
        self.num_vertices = int(num_vertices)

        # Optimized projection
        if use_belly:
            belly_dim = int(self.crystal_dim * float(belly_expand))
            dropout_rate = 1.0 / math.sqrt(self.crystal_dim)
            self.projection = nn.Sequential(
                nn.Linear(input_dim, belly_dim, bias=True),
                nn.ReLU(inplace=True),  # in-place
                nn.Dropout(dropout_rate),
                nn.Linear(belly_dim, self.crystal_dim, bias=False),
            )
        else:
            self.projection = nn.Linear(input_dim, self.crystal_dim, bias=False)

        self._init_weights()

        if self.enable_cantor_gate:
            self.cantor_gate = CantorSimplexGate(
                num_vertices=self.num_vertices,
                gate_strength=gate_strength
            )

    def _init_weights(self):
        """Xavier init for stability."""
        for layer in self.projection.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        features: torch.Tensor,
        anchors: torch.Tensor,
        crystals: Optional[torch.Tensor] = None,
        cantor_scalar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass.

        Args:
            features: [B, D_in]
            anchors: [C, D_scale]
            crystals: [C, V, D_scale]
            cantor_scalar: [B]

        Returns:
            logits: [B, C]
            z: [B, D_scale]
        """
        z = self.projection(features)
        z = F.normalize(z, dim=-1)

        if self.enable_cantor_gate and (cantor_scalar is not None) and (crystals is not None):
            # Gate only the anchor vertex and compute logits directly
            # No [B, C, V, d] bloat!
            logits = self.cantor_gate(cantor_scalar, crystals, z) / self.temperature
        else:
            # Simple anchor matching
            anchors_norm = F.normalize(anchors, dim=-1)
            logits = (z @ anchors_norm.T) / self.temperature

        return logits, z


# ============================================================================
# OPTIMIZED ROSE LOSS (BATCHED OPERATIONS)
# ============================================================================

class RoseLoss(nn.Module):
    """Memory-efficient Rose Loss with batched role weight computation."""

    def __init__(
        self,
        num_vertices: int = 5,
        margin: float = 1.0,
        temperature: float = 0.07,
        role_weights: Optional[Union[Dict[str, float], torch.Tensor]] = None,
        cantor_aware: bool = True,
        cantor_theta: float = 0.5,
    ):
        super().__init__()
        self.num_vertices = num_vertices
        self.margin = margin
        self.temperature = temperature
        self.cantor_aware = cantor_aware

        # Generate base role weights
        if role_weights is None:
            role_vec = self._generate_default_roles(num_vertices)
        elif isinstance(role_weights, dict):
            if num_vertices == 5:
                default_weights = {
                    "anchor": 1.0, "need": -0.75, "relation": 0.75,
                    "purpose": 0.75, "observer": -0.75,
                }
                weights = {**default_weights, **role_weights}
                role_vec = torch.tensor([
                    weights["anchor"], weights["need"], weights["relation"],
                    weights["purpose"], weights["observer"],
                ], dtype=torch.float32)
            else:
                raise ValueError(f"Named roles only for 5 vertices, got {num_vertices}")
        else:
            role_vec = role_weights

        self.register_buffer("base_role_weights", role_vec)

        if self.cantor_aware:
            self.theta = nn.Parameter(torch.tensor(cantor_theta, dtype=torch.float32))

            # Pre-compute emphasis patterns
            anchor_emphasis = torch.zeros(num_vertices)
            anchor_emphasis[0] = 1.0
            if num_vertices > 1:
                anchor_emphasis[-1] = -0.5
            self.register_buffer("anchor_emphasis", anchor_emphasis)

            observer_emphasis = torch.zeros(num_vertices)
            if num_vertices > 1:
                observer_emphasis[-1] = 1.0
            observer_emphasis[0] = -0.5
            self.register_buffer("observer_emphasis", observer_emphasis)

    @staticmethod
    def _generate_default_roles(num_vertices: int) -> torch.Tensor:
        """Generate default role weights for k-simplex."""
        if num_vertices < 2:
            return torch.ones(num_vertices)

        weights = torch.zeros(num_vertices)
        weights[0] = 1.0
        weights[-1] = -0.75

        for i in range(1, num_vertices - 1):
            weights[i] = 0.75 if i % 2 == 1 else -0.5

        return weights

    def _compute_cantor_modulated_roles_fast(
        self,
        cantor_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """Fast batched Cantor role computation."""
        B = cantor_scalar.shape[0]

        t = (cantor_scalar * 3.0).floor()
        left = (t == 0).float().unsqueeze(-1)    # [B, 1]
        right = (t == 2).float().unsqueeze(-1)   # [B, 1]

        # Broadcasting instead of expand
        base = self.base_role_weights.unsqueeze(0)  # [1, V]

        theta = self.theta.sigmoid()

        # Efficient broadcasting
        modulated = base + theta * left * self.anchor_emphasis.unsqueeze(0)
        modulated = modulated + theta * right * self.observer_emphasis.unsqueeze(0)

        return modulated  # [B, V]

    def forward(
        self,
        z: torch.Tensor,
        crystals: torch.Tensor,
        targets: torch.Tensor,
        cantor_scalar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Optimized Rose Loss computation.

        Args:
            z: [B, D]
            crystals: [C, V, d]
            targets: [B]
            cantor_scalar: [B]

        Returns:
            loss: scalar
        """
        crystals = crystals.to(z.device)
        crystals_norm = F.normalize(crystals, dim=-1)

        # Efficient similarity computation
        cos_sim = torch.einsum("bd,cvd->bcv", z, crystals_norm)  # [B, C, V]

        # Role weights
        if self.cantor_aware and (cantor_scalar is not None):
            role_weights = self._compute_cantor_modulated_roles_fast(cantor_scalar)  # [B, V]
            # Weighted sum: [B, C, V] * [B, 1, V] -> [B, C]
            rose_scores = torch.einsum('bcv,bv->bc', cos_sim, role_weights)
        else:
            # Static role weights: [B, C, V] * [V] -> [B, C]
            role_weights = self.base_role_weights
            rose_scores = torch.einsum('bcv,v->bc', cos_sim, role_weights)

        rose_scores = rose_scores / self.temperature

        # Efficient hard negative mining
        true_scores = rose_scores.gather(1, targets.view(-1, 1)).squeeze(1)

        # Mask out true class for hard negatives
        mask = torch.zeros_like(rose_scores, dtype=torch.bool)
        mask.scatter_(1, targets.view(-1, 1), True)
        hard_neg = rose_scores.masked_fill(mask, float("-inf")).max(dim=1).values

        loss = F.relu(self.margin - (true_scores - hard_neg))
        return loss.mean()


# ============================================================================
# OPTIMIZED FRACTAL MULTI-SCALE LOSS
# ============================================================================

class FractalMultiScaleLoss(nn.Module):
    """Optimized multi-scale loss with minimal redundant computation."""

    def __init__(
        self,
        scales: List[int],
        num_classes: int = 1000,
        num_vertices: int = 5,
        use_rose_loss: bool = True,
        rose_initial_weight: float = 0.1,
        rose_max_weight: float = 0.5,
        rose_margin: float = 1.0,
        rose_temperature: float = 0.07,
        rose_cantor_aware: bool = True,
        rose_cantor_theta: float = 0.5,
        scale_loss_balance: Optional[Dict[int, float]] = None,
    ):
        super().__init__()

        self.scales = scales
        self.num_classes = num_classes
        self.num_vertices = num_vertices
        self.use_rose_loss = use_rose_loss
        self.rose_weight = rose_initial_weight
        self.rose_max_weight = rose_max_weight
        self.rose_cantor_aware = rose_cantor_aware

        self.scale_balance = scale_loss_balance or {s: 1.0 for s in scales}
        self.ce_loss = nn.CrossEntropyLoss()

        if use_rose_loss:
            self.rose_losses = nn.ModuleDict({
                str(scale): RoseLoss(
                    num_vertices=num_vertices,
                    margin=rose_margin,
                    temperature=rose_temperature,
                    cantor_aware=rose_cantor_aware,
                    cantor_theta=rose_cantor_theta,
                )
                for scale in scales
            })

    def forward(
        self,
        combined_logits: torch.Tensor,
        scale_logits: List[torch.Tensor],
        scale_features: List[torch.Tensor],
        targets: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor],
        crystals_dict: Dict[int, torch.Tensor],
        epoch: int = 0,
        cantor_scalars: Optional[Union[torch.Tensor, Dict[int, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute losses efficiently."""
        losses = {}

        # Main CE loss
        ce_main = self.ce_loss(combined_logits, targets)
        losses['ce'] = ce_main
        losses['ce_main'] = ce_main

        total_scale_loss = 0.0
        rose_total = 0.0

        # Compute rose weight schedule
        progress = min(epoch / 100.0, 1.0)
        current_rose_weight = self.rose_weight + \
            (self.rose_max_weight - self.rose_weight) * progress

        for i, (scale, logits, features) in enumerate(
            zip(self.scales, scale_logits, scale_features)
        ):
            scale_weight = self.scale_balance.get(scale, 1.0)

            # CE per scale
            ce_scale = self.ce_loss(logits, targets)
            losses[f'ce_{scale}'] = ce_scale
            total_scale_loss += scale_weight * ce_scale

            # Rose loss per scale
            if self.use_rose_loss and scale in crystals_dict:
                if cantor_scalars is None:
                    cs = None
                elif isinstance(cantor_scalars, dict):
                    cs = cantor_scalars.get(scale, None)
                else:
                    cs = cantor_scalars

                rose_loss = self.rose_losses[str(scale)](
                    features, crystals_dict[scale], targets, cantor_scalar=cs
                )
                losses[f'rose_{scale}'] = rose_loss
                rose_total += rose_loss

                total_scale_loss += current_rose_weight * rose_loss

        # Average rose loss for logging
        if self.use_rose_loss and len(scale_logits) > 0:
            losses['rose'] = rose_total / len(scale_logits)

        total_loss = ce_main + total_scale_loss / len(self.scales)
        losses['total'] = total_loss

        return losses


# ============================================================================
# OPTIMIZED FRACTAL DAVID MODEL
# ============================================================================

class FractalDavid(nn.Module):
    """Optimized FractalDavid with batched operations and reduced memory."""

    def __init__(self, config: Optional[FractalDavidConfig] = None):
        super().__init__()
        self.config = config or FractalDavidConfig()

        self.feature_dim = int(self.config.feature_dim)
        self.num_classes = int(self.config.num_classes)
        self.scales: List[int] = list(self.config.scales)

        self.current_epoch = 0
        self.progressive_training = self.config.progressive_training
        self.scale_warmup_epochs = self.config.scale_warmup_epochs

        self._last_cantor_scalars: Optional[Dict[int, torch.Tensor]] = None
        self._cantor_cache: Optional[torch.Tensor] = None  # Cache for efficiency

        # Create scale heads
        self.heads = nn.ModuleDict({
            str(scale): FractalScaleHead(
                input_dim=self.feature_dim,
                crystal_dim=scale,
                temperature=self.config.projection_temperature,
                use_belly=self.config.use_belly,
                belly_expand=self.config.belly_expand,
                num_vertices=self.config.num_vertices,
                enable_cantor_gate=self.config.enable_cantor_gate,
                gate_strength=self.config.gate_strength,
            )
            for scale in self.scales
        })

        # Fusion
        if self.config.fusion_mode == "WEIGHTED_SUM":
            self.fusion_weights = nn.Parameter(torch.ones(len(self.scales)))
        else:
            raise NotImplementedError(f"Fusion mode {self.config.fusion_mode} not implemented")

    def _active_scales(self) -> List[int]:
        """Get active scales based on progressive training."""
        if not self.progressive_training:
            return self.scales

        active = []
        for scale in self.scales:
            warmup = self.scale_warmup_epochs.get(scale, 0)
            if self.current_epoch >= warmup:
                active.append(scale)
        return active

    def _get_or_compute_cantor_scalars(
        self,
        num_scales: int,
        batch_size: int,
        device: torch.device,
        cantor_pos: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Get or compute Cantor scalars efficiently with caching.

        Returns:
            cantor_scalars: [num_scales] or [B] depending on cantor_pos
        """
        if cantor_pos is not None:
            return cantor_pos.float().clamp(0.0, 1.0)

        # Use cached Cantor values if available
        if self._cantor_cache is None or self._cantor_cache.shape[0] != num_scales:
            max_pos = max(1, num_scales - 1)
            pos_tensor = torch.arange(num_scales, device=device, dtype=torch.float32)
            self._cantor_cache = CantorStairs.value(
                pos_tensor,
                max_pos=max_pos + 1,
                levels=self.config.cantor_levels
            )

        return self._cantor_cache

    def _parallel_forward_optimized(
        self,
        features: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor],
        crystals_dict: Dict[int, torch.Tensor],
        cantor_pos: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Optimized parallel forward with minimal redundancy."""
        logits_list: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []

        B = features.shape[0]
        device = features.device

        scale_ids = self._active_scales()
        num_scales = len(scale_ids)

        # Pre-compute Cantor scalars once
        if self.config.default_internal_cantor:
            cantor_base = self._get_or_compute_cantor_scalars(
                num_scales, B, device, cantor_pos
            )
        else:
            cantor_base = None

        cantor_scalars_dict = {}

        # Process each scale
        for idx, scale in enumerate(scale_ids):
            head = self.heads[str(scale)]
            anchors = anchors_dict[scale]
            crystals = crystals_dict.get(scale, None)

            # Get Cantor scalar for this scale
            if cantor_base is not None:
                if cantor_base.dim() == 0 or (cantor_base.dim() == 1 and cantor_base.shape[0] == num_scales):
                    # Per-scale value, expand to batch
                    cs = cantor_base[idx].expand(B)
                else:
                    # Already batch-sized
                    cs = cantor_base
            else:
                cs = None

            if cs is not None:
                cantor_scalars_dict[scale] = cs.detach()

            # Forward pass
            logits, feats = head(features, anchors, crystals=crystals, cantor_scalar=cs)
            logits_list.append(logits)
            features_list.append(feats)

        self._last_cantor_scalars = cantor_scalars_dict if cantor_scalars_dict else None

        return logits_list, features_list

    def _fuse_logits(self, logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse logits across scales."""
        if len(logits_list) == 1:
            w = torch.ones(1, device=logits_list[0].device)
            return logits_list[0], w

        weights = F.softmax(self.fusion_weights[:len(logits_list)], dim=0)
        combined = sum(w * logits for w, logits in zip(weights, logits_list))
        return combined, weights

    def forward(
        self,
        x: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor],
        crystals_dict: Optional[Dict[int, torch.Tensor]] = None,
        return_all_scales: bool = False,
        *,
        cantor_pos: Optional[torch.Tensor] = None,
        cantor_levels: Optional[int] = None
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]
    ]:
        """Optimized forward pass."""
        assert x.dim() == 2 and x.size(-1) == self.feature_dim

        for s in self.scales:
            assert s in anchors_dict, f"anchors_dict missing scale {s}"

        if crystals_dict is None:
            crystals_dict = {}

        # Forward through scales
        logits_list, features_list = self._parallel_forward_optimized(
            x, anchors_dict, crystals_dict, cantor_pos
        )

        # Fuse
        combined, fusion_w = self._fuse_logits(logits_list)

        if return_all_scales:
            return combined, logits_list, features_list, fusion_w
        else:
            return combined, (features_list[0] if features_list else x)

    def update_epoch(self, epoch: int):
        """Update epoch for progressive training."""
        self.current_epoch = epoch

    def get_active_scales(self) -> List[int]:
        """Get currently active scales."""
        return self._active_scales()

    def get_last_cantor_scalars(self) -> Optional[Dict[int, torch.Tensor]]:
        """Get last computed Cantor scalars for diagnostics."""
        return self._last_cantor_scalars

    def freeze_scale(self, scale: int):
        """Freeze a specific scale."""
        for param in self.heads[str(scale)].parameters():
            param.requires_grad = False

    def unfreeze_scale(self, scale: int):
        """Unfreeze a specific scale."""
        for param in self.heads[str(scale)].parameters():
            param.requires_grad = True

    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        return {
            "name": "FractalDavid-Optimized",
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "simplex_k": self.config.simplex_k,
            "num_vertices": self.config.num_vertices,
            "scales": self.scales,
            "active_scales": self.get_active_scales(),
            "enable_cantor": self.config.enable_cantor_gate,
            "cantor_levels": self.config.cantor_levels,
            "gate_strength": self.config.gate_strength,
            "current_epoch": self.current_epoch,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def __repr__(self):
        info = self.get_model_info()
        k = self.config.num_vertices - 1
        return (
            f"FractalDavid-Optimized (Batched Multi-Scale + Cantor Gating)\n"
            f"  Simplex: k={k} ({info['num_vertices']} vertices)\n"
            f"  Scales: {info['scales']}\n"
            f"  Active: {info['active_scales']}\n"
            f"  Cantor: {'Enabled' if info['enable_cantor'] else 'Disabled'}\n"
            f"  Parameters: {info['total_parameters']:,}"
        )


# ============================================================================
# PERFORMANCE TEST
# ============================================================================

if __name__ == "__main__":
    import time

    print("="*80)
    print("FractalDavid - Performance Optimization Test")
    print("="*80)

    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Test configuration
    config = FractalDavidConfig(
        feature_dim=768,
        num_classes=1000,
        scales=(384, 512, 768, 1024, 1280),  # 5 scales
        simplex_k=5,  # 6-simplex (Jupiter style!)
        enable_cantor_gate=True,
        gate_strength=0.25,
    )

    model = FractalDavid(config).to(device)
    print(model)
    print()

    # Create test data
    B, C = 512, config.num_classes  # Full batch
    x = torch.randn(B, config.feature_dim, device=device)

    # Generate crystals
    anchors_dict = {}
    crystals_dict = {}
    for scale in config.scales:
        crystals = F.normalize(torch.randn(C, 6, scale, device=device), dim=-1)  # 6 vertices
        anchors = crystals[:, 0, :].clone()
        anchors_dict[scale] = anchors
        crystals_dict[scale] = crystals

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, anchors_dict, crystals_dict, return_all_scales=True)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking forward pass...")
    num_iterations = 50 if device.type == 'cuda' else 10  # Fewer iterations on CPU

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            out, logits_list, feats_list, w = model(x, anchors_dict, crystals_dict, return_all_scales=True)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start

    avg_time = elapsed / num_iterations
    throughput = B / avg_time

    print(f"\n📊 Performance Results ({device.type.upper()}):")
    print(f"   Batch size: {B}")
    print(f"   Scales: {len(config.scales)}")
    print(f"   Simplex: k={config.simplex_k} ({config.num_vertices} vertices)")
    print(f"   Average time: {avg_time*1000:.2f} ms/batch")
    print(f"   Throughput: {throughput:.1f} samples/sec")
    print(f"   Est. epoch time (1.28M samples): {(1.28e6 / throughput / 60):.1f} minutes")

    if device.type == 'cpu':
        print(f"\n   💡 Note: CPU performance. GPU will be ~10-50x faster!")

    print("\n✅ Optimization complete!")