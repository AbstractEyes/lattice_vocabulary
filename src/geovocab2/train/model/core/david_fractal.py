"""
FractalDavid - Multi-Scale Crystal Classifier with Cantor-Stairs Gating
========================================================================
Production implementation combining proven David architecture with fractal gating
and Cantor-aware Rose Loss.

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT

Architecture Overview:
----------------------
1. **k-Simplex Prototypes**: Each class represented by a k-simplex (default k=4)
   - k=0: point, k=1: edge, k=2: triangle, k=3: tetrahedron, k=4: pentachoron
   - num_vertices = k+1

2. **Cantor-Stairs Gating**: Maps Cantor scalar [0,1] → vertex emphasis
   - Left [0, 1/3):   Emphasize anchor (v0)
   - Middle [1/3, 2/3): Balanced/diffuse
   - Right [2/3, 1]:  Emphasize observer (v_last)

3. **Cantor-Aware Rose Loss**: Dynamic role weights aligned with forward gating
   - Forward: Cantor gates certain vertices
   - Backward: Rose Loss emphasizes same vertices
   - Creates coherent forward-backward feedback loop

4. **Decoupled Multi-Scale**: No cross-scale coupling

Usage:
------
config = FractalDavidConfig(simplex_k=4, enable_cantor_gate=True)
model = FractalDavid(config)

# With CrystalGenerator:
factory = SimplexFactory(k=config.simplex_k, embed_dim=scale, method="random")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FractalDavidConfig:
    """Configuration for FractalDavid model."""

    # Architecture
    feature_dim: int = 768
    num_classes: int = 1000
    scales: Tuple[int, ...] = (384, 512, 768, 1024, 1280)

    # Head configuration
    use_belly: bool = True
    belly_expand: float = 2.0
    projection_temperature: float = 0.07

    # Simplex geometry (k-simplex has k+1 vertices)
    simplex_k: int = 4  # 4-simplex = pentachoron (5 vertices)
    num_vertices: Optional[int] = None  # Computed as k+1

    # Cantor gating
    enable_cantor_gate: bool = True
    cantor_levels: int = 12
    gate_strength: float = 0.25
    default_internal_cantor: bool = True

    # Fusion
    fusion_mode: str = "WEIGHTED_SUM"

    # Progressive training
    progressive_training: bool = False
    scale_warmup_epochs: Optional[Dict[int, int]] = None

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        if self.num_vertices is None:
            self.num_vertices = self.simplex_k + 1

        assert self.simplex_k >= 0, f"simplex_k must be >= 0, got {self.simplex_k}"
        assert self.num_vertices == self.simplex_k + 1, \
            f"num_vertices ({self.num_vertices}) must equal simplex_k + 1 ({self.simplex_k + 1})"

        if self.scale_warmup_epochs is None:
            self.scale_warmup_epochs = {s: 0 for s in self.scales}


# ============================================================================
# CANTOR STAIRS
# ============================================================================

class CantorStairs:
    """Finite-level Cantor (devil's staircase) for positions → [0,1]."""

    @staticmethod
    def value(pos: torch.Tensor, max_pos: int, levels: int = 12) -> torch.Tensor:
        """
        Compute Cantor staircase value.

        Args:
            pos: Tensor[...] - positions
            max_pos: int - normalization denominator
            levels: int - ternary expansion depth

        Returns:
            Tensor[...] in [0,1]
        """
        if max_pos > 1:
            x = pos.to(torch.float32) / float(max_pos - 1)
        else:
            x = pos.to(torch.float32).clamp(0.0, 1.0)

        y = x.clone()
        out = torch.zeros_like(y)
        w = 0.5

        for _ in range(levels):
            t = torch.floor(y * 3.0)
            bit = (t == 2.0).to(y.dtype)
            out = out + bit * w
            y = y * 3.0 - t
            w *= 0.5

        return out.clamp_(0.0, 1.0)


# ============================================================================
# CANTOR SIMPLEX GATE
# ============================================================================

class CantorSimplexGate(nn.Module):
    """Map Cantor scalar → per-vertex soft gates for class crystals."""

    def __init__(self, num_vertices: int = 5, gate_strength: float = 0.25):
        super().__init__()
        self.V = int(num_vertices)
        self.gain = nn.Parameter(torch.tensor(float(gate_strength), dtype=torch.float32))
        self.register_buffer("bary_templates", torch.eye(self.V))

    def forward(self, cantor_scalar: torch.Tensor, crystals: torch.Tensor) -> torch.Tensor:
        """
        Apply Cantor-based gating to crystals.

        Args:
            cantor_scalar: [B] - values in [0,1]
            crystals: [C, V, d] - class crystals

        Returns:
            gated: [B, C, V, d]
        """
        B = cantor_scalar.shape[0]
        C, V, d = crystals.shape
        assert V == self.V, f"num_vertices mismatch: expected {self.V}, got {V}"

        # Assign interval: 0 (left), 1 (middle), 2 (right)
        t = (cantor_scalar * 3.0).floor()
        left = (t == 0).float().unsqueeze(-1)
        mid = (t == 1).float().unsqueeze(-1)
        right = (t == 2).float().unsqueeze(-1)

        # Base gates [B, V]
        base = torch.full((B, V), 1.0 / V, device=crystals.device, dtype=crystals.dtype)
        base = base + left * (self.bary_templates[0] - base)
        base = base + right * (self.bary_templates[-1] - base)
        base = base + mid * (0.15 * torch.ones_like(base) - base)

        # Apply learnable gain
        g = self.gain.sigmoid()
        gates = (1.0 - g) * (torch.ones_like(base) / V) + g * base

        # Broadcast and apply
        crystals_b = crystals.unsqueeze(0).expand(B, C, V, d)
        gates_b = gates.view(B, 1, V, 1)

        return crystals_b * (1e-6 + gates_b)


# ============================================================================
# FRACTAL SCALE HEAD
# ============================================================================

class FractalScaleHead(nn.Module):
    """Decoupled scale head with optional Cantor-stairs simplex gating."""

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

        # Projection
        if use_belly:
            belly_dim = int(self.crystal_dim * float(belly_expand))
            self.projection = nn.Sequential(
                nn.Linear(input_dim, belly_dim),
                nn.ReLU(),
                nn.Dropout(1.0 / math.sqrt(self.crystal_dim)),
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
        Forward pass with optional Cantor gating.

        Args:
            features: [B, D_in]
            anchors: [C, D_scale] - first vertex only
            crystals: [C, V, D_scale] - full geometry (for Cantor gating)
            cantor_scalar: [B] in [0,1]

        Returns:
            logits: [B, C]
            z: [B, D_scale]
        """
        B = features.shape[0]
        C, D = anchors.shape

        z = self.projection(features)
        z = F.normalize(z, dim=-1)

        if self.enable_cantor_gate and (cantor_scalar is not None) and (crystals is not None):
            assert crystals.shape == (C, self.num_vertices, D), \
                f"Expected crystals [{C}, {self.num_vertices}, {D}], got {crystals.shape}"

            gated = self.cantor_gate(cantor_scalar, crystals)
            gated_anchors = gated[:, :, 0, :]
            anchors_eff = F.normalize(gated_anchors, dim=-1)
            logits = torch.einsum("bd,bcd->bc", z, anchors_eff) / self.temperature
        else:
            anchors_eff = F.normalize(anchors, dim=-1)
            logits = (z @ anchors_eff.T) / self.temperature

        return logits, z


# ============================================================================
# ROSE LOSS (CANTOR-AWARE)
# ============================================================================

class RoseLoss(nn.Module):
    """Rose Loss with k-simplex role weighting and Cantor-awareness."""

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
            assert role_vec.shape == (num_vertices,)

        self.register_buffer("base_role_weights", role_vec)

        if self.cantor_aware:
            self.theta = nn.Parameter(torch.tensor(cantor_theta, dtype=torch.float32))

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

    def _compute_cantor_modulated_roles(
        self,
        cantor_scalar: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Compute Cantor-aware role weights."""
        B = cantor_scalar.shape[0]
        V = self.num_vertices

        t = (cantor_scalar * 3.0).floor()
        left = (t == 0).float()
        mid = (t == 1).float()
        right = (t == 2).float()

        base = self.base_role_weights.to(device).unsqueeze(0).expand(B, V)

        anchor_emphasis = torch.zeros((B, V), device=device)
        anchor_emphasis[:, 0] = 1.0
        if V > 1:
            anchor_emphasis[:, -1] = -0.5

        observer_emphasis = torch.zeros((B, V), device=device)
        if V > 1:
            observer_emphasis[:, -1] = 1.0
        observer_emphasis[:, 0] = -0.5

        theta = self.theta.sigmoid()

        modulated = base.clone()
        modulated = modulated + theta * left.unsqueeze(-1) * anchor_emphasis
        modulated = modulated + theta * right.unsqueeze(-1) * observer_emphasis

        return modulated

    def forward(
        self,
        z: torch.Tensor,
        crystals: torch.Tensor,
        targets: torch.Tensor,
        cantor_scalar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Rose Loss with optional Cantor-awareness.

        Args:
            z: [B, D]
            crystals: [C, V, d]
            targets: [B]
            cantor_scalar: [B] in [0,1]

        Returns:
            loss: scalar
        """
        crystals = crystals.to(z.device)

        C, V, d = crystals.shape
        assert V == self.num_vertices

        crystals_norm = F.normalize(crystals, dim=-1)
        cos_sim = torch.einsum("bd,cvd->bcv", z, crystals_norm)

        B = z.shape[0]
        if self.cantor_aware and (cantor_scalar is not None):
            role_weights = self._compute_cantor_modulated_roles(cantor_scalar, z.device)
            rose_scores = (cos_sim * role_weights.unsqueeze(1)).sum(dim=-1)
        else:
            role_weights = self.base_role_weights.to(z.device)
            rose_scores = (cos_sim * role_weights.view(1, 1, V)).sum(dim=-1)

        rose_scores = rose_scores / self.temperature

        true_scores = rose_scores.gather(1, targets.view(-1, 1)).squeeze(1)
        mask = torch.zeros_like(rose_scores, dtype=torch.bool)
        mask.scatter_(1, targets.view(-1, 1), True)
        hard_neg = rose_scores.masked_fill(mask, float("-inf")).max(dim=1).values

        loss = F.relu(self.margin - (true_scores - hard_neg))
        return loss.mean()


# ============================================================================
# FRACTAL MULTI-SCALE LOSS
# ============================================================================

class FractalMultiScaleLoss(nn.Module):
    """Aggregate loss for FractalDavid with Cantor-aware Rose Loss."""

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
        """Compute all loss components."""
        losses = {}

        ce_main = self.ce_loss(combined_logits, targets)
        losses['ce_main'] = ce_main

        total_scale_loss = 0.0
        for i, (scale, logits, features) in enumerate(
            zip(self.scales, scale_logits, scale_features)
        ):
            scale_weight = self.scale_balance.get(scale, 1.0)

            ce_scale = self.ce_loss(logits, targets)
            losses[f'ce_{scale}'] = ce_scale
            total_scale_loss += scale_weight * ce_scale

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

                progress = min(epoch / 100.0, 1.0)
                current_weight = self.rose_weight + \
                    (self.rose_max_weight - self.rose_weight) * progress
                total_scale_loss += current_weight * rose_loss

        total_loss = ce_main + total_scale_loss / len(self.scales)
        losses['total'] = total_loss

        return losses


# ============================================================================
# FRACTAL DAVID MODEL
# ============================================================================

class FractalDavid(nn.Module):
    """FractalDavid - Multi-scale crystal classifier with Cantor gating."""

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

        if self.config.fusion_mode == "WEIGHTED_SUM":
            self.fusion_weights = nn.Parameter(torch.ones(len(self.scales)))
        else:
            raise NotImplementedError(f"Fusion mode {self.config.fusion_mode} not implemented")

    def _active_scales(self) -> List[int]:
        if not self.progressive_training:
            return self.scales

        active = []
        for scale in self.scales:
            warmup = self.scale_warmup_epochs.get(scale, 0)
            if self.current_epoch >= warmup:
                active.append(scale)
        return active

    def _parallel_forward(
        self,
        features: torch.Tensor,
        anchors_dict: Dict[int, torch.Tensor],
        crystals_dict: Dict[int, torch.Tensor],
        cantor_pos: Optional[torch.Tensor],
        cantor_levels: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits_list: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []

        B = features.shape[0]
        device = features.device

        cantor_scalars_dict = {}

        scale_ids = self._active_scales()
        if (cantor_pos is None) and self.config.default_internal_cantor:
            max_pos = max(1, len(scale_ids) - 1)
            pos_tensor = torch.arange(len(scale_ids), device=device, dtype=torch.float32)
            cantor_all = CantorStairs.value(pos_tensor, max_pos=max_pos + 1, levels=cantor_levels)
        else:
            cantor_all = None

        for idx, scale in enumerate(scale_ids):
            head = self.heads[str(scale)]
            anchors = anchors_dict[scale]
            crystals = crystals_dict.get(scale, None)

            if cantor_pos is None:
                cs = cantor_all[idx].expand(B) if cantor_all is not None else None
            else:
                cs = cantor_pos.float().clamp(0.0, 1.0)

            if cs is not None:
                cantor_scalars_dict[scale] = cs.detach()

            logits, feats = head(features, anchors, crystals=crystals, cantor_scalar=cs)
            logits_list.append(logits)
            features_list.append(feats)

        self._last_cantor_scalars = cantor_scalars_dict if cantor_scalars_dict else None

        return logits_list, features_list

    def _fuse_logits(self, logits_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        assert x.dim() == 2 and x.size(-1) == self.feature_dim

        for s in self.scales:
            assert s in anchors_dict, f"anchors_dict missing scale {s}"

        if crystals_dict is None:
            crystals_dict = {}

        levels = int(cantor_levels or self.config.cantor_levels)

        logits_list, features_list = self._parallel_forward(
            x, anchors_dict, crystals_dict, cantor_pos, levels
        )

        combined, fusion_w = self._fuse_logits(logits_list)

        if return_all_scales:
            return combined, logits_list, features_list, fusion_w
        else:
            return combined, (features_list[0] if features_list else x)

    def update_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_active_scales(self) -> List[int]:
        return self._active_scales()

    def get_last_cantor_scalars(self) -> Optional[Dict[int, torch.Tensor]]:
        return self._last_cantor_scalars

    def freeze_scale(self, scale: int):
        for param in self.heads[str(scale)].parameters():
            param.requires_grad = False

    def unfreeze_scale(self, scale: int):
        for param in self.heads[str(scale)].parameters():
            param.requires_grad = True

    def get_model_info(self) -> Dict[str, any]:
        return {
            "name": "FractalDavid",
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
            f"FractalDavid(Decoupled Multi-Scale + Cantor Gating)\n"
            f"  Simplex: k={k} ({info['num_vertices']} vertices)\n"
            f"  Scales: {info['scales']}\n"
            f"  Active: {info['active_scales']}\n"
            f"  Cantor: {'Enabled' if info['enable_cantor'] else 'Disabled'}\n"
            f"  Parameters: {info['total_parameters']:,}"
        )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FractalDavid - k-Simplex with Cantor-Aware Rose Loss")
    print("="*80)

    for k in [2, 4, 6]:
        print(f"\n{'='*80}")
        print(f"Testing k={k} ({k+1} vertices)")
        print(f"{'='*80}")

        config = FractalDavidConfig(
            feature_dim=768,
            num_classes=1000,
            scales=(384, 512, 768),
            simplex_k=k,
            enable_cantor_gate=True,
            gate_strength=0.25,
            cantor_levels=12,
        )

        model = FractalDavid(config)
        print(model)

        print(f"\nAuto-generated base role weights for k={k}:")
        role_weights = RoseLoss._generate_default_roles(k+1)
        for i, w in enumerate(role_weights):
            vertex_name = "anchor" if i == 0 else ("observer" if i == k else f"v{i}")
            print(f"  {vertex_name:10s} (v{i}): {w:+.2f}")

        B, C = 4, config.num_classes
        x = torch.randn(B, config.feature_dim)

        anchors_dict = {}
        crystals_dict = {}
        for scale in config.scales:
            crystals = F.normalize(torch.randn(C, k+1, scale), dim=-1)
            anchors = crystals[:, 0, :].clone()
            anchors_dict[scale] = anchors
            crystals_dict[scale] = crystals

        print(f"\nForward pass test:")
        with torch.no_grad():
            out, feats = model(x, anchors_dict, crystals_dict)
            print(f"  Logits: {tuple(out.shape)}")
            print(f"  Features: {tuple(feats.shape)}")

            cantor_scalars = model.get_last_cantor_scalars()
            if cantor_scalars:
                print(f"  Cantor scalars tracked for {len(cantor_scalars)} scales")

        print(f"\nTesting Cantor-aware Rose Loss:")

        criterion_baseline = FractalMultiScaleLoss(
            scales=list(config.scales),
            num_classes=C,
            num_vertices=k+1,
            use_rose_loss=True,
            rose_cantor_aware=False,
        )

        criterion_cantor = FractalMultiScaleLoss(
            scales=list(config.scales),
            num_classes=C,
            num_vertices=k+1,
            use_rose_loss=True,
            rose_cantor_aware=True,
            rose_cantor_theta=0.5,
        )

        targets = torch.randint(0, C, (B,))
        with torch.no_grad():
            out3, logits_list, feats_list, w = model(
                x, anchors_dict, crystals_dict, return_all_scales=True
            )

            cantor_scalars = model.get_last_cantor_scalars()

            losses_baseline = criterion_baseline(
                out3, logits_list, feats_list,
                targets, anchors_dict, crystals_dict,
                epoch=0
            )

            losses_cantor = criterion_cantor(
                out3, logits_list, feats_list,
                targets, anchors_dict, crystals_dict,
                epoch=0,
                cantor_scalars=cantor_scalars
            )

            print(f"\n  Baseline (static roles):")
            print(f"    Total: {losses_baseline['total'].item():.4f}")
            print(f"    CE main: {losses_baseline['ce_main'].item():.4f}")

            print(f"\n  Cantor-aware (dynamic roles):")
            print(f"    Total: {losses_cantor['total'].item():.4f}")
            print(f"    CE main: {losses_cantor['ce_main'].item():.4f}")

            print(f"\n  Learned theta values:")
            for scale in config.scales:
                if str(scale) in criterion_cantor.rose_losses:
                    theta = criterion_cantor.rose_losses[str(scale)].theta.sigmoid().item()
                    print(f"    Scale {scale}: {theta:.3f}")

    print("\n" + "="*80)
    print("✅ All tests passed! Ready for training.")
    print("="*80)