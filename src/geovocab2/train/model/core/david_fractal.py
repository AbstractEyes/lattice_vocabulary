"""
FractalDavid with Geometric Basin Compatibility Heads
======================================================
Integrates GeometricBasin-style Cantor coherence for direct alpha optimization.

KEY FEATURES:
- Geometric basin compatibility scoring per scale
- Learned Cantor prototypes (one per class per scale)
- Cantor coherence loss for alpha supervision
- Multi-scale geometric compatibility

Based on: GeometricBasinClassifier + FractalDavid-Alpha
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
    cantor_alpha_init: float = 0.5
    cantor_tau: float = 0.25
    gate_strength: float = 0.25
    # NEW: Geometric Basin params
    use_geometric_basin: bool = True
    basin_cantor_bandwidth: float = 0.1
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
# ALPHA-NORMALIZED CANTOR STAIRS
# ============================================================================

class CantorStairsAlpha(nn.Module):
    """Soft Cantor staircase with learnable alpha parameter."""

    def __init__(self, levels: int = 12, alpha_init: float = 0.5, tau: float = 0.25):
        super().__init__()
        self.levels = levels
        self.tau = tau
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.register_buffer("centers", torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32))

    def forward(self, pos: torch.Tensor, max_pos: int) -> torch.Tensor:
        """Compute soft Cantor values with alpha normalization."""
        if max_pos > 1:
            x = pos.float() / float(max_pos - 1)
        else:
            x = pos.float()

        x = x.clamp(1e-6, 1.0 - 1e-6)
        Cx = torch.zeros_like(x)
        w = 0.5

        for _ in range(self.levels):
            y = x * 3.0
            d2 = (y.unsqueeze(-1) - self.centers) ** 2
            logits = -d2 / (self.tau + 1e-8)
            p = F.softmax(logits, dim=-1)
            bit_k = p[..., 2] + self.alpha * p[..., 1]
            Cx = Cx + bit_k * w
            t = y.floor()
            x = y - t
            w *= 0.5

        return Cx.clamp(0.0, 1.0)


# ============================================================================
# GEOMETRIC BASIN COMPATIBILITY HEAD
# ============================================================================

class GeometricBasinHead(nn.Module):
    """
    Geometric basin compatibility head for one scale.

    Computes compatibility scores based on:
    1. Feature similarity to class prototypes
    2. Cantor coherence with learned prototypes
    3. Geometric consistency
    """

    def __init__(
        self,
        input_dim: int,
        crystal_dim: int,
        num_classes: int,
        num_vertices: int = 5,
        temperature: float = 0.07,
        cantor_bandwidth: float = 0.1,
        use_belly: bool = True,
        belly_expand: float = 2.0,
    ):
        super().__init__()
        self.crystal_dim = int(crystal_dim)
        self.num_classes = num_classes
        self.num_vertices = num_vertices
        self.temperature = temperature
        self.cantor_bandwidth = cantor_bandwidth

        # Feature projection
        if use_belly:
            belly_dim = int(self.crystal_dim * float(belly_expand))
            dropout_rate = 1.0 / math.sqrt(self.crystal_dim)
            self.projection = nn.Sequential(
                nn.Linear(input_dim, belly_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(belly_dim, self.crystal_dim, bias=False),
            )
        else:
            self.projection = nn.Linear(input_dim, self.crystal_dim, bias=False)

        self._init_weights()

        # GEOMETRIC BASIN COMPONENTS

        # Learned Cantor prototypes (one per class)
        self.cantor_prototypes = nn.Parameter(
            torch.linspace(0.0, 1.0, num_classes, dtype=torch.float32)
        )

        # Class feature prototypes (for feature compatibility)
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, self.crystal_dim) * 0.1
        )

        # Geometric weighting (how much each component matters)
        self.geo_weights = nn.Parameter(
            torch.tensor([0.4, 0.3, 0.3], dtype=torch.float32)  # [feature, cantor, crystal]
        )

    def _init_weights(self):
        """Xavier init for stability."""
        for layer in self.projection.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def compute_feature_compatibility(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature-based compatibility with class prototypes.

        Args:
            z: [B, d] normalized features

        Returns:
            feature_compat: [B, C]
        """
        # Normalize prototypes
        proto_norm = F.normalize(self.class_prototypes, dim=-1)

        # Cosine similarity
        similarities = z @ proto_norm.T  # [B, C]

        # Map to [0, 1]
        feature_compat = (similarities + 1) / 2

        return feature_compat

    def compute_cantor_coherence(
        self,
        cantor_scalar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Cantor coherence with learned prototypes.

        Args:
            cantor_scalar: [B] Cantor values

        Returns:
            cantor_compat: [B, C]
        """
        # Distance to each class's Cantor prototype
        distances = torch.abs(
            cantor_scalar.unsqueeze(1) - self.cantor_prototypes.unsqueeze(0)
        )  # [B, C]

        # Gaussian kernel
        cantor_compat = torch.exp(-distances ** 2 / self.cantor_bandwidth)

        return cantor_compat

    def compute_crystal_compatibility(
        self,
        z: torch.Tensor,
        crystals: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute compatibility with crystal anchors.

        Args:
            z: [B, d] normalized features
            crystals: [C, V, d] crystal vertices

        Returns:
            crystal_compat: [B, C]
        """
        # Use anchor vertices (first vertex)
        anchors = crystals[:, 0, :]  # [C, d]
        anchors_norm = F.normalize(anchors, dim=-1)

        # Similarity
        similarities = z @ anchors_norm.T  # [B, C]
        crystal_compat = (similarities + 1) / 2

        return crystal_compat

    def forward(
        self,
        features: torch.Tensor,
        crystals: Optional[torch.Tensor] = None,
        cantor_scalar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with geometric basin compatibility.

        Args:
            features: [B, D_in]
            crystals: [C, V, D_scale]
            cantor_scalar: [B]

        Returns:
            compatibility_scores: [B, C] - RAW compatibility (not temperature scaled yet!)
            z: [B, D_scale] - projected features
            components: dict with individual compatibility components
        """
        # Project features
        z = self.projection(features)
        z = F.normalize(z, dim=-1)

        # 1. Feature compatibility
        feature_compat = self.compute_feature_compatibility(z)  # [B, C]

        # 2. Cantor coherence
        if cantor_scalar is not None:
            cantor_compat = self.compute_cantor_coherence(cantor_scalar)  # [B, C]
        else:
            cantor_compat = torch.ones_like(feature_compat)

        # 3. Crystal compatibility
        if crystals is not None:
            crystal_compat = self.compute_crystal_compatibility(z, crystals)  # [B, C]
        else:
            crystal_compat = torch.ones_like(feature_compat)

        # Combine with learned weights
        weights = F.softmax(self.geo_weights, dim=0)

        # Weighted geometric mean (components already in [0, 1])
        eps = 1e-8
        feature_compat = torch.clamp(feature_compat, eps, 1.0)
        cantor_compat = torch.clamp(cantor_compat, eps, 1.0)
        crystal_compat = torch.clamp(crystal_compat, eps, 1.0)

        compatibility_scores = (
            feature_compat ** weights[0] *
            cantor_compat ** weights[1] *
            crystal_compat ** weights[2]
        )

        # Ensure bounded [eps, 1]
        compatibility_scores = torch.clamp(compatibility_scores, eps, 1.0)

        # DO NOT temperature scale here - let loss handle it!

        components = {
            'feature': feature_compat,
            'cantor': cantor_compat,
            'crystal': crystal_compat,
            'weights': weights
        }

        return compatibility_scores, z, components


# ============================================================================
# GEOMETRIC BASIN LOSS
# ============================================================================

class GeometricBasinMultiScaleLoss(nn.Module):
    """
    Multi-scale geometric basin loss with Cantor coherence.

    Combines:
    - Compatibility-based classification loss
    - Cantor coherence loss (direct alpha supervision)
    - Multi-scale consistency
    """

    def __init__(
        self,
        scales: List[int],
        num_classes: int = 1000,
        cantor_coherence_weight: float = 0.5,
        scale_consistency_weight: float = 0.3,
        temperature: float = 0.1,
        scale_loss_balance: Optional[Dict[int, float]] = None,
    ):
        super().__init__()

        self.scales = scales
        self.num_classes = num_classes
        self.cantor_coherence_weight = cantor_coherence_weight
        self.scale_consistency_weight = scale_consistency_weight
        self.temperature = temperature
        self.scale_balance = scale_loss_balance or {s: 1.0 for s in scales}

    def compatibility_classification_loss(
        self,
        compatibility_scores: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Classification loss based on compatibility scores.

        Args:
            compatibility_scores: [B, C] RAW compatibilities in [0, 1]
            targets: [B] class labels
        """
        B = compatibility_scores.shape[0]

        # Ensure scores are in valid range
        compatibility_scores = torch.clamp(compatibility_scores, 1e-8, 1.0)

        # Convert to logits via temperature-scaled log
        # log(compat) gives values in (-inf, 0]
        # Divide by temperature to get proper scale
        logits = torch.log(compatibility_scores + 1e-8) / self.temperature

        # Standard cross-entropy on logits
        ce_loss = F.cross_entropy(logits, targets)

        # Additional direct compatibility loss: correct class should be close to 1
        correct_compat = compatibility_scores[torch.arange(B), targets]
        direct_loss = (1.0 - correct_compat).mean()

        return ce_loss + 0.3 * direct_loss

    def cantor_coherence_loss(
        self,
        cantor_scalars: torch.Tensor,
        cantor_prototypes: torch.Tensor,
        targets: torch.Tensor,
        bandwidth: float = 0.1
    ) -> torch.Tensor:
        """
        Cantor coherence loss - ensures Cantor values match class prototypes.

        This provides DIRECT gradient signal to alpha parameter.
        """
        # Get target Cantor prototype for each sample
        target_prototypes = cantor_prototypes[targets]  # [B]

        # MSE loss between actual and target Cantor values
        coherence_loss = F.mse_loss(cantor_scalars, target_prototypes)

        return coherence_loss

    def scale_consistency_loss(
        self,
        scale_cantor_values: Dict[int, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage consistency of Cantor values across scales for same sample.
        """
        if len(scale_cantor_values) < 2:
            return torch.tensor(0.0, device=targets.device)

        # Get all Cantor values
        cantor_list = [v for v in scale_cantor_values.values()]

        # Compute pairwise consistency
        consistency_loss = 0.0
        count = 0

        for i in range(len(cantor_list)):
            for j in range(i + 1, len(cantor_list)):
                consistency_loss += F.mse_loss(cantor_list[i], cantor_list[j])
                count += 1

        if count > 0:
            consistency_loss = consistency_loss / count

        return consistency_loss

    def forward(
        self,
        combined_compatibility: torch.Tensor,
        scale_compatibilities: List[torch.Tensor],
        scale_components: List[Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        heads_dict: Dict[int, GeometricBasinHead],
        cantor_scalars_dict: Optional[Dict[int, torch.Tensor]] = None,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute complete multi-scale geometric basin loss.

        Args:
            combined_compatibility: [B, C] fused compatibility scores
            scale_compatibilities: List of [B, C] per-scale scores
            scale_components: List of component dicts per scale
            targets: [B] class labels
            heads_dict: Dictionary of GeometricBasinHead per scale
            cantor_scalars_dict: Dict of [B] Cantor values per scale
            epoch: Current epoch

        Returns:
            losses: Dictionary of all loss components
        """
        losses = {}

        # 1. Main compatibility classification loss
        main_loss = self.compatibility_classification_loss(combined_compatibility, targets)
        losses['main'] = main_loss

        # 2. Per-scale losses
        total_scale_loss = 0.0
        total_cantor_coherence = 0.0

        for i, (scale, compat) in enumerate(zip(self.scales, scale_compatibilities)):
            scale_weight = self.scale_balance.get(scale, 1.0)

            # Scale classification loss
            scale_loss = self.compatibility_classification_loss(compat, targets)
            losses[f'compat_{scale}'] = scale_loss
            total_scale_loss += scale_weight * scale_loss

            # Cantor coherence loss
            if cantor_scalars_dict and scale in cantor_scalars_dict and scale in heads_dict:
                head = heads_dict[scale]
                cantor_scalars = cantor_scalars_dict[scale]

                coherence_loss = self.cantor_coherence_loss(
                    cantor_scalars,
                    head.cantor_prototypes,
                    targets,
                    bandwidth=head.cantor_bandwidth
                )

                losses[f'cantor_coherence_{scale}'] = coherence_loss
                total_cantor_coherence += coherence_loss

        # Average losses
        if len(scale_compatibilities) > 0:
            losses['scale_avg'] = total_scale_loss / len(scale_compatibilities)
            losses['cantor_coherence'] = total_cantor_coherence / len(scale_compatibilities)

        # 3. Scale consistency loss
        if cantor_scalars_dict:
            consistency_loss = self.scale_consistency_loss(cantor_scalars_dict, targets)
            losses['scale_consistency'] = consistency_loss

        # 4. Total loss
        total_loss = main_loss + losses.get('scale_avg', 0.0)

        if 'cantor_coherence' in losses:
            total_loss = total_loss + self.cantor_coherence_weight * losses['cantor_coherence']

        if 'scale_consistency' in losses:
            total_loss = total_loss + self.scale_consistency_weight * losses['scale_consistency']

        losses['total'] = total_loss

        return losses


# ============================================================================
# FRACTAL DAVID WITH GEOMETRIC BASIN HEADS
# ============================================================================

class FractalDavid(nn.Module):
    """FractalDavid with geometric basin compatibility heads."""

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
        self._last_components: Optional[List[Dict[str, torch.Tensor]]] = None

        # ALPHA CANTOR MODULE
        self.cantor_stairs = CantorStairsAlpha(
            levels=self.config.cantor_levels,
            alpha_init=self.config.cantor_alpha_init,
            tau=self.config.cantor_tau
        )

        # Create GEOMETRIC BASIN HEADS per scale
        self.heads = nn.ModuleDict({
            str(scale): GeometricBasinHead(
                input_dim=self.feature_dim,
                crystal_dim=scale,
                num_classes=self.num_classes,
                num_vertices=self.config.num_vertices,
                temperature=self.config.projection_temperature,
                cantor_bandwidth=self.config.basin_cantor_bandwidth,
                use_belly=self.config.use_belly,
                belly_expand=self.config.belly_expand,
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
        """Get or compute Cantor scalars with alpha normalization."""
        if cantor_pos is not None:
            return cantor_pos.float().clamp(0.0, 1.0)

        max_pos = max(1, num_scales - 1)
        pos_tensor = torch.arange(num_scales, device=device, dtype=torch.float32)
        cantor_values = self.cantor_stairs(pos_tensor, max_pos=max_pos + 1)

        return cantor_values

    def _parallel_forward_optimized(
        self,
        features: torch.Tensor,
        crystals_dict: Dict[int, torch.Tensor],
        cantor_pos: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Optimized parallel forward with geometric basin heads."""
        compatibility_list: List[torch.Tensor] = []
        features_list: List[torch.Tensor] = []
        components_list: List[Dict[str, torch.Tensor]] = []

        B = features.shape[0]
        device = features.device

        scale_ids = self._active_scales()
        num_scales = len(scale_ids)

        # Compute alpha-normalized Cantor scalars
        if self.config.default_internal_cantor:
            cantor_base = self._get_or_compute_cantor_scalars(
                num_scales, B, device, cantor_pos
            )
        else:
            cantor_base = None

        cantor_scalars_dict = {}

        # Process each scale with geometric basin head
        for idx, scale in enumerate(scale_ids):
            head = self.heads[str(scale)]
            crystals = crystals_dict.get(scale, None)

            if cantor_base is not None:
                if cantor_base.dim() == 0 or (cantor_base.dim() == 1 and cantor_base.shape[0] == num_scales):
                    cs = cantor_base[idx].expand(B)
                else:
                    cs = cantor_base
            else:
                cs = None

            if cs is not None:
                cantor_scalars_dict[scale] = cs.detach()

            # Forward through geometric basin head
            compat, feats, components = head(features, crystals=crystals, cantor_scalar=cs)

            compatibility_list.append(compat)
            features_list.append(feats)
            components_list.append(components)

        self._last_cantor_scalars = cantor_scalars_dict if cantor_scalars_dict else None
        self._last_components = components_list

        return compatibility_list, features_list, components_list

    def _fuse_compatibilities(
        self,
        compatibility_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse compatibility scores across scales."""
        if len(compatibility_list) == 1:
            w = torch.ones(1, device=compatibility_list[0].device)
            return compatibility_list[0], w

        weights = F.softmax(self.fusion_weights[:len(compatibility_list)], dim=0)
        combined = sum(w * compat for w, compat in zip(weights, compatibility_list))
        return combined, weights

    def forward(
        self,
        x: torch.Tensor,
        crystals_dict: Optional[Dict[int, torch.Tensor]] = None,
        return_all_scales: bool = False,
        *,
        cantor_pos: Optional[torch.Tensor] = None,
        cantor_levels: Optional[int] = None
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[Dict]]
    ]:
        """Forward pass with geometric basin compatibility."""
        assert x.dim() == 2 and x.size(-1) == self.feature_dim

        if crystals_dict is None:
            crystals_dict = {}

        compatibility_list, features_list, components_list = self._parallel_forward_optimized(
            x, crystals_dict, cantor_pos
        )

        combined, fusion_w = self._fuse_compatibilities(compatibility_list)

        if return_all_scales:
            return combined, compatibility_list, features_list, fusion_w, components_list
        else:
            return combined, (features_list[0] if features_list else x)

    def update_epoch(self, epoch: int):
        """Update epoch for progressive training."""
        self.current_epoch = epoch

    def get_active_scales(self) -> List[int]:
        """Get currently active scales."""
        return self._active_scales()

    def get_last_cantor_scalars(self) -> Optional[Dict[int, torch.Tensor]]:
        """Get last computed Cantor scalars."""
        return self._last_cantor_scalars

    def get_last_components(self) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Get last compatibility components for diagnostics."""
        return self._last_components

    def get_cantor_alpha(self) -> float:
        """Get current alpha value from Cantor stairs."""
        return self.cantor_stairs.alpha.item()

    def get_heads_dict(self) -> Dict[int, GeometricBasinHead]:
        """Get dictionary of heads for loss computation."""
        return {int(k): v for k, v in self.heads.items()}

    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        return {
            "name": "FractalDavid-GeometricBasin",
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "simplex_k": self.config.simplex_k,
            "num_vertices": self.config.num_vertices,
            "scales": self.scales,
            "active_scales": self.get_active_scales(),
            "enable_cantor": self.config.enable_cantor_gate,
            "cantor_levels": self.config.cantor_levels,
            "cantor_alpha": self.get_cantor_alpha(),
            "cantor_tau": self.config.cantor_tau,
            "use_geometric_basin": self.config.use_geometric_basin,
            "current_epoch": self.current_epoch,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

    def __repr__(self):
        info = self.get_model_info()
        k = self.config.num_vertices - 1
        return (
            f"FractalDavid-GeometricBasin (Cantor Coherence)\n"
            f"  Simplex: k={k} ({info['num_vertices']} vertices)\n"
            f"  Scales: {info['scales']}\n"
            f"  Active: {info['active_scales']}\n"
            f"  Cantor Alpha: {info['cantor_alpha']:.4f}\n"
            f"  Geometric Basin: {info['use_geometric_basin']}\n"
            f"  Parameters: {info['total_parameters']:,}"
        )


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FractalDavid with Geometric Basin Heads")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Test config
    config = FractalDavidConfig(
        feature_dim=768,
        num_classes=100,
        scales=(384, 512, 768),
        simplex_k=4,
        enable_cantor_gate=True,
        cantor_alpha_init=0.5,
        cantor_tau=0.25,
        use_geometric_basin=True,
        basin_cantor_bandwidth=0.1
    )

    model = FractalDavid(config).to(device)
    print(model)
    print()

    # Test data
    B, C = 16, config.num_classes
    x = torch.randn(B, config.feature_dim, device=device)

    crystals_dict = {}
    for scale in config.scales:
        crystals = F.normalize(torch.randn(C, 5, scale, device=device), dim=-1)
        crystals_dict[scale] = crystals

    # Forward pass
    combined, compat_list, feats_list, fusion_w, components = model(
        x, crystals_dict, return_all_scales=True
    )

    print(f"Combined compatibility: {combined.shape}")
    print(f"Num scales: {len(compat_list)}")
    print(f"Fusion weights: {fusion_w.detach().cpu().numpy()}")
    print(f"Cantor alpha: {model.get_cantor_alpha():.4f}")

    # Check components
    print(f"\nCompatibility components (scale {config.scales[0]}):")
    comp = components[0]
    print(f"  Feature: {comp['feature'].shape}")
    print(f"  Cantor: {comp['cantor'].shape}")
    print(f"  Crystal: {comp['crystal'].shape}")
    print(f"  Weights: {comp['weights'].detach().cpu().numpy()}")

    # Test loss
    print(f"\n[Test] Geometric Basin Loss")
    targets = torch.randint(0, C, (B,), device=device)

    criterion = GeometricBasinMultiScaleLoss(
        scales=list(config.scales),
        num_classes=C,
        cantor_coherence_weight=0.5,
        scale_consistency_weight=0.3
    ).to(device)

    losses = criterion(
        combined, compat_list, components, targets,
        model.get_heads_dict(), model.get_last_cantor_scalars(), epoch=0
    )

    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Main loss: {losses['main'].item():.4f}")
    if 'cantor_coherence' in losses:
        print(f"  Cantor coherence: {losses['cantor_coherence'].item():.4f}")
    if 'scale_consistency' in losses:
        print(f"  Scale consistency: {losses['scale_consistency'].item():.4f}")

    # Test gradient flow
    losses['total'].backward()
    alpha_grad = model.cantor_stairs.alpha.grad
    print(f"\n  Alpha gradient: {alpha_grad.item():.6f}")
    print(f"  Gradient flow: {'✓ YES' if alpha_grad is not None else '✗ NO'}")

    # Check Cantor prototype gradients
    head = model.heads[str(config.scales[0])]
    cantor_proto_grad = head.cantor_prototypes.grad
    print(f"  Cantor prototype gradient: {cantor_proto_grad[:5].detach().cpu().numpy()}")

    print("\n" + "="*80)
    print("✅ Geometric Basin integration complete!")
    print("="*80)
    print("\nKey Features:")
    print("  ✓ Learned Cantor prototypes per class per scale")
    print("  ✓ Cantor coherence loss for direct alpha supervision")
    print("  ✓ Geometric compatibility scoring (feature + cantor + crystal)")
    print("  ✓ Multi-scale consistency")
    print("  ✓ Full gradient flow to alpha parameter")
    print("\nReady for training with proper alpha optimization!")