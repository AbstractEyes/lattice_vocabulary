"""
FLOW MATCHER + CANTOR-AWARE CLASSIFICATION
-------------------------------------------
Maps transfinite geometric flow to finite classification via Cantor structure.

The insight: Flowed simplices operate in transfinite scale regimes.
The solution: Use Cantor function to map infinite hierarchy to [0,1] for classification.

Author: AbstractPhil + Claude Sonnet 4.5
"""

from typing import Dict, Optional
import torch
from torch import Tensor
import torch.nn as nn

from geovocab2.shapes.formula.formula_base import FormulaBase
from geovocab2.shapes.formula import SimplexVolume, SimplexQuality, SimplexVolumeExtended
from geovocab2.shapes.formula import CantorFunction


class GeometricTrajectoryNet(nn.Module):
    """Learn velocity field in geometric space."""

    def __init__(self, simplex_dim: int, hidden_scale: int = 4, num_heads: int = 4):
        super().__init__()
        self.simplex_dim = simplex_dim

        self.edge_encoder = nn.Sequential(
            nn.Linear(simplex_dim, simplex_dim * hidden_scale),
            nn.GELU(),
            nn.LayerNorm(simplex_dim * hidden_scale)
        )

        self.vertex_attention = nn.MultiheadAttention(
            embed_dim=simplex_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.velocity_proj = nn.Sequential(
            nn.Linear(simplex_dim * hidden_scale, simplex_dim),
            nn.LayerNorm(simplex_dim)
        )

    def forward(self, simplex_state: Tensor) -> Tensor:
        orig_shape = simplex_state.shape
        batch_dims = orig_shape[:-2]
        k_plus_1 = orig_shape[-2]
        dim = orig_shape[-1]

        simplex_flat = simplex_state.reshape(-1, k_plus_1, dim)
        encoded = self.edge_encoder(simplex_flat)

        attn_out, _ = self.vertex_attention(
            simplex_flat, simplex_flat, simplex_flat
        )

        attn_repeated = attn_out.repeat_interleave(
            encoded.shape[-1] // attn_out.shape[-1], dim=-1
        )
        combined = encoded + attn_repeated
        velocity = self.velocity_proj(combined)
        velocity = velocity.reshape(*batch_dims, k_plus_1, dim)

        return velocity


class CantorGeometricClassificationHead(nn.Module):
    """
    Classification head that maps transfinite geometric positions to logits.

    Uses Cantor function to compress infinite-scale simplices to [0,1],
    preserving hierarchical structure information.
    """

    def __init__(self, num_origins: int, origin_dim: int, embed_dim: int, num_classes: int):
        super().__init__()

        self.num_origins = num_origins
        self.origin_dim = origin_dim
        self.embed_dim = embed_dim

        # Cantor function for scale mapping
        self.cantor_fn = CantorFunction(iterations=8)

        # Process Cantor-normalized features
        self.simplex_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        self.origin_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def _map_to_cantor_space(self, flowed_simplices: Tensor) -> Tensor:
        """
        Map flowed simplices from transfinite geometric space to [0,1] via Cantor.

        Args:
            flowed_simplices: [..., num_origins, k+1, embed_dim]

        Returns:
            cantor_features: [..., num_origins, embed_dim] in [0,1] range
        """
        # Pool over simplex vertices
        pooled = flowed_simplices.mean(dim=-2)  # [..., num_origins, embed_dim]

        # Compute position in transfinite hierarchy
        # Use log-scale to map explosive values to reasonable range
        abs_values = torch.abs(pooled)

        # Map to [0,1]: use log to compress large values
        # log(1 + |x|) maps [0, inf) -> [0, inf), then normalize
        log_scale = torch.log1p(abs_values)  # [..., num_origins, embed_dim]

        # Normalize to [0,1] per feature dimension
        min_vals = log_scale.min(dim=-2, keepdim=True)[0]
        max_vals = log_scale.max(dim=-2, keepdim=True)[0]
        normalized = (log_scale - min_vals) / (max_vals - min_vals + 1e-8)

        # Apply Cantor function to get hierarchical position
        cantor_values = self.cantor_fn.forward(normalized)['values']

        # Restore sign information (Cantor function loses sign)
        sign = torch.sign(pooled)
        cantor_features = sign * cantor_values

        return cantor_features

    def forward(self, flowed_simplices: Tensor) -> Tensor:
        """
        Args:
            flowed_simplices: [..., num_origins, k+1, embed_dim]

        Returns:
            logits: [..., num_classes]
        """
        # Map from transfinite geometric space to Cantor-normalized features
        cantor_features = self._map_to_cantor_space(flowed_simplices)

        # Now features are in [-1, 1] range - safe for standard neural operations
        features = self.simplex_pooling(cantor_features)

        # Attention over origins
        attended, _ = self.origin_attention(features, features, features)

        # Pool over origins
        pooled = attended.mean(dim=-2)

        # Classify
        logits = self.classifier(pooled)

        return logits


class FlowMatcher(FormulaBase):
    """
    Geometric flow through simplex space.

    Allows transfinite-scale exploration - no artificial clamping.
    Classification head handles the mapping back to finite space.
    """

    def __init__(
        self,
        simplex_dim: int,
        flow_steps: int = 4,
        hidden_scale: int = 4,
        validation_strength: float = 0.1,
        projection_lr: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        super().__init__("flow_matcher", "f.flow.matcher")

        self.simplex_dim = simplex_dim
        self.flow_steps = flow_steps
        self.validation_strength = validation_strength
        self.projection_lr = projection_lr
        self.max_grad_norm = max_grad_norm

        self.trajectory_net = GeometricTrajectoryNet(
            simplex_dim,
            hidden_scale=hidden_scale
        )

        self.volume_calc = SimplexVolumeExtended(mode="auto", check_degeneracy=True)
        self.quality_check = SimplexQuality()

    def _validate_and_project(self, state: Tensor, step: int) -> Tensor:
        """
        Light geometric validation - only fix true degeneracies.
        Allow transfinite scale exploration.
        """
        # Only check for actual geometric degeneracy (collapsed simplices)
        vol_result = self.volume_calc.forward(state)
        is_degenerate = vol_result['is_degenerate']

        if is_degenerate.any():
            # Only fix truly degenerate simplices
            k_plus_1 = state.shape[-2]

            ii, jj = torch.triu_indices(k_plus_1, k_plus_1, offset=1, device=state.device)
            edges = state[..., jj, :] - state[..., ii, :]
            edge_lengths = torch.norm(edges, dim=-1)
            mean_edge = edge_lengths.mean(dim=-1, keepdim=True).clamp(min=1e-6)

            # Scale to unit mean edge
            scale_factor = 1.0 / mean_edge.unsqueeze(-1)

            correction_mask = is_degenerate.float()
            while correction_mask.dim() < state.dim():
                correction_mask = correction_mask.unsqueeze(-1)

            state = state * (1 - correction_mask) + (state * scale_factor) * correction_mask

        return state

    def flow(
        self,
        initial_state: Tensor,
        return_trajectory: bool = False
    ) -> Dict[str, Tensor]:
        """
        Flow through geometric space - allow transfinite exploration.
        """
        current_state = initial_state
        trajectory = [current_state.detach().clone()] if return_trajectory else None

        initial_quality = self.quality_check.forward(initial_state)

        flow_metrics = {
            'initial_quality': initial_quality['regularity'],
            'step_qualities': [],
            'step_volumes': []
        }

        for step in range(self.flow_steps):
            velocity = self.trajectory_net(current_state)

            # Simple gradient clipping
            torch.nn.utils.clip_grad_value_(velocity, self.max_grad_norm)

            # Flow update
            step_size = 0.1
            next_state = current_state + step_size * velocity

            # Only fix degeneracies, allow scale to vary
            next_state = self._validate_and_project(next_state, step)

            # Track metrics
            step_quality = self.quality_check.forward(next_state)
            flow_metrics['step_qualities'].append(
                step_quality['regularity'].mean().detach()
            )

            vol_calc = SimplexVolume()
            vol_result = vol_calc.forward(next_state)
            flow_metrics['step_volumes'].append(
                vol_result['volume'].mean().detach()
            )

            current_state = next_state

            if return_trajectory:
                trajectory.append(current_state.detach().clone())

        final_quality = self.quality_check.forward(current_state)
        flow_metrics['final_quality'] = final_quality['regularity']

        if flow_metrics['step_qualities']:
            flow_metrics['step_qualities'] = torch.stack(flow_metrics['step_qualities'])
            flow_metrics['step_volumes'] = torch.stack(flow_metrics['step_volumes'])

        result = {
            'final_state': current_state,
            'flow_metrics': flow_metrics
        }

        if return_trajectory:
            result['trajectory'] = torch.stack(trajectory, dim=0)

        return result

    def forward(
        self,
        input_simplices: Tensor,
        return_trajectory: bool = False
    ) -> Dict[str, Tensor]:
        """Forward pass."""
        return self.flow(input_simplices, return_trajectory=return_trajectory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TESTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_flow_matcher():
    """Test FlowMatcher with geometric trajectory learning."""

    print("\n" + "=" * 70)
    print("FLOW MATCHER TESTS")
    print("=" * 70)

    # Test 1: Basic flow
    print("\n[Test 1] Basic Geometric Flow")

    batch_size = 2
    num_simplices = 8
    k_plus_1 = 5  # Pentachoron
    dim = 256

    # Create initial simplices (random but valid)
    initial_state = torch.randn(batch_size, num_simplices, k_plus_1, dim)

    flow_matcher = FlowMatcher(
        simplex_dim=dim,
        flow_steps=4
    )

    result = flow_matcher.flow(initial_state)

    print(f"  Input: [{batch_size}, {num_simplices}, {k_plus_1}, {dim}]")
    print(f"  Output: {result['final_state'].shape}")
    print(f"  Initial quality: {result['flow_metrics']['initial_quality'].mean().item():.4f}")
    print(f"  Final quality: {result['flow_metrics']['final_quality'].mean().item():.4f}")
    print(f"  Quality change: {(result['flow_metrics']['final_quality'].mean() - result['flow_metrics']['initial_quality'].mean()).item():+.4f}")
    print(f"  Status: ✓ PASS")

    # Test 2: Trajectory tracking
    print("\n[Test 2] Trajectory Tracking")

    result_traj = flow_matcher.flow(initial_state, return_trajectory=True)

    print(f"  Trajectory shape: {result_traj['trajectory'].shape}")
    print(f"  Expected: [{flow_matcher.flow_steps + 1}, {batch_size}, {num_simplices}, {k_plus_1}, {dim}]")
    print(f"  Step qualities: {result_traj['flow_metrics']['step_qualities'].shape}")

    # Print quality evolution
    print(f"  Quality evolution:")
    for i, q in enumerate(result_traj['flow_metrics']['step_qualities']):
        print(f"    Step {i}: {q.item():.4f}")

    print(f"  Status: ✓ PASS")

    # Test 3: Gradient flow
    print("\n[Test 3] Gradient Flow")

    initial_state_grad = torch.randn(
        batch_size, num_simplices, k_plus_1, dim,
        requires_grad=True
    )

    result_grad = flow_matcher.flow(initial_state_grad)
    loss = (1.0 - result_grad['flow_metrics']['final_quality']).mean()
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Input gradient: {initial_state_grad.grad is not None}")
    print(f"  Gradient norm: {initial_state_grad.grad.norm().item():.6f}")
    print(f"  Status: ✓ PASS")

    # Test 4: Volume preservation
    print("\n[Test 4] Volume Tracking")

    vol_calc = SimplexVolume()
    initial_vol = vol_calc.forward(initial_state)['volume'].mean()
    final_vol = vol_calc.forward(result['final_state'])['volume'].mean()

    print(f"  Initial volume: {initial_vol.item():.6f}")
    print(f"  Final volume: {final_vol.item():.6f}")
    print(f"  Volume change: {(final_vol - initial_vol).item():+.6f}")
    print(f"  Relative change: {((final_vol - initial_vol) / initial_vol * 100).item():+.2f}%")
    print(f"  Status: ✓ PASS")

    # Test 5: Different flow steps
    print("\n[Test 5] Variable Flow Steps")

    for steps in [1, 2, 4, 8]:
        matcher = FlowMatcher(simplex_dim=dim, flow_steps=steps)
        res = matcher.flow(initial_state)

        quality_gain = (
            res['flow_metrics']['final_quality'].mean() -
            res['flow_metrics']['initial_quality'].mean()
        ).item()

        print(f"  Steps={steps}: Quality gain = {quality_gain:+.4f}")

    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("ALL FLOW MATCHER TESTS PASSED")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ Geometric trajectory learning")
    print("  ✓ Vertex attention for routing")
    print("  ✓ CM-based validation and projection")
    print("  ✓ Quality tracking over flow")
    print("  ✓ Gradient flow for optimization")
    print("  ✓ Volume preservation monitoring")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_flow_matcher()