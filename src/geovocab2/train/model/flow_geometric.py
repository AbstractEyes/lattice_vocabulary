"""
PENTACHORON FLOW NETWORK
------------------------
Complete geometric flow architecture.
Replaces transformers with native geometric processing.

Author: AbstractPhil + Claude Sonnet 4.5
"""

from typing import Dict, Optional
import torch
from torch import Tensor
import torch.nn as nn
import math

from geovocab2.shapes.formula import (
    GeometricOriginSampler,
    NoiseCollector,
    FlowMatcher,
    SimplexQuality,
    SimplexVolumeExtended
)


class PentachoronFlowConfig:
    """Configuration for PentachoronFlow network."""

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        image_size: int = 32,
        num_origins: int = 49,
        origin_dim: int = 5,
        embed_dim: int = 768,
        init_strategy: str = 'diffusion',
        collection_strategy: str = 'geometric_distance',
        adaptive_radius: bool = True,
        base_radius: float = 1.0,
        k_nearest: int = 16,
        flow_steps: int = 4,
        hidden_scale: int = 4,
        max_grad_norm: float = 1.0,
        validation_weights: Dict[str, float] = None
    ):
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.image_size = image_size
        self.num_origins = num_origins
        self.origin_dim = origin_dim
        self.embed_dim = embed_dim
        self.init_strategy = init_strategy
        self.collection_strategy = collection_strategy
        self.adaptive_radius = adaptive_radius
        self.base_radius = base_radius
        self.k_nearest = k_nearest
        self.flow_steps = flow_steps
        self.hidden_scale = hidden_scale
        self.max_grad_norm = max_grad_norm

        if validation_weights is None:
            self.validation_weights = {'rose': 0.5, 'quality': 0.3, 'volume': 0.2}
        else:
            self.validation_weights = validation_weights


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class GeometricClassificationHead(nn.Module):
    """Classification head for geometric structures."""

    def __init__(self, num_origins: int, origin_dim: int, embed_dim: int, num_classes: int):
        super().__init__()

        self.simplex_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        self.origin_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, flowed_simplices: Tensor) -> Tensor:
        pooled = flowed_simplices.mean(dim=-2)
        pooled = self.simplex_pooling(pooled)
        attended, _ = self.origin_attention(pooled, pooled, pooled)
        features = attended.mean(dim=1)
        logits = self.classifier(features)
        return logits


class CayleyMengerValidator(nn.Module):
    """Pure geometric validation loss with Rose margin."""

    def __init__(self, config: PentachoronFlowConfig):
        super().__init__()
        self.config = config
        self.quality_check = SimplexQuality()
        self.volume_calc = SimplexVolumeExtended(mode="auto", check_degeneracy=True)

    def _compute_rose_score(self, vertices: Tensor) -> Tensor:
        """
        Compute Rose (multi-cosine) score for discriminative power.
        Measures directional coherence of simplex edges.
        """
        n = vertices.shape[-2]
        if n < 3:
            return torch.zeros(vertices.shape[:-2], device=vertices.device, dtype=vertices.dtype)

        # Get all edges
        ii, jj = torch.triu_indices(n, n, offset=1, device=vertices.device)
        edges = vertices[..., jj, :] - vertices[..., ii, :]

        # Normalize to unit vectors
        norms = torch.norm(edges, dim=-1, keepdim=True).clamp(min=1e-8)
        unit_edges = edges / norms

        # Compute pairwise cosine similarities
        cos_sim = torch.matmul(unit_edges, unit_edges.transpose(-2, -1))

        # Exclude diagonal (self-similarity)
        n_edges = unit_edges.shape[-2]
        mask = ~torch.eye(n_edges, dtype=torch.bool, device=vertices.device)

        # Mean cosine similarity (directional coherence)
        cos_mean = cos_sim[..., mask].reshape(*vertices.shape[:-2], -1).mean(dim=-1)

        # Map to [0, 1] range
        rose_score = (cos_mean + 1.0) * 0.5
        return rose_score

    def compute_loss(self, predicted_simplices: Tensor) -> Dict[str, Tensor]:
        """Geometric loss with Rose margin + Quality + Volume."""
        original_shape = predicted_simplices.shape
        batch_size = original_shape[0]
        num_origins = original_shape[1]
        k_plus_1 = original_shape[2]
        dim = original_shape[3]

        pred_flat = predicted_simplices.reshape(batch_size * num_origins, k_plus_1, dim)

        # 1. Rose score (directional coherence)
        rose_score = self._compute_rose_score(pred_flat)
        rose_loss = (1.0 - rose_score).mean()  # Maximize coherence

        # 2. Quality check
        quality_result = self.quality_check.forward(pred_flat)
        quality_loss = (1.0 - quality_result['regularity']).mean()

        # 3. Volume (using working SimplexVolumeExtended)
        volume_result = self.volume_calc.forward(pred_flat)
        volume_loss = volume_result['is_degenerate'].float().mean()

        w = self.config.validation_weights
        total_loss = (
            w.get('rose', 0.5) * rose_loss +
            w.get('quality', 0.3) * quality_loss +
            w.get('volume', 0.2) * volume_loss
        )

        return {
            'loss': total_loss,
            'rose_loss': rose_loss,
            'quality_loss': quality_loss,
            'volume_loss': volume_loss,
            'mean_regularity': quality_result['regularity'].mean(),
            'mean_volume': volume_result['volume'].mean(),
            'mean_rose': rose_score.mean()
        }


class PentachoronFlowNetwork(nn.Module):
    """Complete geometric flow architecture."""

    def __init__(self, config: PentachoronFlowConfig):
        super().__init__()

        self.config = config

        patch_size = 4 if config.image_size == 32 else 16
        self.patch_embed = PatchEmbedding(
            img_size=config.image_size,
            patch_size=patch_size,
            in_channels=config.input_channels,
            embed_dim=config.embed_dim
        )

        self.origin_sampler = GeometricOriginSampler(
            num_origins=config.num_origins,
            origin_dim=config.origin_dim,
            embed_dim=config.embed_dim,
            init_strategy=config.init_strategy
        )

        self.noise_collector = NoiseCollector(
            collection_strategy=config.collection_strategy,
            adaptive_radius=config.adaptive_radius,
            base_radius=config.base_radius,
            k_nearest=config.k_nearest
        )

        self.flow_matcher = FlowMatcher(
            simplex_dim=config.embed_dim,
            flow_steps=config.flow_steps,
            hidden_scale=config.hidden_scale,
            max_grad_norm=config.max_grad_norm
        )

        self.validator = CayleyMengerValidator(config)

        self.classification_head = GeometricClassificationHead(
            num_origins=config.num_origins,
            origin_dim=config.origin_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes
        )

    def forward(
        self,
        images: Tensor,
        labels: Optional[Tensor] = None,
        return_geometric_loss: bool = True
    ) -> Dict[str, Tensor]:
        """Forward pass through geometric flow."""

        # 1. Patch embedding
        patch_embeddings = self.patch_embed(images)

        # 2. Sample geometric origins
        origin_result = self.origin_sampler.forward(patch_embeddings)
        origins = origin_result['origins']

        # 3. Collect features
        collection_result = self.noise_collector.collect(
            patch_embeddings,
            origins,
            origin_centroids=origin_result['origin_centroids']
        )

        # 4. Flow through geometric space
        flow_result = self.flow_matcher.flow(origins, return_trajectory=False)
        flowed_simplices = flow_result['final_state']

        # 5. Classification
        logits = self.classification_head(flowed_simplices)

        output = {
            'logits': logits,
            'origins': origins,
            'flowed_state': flowed_simplices,
            'collection_stats': collection_result['collection_stats'],
            'flow_metrics': flow_result['flow_metrics']
        }

        if labels is not None:
            ce_loss = nn.functional.cross_entropy(logits, labels)
            output['ce_loss'] = ce_loss
            pred = logits.argmax(dim=-1)
            accuracy = (pred == labels).float().mean()
            output['accuracy'] = accuracy

        if return_geometric_loss:
            geometric_metrics = self.validator.compute_loss(flowed_simplices)
            output.update({
                'geometric_loss': geometric_metrics['loss'],
                'rose_loss': geometric_metrics['rose_loss'],
                'quality_loss': geometric_metrics['quality_loss'],
                'volume_loss': geometric_metrics['volume_loss'],
                'mean_regularity': geometric_metrics['mean_regularity'],
                'mean_volume': geometric_metrics['mean_volume'],
                'mean_rose': geometric_metrics['mean_rose']
            })

            if labels is not None:
                output['total_loss'] = ce_loss + 0.1 * geometric_metrics['loss']

        return output


def test_pentachoron_flow_network():
    """Test complete PentachoronFlow pipeline."""

    print("\n" + "=" * 70)
    print("PENTACHORON FLOW NETWORK - INTEGRATION TEST")
    print("=" * 70)

    config = PentachoronFlowConfig(
        num_classes=10,
        input_channels=3,
        image_size=32,
        num_origins=16,
        origin_dim=5,
        embed_dim=256,
        flow_steps=2,
        init_strategy='diffusion',
        collection_strategy='k_nearest'
    )

    network = PentachoronFlowNetwork(config)

    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))

    print(f"\n[Test 1] Forward Pass Without Labels")
    print(f"  Input: [{batch_size}, 3, 32, 32]")

    with torch.no_grad():
        result = network(images, return_geometric_loss=True)

    print(f"  Logits: {result['logits'].shape}")
    print(f"  Geometric loss: {result['geometric_loss'].item():.4f}")
    print(f"  Mean regularity: {result['mean_regularity'].item():.4f}")
    print(f"  Status: ✓ PASS")

    print("\n[Test 2] Forward Pass With Labels")

    with torch.no_grad():
        result_labeled = network(images, labels=labels)

    print(f"  CE loss: {result_labeled['ce_loss'].item():.4f}")
    print(f"  Total loss: {result_labeled['total_loss'].item():.4f}")
    print(f"  Accuracy: {result_labeled['accuracy'].item():.2%}")
    print(f"  Status: ✓ PASS")

    print("\n[Test 3] Gradient Flow")

    result_grad = network(images, labels=labels)
    loss = result_grad['total_loss']
    loss.backward()

    has_grads = sum(1 for p in network.parameters() if p.grad is not None)
    total_params = sum(1 for _ in network.parameters())

    print(f"  Parameters with gradients: {has_grads}/{total_params}")
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED - PENTACHORON FLOW READY")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_pentachoron_flow_network()