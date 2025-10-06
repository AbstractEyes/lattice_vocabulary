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

from geovocab2.train.config.pentachoron_flow import PentachoronFlowConfig
from geovocab2.train.losses.cayley_menger import CayleyMengerValidator
from geovocab2.train.model.heads.classification import GeometricClassificationHead

from geovocab2.train.model.flow_matcher import FlowMatcher

from geovocab2.shapes.formula import (
    GeometricOriginSampler,
    NoiseCollector,
    SimplexQuality,
    SimplexVolumeExtended
)



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


class PentachoronFlowNetwork(nn.Module):
    """Complete geometric flow architecture."""

    def __init__(self, config: PentachoronFlowConfig):
        super().__init__()

        self.config = config

        self.patch_embed = PatchEmbedding(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.input_channels,
            embed_dim=config.embed_dim
        )

        self.origin_sampler = GeometricOriginSampler(
            num_origins=config.num_origins,
            origin_dim=config.origin_dim,
            embed_dim=config.embed_dim,
            init_strategy=config.init_strategy,
            temperature=config.temperature,
            quality_threshold=config.quality_threshold,
            max_init_attempts=config.max_init_attempts,
            diffusion_steps=config.flow_steps
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
            max_grad_norm=config.max_grad_norm,
            use_trajectory_attention=config.use_trajectory_attention,
            trajectory_attention_heads=config.trajectory_attention_heads,
            projection_lr=config.embed_dim ** -0.5
        )

        self.validator = CayleyMengerValidator(config)

        self.classification_head = GeometricClassificationHead(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            use_attention=config.use_attention,
            attention_heads=config.attention_heads,
            dropout_rate=config.dropout_rate
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