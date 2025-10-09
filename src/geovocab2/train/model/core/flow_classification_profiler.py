"""
Profiling Wrapper for PentachoronFlowNetwork
--------------------------------------------
Wraps model to track timing for each component.

Usage:
    model = PentachoronFlowNetwork(config)
    profiled_model = ProfiledPentachoronFlow(model)

    # Use normally
    out = profiled_model(imgs, labels=labels)

    # Print stats
    profiled_model.print_stats()
"""

import torch
import time
from typing import Dict, Optional
from torch import Tensor

from geovocab2.train.config.pentachoron_flow import PentachoronFlowConfig
from geovocab2.train.model.core.flow_classification import PentachoronFlowNetwork

class ProfiledPentachoronFlow(torch.nn.Module):
    """Profiling wrapper for PentachoronFlowNetwork."""

    def __init__(self, model, print_every=10):
        super().__init__()
        self.model = model
        self.print_every = print_every

        # Timing accumulators
        self.timings = {
            'patch_embed': [],
            'origin_sampler': [],
            'noise_collector': [],
            'flow_matcher': [],
            'classification_head': [],
            'geometric_loss': [],
            'ce_loss': [],
            'total': []
        }

        self.batch_count = 0

    def forward(
            self,
            images: Tensor,
            labels: Optional[Tensor] = None,
            return_geometric_loss: bool = True
    ) -> Dict[str, Tensor]:
        """Forward pass with timing."""

        self.batch_count += 1
        batch_start = time.time()

        # Patch embedding
        t0 = time.time()
        patch_embeddings = self.model.patch_embed(images)
        self.timings['patch_embed'].append(time.time() - t0)

        # Origin sampling
        t0 = time.time()
        origin_result = self.model.origin_sampler.forward(patch_embeddings)
        origins = origin_result['origins']
        self.timings['origin_sampler'].append(time.time() - t0)

        # Collection
        t0 = time.time()
        collection_result = self.model.noise_collector.collect(
            patch_embeddings,
            origins,
            origin_centroids=origin_result['origin_centroids']
        )
        self.timings['noise_collector'].append(time.time() - t0)

        # Flow
        t0 = time.time()
        flow_result = self.model.flow_matcher.flow(origins, return_trajectory=False)
        flowed_simplices = flow_result['final_state']
        self.timings['flow_matcher'].append(time.time() - t0)

        # Classification
        t0 = time.time()
        logits = self.model.classification_head(flowed_simplices)
        self.timings['classification_head'].append(time.time() - t0)

        output = {
            'logits': logits,
            'origins': origins,
            'flowed_state': flowed_simplices,
            'collection_stats': collection_result['collection_stats'],
            'flow_metrics': flow_result['flow_metrics']
        }

        # CE loss
        if labels is not None:
            t0 = time.time()
            ce_loss = torch.nn.functional.cross_entropy(logits, labels)
            self.timings['ce_loss'].append(time.time() - t0)

            output['ce_loss'] = ce_loss
            pred = logits.argmax(dim=-1)
            accuracy = (pred == labels).float().mean()
            output['accuracy'] = accuracy

        # Geometric loss
        if return_geometric_loss:
            t0 = time.time()
            geometric_metrics = self.model.validator.compute_loss(flowed_simplices)
            self.timings['geometric_loss'].append(time.time() - t0)

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

        # Total time
        self.timings['total'].append(time.time() - batch_start)

        # Print stats periodically
        if self.batch_count % self.print_every == 0:
            self.print_stats()

        return output

    def print_stats(self, reset=False):
        """Print timing statistics."""
        if self.batch_count == 0:
            print("No batches processed yet")
            return

        print(f"\n{'=' * 60}")
        print(f"PROFILING STATS - Batch {self.batch_count}")
        print(f"{'=' * 60}")

        # Compute averages over last N batches
        window = min(self.print_every, len(self.timings['total']))

        print(f"\nComponent timings (avg over last {window} batches):")
        print(f"{'Component':<20} {'Time (ms)':<12} {'% of Total':<12}")
        print(f"{'-' * 44}")

        total_avg = sum(self.timings['total'][-window:]) / window * 1000

        for component in ['patch_embed', 'origin_sampler', 'noise_collector',
                          'flow_matcher', 'classification_head', 'geometric_loss', 'ce_loss']:
            if self.timings[component]:
                avg_time = sum(self.timings[component][-window:]) / len(self.timings[component][-window:]) * 1000
                pct = (avg_time / total_avg * 100) if total_avg > 0 else 0
                print(f"{component:<20} {avg_time:>8.2f} ms   {pct:>6.1f}%")

        print(f"{'-' * 44}")
        print(f"{'TOTAL':<20} {total_avg:>8.2f} ms   100.0%")

        # Throughput
        if total_avg > 0:
            samples_per_sec = (1000 / total_avg)
            print(f"\nThroughput: {samples_per_sec:.1f} samples/sec (per batch)")

        print(f"{'=' * 60}\n")

        if reset:
            self.reset_stats()

    def reset_stats(self):
        """Reset all timing statistics."""
        for key in self.timings:
            self.timings[key] = []
        self.batch_count = 0

    def get_summary(self):
        """Get summary statistics as dict."""
        if not self.timings['total']:
            return {}

        summary = {}
        total_time = sum(self.timings['total'])

        for component, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times) * 1000
                total_component = sum(times) * 1000
                pct = (total_component / (total_time * 1000) * 100) if total_time > 0 else 0

                summary[component] = {
                    'avg_ms': avg_time,
                    'total_ms': total_component,
                    'percent': pct,
                    'count': len(times)
                }

        return summary


# Usage example
if __name__ == "__main__":

    config = PentachoronFlowConfig(
        num_classes=100,
        input_channels=3,
        image_size=32,
        num_origins=49,
        embed_dim=192,
        flow_steps=3,
    )

    model = PentachoronFlowNetwork(config).cuda()
    profiled_model = ProfiledPentachoronFlow(model, print_every=5)

    # Test with dummy data
    for i in range(20):
        imgs = torch.randn(256, 3, 32, 32).cuda()
        labels = torch.randint(0, 100, (256,)).cuda()

        out = profiled_model(imgs, labels=labels, return_geometric_loss=True)

    # Final summary
    profiled_model.print_stats()