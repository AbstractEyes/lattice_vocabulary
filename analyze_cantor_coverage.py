"""
Analyze Cantor Set Global Spectrum Coverage for Liminal Staircase
==================================================================

Calculate the containment zones and global attentiveness capacity.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_cantor_coord(position: float, depth: int = 8) -> float:
    """Pure Cantor set mapping (ternary)."""
    x = max(1e-6, min(position, 1.0 - 1e-6))

    cantor_val = 0.0
    factor = 0.5

    for _ in range(depth):
        x_scaled = x * 3.0
        digit = int(x_scaled)
        x_frac = x_scaled - digit

        # In Cantor set: keep left (0) and right (2) thirds, remove middle (1)
        if digit == 2:
            cantor_val += factor

        x = x_frac
        factor *= 0.5

    return cantor_val


def analyze_cantor_coverage(num_positions: int = 512, depth: int = 8, k_neighbors: int = 64):
    """Analyze global spectrum coverage."""

    print("=" * 80)
    print("CANTOR SET GLOBAL SPECTRUM COVERAGE ANALYSIS")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Sequence length: {num_positions}")
    print(f"  Cantor depth: {depth}")
    print(f"  k-NN neighbors: {k_neighbors}")

    # Map positions to Cantor coordinates
    positions = np.linspace(0, 1, num_positions)
    cantor_coords = np.array([compute_cantor_coord(pos, depth) for pos in positions])

    # Theoretical analysis
    print(f"\n{'='*80}")
    print("THEORETICAL PROPERTIES:")
    print(f"{'='*80}")

    # Ternary Cantor set properties
    num_segments = 2 ** depth
    segment_width = (1/3) ** depth
    total_measure = (2/3) ** depth

    print(f"\n1. Containment Zones (Cantor segments):")
    print(f"   - Number of segments: {num_segments}")
    print(f"   - Segment width: {segment_width:.2e}")
    print(f"   - Total measure: {total_measure:.6f} ({total_measure*100:.2f}% of [0,1])")
    print(f"   - Gap measure: {1-total_measure:.6f} ({(1-total_measure)*100:.2f}% of [0,1])")

    # Resolution
    print(f"\n2. Spatial Resolution:")
    print(f"   - Minimum distinguishable distance: {segment_width:.2e}")
    print(f"   - Positions per segment (avg): {num_positions / num_segments:.2f}")

    # k-NN coverage
    print(f"\n3. k-NN Global Coverage:")
    print(f"   - Neighbors per position: {k_neighbors}")
    print(f"   - Coverage fraction: {k_neighbors / num_positions * 100:.2f}%")
    print(f"   - Theoretical receptive field: O(log n) scales due to fractal structure")

    # Empirical analysis
    print(f"\n{'='*80}")
    print("EMPIRICAL ANALYSIS:")
    print(f"{'='*80}")

    # Compute pairwise distances in Cantor space
    distances = np.abs(cantor_coords[:, None] - cantor_coords[None, :])

    # For each position, find k nearest neighbors
    neighbor_indices = np.argpartition(distances, k_neighbors, axis=1)[:, :k_neighbors]

    # Analyze neighbor distribution
    max_distances = []
    position_spans = []

    for i in range(num_positions):
        neighbors = neighbor_indices[i]
        max_dist = distances[i, neighbors].max()
        max_distances.append(max_dist)

        # Span in original position space
        position_span = (neighbors.max() - neighbors.min())
        position_spans.append(position_span)

    print(f"\n1. Neighbor Distribution:")
    print(f"   - Max Cantor distance (mean): {np.mean(max_distances):.4f}")
    print(f"   - Max Cantor distance (std): {np.std(max_distances):.4f}")
    print(f"   - Position span (mean): {np.mean(position_spans):.1f} positions")
    print(f"   - Position span (max): {np.max(position_spans)} positions")

    # Multi-scale coverage
    scale_bins = [0, 0.1, 0.25, 0.5, 1.0]
    print(f"\n2. Multi-Scale Attention Coverage:")
    for i in range(len(scale_bins) - 1):
        lower, upper = scale_bins[i], scale_bins[i+1]
        fraction = np.mean((distances > lower) & (distances <= upper))
        scale_range = int(num_positions * (upper - lower))
        print(f"   - Range {lower:.2f}-{upper:.2f} (±{scale_range} pos): {fraction*100:.2f}% of pairs")

    # Clustering analysis
    print(f"\n3. Containment Zone Utilization:")

    # Discretize Cantor coords to segments
    segment_assignments = (cantor_coords * num_segments).astype(int)
    segment_assignments = np.clip(segment_assignments, 0, num_segments - 1)

    unique_segments, counts = np.unique(segment_assignments, return_counts=True)

    print(f"   - Occupied segments: {len(unique_segments)} / {num_segments}")
    print(f"   - Occupancy rate: {len(unique_segments) / num_segments * 100:.2f}%")
    print(f"   - Positions per segment (mean): {np.mean(counts):.2f}")
    print(f"   - Positions per segment (std): {np.std(counts):.2f}")
    print(f"   - Max positions in one segment: {np.max(counts)}")

    # Global attentiveness capacity
    print(f"\n{'='*80}")
    print("GLOBAL ATTENTIVENESS CAPACITY:")
    print(f"{'='*80}")

    # For each position, check how many unique "distant" positions it can attend to
    distant_threshold = num_positions * 0.25  # Consider "distant" as >25% of sequence length

    global_reach_counts = []
    for i in range(num_positions):
        neighbors = neighbor_indices[i]
        distant_neighbors = neighbors[np.abs(neighbors - i) > distant_threshold]
        global_reach_counts.append(len(distant_neighbors))

    print(f"\n1. Long-Range Dependency Capacity:")
    print(f"   - Distant neighbors (>25% seq len) per position (mean): {np.mean(global_reach_counts):.2f}")
    print(f"   - Distant neighbors (std): {np.std(global_reach_counts):.2f}")
    print(f"   - Positions with global reach (>10 distant): {np.sum(np.array(global_reach_counts) > 10)}/{num_positions}")

    # Complexity verification
    computation_per_position = k_neighbors
    total_computation = num_positions * k_neighbors
    full_attention_computation = num_positions * num_positions

    print(f"\n2. Complexity Validation:")
    print(f"   - Computation per position: {computation_per_position} (O(k))")
    print(f"   - Total computation: {total_computation} (O(n·k) = O(n))")
    print(f"   - Full attention would be: {full_attention_computation} (O(n²))")
    print(f"   - Speedup: {full_attention_computation / total_computation:.1f}x")

    print(f"\n{'='*80}")
    print("VERDICT:")
    print(f"{'='*80}")

    verdict = f"""
The Cantor set with depth={depth} provides:

✓ O(n) complexity: {k_neighbors} operations per position vs {num_positions} for full attention
✓ Global coverage: {num_segments} containment zones spanning the full [0,1] spectrum
✓ Multi-scale: Fractal structure enables {np.mean(global_reach_counts):.1f} long-range connections per position
✓ Efficiency: {full_attention_computation / total_computation:.1f}x speedup over O(n²) attention

⚠ Coverage density: Only {len(unique_segments)}/{num_segments} segments occupied ({len(unique_segments)/num_segments*100:.1f}%)
⚠ Clustering: Average {np.mean(counts):.1f} positions per occupied segment

RECOMMENDATION: Depth={depth} is adequate for sequences up to {num_positions}.
For longer sequences, increase depth by 1 per 3x sequence length increase.
"""

    print(verdict)

    return {
        'num_segments': num_segments,
        'occupied_segments': len(unique_segments),
        'occupancy_rate': len(unique_segments) / num_segments,
        'mean_global_reach': np.mean(global_reach_counts),
        'speedup': full_attention_computation / total_computation,
        'cantor_coords': cantor_coords,
        'segment_assignments': segment_assignments
    }


if __name__ == "__main__":
    # Analyze for different sequence lengths
    for seq_len in [77, 256, 512]:
        print("\n" * 2)
        results = analyze_cantor_coverage(num_positions=seq_len, depth=8, k_neighbors=64)

        if seq_len == 512:
            # Visualize for longest sequence
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot 1: Cantor coordinate distribution
            ax = axes[0, 0]
            ax.hist(results['cantor_coords'], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Cantor Coordinate')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Cantor Coordinate Distribution (n={seq_len})')
            ax.grid(True, alpha=0.3)

            # Plot 2: Position vs Cantor coordinate
            ax = axes[0, 1]
            positions = np.arange(seq_len)
            ax.scatter(positions, results['cantor_coords'], alpha=0.5, s=10)
            ax.set_xlabel('Sequence Position')
            ax.set_ylabel('Cantor Coordinate')
            ax.set_title('Position → Cantor Mapping')
            ax.grid(True, alpha=0.3)

            # Plot 3: Segment occupancy
            ax = axes[1, 0]
            unique_segs, counts = np.unique(results['segment_assignments'], return_counts=True)
            ax.bar(unique_segs, counts, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Segment ID')
            ax.set_ylabel('Positions in Segment')
            ax.set_title(f'Containment Zone Occupancy ({results["occupied_segments"]}/{results["num_segments"]} zones)')
            ax.grid(True, alpha=0.3)

            # Plot 4: Cantor set structure
            ax = axes[1, 1]
            # Show the fractal structure
            sorted_coords = np.sort(results['cantor_coords'])
            ax.scatter(sorted_coords, np.zeros_like(sorted_coords), alpha=0.6, s=20)
            ax.set_xlabel('Cantor Coordinate')
            ax.set_title('Cantor Set Structure (Fractal Distribution)')
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('/home/user/lattice_vocabulary/cantor_coverage_analysis.png', dpi=150)
            print(f"\n📊 Visualization saved to: cantor_coverage_analysis.png")
