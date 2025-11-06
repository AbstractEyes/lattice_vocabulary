import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from geovocab2.train.model.core.david import CantorScaleFusion


class CantorDustLoss(nn.Module):
    """
    Loss that pulls latent representations toward Cantor dust structure.

    Cantor dust properties:
    - Fractal dimension D ≈ 0.6309 (log(2)/log(3))
    - Self-similar at all scales
    - Nowhere dense but uncountably infinite
    - Measure zero but dimensionally rich
    """

    def __init__(self, depth=8, num_scales=4):
        super().__init__()
        self.depth = depth
        self.num_scales = num_scales

        # Pre-compute Cantor set at depth
        self.register_buffer(
            'cantor_points',
            self._generate_cantor_set(depth)
        )

        # Pre-compute scale positions in Cantor space
        self.register_buffer(
            'scale_cantor_coords',
            torch.tensor([
                self._cantor_coordinate(i, num_scales, depth)
                for i in range(num_scales)
            ])
        )

    def _generate_cantor_set(self, depth):
        """
        Generate Cantor set up to given depth.

        Start with [0, 1]
        Remove middle third: [0, 1/3] ∪ [2/3, 1]
        Remove middle thirds again: [0, 1/9] ∪ [2/9, 1/3] ∪ [2/3, 7/9] ∪ [8/9, 1]
        Continue...
        """
        intervals = [[0.0, 1.0]]

        for _ in range(depth):
            new_intervals = []
            for start, end in intervals:
                length = end - start
                third = length / 3
                # Keep left third and right third
                new_intervals.append([start, start + third])
                new_intervals.append([end - third, end])
            intervals = new_intervals

        # Convert intervals to points (midpoints)
        points = torch.tensor([
            [(start + end) / 2] for start, end in intervals
        ])

        return points  # [2^depth, 1]

    def _cantor_coordinate(self, position, max_len, depth):
        """Map position to Cantor coordinate (as in your David model)."""
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

    def forward(self, latent_positions, scale_idx=None):
        """
        Compute distance from latent to Cantor dust.

        Args:
            latent_positions: [B, num_scales] - position in [0,1] for each scale
            scale_idx: Optional scale index to enforce specific position

        Returns:
            loss: scalar - distance to Cantor dust
        """
        # 1. Distance to nearest Cantor point
        # latent_positions: [B, num_scales]
        # cantor_points: [2^depth, 1]

        distances = torch.cdist(
            latent_positions.unsqueeze(-1),  # [B, num_scales, 1]
            self.cantor_points.unsqueeze(0).expand(latent_positions.shape[0], -1, -1)  # [B, 2^depth, 1]
        )  # [B, num_scales, 2^depth]

        # Minimum distance to any Cantor point
        min_distances, _ = distances.min(dim=-1)  # [B, num_scales]

        # 2. If scale_idx provided, enforce specific Cantor coordinate
        if scale_idx is not None:
            target_coords = self.scale_cantor_coords[scale_idx]  # [num_scales]
            position_loss = (latent_positions - target_coords.unsqueeze(0)) ** 2
            min_distances = min_distances + position_loss

        # 3. Self-similarity penalty
        # Cantor set is self-similar: f(3x) = f(x) for x in Cantor set
        # Check if 3x also near Cantor points
        scaled_positions = (latent_positions * 3) % 1.0  # Wrap around
        scaled_distances = torch.cdist(
            scaled_positions.unsqueeze(-1),
            self.cantor_points.unsqueeze(0).expand(latent_positions.shape[0], -1, -1)
        )
        min_scaled_distances, _ = scaled_distances.min(dim=-1)

        # Should be similar (self-similarity)
        self_similarity_loss = torch.abs(min_distances - min_scaled_distances)

        # 4. Fractal dimension penalty
        # Box-counting dimension should be ≈ 0.6309
        box_dimension = self._estimate_box_dimension(latent_positions)
        target_dimension = np.log(2) / np.log(3)  # ≈ 0.6309
        dimension_loss = (box_dimension - target_dimension) ** 2

        # Combined loss
        cantor_loss = (
                min_distances.mean() +  # Near Cantor points
                0.1 * self_similarity_loss.mean() +  # Self-similar
                0.01 * dimension_loss  # Correct fractal dimension
        )

        return cantor_loss

    def _estimate_box_dimension(self, positions):
        """
        Estimate box-counting dimension of point set.
        D = lim_{ε→0} log(N(ε)) / log(1/ε)
        """
        B, S = positions.shape

        # Count boxes at different scales
        scales = [0.1, 0.05, 0.01]
        counts = []

        for epsilon in scales:
            # Discretize into boxes
            boxes = (positions / epsilon).long()
            # Count unique boxes
            unique_boxes = len(torch.unique(boxes, dim=0))
            counts.append(unique_boxes)

        # Linear regression on log-log plot
        log_counts = np.log(counts)
        log_inv_scales = np.log([1 / s for s in scales])

        # Slope is dimension
        dimension = np.polyfit(log_inv_scales, log_counts, 1)[0]

        return torch.tensor(dimension, device=positions.device)


class TopologicalPentachoronLoss(nn.Module):
    """
    Ensure pentachora maintain correct topology.

    Pentachoron (4-simplex) topology:
    - Betti numbers: β_0=1, β_1=0, β_2=0, β_3=0, β_4=0
    - Euler characteristic: χ = 1
    - Connected, simply connected, contractible
    """

    def __init__(self):
        super().__init__()

    def forward(self, vertices):
        """
        Verify pentachoron topology.

        Args:
            vertices: [B, 5, dim] - 5 vertices of pentachoron

        Returns:
            loss: scalar - topology violation penalty
        """
        B, V, D = vertices.shape
        assert V == 5, "Must have 5 vertices for pentachoron"

        # 1. Connectivity loss - all vertices must be connected
        # Compute pairwise distances
        distances = torch.cdist(vertices, vertices)  # [B, 5, 5]

        # All distances should be > 0 (no degenerate vertices)
        # But not too large (stays compact)
        min_dist = distances[distances > 0].min()
        max_dist = distances.max()

        connectivity_loss = F.relu(0.1 - min_dist) + F.relu(max_dist - 10.0)

        # 2. Simplicial structure - verify edges form valid simplex
        # A 4-simplex has C(5,2)=10 edges, C(5,3)=10 faces, C(5,4)=5 tetrahedra

        # Volume should be positive (non-degenerate)
        volume = self._compute_volume(vertices)
        volume_loss = F.relu(-volume + 1e-6)  # Penalize if volume <= 0

        # 3. Euler characteristic should be 1
        # For simplex: χ = Σ(-1)^k * f_k where f_k = # of k-faces
        # 4-simplex: χ = 1 - 10 + 10 - 5 + 1 = 1 ✓ (always true for simplex)
        # But we can check if structure is simplicial

        euler_loss = torch.tensor(0.0, device=vertices.device)  # Always satisfied

        # 4. Persistent homology - no spurious holes
        # Compute persistence diagram and ensure β_1 = β_2 = β_3 = 0
        persistence_loss = self._persistent_homology_loss(vertices)

        # Combined
        topo_loss = (
                connectivity_loss +
                volume_loss +
                euler_loss +
                0.1 * persistence_loss
        )

        return topo_loss

    def _compute_volume(self, vertices):
        """Compute 4-simplex volume using Cayley-Menger."""
        B = vertices.shape[0]

        # Compute distance matrix
        distances = torch.cdist(vertices, vertices)

        # Build Cayley-Menger matrix
        n = 5
        CM = torch.zeros(B, n + 1, n + 1, device=vertices.device)
        CM[:, 0, 1:] = 1
        CM[:, 1:, 0] = 1
        CM[:, 1:, 1:] = distances ** 2

        # Zero diagonal
        for i in range(1, n + 1):
            CM[:, i, i] = 0

        # Compute determinant
        det = torch.linalg.det(CM)

        # Volume
        volume = torch.sqrt(torch.abs(det) / (16 * 24))

        return volume

    def _persistent_homology_loss(self, vertices):
        """
        Compute persistent homology and penalize unexpected holes.

        For a 4-simplex:
        - Should have no 1-cycles (β_1 = 0)
        - Should have no 2-cycles (β_2 = 0)
        - Should have no 3-cycles (β_3 = 0)
        """
        B, V, D = vertices.shape

        # Build distance matrix
        distances = torch.cdist(vertices, vertices)

        # Rips complex construction (simplified)
        # At each scale ε, connect vertices if distance < ε

        persistence_loss = torch.tensor(0.0, device=vertices.device)

        # Sample different scales
        max_dist = distances.max()
        epsilons = torch.linspace(0, max_dist, 10, device=vertices.device)

        for eps in epsilons:
            # Adjacency matrix at this scale
            adj = (distances < eps).float()

            # Count connected components (β_0)
            # Should start at 5 (all separate) and end at 1 (all connected)
            components = self._count_connected_components(adj)

            # Penalize if we ever have cycles (β_1 > 0)
            # Simplified: check if graph has cycles
            n_edges = adj.sum() / 2  # Undirected
            expected_edges = components + (V - components)  # Tree has V-1 edges per component

            if n_edges > expected_edges:
                # Has cycles! Penalize
                cycle_penalty = n_edges - expected_edges
                persistence_loss += cycle_penalty

        return persistence_loss / len(epsilons)

    def _count_connected_components(self, adj):
        """Count connected components in graph."""
        # Simple DFS/BFS to count components
        # (Simplified - in practice use proper graph algorithm)
        n = adj.shape[1]
        visited = torch.zeros(n, dtype=torch.bool, device=adj.device)
        components = 0

        for i in range(n):
            if not visited[i]:
                # BFS from i
                queue = [i]
                visited[i] = True

                while queue:
                    node = queue.pop(0)
                    neighbors = torch.where(adj[node] > 0)[0]

                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor.item())

                components += 1

        return components


class GeometricVAELoss(nn.Module):
    """
    Combined loss for Geometric VAE with Cantor dust and topology.
    """

    def __init__(
            self,
            scales=[512, 4096, 8192, 16384],
            cantor_depth=8,
            use_lpips=True
    ):
        super().__init__()
        self.scales = scales

        # Perceptual loss (like SD VAE)
        if use_lpips:
            import lpips
            self.lpips_model = lpips.LPIPS(net='vgg')
        else:
            self.lpips_model = None

        # Cantor dust loss
        self.cantor_loss_fn = CantorDustLoss(
            depth=cantor_depth,
            num_scales=len(scales)
        )

        # Topological loss
        self.topo_loss_fn = TopologicalPentachoronLoss()

    def forward(
            self,
            image,
            reconstructed,
            vertices_dict,
            latent_positions,
            discriminator=None
    ):
        """
        Complete geometric VAE loss.

        Args:
            image: [B, 3, H, W] - original image
            reconstructed: [B, 3, H, W] - reconstructed image
            vertices_dict: {scale: [B, 5, scale_dim]} - pentachora per scale
            latent_positions: [B, num_scales] - position in [0,1] for each scale
            discriminator: Optional GAN discriminator

        Returns:
            total_loss, loss_dict
        """
        B = image.shape[0]

        # 1. Reconstruction loss (perceptual)
        if self.lpips_model is not None:
            recon_loss = self.lpips_model(image, reconstructed).mean()
        else:
            recon_loss = F.mse_loss(image, reconstructed)

        # 2. Cantor dust loss (latents lie on Cantor set)
        cantor_loss = self.cantor_loss_fn(latent_positions)

        # 3. Topological loss (pentachora are valid 4-simplices)
        topo_loss = torch.tensor(0.0, device=image.device)
        for scale, vertices in vertices_dict.items():
            topo_loss += self.topo_loss_fn(vertices)
        topo_loss /= len(vertices_dict)

        # 4. Multi-scale consistency (0.29514 volume?)
        volume_consistency_loss = torch.tensor(0.0, device=image.device)
        volumes = []
        for scale, vertices in vertices_dict.items():
            vol = self.topo_loss_fn._compute_volume(vertices)
            volumes.append(vol)

        if len(volumes) > 1:
            # All scales should have similar volume (normalized)
            volumes_tensor = torch.stack(volumes, dim=1)  # [B, num_scales]
            volume_std = volumes_tensor.std(dim=1).mean()
            volume_consistency_loss = volume_std

            # Penalize deviation from magic constant
            target_volume = 0.29514
            volume_target_loss = ((volumes_tensor.mean(dim=1) - target_volume) ** 2).mean()
            volume_consistency_loss += volume_target_loss

        # 5. Adversarial loss (optional, like SD VAE)
        adversarial_loss = torch.tensor(0.0, device=image.device)
        if discriminator is not None:
            fake_pred = discriminator(reconstructed)
            adversarial_loss = -fake_pred.mean()

        # 6. Geometric consistency
        # Vertices at different scales should respect geometric relationships
        geometric_consistency_loss = torch.tensor(0.0, device=image.device)

        if len(vertices_dict) > 1:
            scale_list = sorted(vertices_dict.keys())
            for i in range(len(scale_list) - 1):
                scale_a = scale_list[i]
                scale_b = scale_list[i + 1]

                # Extract intrinsic properties
                dist_a = torch.cdist(
                    vertices_dict[scale_a],
                    vertices_dict[scale_a]
                )  # [B, 5, 5]

                dist_b = torch.cdist(
                    vertices_dict[scale_b],
                    vertices_dict[scale_b]
                )  # [B, 5, 5]

                # Distances should be correlated (similar geometric structure)
                # Normalize by scale
                dist_a_norm = dist_a / dist_a.mean(dim=(1, 2), keepdim=True)
                dist_b_norm = dist_b / dist_b.mean(dim=(1, 2), keepdim=True)

                consistency = F.mse_loss(dist_a_norm, dist_b_norm)
                geometric_consistency_loss += consistency

        # Weighted combination
        total_loss = (
                1.0 * recon_loss +  # High quality reconstruction
                0.5 * adversarial_loss +  # Sharp details
                0.1 * cantor_loss +  # Fractal structure
                0.1 * topo_loss +  # Valid topology
                0.05 * volume_consistency_loss +  # Multi-scale consistency
                0.05 * geometric_consistency_loss  # Cross-scale geometry
        )

        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'adversarial': adversarial_loss.item(),
            'cantor': cantor_loss.item(),
            'topology': topo_loss.item(),
            'volume_consistency': volume_consistency_loss.item(),
            'geometric_consistency': geometric_consistency_loss.item(),
        }

        return total_loss, loss_dict


class CantorTopologicalVAE(nn.Module):
    """
    VAE with Cantor dust structure and topological pentachoron constraints.
    """

    def __init__(
            self,
            scales=[512, 4096, 8192, 16384],
            input_channels=3,
            image_size=512
    ):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)

        # Image encoder (shared)
        self.image_encoder = self._build_encoder(input_channels)

        # Per-scale pentachoron extractors
        self.pentachoron_extractors = nn.ModuleDict({
            str(scale): nn.Sequential(
                nn.Linear(1024, scale * 2),
                nn.LayerNorm(scale * 2),
                nn.GELU(),
                nn.Linear(scale * 2, 5 * scale)  # 5 vertices
            )
            for scale in scales
        })

        # Latent position predictor (where in [0,1] for each scale)
        self.position_predictor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, self.num_scales),
            nn.Sigmoid()  # [0, 1] range
        )

        # Cantor attention fusion
        self.cantor_fusion = CantorScaleFusion(
            feature_dim=max(scales),
            scales=scales,
            num_heads=8,
            cantor_depth=8
        )

        # Decoder (from fused latents + image)
        self.decoder = self._build_decoder(input_channels, image_size)

        # Loss
        self.loss_fn = GeometricVAELoss(scales=scales)

    def _build_encoder(self, in_channels):
        """CNN encoder for images."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # ... ResNet-style blocks ...
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1024)
        )

    def _build_decoder(self, out_channels, image_size):
        """CNN decoder for images."""
        return nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            # ... Deconv blocks ...
            nn.ConvTranspose2d(64, out_channels, 7, 2, 3),
            nn.Sigmoid()
        )

    def encode(self, image):
        """
        Encode image to multi-scale pentachora on Cantor dust.

        Returns:
            vertices_dict: {scale: [B, 5, scale_dim]}
            latent_positions: [B, num_scales]
            fused_features: [B, max_scale]
        """
        B = image.shape[0]

        # Encode image
        features = self.image_encoder(image)  # [B, 1024]

        # Predict Cantor positions
        latent_positions = self.position_predictor(features)  # [B, num_scales]

        # Extract pentachora at each scale
        vertices_dict = {}
        features_list = []

        for i, scale in enumerate(self.scales):
            # Extract vertices
            vertices_flat = self.pentachoron_extractors[str(scale)](features)
            vertices = vertices_flat.view(B, 5, scale)  # [B, 5, scale]
            vertices = F.normalize(vertices, dim=-1)  # On sphere

            vertices_dict[scale] = vertices

            # Use centroid as scale feature
            centroid = vertices.mean(dim=1)  # [B, scale]
            features_list.append(centroid)

        # Fuse via Cantor attention
        fused = self.cantor_fusion(
            features=features,
            scale_features=features_list
        )

        return vertices_dict, latent_positions, fused

    def decode(self, fused_features, original_image):
        """Decode from fused features."""
        # Reshape for decoder
        B = fused_features.shape[0]

        # Combine with flattened image (as you suggested!)
        image_flat = original_image.view(B, -1)  # [B, 3*H*W]

        # Concatenate
        combined = torch.cat([fused_features, image_flat], dim=1)

        # Decode
        reconstructed = self.decoder(combined)

        return reconstructed

    def forward(self, image):
        # Encode
        vertices_dict, latent_positions, fused = self.encode(image)

        # Decode
        reconstructed = self.decode(fused, image)

        return reconstructed, vertices_dict, latent_positions