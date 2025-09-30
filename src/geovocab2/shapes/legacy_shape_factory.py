# Legacy code from GeoVocab v1/v2 adapted for unified crystal synthesis
import hashlib
import numpy as np
import torch
import warnings
from dataclasses import replace
from torch.nn.functional import cosine_similarity
from typing import Optional, Dict, Any

from data.trie import ArrowDataManager
from defaults import UnifiedCrystalConfig, DimensionType, FormulaType, ContentType, NormType, EPS, Crystal

# ============================================================================
# Crystal Factory
# ============================================================================

def regular_simplex_5():
    """Create regular 4-simplex vertices in 5D with unit edge length"""
    E = np.eye(5, dtype=np.float64)
    centroid = np.mean(E, axis=0, keepdims=True)
    S = E - centroid
    edge_length = np.linalg.norm(S[0] - S[1])
    S = S / edge_length
    return S.astype(np.float32)


def validate_crystal(crystal: np.ndarray) -> Dict[str, Any]:
    """Validate crystal geometry"""
    validation = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check shape
    if crystal.ndim != 2:
        validation['valid'] = False
        validation['errors'].append(f"Crystal must be 2D, got {crystal.ndim}D")
        return validation

    n_vertices, dim = crystal.shape

    # Check centering
    centroid = crystal.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0.1:
        validation['warnings'].append(f"Crystal not centered: centroid norm = {centroid_norm:.4f}")

    # Check edge lengths
    if n_vertices > 1:
        distances = []
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                distances.append(np.linalg.norm(crystal[i] - crystal[j]))

        if distances:
            min_dist = min(distances)
            max_dist = max(distances)

            if min_dist < 1e-6:
                validation['valid'] = False
                validation['errors'].append(f"Degenerate crystal: min edge = {min_dist:.8f}")

            if max_dist / (min_dist + 1e-8) > 100:
                validation['warnings'].append(f"High edge ratio: {max_dist / min_dist:.2f}")

    # Check for NaN or Inf
    if np.any(np.isnan(crystal)):
        validation['valid'] = False
        validation['errors'].append("Crystal contains NaN values")

    if np.any(np.isinf(crystal)):
        validation['valid'] = False
        validation['errors'].append("Crystal contains Inf values")

    # Check volume (if applicable)
    if 2 <= n_vertices <= 6:
        factory = CrystalFactory(UnifiedCrystalConfig())
        volume = factory._compute_volume(crystal)

        if volume < 1e-10:
            validation['warnings'].append(f"Near-degenerate volume: {volume:.8f}")

        validation['volume'] = volume

    return validation

class CrystalFactory:
    """Factory for creating crystals with various configurations"""

    def __init__(self, config: UnifiedCrystalConfig, data_manager: Optional[ArrowDataManager] = None):
        self.config = config
        self.dim = config.embedding_dim
        self.data_manager = data_manager
        self._initialize_handlers()
        self._char_cache: Dict[str, np.ndarray] = {}

    def _initialize_handlers(self):
        """Initialize handler mappings"""
        self.dimension_handlers = {
            DimensionType.D1: self._build_1d,
            DimensionType.D2: self._build_2d,
            DimensionType.D3: self._build_3d,
            DimensionType.D4: self._build_4d,
            DimensionType.D5: self._build_5d,
            DimensionType.D6_PLUS: self._build_nd,
        }

        self.formula_handlers = {
            FormulaType.ROSE_CAYLEY: self._apply_rose_cayley,
            FormulaType.CAYLEY_MENGER: self._apply_cayley_menger,
            FormulaType.CAYLEY: self._apply_cayley,
            FormulaType.MENGER: self._apply_menger,
            FormulaType.EULER: self._apply_euler,
            FormulaType.GRAHAM_INFINITE: self._apply_graham_infinite,
            FormulaType.GRAHAM_FINITE: self._apply_graham_finite,
            FormulaType.GRAHAM_MASKED: self._apply_graham_masked,
            FormulaType.HYBRID_V1V2: self._apply_hybrid_v1v2,
        }

        self.content_handlers = {
            ContentType.SPARSE: self._content_sparse,
            ContentType.ENRICHED: self._content_enriched,
            ContentType.TRAJECTORY: self._content_trajectory,
            ContentType.MAGNITUDE: self._content_magnitude,
            ContentType.VOLUME: self._content_volume,
            ContentType.HYBRID: self._content_hybrid,
        }

    def create_crystal(self, token: str, definition: Optional[str] = None,
                       override_config: Optional[Dict[str, Any]] = None,
                       use_dataset: bool = True) -> Dict[str, Any]:
        """Create crystal with optional dataset lookup"""
        config = self.config
        if override_config:
            config = replace(config, **override_config)

        # Try dataset first if available
        if use_dataset and self.data_manager and config.prefer_dataset and not definition:
            dataset_data = self.data_manager.get_token_data(token)
            if dataset_data and dataset_data.get("crystal") is not None:
                base_crystal = dataset_data["crystal"]

                # Handle different data formats
                if base_crystal.ndim == 1:
                    center = base_crystal
                    base_crystal = self._pentachoron_from_center(center, token)
                elif base_crystal.shape[0] != config.dimension_type.value:
                    base_crystal = self._reshape_crystal(base_crystal, config.dimension_type)

                # Apply transformations
                formula_crystal = self.formula_handlers[config.formula_type](base_crystal, token, definition)
                content_crystal = self.content_handlers[config.content_type](formula_crystal, token, definition)

                # Smart normalization based on content type
                if config.content_type == ContentType.VOLUME:
                    final_crystal = content_crystal
                elif config.norm_type != NormType.NONE:
                    if config.norm_type == NormType.L2:
                        current_scale = np.sqrt(np.mean(content_crystal * content_crystal))
                        target_scale = np.sqrt(config.embedding_dim) * 0.1
                        if current_scale > EPS:
                            final_crystal = content_crystal * (target_scale / current_scale)
                        else:
                            final_crystal = content_crystal
                    else:
                        final_crystal = self._apply_normalization(content_crystal, config.norm_type)
                else:
                    final_crystal = content_crystal

                final_crystal = final_crystal - final_crystal.mean(axis=0, keepdims=True)

                metadata = self._compute_metadata(final_crystal, token, definition)
                metadata["source"] = "dataset"

                return {
                    'crystal': final_crystal.astype(np.float32),
                    'volume': metadata["volume"],
                    'metadata': metadata,
                    'config': config
                }

        # Synthesize
        print(f"Synthesizing crystal for token: {token}")
        print(config)
        print(self.dimension_handlers.keys())
        base_crystal = self.dimension_handlers[config.dimension_type](token, definition)
        base_crystal = base_crystal - base_crystal.mean(axis=0, keepdims=True)

        formula_crystal = self.formula_handlers[config.formula_type](base_crystal, token, definition)
        content_crystal = self.content_handlers[config.content_type](formula_crystal, token, definition)

        # Smart normalization
        if config.content_type == ContentType.VOLUME:
            final_crystal = content_crystal
        elif config.norm_type != NormType.NONE:
            if config.norm_type == NormType.L2:
                current_scale = np.sqrt(np.mean(content_crystal * content_crystal))
                target_scale = np.sqrt(config.embedding_dim) * 0.1
                if current_scale > EPS:
                    final_crystal = content_crystal * (target_scale / current_scale)
                else:
                    final_crystal = content_crystal
            else:
                final_crystal = self._apply_normalization(content_crystal, config.norm_type)
        else:
            final_crystal = content_crystal

        final_crystal = final_crystal - final_crystal.mean(axis=0, keepdims=True)

        metadata = self._compute_metadata(final_crystal, token, definition)
        metadata["source"] = "synthesized"

        return {
            'crystal': final_crystal.astype(np.float32),
            'volume': metadata['volume'],
            'metadata': metadata,
            'config': config
        }

    # Dimension builders
    def _build_1d(self, token: str, definition: Optional[str]) -> Crystal:
        return self._text_to_vec(definition or token).reshape(1, -1)

    def _build_2d(self, token: str, definition: Optional[str]) -> Crystal:
        X = np.zeros((2, self.dim), dtype=np.float32)
        X[0] = self._text_to_vec(token)
        X[1] = self._text_to_vec(definition or token[::-1])
        return X

    def _build_3d(self, token: str, definition: Optional[str]) -> Crystal:
        X = np.zeros((3, self.dim), dtype=np.float32)
        X[0] = self._text_to_vec(token)
        X[1] = self._text_to_vec(token[:len(token) // 2] if len(token) > 1 else token)
        X[2] = self._text_to_vec(definition or token)
        return X

    def _build_4d(self, token: str, definition: Optional[str]) -> Crystal:
        X = np.zeros((4, self.dim), dtype=np.float32)
        base = self._text_to_vec(token)
        for i in range(4):
            X[i] = self._rotate_vector(base, i * np.pi / 2)
        return X

    def _build_5d(self, token: str, definition: Optional[str]) -> Crystal:
        if definition:
            return self._build_5d_with_full_v1(token, definition)
        elif self.config.use_character_composition:
            return self._build_5d_from_characters(token)
        else:
            return self._build_5d_deterministic(token)

    def _build_nd(self, token: str, definition: Optional[str]) -> Crystal:
        n = self.config.dimension_type.value
        X = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            binary = format(i, f'0{min(n, 32)}b')
            X[i] = self._text_to_vec(token + binary)
        return X

    def _build_5d_with_full_v1(self, token: str, definition: str) -> np.ndarray:
        """Full V1-style synthesis with projections and cardinal axes"""
        S5 = regular_simplex_5()

        # Build cardinal axes from definition
        C4 = self._cardinal_axes_from_definition_full(definition, token)

        # Build orthonormal frame
        Q5 = self._orthonormal_frame_5_full(C4, token, definition)

        # Project simplex through frame
        b = (S5 @ Q5.T).astype(np.float32)

        # Base trajectory from definition
        v0 = self._text_to_vec(definition)

        # Length-based scaling
        L = len(definition.encode('utf-8', errors='ignore'))
        base = float(np.clip(np.log1p(L) * 0.5, 0.5, 2.0))

        # Gamma factors for vertex scaling
        gamma = np.array([base, base * 0.9, -0.8 * base, base * 1.1, 1.2 * base], dtype=np.float32)

        # Project simplex vertices onto cardinal axes
        proj = np.zeros((5, 4), dtype=np.float32)
        for i in range(5):
            for k in range(4):
                proj[i, k] = float(np.dot(C4[k], b[i]))

        # Definition-weighted delta factors
        base_vec = np.array([L + 1, 1, 1, 1], dtype=np.float64)
        base_vec = base_vec / base_vec.sum()

        delta = np.tile(base_vec[None, :], (5, 1)).astype(np.float32)
        delta[1, 1] *= 1.2
        delta[2, 2] *= 1.5
        delta[3, 3] *= 1.2
        delta = delta / (delta.sum(axis=1, keepdims=True) + EPS)

        # Construct crystal vertices with full projections
        X = np.zeros((5, self.dim), dtype=np.float32)
        for i in range(5):
            xi = gamma[i] * b[i]
            for k in range(4):
                xi += delta[i, k] * proj[i, k] * C4[k]
            X[i] = v0 + xi

        # Center the crystal
        X -= X.mean(axis=0, keepdims=True)

        # Gentle scaling based on content
        if self.config.norm_type == NormType.L1:
            current_scale = np.mean(np.abs(X))
            target_scale = np.clip(np.log1p(L) * 0.3, 0.4, 2.0)
        else:
            current_scale = np.sqrt(np.mean(X * X))
            target_scale = np.clip(np.log1p(L) * 0.2, 0.25, 1.5)

        if current_scale > EPS:
            X *= (target_scale / current_scale)

        return X

    def _build_5d_from_characters(self, token: str) -> Crystal:
        """Character-based synthesis with exponential positional weighting
        NOTE: exp(-0.3 * i) gives weight 1.0 to first char (i=0), decreasing for later chars
        This is the intended behavior - first characters are most important."""
        char_vecs = []

        for char in token:
            if char in self._char_cache:
                char_vecs.append(self._char_cache[char])
            else:
                # Try dataset lookup
                if self.data_manager and self.data_manager.token_exists(char):
                    char_data = self.data_manager.get_token_data(char)
                    if char_data and char_data.get("crystal") is not None:
                        crystal_data = char_data["crystal"]
                        if crystal_data.ndim > 1:
                            pooled = crystal_data.mean(axis=0)
                        else:
                            pooled = crystal_data
                        self._char_cache[char] = pooled
                        char_vecs.append(pooled)
                        continue

                # Synthesize character
                char_vec = self._create_char_embedding(char)
                self._char_cache[char] = char_vec
                char_vecs.append(char_vec)

        if not char_vecs:
            return self._build_5d_deterministic(token)

        # Exponential decay weights: first char gets weight ~1.0, decreasing exponentially
        weights = np.array([np.exp(-0.3 * i) for i in range(len(char_vecs))])
        weights /= weights.sum()

        # Build center with interaction to ensure different tokens with same chars differ
        center = np.zeros(self.dim, dtype=np.float32)
        interaction_hash = self._sha_u64(token + "_char_interaction")

        for i, (weight, vec) in enumerate(zip(weights, char_vecs)):
            # Perturbation based on token-specific hash
            perturbation = 1.0 + ((interaction_hash >> (i * 4)) & 0xF) / 100.0 - 0.075
            center += weight * vec * perturbation

        return self._pentachoron_from_center(center, token)

    def _build_5d_deterministic(self, token: str) -> Crystal:
        """Pure deterministic synthesis"""
        center = self._deterministic_center(token)
        return self._pentachoron_from_center(center, token)

    def _pentachoron_from_center(self, center: np.ndarray, token: str) -> Crystal:
        """Build geometrically correct pentachoron with reasonable initial scale"""
        dim = len(center)
        token_hash = self._sha_u64(token)

        # Standard 4-simplex vertices in 5D (before embedding)
        simplex_5d = np.array([
            [1, 0, 0, 0, 0],
            [-0.25, 0.968, 0, 0, 0],
            [-0.25, -0.323, 0.913, 0, 0],
            [-0.25, -0.323, -0.457, 0.791, 0],
            [-0.25, -0.323, -0.457, -0.395, 0.686]
        ], dtype=np.float32)

        # Create random orthonormal basis in embedding dimension
        np.random.seed(token_hash % 2 ** 32)

        if dim >= 5:
            Q = np.random.randn(dim, 5).astype(np.float32)
            Q, _ = np.linalg.qr(Q)
            Q = Q[:, :5]
        else:
            Q = np.random.randn(dim, min(dim, 5)).astype(np.float32)
            Q, _ = np.linalg.qr(Q)
            simplex_5d = simplex_5d[:, :dim]

        # Embed simplex in the space
        X = (simplex_5d @ Q.T).astype(np.float32)

        # Scale appropriately for the embedding dimension
        token_length = len(token.encode('utf-8'))
        base_scale = np.sqrt(dim) * 0.5
        length_factor = 1.0 + np.log1p(token_length) * 0.2
        scale = base_scale * length_factor

        X = X * scale

        # Add center and token-specific variations
        state = token_hash
        for i in range(5):
            state = (state * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)

            perturbation = np.zeros(dim, dtype=np.float32)
            for j in range(min(dim, 10)):
                perturbation[state % dim] += (state % 1000 - 500) / 5000.0
                state = (state * 1099511628211) & ((1 << 64) - 1)

            X[i] = X[i] + center + perturbation

        # Center the crystal
        X = X - X.mean(axis=0, keepdims=True)

        return X

    def _cardinal_axes_from_definition_full(self, def_text: str, token: str) -> np.ndarray:
        """Full V1 cardinal axes with QR decomposition"""
        v_def = self._text_to_vec(def_text).astype(np.float64)

        C = np.zeros((4, self.dim), dtype=np.float64)
        built = 0

        # First axis: definition direction
        n = float(np.linalg.norm(v_def) if self.config.norm_type == NormType.L2 else np.abs(v_def).sum())
        if n > EPS:
            C[0] = v_def / n
            built = 1

        # Build remaining axes deterministically
        state = self._sha_u64(token) ^ 0xD1F2C3B4A5968778
        mask = (1 << 64) - 1

        while built < 4:
            h = np.zeros(self.dim, dtype=np.float64)
            for _ in range(8):
                state ^= 0x9E3779B97F4A7C15
                state = (state * 1099511628211) & mask
                h[state % self.dim] += 1.0

            # Orthogonalize against existing axes
            vk = h
            for j in range(built):
                vk -= np.dot(vk, C[j]) * C[j]

            n = float(np.linalg.norm(vk) if self.config.norm_type == NormType.L2 else np.abs(vk).sum())
            if n <= EPS:
                idx = (state >> 5) % self.dim
                vk = np.zeros(self.dim)
                vk[idx] = 1.0
                for j in range(built):
                    vk -= np.dot(vk, C[j]) * C[j]
                n = float(np.linalg.norm(vk) if self.config.norm_type == NormType.L2 else np.abs(vk).sum())

            C[built] = vk / (n + EPS)
            built += 1

        # QR decomposition for numerical stability
        M = C.T
        Qr, _ = np.linalg.qr(M, mode='reduced')
        return Qr.T.astype(np.float32)

    def _orthonormal_frame_5_full(self, C4: np.ndarray, token: str, def_text: str) -> np.ndarray:
        """Full orthonormal frame extension with QR"""
        Q = np.zeros((self.dim, 5), dtype=np.float64)

        # Copy cardinal axes
        for k in range(4):
            Q[:, k] = C4[k].astype(np.float64)

        # Fifth axis: orthogonal to all cardinals
        v5 = self._text_to_vec(def_text if def_text else token).astype(np.float64)

        # Orthogonalize
        for k in range(4):
            v5 -= np.dot(v5, Q[:, k]) * Q[:, k]

        n = float(np.linalg.norm(v5) if self.config.norm_type == NormType.L2 else np.abs(v5).sum())
        if n <= EPS:
            # Deterministic fallback
            state = self._sha_u64(token) ^ 0xABCDEF9876543210
            mask = (1 << 64) - 1
            h = np.zeros(self.dim, dtype=np.float64)

            for _ in range(12):
                state ^= 0x9E3779B97F4A7C15
                state = (state * 1099511628211) & mask
                h[state % self.dim] += 1.0

            v5 = h
            for k in range(4):
                v5 -= np.dot(v5, Q[:, k]) * Q[:, k]
            n = float(np.linalg.norm(v5) if self.config.norm_type == NormType.L2 else np.abs(v5).sum())

            if n <= EPS:
                idx = (state >> 7) % self.dim
                v5 = np.zeros(self.dim)
                v5[idx] = 1.0
                for k in range(4):
                    v5 -= np.dot(v5, Q[:, k]) * Q[:, k]
                n = float(np.linalg.norm(v5) if self.config.norm_type == NormType.L2 else np.abs(v5).sum())

        Q[:, 4] = v5 / (n + EPS)

        # Final QR for numerical stability
        Qr, _ = np.linalg.qr(Q, mode='reduced')
        return Qr.astype(np.float32)

    # Formula applications
    def _apply_rose_cayley(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Rose-Cayley with proper semantic distance preservation"""
        if crystal.shape[0] != 5:
            return crystal

        rose = crystal.copy()
        anchor = rose[0].copy()

        for i in range(1, 5):
            # Semantic distance scaling
            semantic_scale = 1.0 / (1.0 + i * 0.25)

            # Angular position in rose pattern
            angle = i * (2 * np.pi / 4)

            # Apply scaling from anchor
            direction = rose[i] - anchor
            rose[i] = anchor + direction * semantic_scale

            # Apply rotation in primary plane
            if rose.shape[1] >= 2:
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                rel_pos = rose[i] - anchor
                rotated = rel_pos.copy()
                rotated[0] = cos_a * rel_pos[0] - sin_a * rel_pos[1]
                rotated[1] = sin_a * rel_pos[0] + cos_a * rel_pos[1]
                rose[i] = anchor + rotated

        # Re-center
        rose = rose - rose.mean(axis=0, keepdims=True)

        return rose

    def _apply_cayley_menger(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        n = crystal.shape[0]
        crystal = crystal - crystal.mean(axis=0, keepdims=True)

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(np.linalg.norm(crystal[i] - crystal[j]))

        if distances:
            target_edge = 1.5
            current_edge = np.median(distances)
            if current_edge > EPS:
                crystal *= (target_edge / current_edge)

        return crystal

    def _apply_cayley(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Pure Cayley determinant optimization"""
        n = crystal.shape[0]

        # Build distance matrix
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = np.linalg.norm(crystal[i] - crystal[j]) ** 2

        # Optimize Cayley determinant
        det = np.linalg.det(D)

        # Scale to optimize determinant
        if abs(det) > EPS:
            target_det = (2.0 ** n) * (n ** n)
            scale = np.power(abs(target_det / det), 1.0 / (2 * n))
            crystal *= scale
        else:
            # Degenerate - add small perturbation
            crystal += np.random.randn(*crystal.shape) * 0.01

        return crystal

    def _apply_menger(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Menger curvature optimization"""
        n = crystal.shape[0]

        if n >= 4:
            curvatures = []

            for i in range(n - 3):
                p1, p2, p3, p4 = crystal[i:i + 4]

                edges = [
                    np.linalg.norm(p2 - p1),
                    np.linalg.norm(p3 - p2),
                    np.linalg.norm(p4 - p3),
                    np.linalg.norm(p4 - p1),
                    np.linalg.norm(p3 - p1),
                    np.linalg.norm(p4 - p2)
                ]

                if min(edges) > EPS:
                    curv = 1.0 / min(edges)
                    curvatures.append(curv)

            if curvatures:
                target_curv = 0.5
                avg_curv = np.mean(curvatures)

                if avg_curv > EPS:
                    scale = target_curv / avg_curv
                    crystal *= scale

        return crystal

    def _apply_euler(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        angles = [np.pi / 6, np.pi / 4, np.pi / 3]

        for angle in angles[:1]:
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            for i in range(crystal.shape[0]):
                temp = crystal[i, 0] * cos_a - crystal[i, 1] * sin_a
                crystal[i, 1] = crystal[i, 0] * sin_a + crystal[i, 1] * cos_a
                crystal[i, 0] = temp

        return crystal

    def _apply_graham_infinite(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Graham infinite with controlled tetration growth"""
        n_vertices = crystal.shape[0]

        for i in range(n_vertices):
            if i == 0:
                g_level = 3
            elif i == 1:
                g_level = 27
            elif i == 2:
                g_level = min(3 ** 27, 1e15)  # Cap for stability
            elif i == 3:
                g_level = 1e20
            else:
                g_level = 10 ** (10 + i * 2)

            scale = np.log1p(g_level) / ((i + 1) * 2.0)
            crystal[i] *= scale

        return crystal

    def _apply_graham_finite(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Graham finite with bounded growth"""
        n_vertices = crystal.shape[0]

        for i in range(n_vertices):
            if i == 0:
                g_level = 3
            elif i == 1:
                g_level = 27
            elif i == 2:
                g_level = 7625
            else:
                g_level = 1000 * (i + 1) ** 3

            scale = np.log1p(g_level) / (i + 2)
            crystal[i] *= scale

        return crystal

    def _apply_graham_masked(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        mask = self.config.graham_mask if self.config.graham_mask is not None else np.random.binomial(1, 0.7,
                                                                                                      crystal.shape)
        graham = self._apply_graham_finite(crystal, token, definition)
        return crystal * (1 - mask) + graham * mask

    def _apply_hybrid_v1v2(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        if definition:
            v_def = self._text_to_vec(definition)
            for i in range(crystal.shape[0]):
                crystal[i] = crystal[i] * 0.7 + v_def * 0.3
        return crystal

    # Content applications
    def _content_sparse(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        for i in range(crystal.shape[0]):
            char = token[i % len(token)] if token else ' '
            crystal[i] = self._create_char_embedding(char)
        return crystal

    def _content_enriched(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        if definition:
            def_vec = self._text_to_vec(definition)
            for i in range(crystal.shape[0]):
                crystal[i] = crystal[i] * 0.8 + def_vec * 0.2
        return crystal

    def _content_trajectory(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Create smooth trajectory through vertices"""
        if crystal.shape[0] <= 1:
            return crystal

        trajectory = crystal.copy()
        alpha = 0.7  # smoothing factor

        for i in range(1, crystal.shape[0]):
            trajectory[i] = alpha * trajectory[i - 1] + (1 - alpha) * crystal[i]

        return trajectory

    def _content_magnitude(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        magnitudes = np.linalg.norm(crystal, axis=1)
        target_mag = np.mean(magnitudes)
        for i in range(crystal.shape[0]):
            if magnitudes[i] > EPS:
                crystal[i] *= (target_mag / magnitudes[i])
        return crystal

    def _content_volume(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        """Volume regularization - scale to target volume"""
        vol = self._compute_volume(crystal)
        target_vol = 1.0

        if vol > EPS:
            # Scale to achieve target volume
            n = crystal.shape[0] - 1  # Dimension of simplex
            if n > 0:
                scale = np.power(target_vol / vol, 1.0 / n)
                crystal *= scale

        return crystal

    def _content_hybrid(self, crystal: Crystal, token: str, definition: Optional[str]) -> Crystal:
        if definition:
            crystal = self._content_enriched(crystal, token, definition)
        if crystal.shape[0] > 3:
            crystal = self._content_trajectory(crystal, token, definition)
        return self._content_volume(crystal, token, definition)

    # Utilities
    def _apply_normalization(self, crystal: Crystal, norm_type: NormType) -> Crystal:
        """Apply normalization to the whole crystal structure"""
        if norm_type == NormType.NONE:
            return crystal

        # Handle both 1D and 2D crystals
        if crystal.ndim == 1:
            if norm_type == NormType.L1:
                total_norm = np.abs(crystal).sum()
            elif norm_type == NormType.L2:
                total_norm = np.linalg.norm(crystal)
            elif norm_type == NormType.LINF:
                total_norm = np.max(np.abs(crystal))
            else:
                return crystal

            if total_norm > EPS:
                return crystal / total_norm
            return crystal

        # Normalize the entire crystal structure
        if norm_type == NormType.L1:
            total_norm = np.abs(crystal).sum()
            if total_norm > EPS:
                return crystal / total_norm
        elif norm_type == NormType.L2:
            # Frobenius norm for matrices
            total_norm = np.linalg.norm(crystal, 'fro')
            if total_norm > EPS:
                return crystal / total_norm
        elif norm_type == NormType.LINF:
            max_val = np.max(np.abs(crystal))
            if max_val > EPS:
                return crystal / max_val

        return crystal

    def _text_to_vec(self, text: str) -> np.ndarray:
        """FNV hash-based text vectorization"""
        acc = np.zeros(self.dim, dtype=np.float64)
        b = text.encode('utf-8', errors='ignore')
        state = 1469598103934665603
        FNV = 1099511628211
        mask = (1 << 64) - 1

        for by in b:
            state ^= by
            state = (state * FNV) & mask
            acc[state % self.dim] += 1.0

        n = float(np.linalg.norm(acc))
        return (acc / n if n > EPS else acc).astype(np.float32)

    def _sha_u64(self, s: str) -> int:
        h = hashlib.sha256(s.encode('utf-8')).digest()
        return int.from_bytes(h[:8], 'little', signed=False)

    def _create_char_embedding(self, char: str) -> np.ndarray:
        seed = ord(char) if len(char) == 1 else hash(char)
        np.random.seed(seed % 2 ** 32)
        vec = np.random.randn(self.dim).astype(np.float32)
        return vec / (np.abs(vec).sum() + EPS)

    def _deterministic_center(self, token: str) -> np.ndarray:
        state = self._sha_u64(token)
        np.random.seed(state % 2 ** 32)
        vec = np.random.randn(self.dim).astype(np.float32)
        return vec / (np.abs(vec).sum() + EPS)

    def _rotate_vector(self, vec: np.ndarray, angle: float) -> np.ndarray:
        rotated = vec.copy()
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        temp = rotated[0] * cos_a - rotated[1] * sin_a
        rotated[1] = rotated[0] * sin_a + rotated[1] * cos_a
        rotated[0] = temp
        return rotated

    def _reshape_crystal(self, crystal: Crystal, target_dim: DimensionType) -> Crystal:
        current_n = crystal.shape[0]
        target_n = target_dim.value

        if current_n == target_n:
            return crystal

        if target_n < current_n:
            indices = np.linspace(0, current_n - 1, target_n, dtype=int)
            return crystal[indices]
        else:
            new_crystal = np.zeros((target_n, crystal.shape[1]), dtype=np.float32)
            for i in range(min(current_n, target_n)):
                new_crystal[i] = crystal[i]
            for i in range(current_n, target_n):
                idx1 = i % current_n
                idx2 = (i + 1) % current_n
                alpha = (i - current_n) / (target_n - current_n)
                new_crystal[i] = crystal[idx1] * (1 - alpha) + crystal[idx2] * alpha
            return new_crystal

    def _compute_volume(self, crystal: Crystal) -> float:
        """CORRECTED Cayley-Menger volume calculation"""
        n = crystal.shape[0]

        # Handle special case for D1 (single vertex has no volume)
        if n == 1:
            return 0.0

        if n < 2:
            return 0.0

        # Build squared distance matrix
        d2 = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                v = crystal[i] - crystal[j]
                d2[i, j] = d2[j, i] = float(np.dot(v, v))

        if n == 2:
            return np.sqrt(d2[0, 1])

        # Cayley-Menger matrix
        M = np.ones((n + 1, n + 1), dtype=np.float64)
        M[0, 0] = 0.0
        M[1:, 1:] = d2

        det = np.linalg.det(M)

        # CRITICAL FIX: Correct volume formulas for (n-1)-simplex
        k = n - 1

        if k == 1:  # Line
            vol2 = abs(det) / 4.0
        elif k == 2:  # Triangle
            vol2 = -det / 16.0
        elif k == 3:  # Tetrahedron
            vol2 = det / 288.0
        elif k == 4:  # 4-simplex
            vol2 = -det / 9216.0
        elif k == 5:  # 5-simplex
            vol2 = det / 460800.0
        elif k == 6:  # 6-simplex
            vol2 = -det / 33177600.0
        else:
            # General formula
            sign = (-1) ** (k + 1)
            factorial_k = 1
            for i in range(1, k + 1):
                factorial_k *= i
            divisor = (2 ** k) * (factorial_k ** 2)
            vol2 = sign * det / divisor

        return float(np.sqrt(abs(vol2))) if abs(vol2) > EPS else 0.0

    def _compute_metadata(self, crystal: Crystal, token: str, definition: Optional[str]) -> Dict[str, Any]:
        metadata = {
            'token': token,
            'has_definition': definition is not None,
        }

        # Handle both 1D and 2D crystals
        if crystal.ndim == 1:
            metadata['n_vertices'] = 1
            metadata['embedding_dim'] = crystal.shape[0]
            metadata['volume'] = 0.0
        else:
            metadata['n_vertices'] = crystal.shape[0]
            metadata['embedding_dim'] = crystal.shape[1]
            metadata['volume'] = self._compute_volume(crystal)

            # Edge statistics
            distances = []
            for i in range(crystal.shape[0]):
                for j in range(i + 1, crystal.shape[0]):
                    distances.append(np.linalg.norm(crystal[i] - crystal[j]))

            if distances:
                metadata['edge_mean'] = float(np.mean(distances))
                metadata['edge_std'] = float(np.std(distances))

        return metadata

    def validate_and_fix_crystal(self, crystal: np.ndarray) -> np.ndarray:
        """Validate and attempt to fix crystal issues"""
        validation = validate_crystal(crystal)

        if not validation['valid']:
            warnings.warn(f"Invalid crystal: {validation['errors']}", RuntimeWarning)
            # Attempt to fix
            crystal = np.nan_to_num(crystal, nan=0.0, posinf=1.0, neginf=-1.0)

        # Re-center if needed
        crystal = crystal - crystal.mean(axis=0, keepdims=True)

        return crystal


if __name__ == "__main__":
    # Example usage
    config = UnifiedCrystalConfig(
        dimension_type=DimensionType.D5,
        formula_type=FormulaType.ROSE_CAYLEY,
        content_type=ContentType.HYBRID,
        norm_type=NormType.L2,
        embedding_dim=128,
        use_character_composition=True
    )

    factory = CrystalFactory(config)
    result = factory.create_crystal("example", "An example definition")

    print("Crystal shape:", result['crystal'].shape)
    print("Volume:", result['volume'])
    print("Metadata:", result['metadata'])

    print("Config:", result['config'])
    print("Validating with cayley menger...")
    valid = validate_crystal(result['crystal'])
    factory.validate_and_fix_crystal(result['crystal'])
    tensor = torch.tensor(result['crystal']).clone()
    tensor_smaller = torch.tensor(result['crystal']).clone() * 0.5
    # rotate the theta of smaller tensor with torch
    tensor_smaller = tensor_smaller @ torch.rot90(torch.eye(tensor.shape[1]), 1, [0, 1])
    print("Validation result:", valid)
    print(tensor)
    print(tensor_smaller)
    sim = cosine_similarity(tensor, tensor_smaller)

    print("Self-similarity matrix shape: ", sim)