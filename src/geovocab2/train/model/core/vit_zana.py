"""
Baseline Vision Transformer with Frozen Pentachora Embeddings
Adapted for L1-normalized pentachora vertices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any

from geovocab2.train.utils.pentachoron_stabilizer import PentachoronStabilizer


class PentachoraEmbedding(nn.Module):
    """
    A single frozen pentachora embedding (5 vertices in geometric space).
    Supports both L1 and L2 normalized vertices.
    """

    def __init__(self, vertices: torch.Tensor, norm_type: str = 'l1'):
        super().__init__()

        self.embed_dim = vertices.shape[-1]
        self.norm_type = norm_type

        # Store provided vertices as frozen buffer
        self.register_buffer('vertices', vertices)
        self.vertices.requires_grad = False

        # Precompute normalized versions and centroid
        with torch.no_grad():
            # For L1-normalized data, use L1 norm for consistency
            if norm_type == 'l1':
                # L1 normalize (sum of abs values = 1)
                self.register_buffer('vertices_norm',
                                     vertices / (vertices.abs().sum(dim=-1, keepdim=True) + 1e-8))
            else:
                # L2 normalize (euclidean norm = 1)
                self.register_buffer('vertices_norm', F.normalize(self.vertices, dim=-1))

            self.register_buffer('centroid', self.vertices.mean(dim=0))

            # Centroid normalization matches vertex normalization
            if norm_type == 'l1':
                self.register_buffer('centroid_norm',
                                     self.centroid / (self.centroid.abs().sum() + 1e-8))
            else:
                self.register_buffer('centroid_norm', F.normalize(self.centroid, dim=-1))

    def get_vertices(self) -> torch.Tensor:
        """Get all 5 vertices."""
        return self.vertices

    def get_centroid(self) -> torch.Tensor:
        """Get the centroid of the pentachora."""
        return self.centroid

    def compute_rose_score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Rose similarity score with this pentachora.
        Scaled appropriately for L1 norm.
        """
        verts = self.vertices.unsqueeze(0)  # [1, 5, D]
        if features.dim() == 1:
            features = features.unsqueeze(0)

        B = features.shape[0]
        if B > 1:
            verts = verts.expand(B, -1, -1)

        # For L1 norm, scale the rose score appropriately
        score = PentachoronStabilizer.rose_score_magnitude(features, verts)
        if self.norm_type == 'l1':
            # L1 norm produces smaller values, so amplify the signal
            score = score * 10.0
        return score

    def compute_similarity(self, features: torch.Tensor, mode: str = 'centroid') -> torch.Tensor:
        """
        Compute similarity between features and this pentachora.
        """
        if mode == 'rose':
            return self.compute_rose_score(features)

        # Normalize features according to norm type
        if self.norm_type == 'l1':
            features_norm = features / (features.abs().sum(dim=-1, keepdim=True) + 1e-8)
        else:
            features_norm = F.normalize(features, dim=-1)

        if mode == 'centroid':
            # Dot product with centroid
            sim = torch.sum(features_norm * self.centroid_norm, dim=-1)
            # Scale up L1 similarities to be comparable to L2
            if self.norm_type == 'l1':
                sim = sim * 10.0
            return sim
        else:  # mode == 'max'
            # Max similarity across vertices
            sims = torch.matmul(features_norm, self.vertices_norm.T)
            if self.norm_type == 'l1':
                sims = sims * 10.0
            return sims.max(dim=-1)[0]


class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention and MLP."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attn_dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attn_dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class BaselineViT(nn.Module):
    """
    Vision Transformer with frozen pentachora embeddings.
    - Preserves L1 law for pentachora geometry.
    - Uses L2 angles for RoseFace (ArcFace/CosFace/SphereFace) classification.
    """

    def __init__(
            self,
            pentachora_list: list,  # List of torch.Tensor, each [5, vocab_dim]
            vocab_dim: int = 256,
            img_size: int = 32,
            patch_size: int = 4,
            embed_dim: int = 512,
            depth: int = 12,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attn_dropout: float = 0.0,
            similarity_mode: str = 'rose',  # legacy similarity (kept for compatibility)
            norm_type: str = 'l1',  # 'l1' or 'l2' normalization for pentachora law
            # --- New RoseFace config ---
            head_type: str = 'roseface',  # 'roseface' | 'legacy'
            prototype_mode: str = 'centroid',  # 'centroid' | 'rose5' | 'max_vertex'
            margin_type: str = 'cosface',  # 'arcface' | 'cosface' | 'sphereface'
            margin_m: float = 0.30,
            scale_s: float = 30.0,
            apply_margin_train_only: bool = False,
    ):
        super().__init__()

        # Validate pentachora list
        assert isinstance(pentachora_list, list), f"Expected list, got {type(pentachora_list)}"
        assert len(pentachora_list) > 0, "Empty pentachora list"
        for i, penta in enumerate(pentachora_list):
            assert isinstance(penta, torch.Tensor), f"Item {i} is not a tensor"

        self.num_classes = len(pentachora_list)
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.similarity_mode = similarity_mode
        self.pentachora_dim = vocab_dim
        self.norm_type = norm_type

        # --- RoseFace config ---
        self.head_type = head_type
        self.prototype_mode = prototype_mode
        self.margin_type = margin_type
        self.margin_m = float(margin_m)
        self.scale_s = float(scale_s)
        self.apply_margin_train_only = apply_margin_train_only

        # Create individual pentachora embeddings from list
        self.class_pentachora = nn.ModuleList([
            PentachoraEmbedding(vertices=penta, norm_type=norm_type)
            for penta in pentachora_list
        ])

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # CLS token - learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Project to pentachora dimension if needed
        if self.pentachora_dim != embed_dim:
            self.to_pentachora_dim = nn.Linear(embed_dim, self.pentachora_dim)
        else:
            self.to_pentachora_dim = nn.Identity()

        # Legacy temperature (used only if head_type == 'legacy')
        if norm_type == 'l1':
            self.temperature = nn.Parameter(torch.zeros(1))  # exp(0)=1
        else:
            self.temperature = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))

        # Precompute all centroids (buffers) for legacy path
        self.register_buffer(
            'all_centroids',
            torch.stack([penta.centroid for penta in self.class_pentachora])
        )
        if norm_type == 'l1':
            centroids_normalized = self.all_centroids / (
                    self.all_centroids.abs().sum(dim=-1, keepdim=True) + 1e-8)
        else:
            centroids_normalized = F.normalize(self.all_centroids, dim=-1)
        self.register_buffer('all_centroids_norm', centroids_normalized)

        # Face weights for rose5 prototypes (10 triads)
        face_triplets = torch.tensor([
            [0, 1, 2], [0, 1, 3], [0, 1, 4],
            [0, 2, 3], [0, 2, 4], [0, 3, 4],
            [1, 2, 3], [1, 2, 4], [1, 3, 4],
            [2, 3, 4]
        ], dtype=torch.long)
        face_weights = torch.zeros(10, 5, dtype=torch.float32)
        for r, (i, j, k) in enumerate(face_triplets):
            face_weights[r, i] = face_weights[r, j] = face_weights[r, k] = 1.0 / 3.0
        self.register_buffer('rose_face_weights', face_weights, persistent=False)

        # Initialize weights
        self.init_weights()

        # Record config for checkpoint saving
        self.config = getattr(self, 'config', {})
        self.config.update({
            'head_type': self.head_type,
            'prototype_mode': self.prototype_mode,
            'margin_type': self.margin_type,
            'margin_m': self.margin_m,
            'scale_s': self.scale_s,
            'apply_margin_train_only': self.apply_margin_train_only,
            'norm_type': self.norm_type,
            'similarity_mode': self.similarity_mode,
            'pentachora_dim': self.pentachora_dim,
        })

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---- Legacy helper (kept) ----
    def get_class_centroids(self) -> torch.Tensor:
        return self.all_centroids_norm

    # ---- Legacy similarity (kept for compatibility & debugging) ----
    def compute_pentachora_similarities(self, features: torch.Tensor) -> torch.Tensor:
        if self.similarity_mode == 'rose':
            all_vertices = torch.stack([penta.vertices for penta in self.class_pentachora])
            features_exp = features.unsqueeze(1).expand(-1, self.num_classes, -1)
            scores = PentachoronStabilizer.rose_score_magnitude(
                features_exp.reshape(-1, self.pentachora_dim),
                all_vertices.repeat(features.shape[0], 1, 1)
            ).reshape(features.shape[0], -1)
            if self.norm_type == 'l1':
                scores = scores * 10.0
            return scores
        else:
            if self.norm_type == 'l1':
                features_norm = features / (features.abs().sum(dim=-1, keepdim=True) + 1e-8)
            else:
                features_norm = F.normalize(features, dim=-1)
            centroids = self.get_class_centroids()
            sims = torch.matmul(features_norm, centroids.T)
            if self.norm_type == 'l1':
                sims = sims * 10.0
            return sims

    # ---- RoseFace utilities ----
    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    def _get_class_vertices_l2(self) -> torch.Tensor:
        """[C,5,D] L2-normalized vertices for all classes."""
        V = torch.stack([p.vertices for p in self.class_pentachora], dim=0)
        V = V.to(self.pos_embed.device, dtype=self.pos_embed.dtype)
        return self._l2_norm(V)

    def _get_prototypes(self, mode: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        Prototypes [C,D] for 'centroid'/'rose5'; None for 'max_vertex'.
        """
        mode = mode or self.prototype_mode
        device = self.pos_embed.device
        dtype = self.pos_embed.dtype

        if mode == 'centroid':
            C = torch.stack([p.centroid for p in self.class_pentachora], dim=0).to(device, dtype)
            return self._l2_norm(C)

        elif mode == 'rose5':
            V_l2 = self._get_class_vertices_l2()  # [C,5,D]
            W = self.rose_face_weights.to(device=device, dtype=dtype)  # [10,5]
            faces = torch.einsum('tf,cfd->ctd', W, V_l2)  # [C,10,D]
            verts_mean = V_l2.mean(dim=1)  # [C,D]
            faces_mean = faces.mean(dim=1)  # [C,D]
            alpha, beta = 1.0, 0.5
            proto = alpha * verts_mean + beta * faces_mean
            return self._l2_norm(proto)

        elif mode == 'max_vertex':
            return None

        else:
            raise ValueError(f"Unknown prototype_mode: {mode}")

    def _cosine_matrix(self, z_l2: torch.Tensor) -> torch.Tensor:
        """
        Pre-margin cosine [B,C] based on prototype_mode.
        """
        if self.prototype_mode in ('centroid', 'rose5'):
            P = self._get_prototypes(self.prototype_mode)  # [C,D]
            return torch.matmul(z_l2, P.t())  # [B,C]
        elif self.prototype_mode == 'max_vertex':
            V_l2 = self._get_class_vertices_l2()  # [C,5,D]
            cos_cv = torch.einsum('bd,cvd->bcv', z_l2, V_l2)  # [B,C,5]
            cos_max, _ = cos_cv.max(dim=2)  # [B,C]
            return cos_max
        else:
            raise ValueError(f"Unknown prototype_mode: {self.prototype_mode}")

    @staticmethod
    def _apply_margin(cosine: torch.Tensor, targets: torch.Tensor, m: float, kind: str = 'cosface') -> torch.Tensor:
        """
        Apply margin to target class cosines. Returns adjusted cosines [B,C].
        """
        eps = 1e-7
        B, C = cosine.shape
        y = targets.view(-1, 1)  # [B,1]

        if kind == 'cosface':
            cos_m = cosine.clone()
            cos_m.scatter_(1, y, (cosine.gather(1, y) - m))
            return cos_m

        theta = torch.acos(torch.clamp(cosine.gather(1, y), -1.0 + eps, 1.0 - eps))  # [B,1]
        if kind == 'arcface':
            cos_margin = torch.cos(theta + m)
        elif kind == 'sphereface':
            cos_margin = torch.cos(m * theta)
        else:
            raise ValueError(f"Unknown margin type: {kind}")

        cos_m = cosine.clone()
        cos_m.scatter_(1, y, cos_margin)
        return cos_m

    def schedule_roseface(
            self, epoch: int, warmup_epochs: int = 15, s_start: float = 10.0, s_final: float = 30.0,
            m_start: Optional[float] = None, m_final: Optional[float] = None
    ):
        """
        Deterministic cosine ramp for scale s (and optional margin m).
        """
        t = max(0.0, min(1.0, epoch / max(1, warmup_epochs)))
        # cosine ramp from s_start -> s_final
        self.scale_s = float(s_final - 0.5 * (1.0 + np.cos(np.pi * t)) * (s_final - s_start))
        if (m_start is not None) and (m_final is not None):
            self.margin_m = float(m_final - 0.5 * (1.0 + np.cos(np.pi * t)) * (m_final - m_start))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(
            self,
            x: torch.Tensor,
            return_features: bool = False,
            targets: Optional[torch.Tensor] = None  # NEW: required for margin at train time
    ) -> Dict[str, torch.Tensor]:

        features = self.forward_features(x)
        output: Dict[str, torch.Tensor] = {}

        # Project to pentachora dimension (L1 law applies here)
        features_proj = self.to_pentachora_dim(features)
        if self.norm_type == 'l1':
            features_proj = features_proj / (features_proj.abs().sum(dim=-1, keepdim=True) + 1e-8)

        if self.head_type == 'roseface':
            # L2 angles for classification head (dual-norm bridge)
            z_l2 = features_proj / (features_proj.norm(p=2, dim=-1, keepdim=True) + 1e-12)

            # Pre-margin cosines [B,C]
            cos_pre = self._cosine_matrix(z_l2)

            # Apply margin (train-time if configured)
            if (self.apply_margin_train_only and not self.training) or (targets is None):
                cos_post = cos_pre
            else:
                cos_post = self._apply_margin(cos_pre, targets, self.margin_m, self.margin_type)

            # Scaled logits
            logits = self.scale_s * cos_post

            # Emit outputs
            output['logits'] = logits  # for CE
            output['similarities'] = cos_pre  # pre-margin (for alignment / diagnostics)
            if return_features:
                output['features'] = features
                output['features_proj'] = features_proj

        else:
            # Legacy path (kept for compatibility)
            similarities = self.compute_pentachora_similarities(features_proj)
            logits = similarities * self.temperature.exp()
            output['logits'] = logits
            output['similarities'] = similarities
            if return_features:
                output['features'] = features
                output['features_proj'] = features_proj

        return output


# Test - requires external setup
if __name__ == "__main__":
    print("BaselineViT requires:")
    print("  1. PentachoronStabilizer loaded externally")
    print("  2. pentachora_batch tensor [num_classes, 5, vocab_dim]")
    print("\nNo random initialization. No fallbacks.")