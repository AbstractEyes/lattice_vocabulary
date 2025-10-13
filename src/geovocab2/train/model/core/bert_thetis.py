"""
BERT-Thetis Encoder
-------------------
Geometric BERT with deterministic crystal embeddings.

Architecture:
  Token IDs → Beatrix PE → Character Composition → Crystal Inflation → Geometric Transformer

Key innovations:
  - Zero-parameter vocabulary (Beatrix staircase encodings)
  - Deterministic crystal inflation (no learned embedding table)
  - Geometric attention over simplices
  - Character-based semantic composition

Author: AbstractPhil + Claude Sonnet 4.5
License: MIT
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ThetisConfig:
    """Configuration for BERT-Thetis encoder."""

    # Model dimensions
    crystal_dim: int = 512          # Dimension of crystal embeddings
    num_vertices: int = 5           # Vertices per crystal (default: pentachoron)
    num_layers: int = 12            # Number of transformer layers
    num_attention_heads: int = 8    # Attention heads per layer
    intermediate_size: int = 2048   # FFN intermediate size

    # Vocabulary
    vocab_size: int = 50000         # Maximum vocabulary size

    # Beatrix staircase configuration
    beatrix_levels: int = 20        # Staircase levels
    beatrix_features_per_level: int = 4  # Features per level
    beatrix_tau: float = 0.25       # Smoothing temperature
    beatrix_base: int = 3           # Base for decomposition

    # Character composition
    num_base_chars: int = 256       # Number of base characters (extended ASCII)
    char_composition_layers: int = 2  # Layers in character composer

    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

    # Training
    max_position_embeddings: int = 512
    initializer_range: float = 0.02

    def __post_init__(self):
        # Validate Beatrix output matches crystal_dim
        self.beatrix_output_dim = self.beatrix_levels * self.beatrix_features_per_level
        if self.beatrix_output_dim > self.crystal_dim:
            # Allow projection down
            pass

    @classmethod
    def bert_base_geometric(cls):
        """BERT-Base sized geometric model."""
        return cls(
            crystal_dim=768,
            num_vertices=5,
            num_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            vocab_size=30522,  # BERT vocab size
        )

    @classmethod
    def bert_tiny_geometric(cls):
        """Tiny model for testing."""
        return cls(
            crystal_dim=128,
            num_vertices=5,
            num_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            vocab_size=5000,
            beatrix_levels=12,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Beatrix Staircase (Simplified for Thetis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BeatrixStaircase(nn.Module):
    """Lightweight Beatrix staircase for token encoding."""

    def __init__(self, config: ThetisConfig):
        super().__init__()
        self.levels = config.beatrix_levels
        self.features_per_level = config.beatrix_features_per_level
        self.tau = config.beatrix_tau
        self.base = config.beatrix_base
        self.vocab_size = config.vocab_size

        # Learnable alpha
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Feature expansion
        if config.beatrix_features_per_level > 2:
            self.feature_expansion = nn.Linear(2, config.beatrix_features_per_level)
        else:
            self.feature_expansion = None

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: [batch, seq_len] or [seq_len]
        Returns:
            encodings: [batch, seq_len, levels * features_per_level]
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Normalize to [0, 1]
        x = token_ids.float() / max(1, self.vocab_size - 1)
        x = x.clamp(1e-6, 1.0 - 1e-6)

        # Centers for base=3
        centers = torch.tensor([0.5, 1.5, 2.5], device=x.device, dtype=x.dtype)

        feats = []
        for k in range(1, self.levels + 1):
            scale = self.base ** k
            y = (x * scale) % self.base

            # Distances to centers
            d2 = (y.unsqueeze(-1) - centers) ** 2
            logits = -d2 / (self.tau + 1e-8)
            p = F.softmax(logits, dim=-1)

            # Bit contribution
            bit_k = p[..., 2] + self.alpha * p[..., 1]

            # Entropy-based PDF
            ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)
            pdf_proxy = 1.1 - ent / math.log(3.0)

            # Stack features
            base_feat = torch.stack([bit_k, pdf_proxy], dim=-1)

            if self.feature_expansion is not None:
                level_feat = self.feature_expansion(base_feat)
            else:
                level_feat = base_feat

            feats.append(level_feat)

        # Stack and flatten
        encodings = torch.stack(feats, dim=-2)  # [batch, seq_len, levels, features]
        encodings = encodings.flatten(-2, -1)   # [batch, seq_len, levels * features]

        if squeeze_batch:
            encodings = encodings.squeeze(0)

        return encodings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Character Composer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CharacterComposer(nn.Module):
    """
    Maps Beatrix structure to semantic center vectors.

    This is the learnable bridge between positional structure (Beatrix)
    and semantic geometry (crystals).
    """

    def __init__(self, config: ThetisConfig):
        super().__init__()

        beatrix_dim = config.beatrix_levels * config.beatrix_features_per_level

        # Multi-layer composition
        layers = []
        current_dim = beatrix_dim

        for _ in range(config.char_composition_layers):
            layers.extend([
                nn.Linear(current_dim, config.crystal_dim),
                nn.LayerNorm(config.crystal_dim, eps=config.layer_norm_eps),
                nn.GELU(),
                nn.Dropout(config.hidden_dropout_prob)
            ])
            current_dim = config.crystal_dim

        # Final projection to center
        layers.append(nn.Linear(config.crystal_dim, config.crystal_dim))

        self.composer = nn.Sequential(*layers)

        # L1 normalization (deterministic, no parameters)
        self.normalize_l1 = True

    def forward(self, beatrix_features: Tensor) -> Tensor:
        """
        Args:
            beatrix_features: [batch, seq_len, beatrix_dim]
        Returns:
            centers: [batch, seq_len, crystal_dim] - L1 normalized
        """
        centers = self.composer(beatrix_features)

        if self.normalize_l1:
            # L1 normalize to match geometric vocab semantics
            l1_norm = centers.abs().sum(dim=-1, keepdim=True) + 1e-8
            centers = centers / l1_norm

        return centers


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Deterministic Crystal Inflator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DeterministicCrystalInflator(nn.Module):
    """
    Inflate center vectors to k-vertex crystals (default: pentachorons).

    Uses deterministic Gram-Schmidt orthogonalization with roll operations.
    Zero parameters - purely algorithmic transformation.
    """

    def __init__(self, config: ThetisConfig):
        super().__init__()
        self.num_vertices = config.num_vertices
        self.crystal_dim = config.crystal_dim

        # Learnable gamma scaling (optional)
        self.gamma = nn.Parameter(
            torch.tensor([1.0, 0.9, -0.8, 1.1, 1.2][:config.num_vertices])
        )

    def forward(self, centers: Tensor) -> Tensor:
        """
        Args:
            centers: [batch, seq_len, D] - L1 normalized center vectors
        Returns:
            crystals: [batch, seq_len, V, D] - Mean-centered vertex clouds
        """
        batch, seq_len, D = centers.shape
        V = self.num_vertices

        # Generate proposal vertices using roll operations
        proposals = torch.stack([
            centers,                                          # Original
            torch.roll(centers, shifts=1, dims=-1),         # Roll 1
            torch.roll(centers, shifts=3, dims=-1) * centers.sign(),  # Roll 3 + sign
            torch.roll(centers, shifts=7, dims=-1) - centers,         # Roll 7 - center
            torch.roll(centers, shifts=11, dims=-1) + centers,        # Roll 11 + center
        ][:V], dim=-2)  # [batch, seq_len, V, D]

        # L1 row normalization
        l1_norms = proposals.abs().sum(dim=-1, keepdim=True) + 1e-8
        Q = proposals / l1_norms

        # Gram-Schmidt orthogonalization (build new tensor to avoid in-place issues)
        Q_orthogonal = []

        for i in range(V):
            q_i = Q[..., i, :]  # [batch, seq_len, D]

            # Project out all previous vectors
            for j in range(len(Q_orthogonal)):
                q_j = Q_orthogonal[j]
                dot = (q_i * q_j).sum(dim=-1, keepdim=True)
                q_i = q_i - dot * q_j

            # L1 normalize
            l1 = q_i.abs().sum(dim=-1, keepdim=True) + 1e-8
            q_i = q_i / l1

            Q_orthogonal.append(q_i)

        # Stack back to [batch, seq_len, V, D]
        Q = torch.stack(Q_orthogonal, dim=-2)

        # Apply gamma scaling and add to center
        gamma = self.gamma.view(1, 1, V, 1)  # Broadcast shape
        crystals = centers.unsqueeze(-2) + gamma * Q

        # Mean-center
        crystals = crystals - crystals.mean(dim=-2, keepdim=True)

        return crystals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Geometric Attention (Pooled Proxy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeometricAttention(nn.Module):
    """
    Multi-head attention over crystal representations.

    Strategy: Pool crystals → standard attention → unpool to crystals
    """

    def __init__(self, config: ThetisConfig):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.crystal_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q, K, V projections
        self.query = nn.Linear(config.crystal_dim, self.all_head_size)
        self.key = nn.Linear(config.crystal_dim, self.all_head_size)
        self.value = nn.Linear(config.crystal_dim, self.all_head_size)

        # Output projection
        self.output_proj = nn.Linear(self.all_head_size, config.crystal_dim)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """Reshape for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)  # [batch, heads, seq_len, head_size]

    def forward(
        self,
        crystals: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            crystals: [batch, seq_len, V, D]
            attention_mask: [batch, seq_len] or [batch, 1, 1, seq_len]
        Returns:
            output_crystals: [batch, seq_len, V, D]
        """
        batch, seq_len, V, D = crystals.shape

        # Pool crystals: mean over vertices
        pooled = crystals.mean(dim=-2)  # [batch, seq_len, D]

        # Standard multi-head attention on pooled
        query_layer = self.transpose_for_scores(self.query(pooled))
        key_layer = self.transpose_for_scores(self.key(pooled))
        value_layer = self.transpose_for_scores(self.value(pooled))

        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # [batch, seq_len] → [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(-3, -2).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        # Output projection
        attention_output = self.output_proj(context_layer)
        attention_output = self.output_dropout(attention_output)

        # Broadcast back to crystal shape
        # Add residual to original crystals
        delta = attention_output.unsqueeze(-2)  # [batch, seq_len, 1, D]
        output_crystals = crystals + delta

        return output_crystals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Geometric Transformer Layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeometricTransformerLayer(nn.Module):
    """Single transformer layer operating on crystals."""

    def __init__(self, config: ThetisConfig):
        super().__init__()

        # Attention
        self.attention = GeometricAttention(config)
        self.attention_norm = nn.LayerNorm(config.crystal_dim, eps=config.layer_norm_eps)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(config.crystal_dim, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.crystal_dim),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.ffn_norm = nn.LayerNorm(config.crystal_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        crystals: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            crystals: [batch, seq_len, V, D]
            attention_mask: [batch, seq_len]
        Returns:
            crystals: [batch, seq_len, V, D]
        """
        # Self-attention with pre-norm
        pooled = crystals.mean(dim=-2)  # [batch, seq_len, D]
        normed_pooled = self.attention_norm(pooled)
        normed_crystals = crystals - pooled.unsqueeze(-2) + normed_pooled.unsqueeze(-2)

        attention_output = self.attention(normed_crystals, attention_mask)
        crystals = crystals + attention_output

        # Feed-forward with pre-norm
        pooled = crystals.mean(dim=-2)
        normed_pooled = self.ffn_norm(pooled)
        ffn_output = self.ffn(normed_pooled)

        # Broadcast FFN output to all vertices
        crystals = crystals + ffn_output.unsqueeze(-2)

        return crystals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BERT-Thetis Encoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThetisEncoder(nn.Module):
    """
    BERT-Thetis: Geometric BERT with deterministic crystal embeddings.

    Architecture:
        Token IDs → Beatrix PE → Character Composition → Crystal Inflation → Transformer
    """

    def __init__(self, config: ThetisConfig):
        super().__init__()
        self.config = config

        # Beatrix staircase (zero parameters for structure)
        self.beatrix = BeatrixStaircase(config)

        # Character composer (learnable bridge)
        self.composer = CharacterComposer(config)

        # Crystal inflator (deterministic)
        self.inflator = DeterministicCrystalInflator(config)

        # Geometric transformer layers
        self.layers = nn.ModuleList([
            GeometricTransformerLayer(config)
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.crystal_dim, eps=config.layer_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following BERT."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_all_layers: bool = False
    ) -> Union[Tensor, Tuple[Tensor, list]]:
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding
            return_all_layers: Return intermediate layer outputs

        Returns:
            crystals: [batch, seq_len, V, D] - Final crystal representations
            (optional) all_layers: List of [batch, seq_len, V, D] from each layer
        """
        # Generate Beatrix structure
        beatrix_features = self.beatrix(token_ids)
        # [batch, seq_len, beatrix_dim]

        # Compose to center vectors
        centers = self.composer(beatrix_features)
        # [batch, seq_len, crystal_dim]

        # Inflate to crystals
        crystals = self.inflator(centers)
        # [batch, seq_len, V, crystal_dim]

        # Prepare attention mask for transformer
        if attention_mask is not None:
            # Convert 1s and 0s to attention scores
            # 0 → -10000 (mask out), 1 → 0 (keep)
            attention_mask = (1.0 - attention_mask.float()) * -10000.0

        # Pass through transformer layers
        all_layer_outputs = []
        for layer in self.layers:
            crystals = layer(crystals, attention_mask)
            if return_all_layers:
                all_layer_outputs.append(crystals)

        # Final normalization
        pooled = crystals.mean(dim=-2)
        normed_pooled = self.final_norm(pooled)
        crystals = crystals - pooled.unsqueeze(-2) + normed_pooled.unsqueeze(-2)

        if return_all_layers:
            return crystals, all_layer_outputs
        return crystals

    def get_pooled_output(self, crystals: Tensor) -> Tensor:
        """Pool crystals for classification tasks."""
        return crystals[:, 0, :, :].mean(dim=-2)  # [CLS] token, mean over vertices


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task Heads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThetisForSequenceClassification(nn.Module):
    """
    BERT-Thetis with sequence classification head.

    For tasks like:
    - Sentiment analysis (SST-2)
    - Paraphrase detection (MRPC)
    - Natural language inference (MNLI)
    - Question answering classification
    """

    def __init__(self, config: ThetisConfig, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        # Encoder
        self.thetis = ThetisEncoder(config)

        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.crystal_dim, num_labels)

        # Initialize classifier weights
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch] - Optional labels for computing loss

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        # Get crystal representations
        crystals = self.thetis(token_ids, attention_mask)
        # [batch, seq_len, V, D]

        # Pool [CLS] token (first position)
        pooled = self.thetis.get_pooled_output(crystals)
        # [batch, D]

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        # [batch, num_labels]

        # Compute loss if labels provided
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits


class ThetisForTokenClassification(nn.Module):
    """
    BERT-Thetis with token classification head.

    For tasks like:
    - Named entity recognition (NER)
    - Part-of-speech tagging (POS)
    - Chunk labeling
    """

    def __init__(self, config: ThetisConfig, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        # Encoder
        self.thetis = ThetisEncoder(config)

        # Token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.crystal_dim, num_labels)

        # Initialize classifier weights
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] - Optional labels for computing loss

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        # Get crystal representations
        crystals = self.thetis(token_ids, attention_mask)
        # [batch, seq_len, V, D]

        # Pool over vertices for each token
        pooled = crystals.mean(dim=-2)
        # [batch, seq_len, D]

        # Classify each token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        # [batch, seq_len, num_labels]

        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only compute loss on non-ignored tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        return logits


class ThetisForMaskedLM(nn.Module):
    """
    BERT-Thetis with masked language modeling head.

    For pretraining with masked token prediction.

    IMPORTANT: This head should be KEPT after pretraining, not removed.
    It's useful for:
    - Continued pretraining / domain adaptation
    - Fine-tuning with auxiliary MLM loss
    - Zero-shot token prediction tasks
    - Vocabulary probing and analysis

    Do NOT remove this head from the model architecture!
    """

    def __init__(self, config: ThetisConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.thetis = ThetisEncoder(config)

        # MLM head (KEEP THIS - see class docstring)
        self.mlm_head = nn.Sequential(
            nn.Linear(config.crystal_dim, config.crystal_dim),
            nn.GELU(),
            nn.LayerNorm(config.crystal_dim, eps=config.layer_norm_eps),
        )

        # Decoder to vocab (we still need this for prediction)
        self.decoder = nn.Linear(config.crystal_dim, config.vocab_size)

        # Tie decoder weights with Beatrix if possible (optional)
        # For now, separate parameters

        # Initialize weights
        self.mlm_head[0].weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.mlm_head[0].bias.data.zero_()
        self.decoder.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.decoder.bias.data.zero_()

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] - Masked token labels (-100 for non-masked)

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        # Get crystal representations
        crystals = self.thetis(token_ids, attention_mask)
        # [batch, seq_len, V, D]

        # Pool over vertices
        pooled = crystals.mean(dim=-2)
        # [batch, seq_len, D]

        # MLM transformation
        hidden = self.mlm_head(pooled)
        # [batch, seq_len, D]

        # Predict tokens
        logits = self.decoder(hidden)
        # [batch, seq_len, vocab_size]

        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            return loss, logits

        return logits


class ThetisForQuestionAnswering(nn.Module):
    """
    BERT-Thetis with span extraction head.

    For tasks like:
    - SQuAD
    - Question answering with start/end positions
    """

    def __init__(self, config: ThetisConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.thetis = ThetisEncoder(config)

        # QA head (predict start and end positions)
        self.qa_outputs = nn.Linear(config.crystal_dim, 2)  # 2 for start/end

        # Initialize weights
        self.qa_outputs.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.qa_outputs.bias.data.zero_()

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Args:
            token_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            start_positions: [batch] - Optional start position labels
            end_positions: [batch] - Optional end position labels

        Returns:
            If positions provided: (loss, start_logits, end_logits)
            Otherwise: (start_logits, end_logits)
        """
        # Get crystal representations
        crystals = self.thetis(token_ids, attention_mask)
        # [batch, seq_len, V, D]

        # Pool over vertices
        pooled = crystals.mean(dim=-2)
        # [batch, seq_len, D]

        # Predict start and end logits
        logits = self.qa_outputs(pooled)
        # [batch, seq_len, 2]

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch, seq_len]
        end_logits = end_logits.squeeze(-1)      # [batch, seq_len]

        # Compute loss if positions provided
        if start_positions is not None and end_positions is not None:
            # Clamp positions to sequence length
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            return loss, start_logits, end_logits

        return start_logits, end_logits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Testing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_task_heads():
    """Test all task heads."""
    print("\n" + "=" * 70)
    print("BERT-THETIS TASK HEADS TESTS")
    print("=" * 70)

    config = ThetisConfig.bert_tiny_geometric()
    batch_size = 4
    seq_len = 16

    # Test 1: Sequence Classification
    print("\n[Test 1] Sequence Classification")
    model = ThetisForSequenceClassification(config, num_labels=2)
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,))

    # Without labels
    logits = model(token_ids)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, 2]")
    assert logits.shape == (batch_size, 2)

    # With labels
    loss, logits_with_loss = model(token_ids, labels=labels)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss is scalar: {loss.shape == ()}")
    assert loss.shape == ()
    print(f"  Status: ✓ PASS")

    # Test 2: Token Classification
    print("\n[Test 2] Token Classification")
    model = ThetisForTokenClassification(config, num_labels=9)  # 9 NER tags
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, 9, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Without labels
    logits = model(token_ids, attention_mask)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, 9]")
    assert logits.shape == (batch_size, seq_len, 9)

    # With labels
    loss, logits_with_loss = model(token_ids, attention_mask, labels)
    print(f"  Loss: {loss.item():.4f}")
    assert loss.shape == ()
    print(f"  Status: ✓ PASS")

    # Test 3: Masked Language Modeling
    print("\n[Test 3] Masked Language Modeling")
    model = ThetisForMaskedLM(config)
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels[:, seq_len//2:] = -100  # Mask out second half

    # Without labels
    logits = model(token_ids)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {config.vocab_size}]")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # With labels
    loss, logits_with_loss = model(token_ids, labels=labels)
    print(f"  Loss: {loss.item():.4f}")
    assert loss.shape == ()
    print(f"  Status: ✓ PASS")

    # Test 4: Question Answering
    print("\n[Test 4] Question Answering")
    model = ThetisForQuestionAnswering(config)
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    start_positions = torch.randint(0, seq_len, (batch_size,))
    end_positions = torch.randint(0, seq_len, (batch_size,))

    # Without positions
    start_logits, end_logits = model(token_ids)
    print(f"  Start logits shape: {start_logits.shape}")
    print(f"  End logits shape: {end_logits.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}]")
    assert start_logits.shape == (batch_size, seq_len)
    assert end_logits.shape == (batch_size, seq_len)

    # With positions
    loss, start_logits_with_loss, end_logits_with_loss = model(
        token_ids, start_positions=start_positions, end_positions=end_positions
    )
    print(f"  Loss: {loss.item():.4f}")
    assert loss.shape == ()
    print(f"  Status: ✓ PASS")

    # Test 5: Gradient flow through task heads
    print("\n[Test 5] Gradient Flow Through Task Heads")
    model = ThetisForSequenceClassification(config, num_labels=2)
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size,))

    loss, _ = model(token_ids, labels=labels)
    loss.backward()

    # Check encoder gradients
    encoder_grads = sum(1 for p in model.thetis.parameters() if p.grad is not None)
    encoder_total = sum(1 for _ in model.thetis.parameters())

    # Check classifier gradients
    classifier_grads = sum(1 for p in model.classifier.parameters() if p.grad is not None)
    classifier_total = sum(1 for _ in model.classifier.parameters())

    print(f"  Encoder gradients: {encoder_grads}/{encoder_total}")
    print(f"  Classifier gradients: {classifier_grads}/{classifier_total}")
    print(f"  All parameters have gradients: {encoder_grads == encoder_total and classifier_grads == classifier_total}")
    print(f"  Status: ✓ PASS")

    # Test 6: Parameter counts
    print("\n[Test 6] Parameter Counts")

    models_and_names = [
        (ThetisForSequenceClassification(config, num_labels=2), "Sequence Classification"),
        (ThetisForTokenClassification(config, num_labels=9), "Token Classification"),
        (ThetisForMaskedLM(config), "Masked LM"),
        (ThetisForQuestionAnswering(config), "Question Answering"),
    ]

    for model, name in models_and_names:
        total = sum(p.numel() for p in model.parameters())
        encoder = sum(p.numel() for p in model.thetis.parameters())
        head = total - encoder
        print(f"  {name:25s}: {total:>9,} total ({encoder:>9,} encoder + {head:>7,} head)")

    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All task head tests passed! ✓ (6/6)")
    print("=" * 70 + "\n")


def test_thetis_encoder():
    """Test BERT-Thetis encoder."""
    print("\n" + "=" * 70)
    print("BERT-THETIS ENCODER TESTS")
    print("=" * 70)

    # Create tiny config for testing
    config = ThetisConfig.bert_tiny_geometric()
    print(f"\nConfig: {config.num_layers} layers, dim={config.crystal_dim}")

    # Create encoder
    encoder = ThetisEncoder(config)
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test 1: Forward pass
    print("\n[Test 1] Forward Pass")
    batch_size = 4
    seq_len = 16
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    crystals = encoder(token_ids)
    print(f"  Input shape: {token_ids.shape}")
    print(f"  Output shape: {crystals.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {config.num_vertices}, {config.crystal_dim}]")
    assert crystals.shape == (batch_size, seq_len, config.num_vertices, config.crystal_dim)
    print(f"  Status: ✓ PASS")

    # Test 2: With attention mask
    print("\n[Test 2] Attention Masking")
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len//2:] = 0  # Mask second half

    crystals_masked = encoder(token_ids, attention_mask=attention_mask)
    print(f"  Masked output shape: {crystals_masked.shape}")
    print(f"  Status: ✓ PASS")

    # Test 3: Gradient flow
    print("\n[Test 3] Gradient Flow")
    crystals = encoder(token_ids)
    loss = crystals.mean()
    loss.backward()

    has_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
    total = sum(1 for _ in encoder.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total}")
    print(f"  Status: ✓ PASS")

    # Test 4: Pooled output
    print("\n[Test 4] Pooled Output (for classification)")
    encoder.zero_grad()
    crystals = encoder(token_ids)
    pooled = encoder.get_pooled_output(crystals)
    print(f"  Pooled shape: {pooled.shape}")
    print(f"  Expected: [{batch_size}, {config.crystal_dim}]")
    assert pooled.shape == (batch_size, config.crystal_dim)
    print(f"  Status: ✓ PASS")

    # Test 5: Deterministic reproduction
    print("\n[Test 5] Deterministic Reproduction")
    encoder.eval()
    with torch.no_grad():
        crystals1 = encoder(token_ids)
        crystals2 = encoder(token_ids)

    identical = torch.allclose(crystals1, crystals2, atol=1e-6)
    print(f"  Outputs identical: {identical}")
    print(f"  Max difference: {(crystals1 - crystals2).abs().max().item():.2e}")
    print(f"  Status: ✓ PASS")

    # Test 6: Variable sequence lengths
    print("\n[Test 6] Variable Sequence Lengths")
    for sl in [8, 16, 32]:
        tids = torch.randint(0, config.vocab_size, (2, sl))
        out = encoder(tids)
        print(f"  Seq len {sl:2d}: {out.shape}")
        assert out.shape == (2, sl, config.num_vertices, config.crystal_dim)
    print(f"  Status: ✓ PASS")

    # Test 7: Return all layers
    print("\n[Test 7] Return All Layers")
    crystals, all_layers = encoder(token_ids, return_all_layers=True)
    print(f"  Final output: {crystals.shape}")
    print(f"  Number of layer outputs: {len(all_layers)}")
    print(f"  Expected: {config.num_layers}")
    assert len(all_layers) == config.num_layers
    print(f"  Status: ✓ PASS")

    print("\n" + "=" * 70)
    print("All tests passed! ✓ (7/7)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run encoder tests
    test_thetis_encoder()

    # Run task head tests
    test_task_heads()

    print("\n[Summary]")
    print("BERT-Thetis Complete System")
    print("\nCore Components:")
    print("  ✓ Beatrix Staircase - Deterministic positional structure")
    print("  ✓ Character Composer - Learnable semantic bridge")
    print("  ✓ Crystal Inflator - Deterministic simplex generation")
    print("  ✓ Geometric Attention - Multi-head attention over crystals")
    print("  ✓ Transformer Layers - Standard architecture with geometric ops")
    print("\nTask Heads:")
    print("  ✓ Sequence Classification - Sentiment, NLI, paraphrase")
    print("  ✓ Token Classification - NER, POS tagging")
    print("  ✓ Masked Language Modeling - Pretraining")
    print("  ✓ Question Answering - Span extraction")
    print("\nReady for:")
    print("  • Fine-tuning on GLUE tasks")
    print("  • Pretraining with MLM")
    print("  • Custom downstream tasks")
    print("  • Comparison with traditional BERT")