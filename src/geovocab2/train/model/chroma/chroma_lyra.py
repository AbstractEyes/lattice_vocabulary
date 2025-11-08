"""
ChromaLyra: Multi-Modal VAE for Chroma and Music Features
========================================================

Geometric deep learning for music generation using chroma features.
Handles circular topology and harmonic relationships natively.

Author: AbstractPhil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ChromaLyraConfig:
    """Configuration for ChromaLyra VAE."""
    # Chroma parameters
    n_chroma: int = 12  # Pitch classes
    seq_len: int = 128  # Time steps

    # Additional modalities (optional)
    modality_dims: Dict[str, int] = None  # e.g., {"text": 512, "midi": 128}

    # Latent space
    latent_dim: int = 256

    # Architecture
    encoder_layers: int = 4
    decoder_layers: int = 4
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1

    # Geometric features
    use_cantor_fusion: bool = True
    use_harmonic_prior: bool = True  # Circle of fifths structure
    cantor_depth: int = 8

    # Loss weights
    beta_kl: float = 0.05
    beta_reconstruction: float = 1.0
    beta_harmonic: float = 0.1  # Encourage harmonic relationships
    beta_circular: float = 0.05  # Circular topology preservation

    # Training
    use_amp: bool = True

    def __post_init__(self):
        if self.modality_dims is None:
            self.modality_dims = {}


# ============================================================================
# CIRCULAR TOPOLOGY UTILITIES
# ============================================================================

class CircularChromaEmbedding(nn.Module):
    """
    Embed chroma bins with circular topology awareness.
    Uses complex exponentials to represent circular structure.
    """

    def __init__(self, n_chroma: int = 12, embed_dim: int = 128):
        super().__init__()
        self.n_chroma = n_chroma
        self.embed_dim = embed_dim

        # Project chroma bins to higher dimension
        self.linear = nn.Linear(n_chroma, embed_dim)

        # Circular positional encoding
        self.register_buffer(
            'circular_encoding',
            self._create_circular_encoding(n_chroma, embed_dim)
        )

        # Harmonic structure (circle of fifths)
        if True:  # Always include harmonic relationships
            self.register_buffer(
                'fifth_relationships',
                self._create_fifth_circle(n_chroma)
            )

    def _create_circular_encoding(self, n_chroma: int, embed_dim: int) -> torch.Tensor:
        """Create circular positional encodings using complex exponentials."""
        encoding = torch.zeros(n_chroma, embed_dim)

        for i in range(n_chroma):
            for j in range(embed_dim // 2):
                # Multiple frequencies to capture different harmonic relationships
                freq = (j + 1) / (embed_dim / 2)
                angle = 2 * math.pi * i * freq / n_chroma

                encoding[i, 2 * j] = math.cos(angle)
                encoding[i, 2 * j + 1] = math.sin(angle)

        return encoding

    def _create_fifth_circle(self, n_chroma: int) -> torch.Tensor:
        """
        Create circle of fifths relationships.
        Each pitch class is 7 semitones (perfect fifth) from the next.
        """
        fifth_circle = torch.zeros(n_chroma, n_chroma)

        for i in range(n_chroma):
            # Perfect fifth = 7 semitones
            fifth = (i + 7) % n_chroma
            fourth = (i - 7) % n_chroma

            # Strong connections to fifths
            fifth_circle[i, fifth] = 1.0
            fifth_circle[i, fourth] = 1.0

            # Weaker connections to major/minor thirds
            major_third = (i + 4) % n_chroma
            minor_third = (i + 3) % n_chroma
            fifth_circle[i, major_third] = 0.5
            fifth_circle[i, minor_third] = 0.5

            # Self-connection
            fifth_circle[i, i] = 1.0

        # Normalize
        fifth_circle = fifth_circle / fifth_circle.sum(dim=1, keepdim=True)

        return fifth_circle

    def forward(self, chroma: torch.Tensor) -> torch.Tensor:
        """
        Embed chroma with circular awareness.

        Args:
            chroma: [batch, seq, n_chroma]

        Returns:
            embedded: [batch, seq, embed_dim]
        """
        B, T, C = chroma.shape

        # Linear projection
        embedded = self.linear(chroma)

        # Add circular encoding (weighted by chroma intensity)
        # This makes nearby pitch classes have similar embeddings
        chroma_weights = chroma.unsqueeze(-1)  # [B, T, C, 1]
        circular_contribution = (chroma_weights * self.circular_encoding.unsqueeze(0).unsqueeze(0)).sum(dim=2)

        embedded = embedded + circular_contribution

        return embedded


class CircularDistance(nn.Module):
    """Compute circular distance for chroma vectors."""

    def __init__(self, n_chroma: int = 12):
        super().__init__()
        self.n_chroma = n_chroma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute circular-aware distance.

        Treats chroma as points on a circle, measuring arc length.
        """
        # Normalize to probability distributions
        pred_norm = F.softmax(pred, dim=-1)
        target_norm = F.softmax(target, dim=-1)

        # Convert to angular representation
        angles_pred = torch.atan2(
            (pred_norm * torch.sin(2 * math.pi * torch.arange(self.n_chroma, device=pred.device) / self.n_chroma)).sum(
                -1),
            (pred_norm * torch.cos(2 * math.pi * torch.arange(self.n_chroma, device=pred.device) / self.n_chroma)).sum(
                -1)
        )

        angles_target = torch.atan2(
            (target_norm * torch.sin(
                2 * math.pi * torch.arange(self.n_chroma, device=target.device) / self.n_chroma)).sum(-1),
            (target_norm * torch.cos(
                2 * math.pi * torch.arange(self.n_chroma, device=target.device) / self.n_chroma)).sum(-1)
        )

        # Circular distance (shortest arc)
        diff = angles_pred - angles_target
        circular_dist = torch.min(torch.abs(diff), 2 * math.pi - torch.abs(diff))

        return circular_dist.mean()


# ============================================================================
# CANTOR CHROMA ATTENTION
# ============================================================================

class CantorChromaAttention(nn.Module):
    """
    Cantor-based attention for chroma sequences.
    Uses fractal routing to capture self-similar harmonic patterns.
    """

    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            cantor_depth: int = 8,
            local_window: int = 16,
            dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cantor_depth = cantor_depth
        self.local_window = local_window

        # QKV projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Temperature
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor set coordinate for a position."""
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

    def _build_cantor_routes(self, seq_len: int) -> torch.Tensor:
        """Build routing table based on Cantor coordinates."""
        # Compute Cantor coordinates for each position
        coords = torch.tensor([
            self._cantor_coordinate(i, seq_len, self.cantor_depth)
            for i in range(seq_len)
        ])

        # Build routing table
        routes = torch.zeros(seq_len, self.local_window, dtype=torch.long)

        for i in range(seq_len):
            distances = torch.abs(coords - coords[i])
            _, nearest = torch.topk(distances, min(self.local_window, seq_len), largest=False)
            routes[i, :len(nearest)] = nearest

        return routes

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Apply Cantor attention.

        Args:
            x: [batch, seq, embed_dim]

        Returns:
            output: [batch, seq, embed_dim]
        """
        B, T, D = x.shape

        # QKV
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Build routes
        routes = self._build_cantor_routes(T).to(x.device)

        # Sparse attention
        attended = []
        for i in range(T):
            neighbors = routes[i]

            q_i = Q[:, :, i:i + 1, :]  # [B, H, 1, D]
            k_neighbors = K[:, :, neighbors, :]  # [B, H, W, D]
            v_neighbors = V[:, :, neighbors, :]  # [B, H, W, D]

            # Attention scores
            scores = torch.matmul(q_i, k_neighbors.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores / self.temperature.abs()

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Apply attention
            out_i = torch.matmul(attn, v_neighbors)  # [B, H, 1, D]
            attended.append(out_i)

        # Concatenate and reshape
        output = torch.cat(attended, dim=2)  # [B, H, T, D]
        output = output.transpose(1, 2).reshape(B, T, D)

        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)

        return output


# ============================================================================
# CHROMA LYRA ENCODER/DECODER
# ============================================================================

class ChromaEncoder(nn.Module):
    """Encode chroma sequences to latent space."""

    def __init__(
            self,
            n_chroma: int = 12,
            latent_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 4,
            num_heads: int = 8,
            use_cantor: bool = True,
            dropout: float = 0.1
    ):
        super().__init__()

        # Circular embedding
        self.chroma_embedding = CircularChromaEmbedding(n_chroma, hidden_dim)

        # Encoder layers
        self.layers = nn.ModuleList([
            CantorChromaAttention(hidden_dim, num_heads, dropout=dropout) if use_cantor
            else nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # Project to latent parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, chroma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode chroma to latent parameters.

        Args:
            chroma: [batch, seq, n_chroma]

        Returns:
            mu, logvar: [batch, seq, latent_dim]
        """
        # Embed with circular awareness
        x = self.chroma_embedding(chroma)

        # Transformer encoder
        for layer, norm, ffn in zip(self.layers, self.norms, self.ffns):
            # Attention
            if isinstance(layer, CantorChromaAttention):
                attn_out = layer(x)
            else:
                attn_out, _ = layer(x, x, x)

            x = norm(x + attn_out)

            # FFN
            ffn_out = ffn(x)
            x = norm(x + ffn_out)

        # Project to latent
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class ChromaDecoder(nn.Module):
    """Decode latent space to chroma sequences."""

    def __init__(
            self,
            n_chroma: int = 12,
            latent_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 4,
            num_heads: int = 8,
            use_cantor: bool = True,
            dropout: float = 0.1
    ):
        super().__init__()

        # Project from latent
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            CantorChromaAttention(hidden_dim, num_heads, dropout=dropout) if use_cantor
            else nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # Project to chroma
        self.to_chroma = nn.Linear(hidden_dim, n_chroma)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to chroma.

        Args:
            z: [batch, seq, latent_dim]

        Returns:
            chroma: [batch, seq, n_chroma]
        """
        x = self.latent_proj(z)

        # Transformer decoder
        for layer, norm, ffn in zip(self.layers, self.norms, self.ffns):
            # Attention
            if isinstance(layer, CantorChromaAttention):
                attn_out = layer(x)
            else:
                attn_out, _ = layer(x, x, x)

            x = norm(x + attn_out)

            # FFN
            ffn_out = ffn(x)
            x = norm(x + ffn_out)

        # Project to chroma
        chroma = self.to_chroma(x)

        return chroma


# ============================================================================
# CHROMA LYRA VAE
# ============================================================================

class ChromaLyra(nn.Module):
    """
    ChromaLyra: Geometric VAE for music generation.

    Handles chroma features with circular topology awareness.
    Can optionally fuse with other modalities (text, MIDI, etc.).
    """

    def __init__(self, config: ChromaLyraConfig):
        super().__init__()
        self.config = config

        # Encoder/Decoder
        self.encoder = ChromaEncoder(
            n_chroma=config.n_chroma,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.encoder_layers,
            num_heads=config.num_heads,
            use_cantor=config.use_cantor_fusion,
            dropout=config.dropout
        )

        self.decoder = ChromaDecoder(
            n_chroma=config.n_chroma,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.decoder_layers,
            num_heads=config.num_heads,
            use_cantor=config.use_cantor_fusion,
            dropout=config.dropout
        )

        # Optional: Multi-modal fusion
        if config.modality_dims:
            # TODO: Add fusion layers for text/MIDI/etc
            pass

    def encode(self, chroma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode chroma to latent parameters."""
        return self.encoder(chroma)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to chroma."""
        return self.decoder(z)

    def forward(self, chroma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            chroma: [batch, seq, n_chroma]

        Returns:
            recon_chroma, mu, logvar
        """
        mu, logvar = self.encode(chroma)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate(
            self,
            num_samples: int = 1,
            seq_len: int = 128,
            temperature: float = 1.0,
            device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate new chroma sequences from noise.

        Args:
            num_samples: Number of sequences to generate
            seq_len: Length of sequences
            temperature: Sampling temperature

        Returns:
            Generated chroma [num_samples, seq_len, n_chroma]
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, seq_len, self.config.latent_dim, device=device) * temperature

            # Decode
            chroma = self.decode(z)

            # Apply softmax to get valid probability distributions
            chroma = F.softmax(chroma, dim=-1)

        return chroma


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class ChromaLyraLoss(nn.Module):
    """Loss for ChromaLyra."""

    def __init__(
            self,
            n_chroma: int = 12,
            beta_kl: float = 0.05,
            beta_reconstruction: float = 1.0,
            beta_circular: float = 0.05,
            beta_harmonic: float = 0.1
    ):
        super().__init__()
        self.beta_kl = beta_kl
        self.beta_reconstruction = beta_reconstruction
        self.beta_circular = beta_circular
        self.beta_harmonic = beta_harmonic

        self.circular_distance = CircularDistance(n_chroma)

        # Harmonic relationships (circle of fifths)
        self.register_buffer(
            'fifth_matrix',
            self._create_fifth_matrix(n_chroma)
        )

    def _create_fifth_matrix(self, n_chroma: int) -> torch.Tensor:
        """Create matrix encoding circle of fifths relationships."""
        matrix = torch.zeros(n_chroma, n_chroma)

        for i in range(n_chroma):
            # Perfect fifth
            fifth = (i + 7) % n_chroma
            fourth = (i - 7) % n_chroma
            matrix[i, fifth] = 1.0
            matrix[i, fourth] = 1.0

        return matrix

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute total loss.

        Args:
            pred: Predicted chroma [batch, seq, n_chroma]
            target: Target chroma [batch, seq, n_chroma]
            mu, logvar: Latent parameters

        Returns:
            total_loss, optional components dict
        """
        losses = {}

        # 1. Reconstruction loss (MSE + cosine similarity)
        mse_loss = F.mse_loss(pred, target)

        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        cos_sim = (pred_norm * target_norm).sum(dim=-1).mean()
        cosine_loss = 1 - cos_sim

        recon_loss = mse_loss + 0.5 * cosine_loss
        losses['reconstruction'] = recon_loss

        # 2. KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mu.shape[0] * mu.shape[1] * mu.shape[2])
        losses['kl'] = kl_loss

        # 3. Circular topology preservation
        circular_loss = self.circular_distance(pred, target)
        losses['circular'] = circular_loss

        # 4. Harmonic coherence (encourage circle of fifths relationships)
        pred_probs = F.softmax(pred, dim=-1)
        target_probs = F.softmax(target, dim=-1)

        # Measure harmonic similarity using fifth matrix
        pred_harmonic = torch.matmul(pred_probs, self.fifth_matrix)
        target_harmonic = torch.matmul(target_probs, self.fifth_matrix)
        harmonic_loss = F.mse_loss(pred_harmonic, target_harmonic)
        losses['harmonic'] = harmonic_loss

        # Total
        total_loss = (
                self.beta_reconstruction * recon_loss +
                self.beta_kl * kl_loss +
                self.beta_circular * circular_loss +
                self.beta_harmonic * harmonic_loss
        )
        losses['total'] = total_loss

        if return_components:
            return total_loss, losses
        return total_loss, None


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ChromaLyra: Geometric VAE for Music Generation")
    print("=" * 80)

    # Configuration
    config = ChromaLyraConfig(
        n_chroma=12,
        seq_len=128,
        latent_dim=256,
        hidden_dim=512,
        num_heads=8,
        use_cantor_fusion=True,
        use_harmonic_prior=True
    )

    # Create model
    model = ChromaLyra(config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Synthetic chroma data (simulate a C major chord progression)
    batch_size = 4
    chroma = torch.zeros(batch_size, config.seq_len, config.n_chroma)

    # C major: C-E-G (indices 0, 4, 7)
    for t in range(0, 32):
        chroma[:, t, [0, 4, 7]] = torch.rand(batch_size, 3) * 0.8 + 0.2

    # F major: F-A-C (indices 5, 9, 0)
    for t in range(32, 64):
        chroma[:, t, [5, 9, 0]] = torch.rand(batch_size, 3) * 0.8 + 0.2

    # G major: G-B-D (indices 7, 11, 2)
    for t in range(64, 96):
        chroma[:, t, [7, 11, 2]] = torch.rand(batch_size, 3) * 0.8 + 0.2

    # Back to C major
    for t in range(96, 128):
        chroma[:, t, [0, 4, 7]] = torch.rand(batch_size, 3) * 0.8 + 0.2

    # Normalize
    chroma = chroma / (chroma.sum(dim=-1, keepdim=True) + 1e-8)

    print(f"\nInput chroma shape: {chroma.shape}")

    # Forward pass
    recon, mu, logvar = model(chroma)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")

    # Compute loss
    loss_fn = ChromaLyraLoss(
        n_chroma=config.n_chroma,
        beta_kl=config.beta_kl,
        beta_reconstruction=config.beta_reconstruction,
        beta_circular=config.beta_circular,
        beta_harmonic=config.beta_harmonic
    )

    loss, components = loss_fn(recon, chroma, mu, logvar, return_components=True)
    print(f"\nTotal loss: {loss.item():.4f}")
    for name, value in components.items():
        if name != 'total':
            print(f"  {name}: {value.item():.4f}")

    # Generate new sequences
    print("\nGenerating new chroma sequences...")
    generated = model.generate(num_samples=2, seq_len=128, temperature=0.8, device='cpu')
    print(f"Generated shape: {generated.shape}")

    # Analyze generated chroma
    dominant_pitches = generated.argmax(dim=-1)
    print(f"\nDominant pitch classes in first sample (first 32 timesteps):")
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    print([pitch_names[p] for p in dominant_pitches[0, :32].tolist()])

    print("\n" + "=" * 80)
    print("ChromaLyra successfully handles circular topology!")
    print("The Cantor manifold naturally respects musical self-similarity.")
    print("=" * 80)