"""
    Classification head for experimental geometric structures.
    Author: AbstractPhil + Claude Sonnet 4.5 + GPT-4o + GPT-5

"""
from torch import nn, Tensor


class GeometricClassificationHead(nn.Module):
    """Classification head for geometric structures."""

    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 use_attention: bool = True,
                 attention_heads: int = 8,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.use_attention = use_attention

        self.simplex_pooling = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        if use_attention:
            self.origin_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attention_heads,
                batch_first=True
            )
        else:
            self.origin_attention = None

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, flowed_simplices: Tensor) -> Tensor:
        pooled = flowed_simplices.mean(dim=-2)
        pooled = self.simplex_pooling(pooled)

        if self.use_attention:
            attended, _ = self.origin_attention(pooled, pooled, pooled)
        else:
            attended = pooled

        features = attended.mean(dim=1)
        logits = self.classifier(features)
        return logits

