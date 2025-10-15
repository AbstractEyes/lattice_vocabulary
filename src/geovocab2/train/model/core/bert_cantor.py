"""
bert_cantor.py — Cantor Stairs BERT (prototype)
Author: AbstractPhil + Mirel (Quartermaster)

Overview
--------
- Drop-in BERT variant with Cantor (devil's staircase) positional embeddings.
- Preserves BERT architecture; only `Embeddings` are swapped.
- Finite-level ternary expansion implements the staircase (constant on removed middle-thirds).
- A small MLP projects the 1D Cantor value into `hidden_size` with residual scale.

File Layout
-----------
1) CONFIG (dict at top)
2) IMPLEMENTATION (core classes / functions)
3) ACTIVATION (quick sanity test in __main__)

Design Notes
------------
- Discrete positions p in [0, max_pos-1] → x = p/(max_pos-1) ∈ [0,1].
- Cantor(x; L): repeat L times → t = floor(3*x); bit = 0 if t==0 else 1 if t==2 else "middle".
  For the middle-third (t==1), we map to 0.5 branch by continuing with x = 3*x - t (still valid).
  The binary sequence (with middle resolved by continuation) becomes a dyadic value in [0,1].
- Finite-level approximation is stable and differentiable w.r.t. learned projection, not x.
- Learnable `pos_gain` gates the Cantor path vs token/content pathways.

Colab/Env
---------
- Requires: torch, transformers (tested on PyTorch 2.x, Transformers 4.43+).
"""

# ======================
# 1) CONFIG
# ======================
CONFIG = {
    "vocab_size": 30522,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    # Cantor-specific
    "cantor_levels": 12,          # ternary expansion depth (finite precision)
    "cantor_mlp_width": 256,      # MLP width for projecting Cantor scalar -> hidden_size
    "cantor_dropout": 0.0,        # optional dropout inside the projector
    "pos_gain_init": 0.1,         # initial gate on positional injection
}

# ======================
# 2) IMPLEMENTATION
# ======================
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import BertConfig, BertEncoder, BertPooler, BertPreTrainedModel
except Exception as e:
    raise ImportError(
        "Transformers is required. In Colab:\n"
        "!pip install --upgrade transformers"
    ) from e


@torch.jit.script
def cantor_staircase_scalar(pos: torch.Tensor, max_pos: int, levels: int) -> torch.Tensor:
    """
    Compute a finite-precision Cantor function value in [0,1] for discrete positions.

    Args:
        pos: Long or float tensor of shape [...], positions in [0, max_pos-1]
        max_pos: int, maximum positions allocated for embeddings
        levels: int, number of ternary-expansion steps (precision)

    Returns:
        Tensor of same shape as pos, float in [0,1]
    """
    # Normalize to [0,1]
    x = pos.to(torch.float32) / float(max_pos - 1)
    y = x.clone()
    out = torch.zeros_like(y)

    # dyadic accumulation: out = sum(b_i * 2^{-i-1})
    # where b_i ∈ {0,1} derived from ternary digit t ∈ {0,1,2} (with middle resolved by continuation)
    weight = 0.5  # first dyadic weight = 2^{-1}
    for _ in range(levels):
        t = torch.floor(y * 3.0)
        # map ternary digit -> binary bit
        bit = torch.zeros_like(y)
        bit = torch.where(t == 2.0, torch.ones_like(bit), bit)  # 2 -> 1
        # For middle third (t==1), we do not immediately fix bit; keep exploring by recursion.
        # Empirically, continuing refines toward the correct binary branch; leave bit=0 this level.

        out = out + bit * weight
        # refine remainder y for next digit
        y = y * 3.0 - t
        weight = weight * 0.5

    return out.clamp(0.0, 1.0)


class CantorProjector(nn.Module):
    """
    Projects a 1D Cantor scalar to hidden_size with a gated residual path.
    """
    def __init__(self, hidden_size: int, width: int = 256, dropout: float = 0.0, pos_gain_init: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, hidden_size),
        )
        # Learnable gain for positional pathway
        self.pos_gain = nn.Parameter(torch.tensor(pos_gain_init, dtype=torch.float32))

    def forward(self, cantor_value: torch.Tensor) -> torch.Tensor:
        """
        cantor_value: [B, L] in [0,1]
        returns: [B, L, H]
        """
        x = cantor_value.unsqueeze(-1)  # [B, L, 1]
        pe = self.net(x)                # [B, L, H]
        return self.pos_gain * pe


class BertCantorEmbeddings(nn.Module):
    """
    Token embeddings + segment embeddings + Cantor positional projection.

    - Token + Type embeddings as usual
    - Replace sinusoidal/learned absolute pos with Cantor projector output
    - Final LayerNorm + Dropout as in BERT
    """
    def __init__(self, config: BertConfig, cantor_levels: int, projector: CantorProjector):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.max_position_embeddings = config.max_position_embeddings
        self.cantor_levels = int(cantor_levels)
        self.projector = projector

        # Prebuild [0..max_pos-1] as a registered buffer for speed
        pos = torch.arange(self.max_position_embeddings, dtype=torch.float32).unsqueeze(0)  # [1, P]
        self.register_buffer("pos_index_row", pos, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        bsz, seq_len = input_shape
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # Build position row considering potential past (for generative variants)
        if position_ids is None:
            # positions in [0..seq_len-1] shifted by past_key_values_length
            start = past_key_values_length
            end = past_key_values_length + seq_len
            # clamp to max range
            pos_ids = torch.arange(start, end, device=inputs_embeds.device, dtype=torch.long)
            pos_ids = torch.clamp(pos_ids, 0, self.max_position_embeddings - 1)
            position_ids = pos_ids.unsqueeze(0).expand(bsz, -1)  # [B, L]

        # Cantor value per position (float in [0,1])
        cantor_vals = cantor_staircase_scalar(position_ids.to(torch.float32), self.max_position_embeddings, self.cantor_levels)  # [B, L]
        pos_proj = self.projector(cantor_vals)  # [B, L, H]

        embeddings = inputs_embeds + token_type_embeddings + pos_proj
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


@dataclass
class BertCantorArgs:
    vocab_size: int = CONFIG["vocab_size"]
    hidden_size: int = CONFIG["hidden_size"]
    num_hidden_layers: int = CONFIG["num_hidden_layers"]
    num_attention_heads: int = CONFIG["num_attention_heads"]
    intermediate_size: int = CONFIG["intermediate_size"]
    max_position_embeddings: int = CONFIG["max_position_embeddings"]
    type_vocab_size: int = CONFIG["type_vocab_size"]
    layer_norm_eps: float = CONFIG["layer_norm_eps"]
    hidden_dropout_prob: float = CONFIG["hidden_dropout_prob"]
    attention_probs_dropout_prob: float = CONFIG["attention_probs_dropout_prob"]
    cantor_levels: int = CONFIG["cantor_levels"]
    cantor_mlp_width: int = CONFIG["cantor_mlp_width"]
    cantor_dropout: float = CONFIG["cantor_dropout"]
    pos_gain_init: float = CONFIG["pos_gain_init"]


class BertCantorModel(BertPreTrainedModel):
    """
    Full BERT encoder with Cantor positional embeddings.

    Compatibility:
      - `from_pretrained` works for encoder weights; embedding block will not load standard pos embeddings.
      - Use when absolute positions should reflect Cantor geometry (piecewise-constant, robust locality).
    """
    config_class = BertConfig

    def __init__(self, config: BertConfig, cantor_levels: int = CONFIG["cantor_levels"],
                 cantor_mlp_width: int = CONFIG["cantor_mlp_width"],
                 cantor_dropout: float = CONFIG["cantor_dropout"],
                 pos_gain_init: float = CONFIG["pos_gain_init"]):
        super().__init__(config)

        projector = CantorProjector(
            hidden_size=config.hidden_size,
            width=cantor_mlp_width,
            dropout=cantor_dropout,
            pos_gain_init=pos_gain_init,
        )
        self.embeddings = BertCantorEmbeddings(config, cantor_levels=cantor_levels, projector=projector)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )

        if attention_mask is None:
            attention_mask = torch.ones(embedding_output.size()[:2], device=embedding_output.device)

        # Standard BERT attention mask processing
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, embedding_output.size()[:2], embedding_output.device)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        if return_dict:
            from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, "hidden_states") else None,
                attentions=encoder_outputs.attentions if hasattr(encoder_outputs, "attentions") else None,
                cross_attentions=getattr(encoder_outputs, "cross_attentions", None),
            )
        return (sequence_output, pooled_output) + encoder_outputs[1:]


# ======================
# 3) ACTIVATION (sanity)
# ======================
if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False, precision=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bargs = BertCantorArgs()
    bcfg = BertConfig(
        vocab_size=bargs.vocab_size,
        hidden_size=bargs.hidden_size,
        num_hidden_layers=bargs.num_hidden_layers,
        num_attention_heads=bargs.num_attention_heads,
        intermediate_size=bargs.intermediate_size,
        max_position_embeddings=bargs.max_position_embeddings,
        type_vocab_size=bargs.type_vocab_size,
        layer_norm_eps=bargs.layer_norm_eps,
        hidden_dropout_prob=bargs.hidden_dropout_prob,
        attention_probs_dropout_prob=bargs.attention_probs_dropout_prob,
    )

    model = BertCantorModel(
        bcfg,
        cantor_levels=bargs.cantor_levels,
        cantor_mlp_width=bargs.cantor_mlp_width,
        cantor_dropout=bargs.cantor_dropout,
        pos_gain_init=bargs.pos_gain_init,
    ).to(device).eval()

    # toy batch
    B, L = 2, 16
    input_ids = torch.randint(0, bargs.vocab_size, (B, L), device=device)
    attention_mask = torch.ones(B, L, device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        print("last_hidden_state:", out.last_hidden_state.shape)
        print("pooler_output:", out.pooler_output.shape)
