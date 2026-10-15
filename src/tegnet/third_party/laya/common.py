# Adapted Laya components; source and modifications are documented in README.md.
"""Laya option scoring, checkpoint loading support, and input batching."""
import threading
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

# Serialize access to tokenizer state.
_TOKENIZE_LOCK = threading.RLock()


def encode_text(tok, text, **kwargs):
    """Tokenize text with serialized access to the shared tokenizer."""
    with _TOKENIZE_LOCK:
        return tok(text, **kwargs)


class _DynamicMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention with sequence-length-independent shape operations."""

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if (need_weights or self.in_proj_weight is None or self.bias_k is not None
                or self.bias_v is not None
                or (attn_mask is not None and attn_mask.dtype != torch.bool)):
            # Use the general attention implementation for optional features.
            return super().forward(query, key, value, key_padding_mask=key_padding_mask,
                                   need_weights=need_weights, attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights, is_causal=is_causal)
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        # Attention uses [batch,head,time,dimension] tensors.
        if query is key is value:
            q, k, v = (part.unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                       for part in F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1))
        else:
            embed_dim = query.shape[-1]
            wq, wk, wv = self.in_proj_weight.split(embed_dim, dim=0)
            bq, bk, bv = ((None, None, None) if self.in_proj_bias is None
                          else self.in_proj_bias.split(embed_dim, dim=0))
            q, k, v = (
                F.linear(t, w, b).unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                for t, w, b in ((query, wq, bq), (key, wk, bk), (value, wv, bv))
            )
        mask = None
        if attn_mask is not None:
            mask = ~attn_mask
        if key_padding_mask is not None:
            # Padding masks use either booleans or zero/negative-infinity values.
            keep = key_padding_mask[:, None, None, :] == 0
            mask = keep if mask is None else mask & keep
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal and mask is None,
                                              dropout_p=self.dropout if self.training else 0.0)
        attn = attn.permute(2, 0, 1, 3).flatten(-2)
        if self.batch_first:
            attn = attn.transpose(0, 1)
        return self.out_proj(attn), None


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder producing raw option logits."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1,
                 no_init: bool = False):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        # Construct checkpoint parameters without random initialization.
        with torch.device("meta") if no_init else nullcontext():
            nhead = max(1, d // 64)
            layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
            # Preserve variable sequence lengths in the attention implementation.
            layer.self_attn = _DynamicMultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
            self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            # Preserve the original checkpoint structure without executing the action head.
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.register_buffer("temperature", torch.ones(3))
        if no_init:
            # Materialize only newly constructed head parameters.
            for module in (self.head, self.type_emb, self.scorer, self.act_head):
                if module is not None:
                    module.to_empty(device="cpu")
            self.temperature = torch.empty_like(self.temperature, device="cpu")

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def _apply_rope_config(ecfg) -> None:
    """Apply checkpoint RoPE parameters to the encoder configuration."""
    rope = getattr(ecfg, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    flat = rope.get("rope_theta")
    for layer_type, attr in (("full_attention", "global_rope_theta"),
                             ("sliding_attention", "local_rope_theta")):
        params = rope.get(layer_type)
        theta = params.get("rope_theta") if isinstance(params, dict) else flat
        if theta is not None and hasattr(ecfg, attr):
            setattr(ecfg, attr, float(theta))


def _no_init_weights():
    """Disable parameter initialization during checkpoint construction."""
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers 4.x
        from transformers.modeling_utils import no_init_weights
    return no_init_weights()


def collate_items(items, pad_id: int):
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True

    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }
