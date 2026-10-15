"""Laya decision logits and differentiable graph edge weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import nn

from .prompts import CLASS_NAMES, PairContext, decision_question, render_state
from .third_party.laya import common


_THETA_PREFIXES = ("head.", "type_emb.", "scorer.")
_INPUT_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def _internal_question() -> dict:
    external = decision_question()
    if tuple(external["criteria"]) != CLASS_NAMES:
        raise ValueError("Laya option order must be reject=0 and keep=1")
    return {"t": external["type"], "ins": external["instructions"], "crit": external["criteria"]}


def encode_context(tokenizer, context: PairContext, *, max_len: int, head_max_len: int) -> dict:
    """Encode a complete candidate-pair context within the Laya input budgets."""
    state, question = render_state(context), _internal_question()
    mask = tokenizer.mask_token
    encode = lambda text: common.encode_text(tokenizer, text.replace(mask, " "), add_special_tokens=False)["input_ids"]
    instruction_ids = encode(f"{question['t']} question: {question['ins']}")
    option_ids = [encode(f" {label}: {criterion}") for label, criterion in question["crit"].items()]
    state_ids = encode(state)
    # Each option has one mask marker and a maximum of 48 tokens.
    option_count = sum(1 + len(option) for option in option_ids)
    if any(len(option) > 48 for option in option_ids):
        raise ValueError("Laya options must fit within 48 tokens; shorten the shared template")
    if head_max_len - option_count < 16 or len(instruction_ids) > head_max_len - option_count:
        raise ValueError("Laya instructions and options exceed head_max_len")
    required = 4 + len(instruction_ids) + option_count + len(state_ids)
    if required > max_len:
        raise ValueError(f"complete Laya input requires {required} tokens, exceeding max_len={max_len}")
    ids = [tokenizer.cls_token_id] + instruction_ids + [tokenizer.sep_token_id]
    markers = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend([tokenizer.mask_token_id] + option)
    ids.extend([tokenizer.sep_token_id] + state_ids + [tokenizer.sep_token_id])
    if len(ids) != required or len(markers) != 2:
        raise ValueError("encoded sequence length or option count does not match the complete input")
    return {"ids": ids, "markers": markers, "qtype": 0}


class LayaDecisionModel(nn.Module):
    """Laya with trainable head, type embedding, and option scorer parameters."""

    def __init__(self, model: nn.Module, tokenizer, *, max_len: int = 4096, head_max_len: int = 192):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (max_len, head_max_len)):
            raise ValueError("Laya input budgets must be positive integers")
        for name in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id", "mask_token"):
            if getattr(tokenizer, name, None) is None:
                raise ValueError(f"tokenizer is missing {name}")
        if not hasattr(model, "encoder") or not hasattr(model, "act_head"):
            raise ValueError("model must contain the Laya encoder and action head")
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len
        names = tuple(name for name, _ in self.model.named_parameters())
        if any(not any(name.startswith(prefix) for name in names) for prefix in _THETA_PREFIXES):
            raise ValueError("model must contain head, type_emb, and scorer parameters")
        self._decision_trainable = False
        self.set_trainable(False)

    def decision_parameters(self) -> Iterator[nn.Parameter]:
        """Return the task decision parameters."""
        return (parameter for name, parameter in self.model.named_parameters() if name.startswith(_THETA_PREFIXES))

    def set_trainable(self, enabled: bool) -> None:
        """Control task decision parameters while keeping the remaining parameters fixed."""
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        self._decision_trainable = enabled
        self.model.requires_grad_(False)
        for parameter in self.decision_parameters():
            parameter.requires_grad_(enabled)
        self.train(self.training)

    def train(self, mode: bool = True):
        """Keep the frozen encoder and action head in evaluation mode."""
        super().train(mode)
        self.model.train(mode and self._decision_trainable)
        self.model.encoder.eval()
        self.model.act_head.eval()
        return self

    def forward(self, contexts: Sequence[PairContext]) -> torch.Tensor:
        """Return differentiable raw option logits with shape [candidate,2]."""
        parameter = next(self.model.parameters())
        if not contexts:
            return parameter.new_empty((0, 2))
        items = [encode_context(self.tokenizer, context, max_len=self.max_len, head_max_len=self.head_max_len)
                 for context in contexts]
        batch = common.collate_items(items, self.tokenizer.pad_token_id)
        inputs = {key: batch[key].to(parameter.device) for key in _INPUT_KEYS}
        logits = self.model(**inputs, detach_encoder=True)
        if logits.shape != (len(contexts), 2) or not torch.isfinite(logits).all():
            raise ValueError("Laya must return two finite raw option logits per candidate")
        return logits

    def retention_probabilities(self, contexts: Sequence[PairContext]) -> torch.Tensor:
        """Compute retention probabilities from the raw option logits."""
        return torch.softmax(self(contexts), dim=-1)[:, 1]

    @classmethod
    def from_local(cls, model_dir, device: str = "cpu", *, tokenizer_dir=None,
                   max_len: int = 4096, head_max_len: int = 192) -> "LayaDecisionModel":
        """Load a Laya checkpoint and tokenizer from local files."""
        model_dir = Path(model_dir).expanduser().resolve()
        tokenizer_dir = Path(tokenizer_dir).expanduser().resolve() if tokenizer_dir else model_dir / "tokenizer"
        required = (model_dir / "rl_agent_config.json", model_dir / "encoder/config.json", model_dir / "model.safetensors")
        missing = [str(path) for path in required if not path.is_file()]
        if not tokenizer_dir.is_dir():
            missing.append(str(tokenizer_dir))
        if missing:
            raise FileNotFoundError("missing local Laya files: " + ", ".join(missing))
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = json.loads(required[0].read_text(encoding="utf-8"))
        encoder_config = AutoConfig.from_pretrained(str(model_dir / "encoder"), local_files_only=True, trust_remote_code=False)
        common._apply_rope_config(encoder_config)
        encoder_config.reference_compile = False
        if getattr(encoder_config, "max_position_embeddings", max_len) < max_len:
            raise ValueError("encoder position capacity is smaller than max_len")
        with common._no_init_weights():
            encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa", trust_remote_code=False)
        model = common.DecisionModel(encoder, head_layers=config["head_layers"], n_act=len(config["act_costs"]) + 1, no_init=True)
        model.load_state_dict(load_file(str(required[2]), device="cpu"), strict=True)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
        model.to(device=torch.device(device), dtype=torch.float32)
        result = cls(model, tokenizer, max_len=max_len, head_max_len=head_max_len)
        result.eval()
        return result
