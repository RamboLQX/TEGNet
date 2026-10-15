"""Multi-scale temporal encoding, text-guided graph forecasting, and calibration."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ForecastOutput:
    """Forecasts on the model computation scale and embeddings of shape [B,N,D]."""

    prediction: torch.Tensor
    base: torch.Tensor
    graph_correction: torch.Tensor
    embeddings: torch.Tensor


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _initialize(root: nn.Module) -> None:
    for module in root.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class MultiScaleTemporalEncoder(nn.Module):
    """Concatenate five convolutional scales after ReLU and temporal mean pooling."""

    def __init__(self, input_weeks: int, filters_per_scale: int):
        super().__init__()
        specifications = [(3, 1), (5, 1), (3, 2), (5, 2), (input_weeks, 1)]
        self.branches = nn.ModuleList()
        for index, (kernel_size, dilation) in enumerate(specifications):
            padding = dilation * (kernel_size - 1) // 2 if index < 4 else 0
            self.branches.append(nn.Sequential(
                nn.Conv1d(1, filters_per_scale, kernel_size, dilation=dilation, padding=padding),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            ))

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(sequences).squeeze(-1) for branch in self.branches], dim=-1)


class TEGNetCore(nn.Module):
    """MTM and TGM with directed edges indexed as [target,source].

    Retention probabilities are zero outside the candidate set. Source
    aggregation divides by the candidate count once in this module.
    """

    def __init__(self, node_count: int, input_weeks: int = 20,
                 filters_per_scale: int = 8, top_k: int = 10):
        super().__init__()
        for name, value in (("node_count", node_count), ("input_weeks", input_weeks),
                            ("filters_per_scale", filters_per_scale),
                            ("top_k", top_k)):
            _positive_int(value, name)
        if node_count < 2 or input_weeks < 5 or top_k > 10:
            raise ValueError("At least two nodes and five input weeks are required; top_k must be at most 10")
        self.node_count = node_count
        self.input_weeks = input_weeks
        self.embedding_dim = 5 * filters_per_scale
        self.top_k = top_k
        dim = self.embedding_dim
        self.temporal_encoder = MultiScaleTemporalEncoder(input_weeks, filters_per_scale)
        self.local_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.ar_head = nn.Linear(input_weeks, 1)
        self.message_projection = nn.Linear(dim, dim)
        self.graph_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.graph_gate = nn.Parameter(torch.zeros(()))
        _initialize(self)

    def forward(self, cases: torch.Tensor, probabilities: torch.Tensor,
                candidates: torch.Tensor) -> ForecastOutput:
        if (cases.ndim != 3 or tuple(cases.shape[1:]) != (self.input_weeks, self.node_count)
                or len(cases) == 0 or not cases.is_floating_point()
                or not bool(torch.isfinite(cases).all())):
            raise ValueError("cases must be a nonempty finite floating-point tensor of shape [B,L,N]")
        expected = (len(cases), self.node_count, self.node_count)
        if (probabilities.shape != expected or candidates.shape != expected
                or candidates.dtype != torch.bool):
            raise ValueError("probabilities and candidates must have shape [B,N,N]; candidates must be boolean")
        if (probabilities.device != cases.device or candidates.device != cases.device
                or probabilities.dtype != cases.dtype):
            raise ValueError("Inputs must share a device; cases and probabilities must have the same floating-point dtype")
        if (not bool(torch.isfinite(probabilities).all())
                or bool(((probabilities < 0) | (probabilities > 1)).any())
                or bool(probabilities.masked_select(~candidates).ne(0).any())):
            raise ValueError("Probabilities must lie in [0,1] and be zero outside the candidate set")
        if bool(torch.diagonal(candidates, dim1=-2, dim2=-1).any()):
            raise ValueError("Candidate sets must exclude self-loops")
        counts = candidates.sum(-1, keepdim=True)
        if bool((counts > self.top_k).any()):
            raise ValueError("Candidate count exceeds top_k")

        batch = len(cases)
        sequences = cases.permute(0, 2, 1).reshape(batch * self.node_count, 1, self.input_weeks)
        embeddings = self.temporal_encoder(sequences).reshape(batch, self.node_count, -1)
        base = self.local_head(embeddings).squeeze(-1)
        base = base + self.ar_head(sequences.squeeze(1)).reshape(batch, self.node_count)
        weights = probabilities * candidates.to(probabilities.dtype) / counts.clamp_min(1)
        message = torch.relu(self.message_projection(torch.bmm(weights, embeddings)))
        message = message * counts.gt(0).to(message.dtype)
        raw = self.graph_head(torch.cat([embeddings, message], dim=-1)).squeeze(-1)
        correction = torch.tanh(self.graph_gate) * torch.tanh(raw)
        # Empty candidate sets receive no graph correction.
        correction = torch.where(counts.squeeze(-1).gt(0), correction, torch.zeros_like(correction))
        return ForecastOutput(base + correction, base, correction, embeddings)


class CalibrationModule(nn.Module):
    """Compute a bounded local correction from each target's temporal embedding."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None):
        super().__init__()
        _positive_int(embedding_dim, "embedding_dim")
        hidden_dim = embedding_dim if hidden_dim is None else hidden_dim
        _positive_int(hidden_dim, "hidden_dim")
        self.embedding_dim = embedding_dim
        self.residual_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )
        self.local_gate = nn.Parameter(torch.zeros(()))
        _initialize(self)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if (embeddings.ndim != 3 or embeddings.shape[-1] != self.embedding_dim
                or not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all())):
            raise ValueError("embeddings must be a finite floating-point tensor of shape [B,N,D]")
        raw = self.residual_head(embeddings).squeeze(-1)
        return torch.tanh(self.local_gate) * torch.tanh(raw)
