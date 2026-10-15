"""Case-rate transformations, fixed regional scaling, and evaluation metrics."""

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import torch


def log_case_rates(weekly_cases: torch.Tensor, populations: torch.Tensor) -> torch.Tensor:
    """Return ln(1 + weekly cases per 10,000 people), preserving NaNs."""
    if weekly_cases.ndim != 2 or not weekly_cases.is_floating_point():
        raise ValueError("weekly_cases must be a floating-point tensor of shape [week,region]")
    if populations.shape != (weekly_cases.shape[1],):
        raise ValueError("populations must match the region axis")
    if not torch.isfinite(populations).all() or (populations <= 0).any():
        raise ValueError("Populations must be finite and positive")
    if torch.isinf(weekly_cases).any() or (weekly_cases < 0).any():
        raise ValueError("Cases must be nonnegative or NaN")
    return torch.log1p(weekly_cases * 10000.0 / populations.to(weekly_cases))


@dataclass(frozen=True)
class RegionScaler:
    """Fixed regional min-max scaling with no clipping of later observations."""

    minimum: torch.Tensor
    span: torch.Tensor
    fit_start: date
    fit_end: date

    @classmethod
    def fit(cls, log_rates: torch.Tensor, week_ends: Sequence[date], *,
            fit_start: date, fit_end: date) -> "RegionScaler":
        if log_rates.ndim != 2 or not log_rates.is_floating_point():
            raise ValueError("log_rates must be a floating-point tensor of shape [week,region]")
        if len(week_ends) != len(log_rates) or fit_start > fit_end:
            raise ValueError("Invalid fitting interval or number of observation dates")
        if any(type(d) is not date for d in week_ends):
            raise ValueError("Observation dates must be date objects representing complete week ends")
        if any(a >= b for a, b in zip(week_ends, week_ends[1:])):
            raise ValueError("Observation dates must be strictly increasing")
        if torch.isinf(log_rates).any() or (log_rates < 0).any():
            raise ValueError("Log case rates must be nonnegative or NaN")
        selected = torch.tensor([fit_start <= d <= fit_end for d in week_ends],
                                dtype=torch.bool, device=log_rates.device)
        values = log_rates[selected]
        if not len(values) or not torch.isfinite(values).any(dim=0).all():
            raise ValueError("Each region needs at least one finite observation in the fitting interval")
        lo = values.nan_to_num(nan=float("inf")).amin(dim=0)
        hi = values.nan_to_num(nan=-float("inf")).amax(dim=0)
        span = hi - lo
        span = torch.where(span > 0, span, torch.ones_like(span))
        return cls(lo.detach().clone(), span.detach().clone(), fit_start, fit_end)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 0 or not values.is_floating_point():
            raise ValueError("Scaler inputs must be floating-point tensors with at least one dimension")
        if values.shape[-1] != len(self.minimum):
            raise ValueError("The final axis must match the scaler's region axis")
        return (values - self.minimum.to(values)) / self.span.to(values)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 0 or not values.is_floating_point():
            raise ValueError("Scaler inputs must be floating-point tensors with at least one dimension")
        if values.shape[-1] != len(self.minimum):
            raise ValueError("The final axis must match the scaler's region axis")
        return values * self.span.to(values) + self.minimum.to(values)


def macro_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                  valid: torch.Tensor) -> dict[str, float]:
    """Average regional MAE and RMSE equally on the inverse-normalized log1p scale."""
    if predictions.ndim != 2 or targets.shape != predictions.shape or valid.shape != targets.shape:
        raise ValueError("predictions, targets, and valid must have shape [sample,region]")
    if valid.dtype != torch.bool or not valid.any(dim=0).all():
        raise ValueError("valid must be boolean and include at least one evaluation sample per region")
    if not torch.isfinite(predictions[valid]).all() or not torch.isfinite(targets[valid]).all():
        raise ValueError("Predictions and targets must be finite at valid evaluation positions")
    errors = torch.where(valid, predictions - targets, torch.zeros_like(predictions))
    count = valid.sum(dim=0)
    return {"MAE": float((errors.abs().sum(dim=0) / count).mean()),
            "RMSE": float((errors.square().sum(dim=0) / count).sqrt().mean())}
