"""Rolling air-passenger windows and candidate sources indexed as [target,source]."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import torch


@dataclass(frozen=True)
class CandidateGraph:
    support: torch.Tensor
    pressure: torch.Tensor
    inbound_share: torch.Tensor
    observed_flow: torch.Tensor
    observed_month_counts: torch.Tensor | None = None
    months: tuple[date, ...] = ()


def select_candidates(flow: torch.Tensor, recent_two_weeks: torch.Tensor, *,
                      top_k: int = 10) -> CandidateGraph:
    """Rank sources by inbound flow share times their two-week case-rate sum.

    flow contains observed passenger totals. recent_two_weeks contains raw
    cases per 10,000 people with shape [2,region]. Sources with missing case
    rates are excluded, while their flows remain in the inbound denominator.
    Ties are resolved by descending flow and then ascending region index.
    """
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 to 10")
    if flow.ndim != 2 or flow.shape[0] != flow.shape[1] or len(flow) < 2:
        raise ValueError("flow must be a square [target,source] matrix with at least two regions")
    if not flow.is_floating_point() or not torch.isfinite(flow).all() or (flow < 0).any():
        raise ValueError("Observed passenger totals must be finite nonnegative floating-point values")
    if flow.diagonal().ne(0).any():
        raise ValueError("Air-passenger inputs must exclude self-loops")
    if recent_two_weeks.shape != (2, len(flow)) or not recent_two_weeks.is_floating_point():
        raise ValueError("Case rates must be a floating-point tensor of shape [2,region]")
    if torch.isinf(recent_two_weeks).any() or (recent_two_weeks < 0).any():
        raise ValueError("Two-week case rates must be nonnegative or NaN")
    recent = recent_two_weeks.to(flow).sum(dim=0)
    inbound = flow.sum(dim=1, keepdim=True)
    if not torch.isfinite(inbound).all():
        raise ValueError("Inbound passenger totals overflowed")
    shares = flow / torch.where(inbound > 0, inbound, torch.ones_like(inbound))
    pressure = shares * recent.unsqueeze(0)
    known = torch.isfinite(recent)
    if not torch.isfinite(pressure[:, known]).all():
        raise ValueError("Importation pressure scores overflowed")
    eligible = (flow > 0) & known.unsqueeze(0)
    support = torch.zeros_like(eligible)
    for target in range(len(flow)):
        sources = eligible[target].nonzero().flatten().tolist()
        chosen = sorted(sources, key=lambda source: (
            -float(pressure[target, source]), -float(flow[target, source]), source))[:top_k]
        support[target, chosen] = True
    return CandidateGraph(support, pressure, shares, flow.detach().clone())


def _shift_month(value: date, delta: int) -> date:
    offset = value.year * 12 + value.month - 1 + delta
    year, month = divmod(offset, 12)
    return date(year, month + 1, 1)


def rolling_candidates(monthly_flows: torch.Tensor, months: Sequence[date],
                       observed: torch.Tensor, available_at: Sequence[date],
                       recent_two_weeks: torch.Tensor, *, forecast_origin: date,
                       top_k: int = 10) -> CandidateGraph:
    """Select candidates using the 12 complete months before the issue month.

    The issue date is the day after the complete observation week ends.
    Monthly records must be available by forecast_origin. observed marks
    measured entries; their coverage counts accompany the passenger totals.
    """
    if type(forecast_origin) is not date:
        raise ValueError("forecast_origin must be a date representing a complete week end")
    if (monthly_flows.ndim != 3 or len(months) != len(monthly_flows)
            or len(available_at) != len(months) or observed.shape != monthly_flows.shape
            or observed.dtype != torch.bool):
        raise ValueError("Monthly flows, months, availability dates, and the boolean observation mask must align")
    if any(type(d) is not date or d.day != 1 for d in months) or len(set(months)) != len(months):
        raise ValueError("Months must be unique date objects representing the first day of each month")
    if any(type(d) is not date for d in available_at):
        raise ValueError("Monthly availability dates must be date objects")
    issue_month = (forecast_origin + timedelta(days=1)).replace(day=1)
    required = tuple(_shift_month(issue_month, delta) for delta in range(-12, 0))
    mapping = {month: index for index, month in enumerate(months)}
    if any(month not in mapping for month in required):
        raise ValueError("All 12 complete months must be present in the passenger-flow window")
    indices = [mapping[month] for month in required]
    if any(available_at[i] > forecast_origin for i in indices):
        raise ValueError("The passenger-flow window includes records unavailable at forecast_origin")
    values, mask = monthly_flows[indices], observed[indices]
    if not values.is_floating_point() or not torch.isfinite(values[mask]).all() or (values[mask] < 0).any():
        raise ValueError("Observed passenger flows must be finite nonnegative floating-point values")
    totals = torch.where(mask, values, torch.zeros_like(values)).sum(dim=0)
    selected = select_candidates(totals, recent_two_weeks, top_k=top_k)
    return CandidateGraph(selected.support, selected.pressure, selected.inbound_share,
                          selected.observed_flow, mask.sum(dim=0), required)
