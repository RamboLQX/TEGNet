"""Historical summaries and shared context for teachers and the decision model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from importlib.resources import files
import json
import math
import re
from typing import Protocol


_TEMPLATES = json.loads(files("tegnet").joinpath("resources/prompt_templates.json").read_text(encoding="utf-8"))
NO_EVENT_TEXT = "no event text available"
CLASS_NAMES = ("reject", "keep")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _date(value: date, name: str) -> None:
    if type(value) is not date:
        raise ValueError(f"{name} must be a date, without a time component")


@dataclass(frozen=True)
class EventText:
    """Event text with its regional scope, availability date, and effective date."""

    text: str
    regions: tuple[str, ...]
    available_at: date
    effective_at: date | None = None

    def __post_init__(self) -> None:
        _text(self.text, "Event text")
        if isinstance(self.regions, str):
            raise ValueError("Event regions must be a sequence of region names")
        object.__setattr__(self, "regions", tuple(self.regions))
        if not self.regions or len(set(self.regions)) != len(self.regions):
            raise ValueError("Event regions must be nonempty and unique")
        for region in self.regions:
            _text(region, "Event region")
        _date(self.available_at, "Event availability date")
        if self.effective_at is not None:
            _date(self.effective_at, "Event effective date")


@dataclass(frozen=True)
class PairContext:
    """Source-target context at the end of the latest complete observation week."""

    target_region: str
    source_region: str
    forecast_origin: date
    horizon: int
    historical_summary: str
    events: tuple[EventText, ...] = ()

    def __post_init__(self) -> None:
        _text(self.target_region, "Target region")
        _text(self.source_region, "Source region")
        if self.target_region == self.source_region:
            raise ValueError("The candidate source must differ from the target region")
        _date(self.forecast_origin, "Forecast origin")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("The forecast horizon must be a positive integer number of weeks")
        _text(self.historical_summary, "Historical summary")
        object.__setattr__(self, "events", tuple(self.events))
        for event in self.events:
            if not isinstance(event, EventText):
                raise ValueError("events must contain only EventText instances")
            if event.available_at > self.forecast_origin:
                raise ValueError("Event text must be available by the forecast origin")

    @property
    def target_week(self) -> date:
        """Return the end date of the single forecast target week."""
        return self.forecast_origin + timedelta(weeks=self.horizon)


@dataclass(frozen=True)
class HistorySeries:
    """Weekly case history with explicit dates, units, and missing values."""

    weeks: tuple[date, ...]
    values: tuple[float | None, ...]
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "weeks", tuple(self.weeks))
        object.__setattr__(self, "values", tuple(self.values))
        _text(self.unit, "Case unit and scale")
        if not self.weeks or len(self.weeks) != len(self.values):
            raise ValueError("History dates and values must be nonempty and have equal lengths")
        for week in self.weeks:
            _date(week, "History week end date")
        if any(b - a != timedelta(weeks=1) for a, b in zip(self.weeks, self.weeks[1:])):
            raise ValueError("History weeks must be consecutive and in chronological order")
        for value in self.values:
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Case values must be finite numbers; use None for missing values")


@dataclass(frozen=True)
class SummaryInput:
    """Regional histories and source-to-target importation pressure for a shared summary."""

    target_region: str
    source_region: str
    forecast_origin: date
    target_history: HistorySeries
    source_history: HistorySeries
    importation_pressure: float

    def __post_init__(self) -> None:
        _text(self.target_region, "Target region")
        _text(self.source_region, "Source region")
        if self.target_region == self.source_region:
            raise ValueError("The summary source and target regions must differ")
        _date(self.forecast_origin, "Forecast origin")
        if not isinstance(self.target_history, HistorySeries) or not isinstance(self.source_history, HistorySeries):
            raise ValueError("Summary histories must be HistorySeries instances")
        if self.target_history.weeks != self.source_history.weeks:
            raise ValueError("Both summary histories must use the same weeks")
        if self.target_history.unit != self.source_history.unit:
            raise ValueError("Both summary histories must use the same case unit and scale")
        if self.target_history.weeks[-1] != self.forecast_origin:
            raise ValueError("The summary window must end at the forecast origin")
        if (type(self.importation_pressure) not in (int, float)
                or not math.isfinite(self.importation_pressure) or self.importation_pressure < 0):
            raise ValueError("Importation pressure must be a finite nonnegative number")


class SummaryProvider(Protocol):
    """Callable interface for generating a historical summary from its prompt."""

    def __call__(self, prompt: str) -> str: ...


def _fill(template: str, replacements: dict[str, str]) -> str:
    # Apply substitutions once so that braces in supplied text remain literal.
    return re.sub(r"\{([a-z_]+)\}", lambda match: replacements.get(match[1], match[0]), template)


def _history_text(history: HistorySeries) -> str:
    return "; ".join(
        f"{week.isoformat()}: {'unknown' if value is None else value}"
        for week, value in zip(history.weeks, history.values)
    )


def render_summary_prompt(inputs: SummaryInput) -> str:
    """Render the four-field summary prompt with histories and importation pressure."""
    sections = _TEMPLATES["summary_prompt"]
    replacements = {
        "target_region": inputs.target_region,
        "source_region": inputs.source_region,
        "window_start": inputs.target_history.weeks[0].isoformat(),
        "forecast_origin": inputs.forecast_origin.isoformat(),
        "case_unit_and_scale": inputs.target_history.unit,
        "target_case_history": _history_text(inputs.target_history),
        "source_case_history": _history_text(inputs.source_history),
        "importation_pressure": str(inputs.importation_pressure),
    }
    return "\n\n".join(f"{label}\n{_fill(text, replacements)}" for label, text in sections.items())


def generate_summary(inputs: SummaryInput, provider: SummaryProvider) -> str:
    """Generate the shared summary S used by teachers and the decision model."""
    result = provider(render_summary_prompt(inputs))
    _text(result, "Summary provider output")
    fields = list(re.finditer(r"\b(Target|Source|Comparison|Importation pressure):", result))
    if [field[1] for field in fields] != ["Target", "Source", "Comparison", "Importation pressure"] or result[:fields[0].start()].strip():
        raise ValueError("The summary must contain exactly Target:, Source:, Comparison:, and Importation pressure:, in that order")
    ends = [field.start() for field in fields[1:]] + [len(result)]
    if any(not result[field.end():end].strip() for field, end in zip(fields, ends)):
        raise ValueError("All four summary fields must be nonempty")
    return result


def render_state(context: PairContext) -> str:
    """Render shared input context with event dates and regional scope."""
    if not isinstance(context, PairContext):
        raise TypeError("context must be a PairContext")
    events = "\n".join(
        f"Available at: {event.available_at.isoformat()}; "
        f"Effective at: {event.effective_at.isoformat() if event.effective_at else 'unknown'}; "
        f"Regions: {', '.join(event.regions)}; Text: {event.text}"
        for event in context.events
    ) or NO_EVENT_TEXT
    return _fill(_TEMPLATES["teacher_annotation_prompt"]["[USER / INPUT]"], {
        "target_region": context.target_region,
        "source_region": context.source_region,
        "forecast_origin": context.forecast_origin.isoformat(),
        "forecast_horizon": str(context.horizon),
        "target_week": context.target_week.isoformat(),
        "historical_summary": context.historical_summary,
        "dated_event_text_available_by_forecast_origin": events,
    })


def decision_question() -> dict:
    """Return the Laya choice question in reject, keep option order."""
    sections = _TEMPLATES["teacher_annotation_prompt"]
    reject, keep = sections["[OPTIONS]"].split("1 - Retain:", 1)
    reject = " ".join(reject.removeprefix("0 - Reject:").split())
    keep = " ".join(keep.split())
    return {
        "type": "choice",
        "instructions": " ".join(sections["[TASK]"].split()),
        "criteria": {"reject": reject[0].upper() + reject[1:], "keep": keep[0].upper() + keep[1:]},
    }


def render_teacher_prompt(context: PairContext) -> str:
    """Render the same annotation prompt for each of the three teachers."""
    sections = dict(_TEMPLATES["teacher_annotation_prompt"])
    sections["[USER / INPUT]"] = render_state(context)
    return "\n\n".join(f"{label}\n{text}" for label, text in sections.items())
