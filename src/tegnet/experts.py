"""Shared relation prompts and majority-vote supervision from three teacher models."""

from dataclasses import dataclass
import json
from typing import Callable, Sequence

from .prompts import PairContext, render_teacher_prompt


@dataclass(frozen=True)
class Teacher:
    """A teacher model identifier and a prompt-to-response callable."""

    provider_id: str
    generate: Callable[[str], str]

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise ValueError("teacher provider_id must be nonempty")
        if not callable(self.generate):
            raise ValueError("teacher generate must be callable")


@dataclass(frozen=True)
class RelationAnnotation:
    """A binary majority label with the three teacher votes and identifiers."""

    label: int
    votes: tuple[int, int, int]
    provider_ids: tuple[str, str, str]


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("teacher JSON must not contain duplicate fields")
        result[key] = value
    return result


def parse_teacher_label(response: str) -> int:
    """Parse a teacher response containing one integer label, either 0 or 1."""
    if not isinstance(response, str):
        raise ValueError("the teacher must return a JSON string")
    try:
        payload = json.loads(response, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("the teacher response must be a valid JSON object") from error
    if not isinstance(payload, dict) or set(payload) != {"label"}:
        raise ValueError("teacher JSON must contain only the label field")
    label = payload["label"]
    if type(label) is not int or label not in (0, 1):
        raise ValueError("the teacher label must be the integer 0 or 1")
    return label


def annotate_relation(context: PairContext, teachers: Sequence[Teacher]) -> RelationAnnotation:
    """Obtain three valid teacher judgments and return their majority label."""
    teachers = tuple(teachers)
    if len(teachers) != 3 or any(not isinstance(teacher, Teacher) for teacher in teachers):
        raise ValueError("relation annotation requires three Teacher instances")
    provider_ids = tuple(teacher.provider_id for teacher in teachers)
    if len({name.strip().casefold() for name in provider_ids}) != 3:
        raise ValueError("the three teacher model identifiers must be distinct")
    prompt = render_teacher_prompt(context)
    votes = tuple(parse_teacher_label(teacher.generate(prompt)) for teacher in teachers)
    return RelationAnnotation(int(sum(votes) >= 2), votes, provider_ids)
