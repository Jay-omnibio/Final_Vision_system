"""Shared prototype-classifier result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "scores": dict(self.scores),
        }


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    score: float
    gallery_type: str
    class_name: str | None = None
    subclass_name: str | None = None
    class_score: float | None = None
    subclass_score: float | None = None
    class_scores: dict[str, float] = field(default_factory=dict)
    subclass_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "gallery_type": self.gallery_type,
            "class_name": self.class_name,
            "subclass_name": self.subclass_name,
            "class_score": self.class_score,
            "subclass_score": self.subclass_score,
            "class_scores": dict(self.class_scores),
            "subclass_scores": dict(self.subclass_scores),
        }

