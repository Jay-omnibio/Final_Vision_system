"""Shared novelty detector result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoveltyResult:
    status: str
    final_label: str
    predicted_label: str
    predicted_class: str
    predicted_subclass: str
    class_score: float
    subclass_score: float
    subclass_margin: float
    novelty_score: float
    novelty_threshold: float
    new_group: str | None = None
    new_group_created: bool = False
    new_group_similarity: float = 0.0
    new_group_count: int = 0
    suggested_class: str | None = None
    nearest_known_subclass: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_label": self.final_label,
            "predicted_label": self.predicted_label,
            "predicted_class": self.predicted_class,
            "predicted_subclass": self.predicted_subclass,
            "class_score": self.class_score,
            "subclass_score": self.subclass_score,
            "subclass_margin": self.subclass_margin,
            "novelty_score": self.novelty_score,
            "novelty_threshold": self.novelty_threshold,
            "new_group": self.new_group,
            "new_group_created": self.new_group_created,
            "new_group_similarity": self.new_group_similarity,
            "new_group_count": self.new_group_count,
            "suggested_class": self.suggested_class,
            "nearest_known_subclass": self.nearest_known_subclass,
        }
