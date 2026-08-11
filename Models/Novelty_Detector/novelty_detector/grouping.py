"""Online grouping for objects rejected as new."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import l2_normalize


@dataclass
class UnknownGroup:
    group_id: int
    vector_sum: np.ndarray
    count: int

    @property
    def label(self) -> str:
        return f"new_{self.group_id}"

    @property
    def prototype(self) -> np.ndarray:
        return l2_normalize(self.vector_sum)

    def update(self, embedding: np.ndarray) -> None:
        self.vector_sum = self.vector_sum + l2_normalize(embedding)
        self.count += 1


class OnlineUnknownGrouper:
    def __init__(self, similarity_threshold: float = 0.55) -> None:
        self.similarity_threshold = similarity_threshold
        self.groups: list[UnknownGroup] = []

    def best_match(self, embedding: np.ndarray) -> tuple[UnknownGroup | None, float]:
        embedding = l2_normalize(embedding)
        if not self.groups:
            return None, 0.0
        similarities = np.asarray([float(group.prototype @ embedding) for group in self.groups])
        best_index = int(np.argmax(similarities))
        return self.groups[best_index], float(similarities[best_index])

    def assign(self, embedding: np.ndarray) -> tuple[UnknownGroup, float, bool]:
        embedding = l2_normalize(embedding)
        match, similarity = self.best_match(embedding)
        if match is not None and similarity >= self.similarity_threshold:
            match.update(embedding)
            return match, similarity, False

        group = UnknownGroup(len(self.groups) + 1, embedding.copy(), 1)
        self.groups.append(group)
        return group, similarity, True

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.groups:
            sums = np.stack([group.vector_sum for group in self.groups])
            counts = np.asarray([group.count for group in self.groups], dtype=np.int64)
            ids = np.asarray([group.group_id for group in self.groups], dtype=np.int64)
        else:
            sums = np.empty((0, 0), dtype=np.float32)
            counts = np.empty((0,), dtype=np.int64)
            ids = np.empty((0,), dtype=np.int64)
        np.savez_compressed(
            output_path,
            vector_sums=sums,
            counts=counts,
            group_ids=ids,
            similarity_threshold=np.float64(self.similarity_threshold),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OnlineUnknownGrouper":
        data = np.load(Path(path), allow_pickle=True)
        grouper = cls(float(data["similarity_threshold"]))
        for group_id, vector_sum, count in zip(data["group_ids"], data["vector_sums"], data["counts"]):
            grouper.groups.append(UnknownGroup(int(group_id), vector_sum.astype(np.float32), int(count)))
        return grouper
