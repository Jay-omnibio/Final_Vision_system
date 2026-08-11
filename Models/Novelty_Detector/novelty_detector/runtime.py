"""Runtime known/new detector over DINO embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .calibration import NoveltyCalibration, ensure_calibrated, l2_normalize
from .grouping import OnlineUnknownGrouper, UnknownGroup
from .types import NoveltyResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_ROOT = PROJECT_ROOT / "Models" / "Prototype_Classifier"
if str(CLASSIFIER_ROOT) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_ROOT))

from prototype_classifier.gallery import HierarchicalPrototypeGallery, load_gallery  # noqa: E402


class NoveltyRuntime:
    def __init__(
        self,
        gallery: HierarchicalPrototypeGallery,
        calibration: NoveltyCalibration,
        *,
        group_similarity_threshold: float = 0.55,
        known_cosine_override: float | None = 0.65,
        known_min_margin: float = 0.06,
        known_min_subclass_score: float = 0.50,
    ) -> None:
        self.gallery = gallery
        self.calibration = calibration
        self.grouper = OnlineUnknownGrouper(group_similarity_threshold)
        self.known_cosine_override = known_cosine_override
        self.known_min_margin = known_min_margin
        self.known_min_subclass_score = known_min_subclass_score

    def process(self, embedding: np.ndarray, *, commit: bool = True) -> NoveltyResult:
        embedding = l2_normalize(embedding)
        prediction = self.gallery.classify(embedding)
        predicted_label = prediction.label
        is_new, novelty_score = self.calibration.is_new(embedding, predicted_label)
        top1, top2, margin = self._subclass_top2(embedding)
        subclass_score = float(prediction.subclass_score or prediction.score)
        confident = subclass_score >= self.known_min_subclass_score and margin >= self.known_min_margin

        if (
            is_new
            and self.known_cosine_override is not None
            and subclass_score >= self.known_cosine_override
            and confident
        ):
            is_new = False
        if not is_new and not confident:
            is_new = True

        base = {
            "predicted_label": predicted_label,
            "predicted_class": str(prediction.class_name),
            "predicted_subclass": str(prediction.subclass_name),
            "class_score": float(prediction.class_score or 0.0),
            "subclass_score": subclass_score,
            "subclass_margin": float(top1 - top2 if np.isfinite(top2) else top1),
            "novelty_score": float(novelty_score),
            "novelty_threshold": float(self.calibration.threshold),
        }
        if not is_new:
            return NoveltyResult(status="known", final_label=predicted_label, **base)

        if commit:
            group, similarity, created = self.grouper.assign(embedding)
            suggestion = self._suggest_group(group)
            return NoveltyResult(
                status="new",
                final_label=group.label,
                new_group=group.label,
                new_group_created=created,
                new_group_similarity=float(similarity),
                new_group_count=group.count,
                **suggestion,
                **base,
            )

        match, similarity = self.grouper.best_match(embedding)
        if match is not None and similarity >= self.grouper.similarity_threshold:
            suggestion = self._suggest_group(match)
            return NoveltyResult(
                status="new",
                final_label=match.label,
                new_group=match.label,
                new_group_similarity=float(similarity),
                new_group_count=match.count,
                **suggestion,
                **base,
            )

        return NoveltyResult(
            status="new",
            final_label=f"new_{len(self.grouper.groups) + 1}",
            new_group=f"new_{len(self.grouper.groups) + 1}",
            new_group_created=True,
            new_group_similarity=float(similarity),
            suggested_class=str(prediction.class_name),
            nearest_known_subclass=predicted_label,
            **base,
        )

    def _subclass_top2(self, embedding: np.ndarray) -> tuple[float, float, float]:
        scores = self.gallery.subclass_vectors @ embedding
        if scores.size == 0:
            return 0.0, 0.0, 0.0
        if scores.size == 1:
            return float(scores[0]), float("-inf"), float(scores[0])
        order = np.argsort(scores)[::-1]
        top1 = float(scores[order[0]])
        top2 = float(scores[order[1]])
        return top1, top2, top1 - top2

    def _suggest_group(self, group: UnknownGroup) -> dict:
        embedding = group.prototype
        class_scores = self.gallery.class_gallery.vectors @ embedding
        subclass_scores = self.gallery.subclass_vectors @ embedding
        class_index = int(np.argmax(class_scores))
        subclass_index = int(np.argmax(subclass_scores))
        return {
            "suggested_class": self.gallery.class_names[class_index],
            "nearest_known_subclass": (
                f"{self.gallery.subclass_class_names[subclass_index]}/"
                f"{self.gallery.subclass_names[subclass_index]}"
            ),
        }


def load_novelty_runtime(
    *,
    gallery_path: str | Path,
    known_embeddings_path: str | Path,
    calibration_path: str | Path,
    known_percentile: float = 99.0,
    shrinkage: float = 0.1,
    force_calibration: bool = False,
) -> NoveltyRuntime:
    gallery = load_gallery(gallery_path)
    if not isinstance(gallery, HierarchicalPrototypeGallery):
        raise ValueError("Novelty runtime requires a hierarchical prototype gallery")
    calibration, _ = ensure_calibrated(
        gallery_path=gallery_path,
        known_embeddings_path=known_embeddings_path,
        calibration_path=calibration_path,
        known_percentile=known_percentile,
        shrinkage=shrinkage,
        force=force_calibration,
    )
    return NoveltyRuntime(gallery, calibration)
