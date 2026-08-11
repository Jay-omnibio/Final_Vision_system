"""Known-only Mahalanobis calibration for novelty detection."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embedding_store(path: str | Path) -> dict:
    data = np.load(Path(path), allow_pickle=True)
    labels = data["labels"] if "labels" in data.files else data["subclass_labels"]
    return {
        "embeddings": np.asarray(data["embeddings"], dtype=np.float32),
        "labels": [str(label) for label in labels.tolist()],
    }


@dataclass
class NoveltyCalibration:
    labels: list[str]
    means: np.ndarray
    inverse_covariance: np.ndarray
    threshold: float
    known_percentile: float
    shrinkage: float
    gallery_sha256: str
    embedding_dim: int
    known_count: int

    def score(self, embedding: np.ndarray, predicted_label: str) -> float:
        try:
            index = self.labels.index(predicted_label)
        except ValueError:
            return float("inf")
        diff = l2_normalize(embedding) - self.means[index]
        return float(diff @ self.inverse_covariance @ diff)

    def is_new(self, embedding: np.ndarray, predicted_label: str) -> tuple[bool, float]:
        score = self.score(embedding, predicted_label)
        return score > self.threshold, score

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            labels=np.array(self.labels, dtype=object),
            means=self.means.astype(np.float32),
            inverse_covariance=self.inverse_covariance.astype(np.float32),
            threshold=np.float64(self.threshold),
            known_percentile=np.float64(self.known_percentile),
            shrinkage=np.float64(self.shrinkage),
            gallery_sha256=np.array(self.gallery_sha256),
            embedding_dim=np.int64(self.embedding_dim),
            known_count=np.int64(self.known_count),
        )
        meta = {
            "format_version": 1,
            "method": "shared_within_subclass_mahalanobis",
            "known_percentile": self.known_percentile,
            "threshold": self.threshold,
            "shrinkage": self.shrinkage,
            "gallery_sha256": self.gallery_sha256,
            "embedding_dim": self.embedding_dim,
            "known_count": self.known_count,
            "num_subclasses": len(self.labels),
            "labels": self.labels,
        }
        output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NoveltyCalibration":
        data = np.load(Path(path), allow_pickle=True)
        return cls(
            labels=[str(item) for item in data["labels"].tolist()],
            means=np.asarray(data["means"], dtype=np.float32),
            inverse_covariance=np.asarray(data["inverse_covariance"], dtype=np.float32),
            threshold=float(data["threshold"]),
            known_percentile=float(data["known_percentile"]),
            shrinkage=float(data["shrinkage"]),
            gallery_sha256=str(data["gallery_sha256"].item()),
            embedding_dim=int(data["embedding_dim"]),
            known_count=int(data["known_count"]),
        )


def fit_known_only_calibration(
    embeddings: np.ndarray,
    labels: list[str],
    *,
    gallery_sha256: str,
    known_percentile: float = 99.0,
    shrinkage: float = 0.1,
) -> NoveltyCalibration:
    embeddings = l2_normalize_rows(embeddings)
    indexes_by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        indexes_by_label[label].append(index)

    grouped = {
        label: embeddings[np.asarray(indexes)]
        for label, indexes in indexes_by_label.items()
    }
    ordered_labels = sorted(grouped)
    means = np.stack([grouped[label].mean(axis=0) for label in ordered_labels])
    residuals = np.concatenate(
        [grouped[label] - grouped[label].mean(axis=0) for label in ordered_labels],
        axis=0,
    )
    covariance = np.cov(residuals, rowvar=False)
    dimension = covariance.shape[0]
    scale = float(np.trace(covariance) / dimension)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.eye(dimension) * scale
    inverse_covariance = np.linalg.pinv(covariance).astype(np.float32)

    known_scores: list[float] = []
    for label in ordered_labels:
        group = grouped[label]
        count = len(group)
        if count <= 1:
            loo_means = np.repeat(group.mean(axis=0)[None, :], count, axis=0)
        else:
            total = group.sum(axis=0)
            loo_means = (total[None, :] - group) / (count - 1)
        differences = group - loo_means
        scores = np.einsum("ni,ij,nj->n", differences, inverse_covariance, differences)
        known_scores.extend(float(score) for score in scores)

    return NoveltyCalibration(
        labels=ordered_labels,
        means=means.astype(np.float32),
        inverse_covariance=inverse_covariance,
        threshold=float(np.percentile(np.asarray(known_scores), known_percentile)),
        known_percentile=known_percentile,
        shrinkage=shrinkage,
        gallery_sha256=gallery_sha256,
        embedding_dim=embeddings.shape[1],
        known_count=len(embeddings),
    )


def ensure_calibrated(
    *,
    gallery_path: str | Path,
    known_embeddings_path: str | Path,
    calibration_path: str | Path,
    known_percentile: float = 99.0,
    shrinkage: float = 0.1,
    force: bool = False,
) -> tuple[NoveltyCalibration, bool]:
    gallery_hash = file_sha256(gallery_path)
    calibration_path = Path(calibration_path)
    if calibration_path.is_file() and not force:
        calibration = NoveltyCalibration.load(calibration_path)
        if (
            calibration.gallery_sha256 == gallery_hash
            and calibration.known_percentile == known_percentile
            and calibration.shrinkage == shrinkage
        ):
            return calibration, False

    known = load_embedding_store(known_embeddings_path)
    calibration = fit_known_only_calibration(
        known["embeddings"],
        known["labels"],
        gallery_sha256=gallery_hash,
        known_percentile=known_percentile,
        shrinkage=shrinkage,
    )
    calibration.save(calibration_path)
    return calibration, True
