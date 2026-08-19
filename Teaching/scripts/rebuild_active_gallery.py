#!/usr/bin/env python
"""Build an active prototype gallery from operator-taught images."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINO_ROOT = PROJECT_ROOT / "Models" / "DINO"
CLASSIFIER_ROOT = PROJECT_ROOT / "Models" / "Prototype_Classifier"
NOVELTY_ROOT = PROJECT_ROOT / "Models" / "Novelty_Detector"
for path in (PROJECT_ROOT, DINO_ROOT, CLASSIFIER_ROOT, NOVELTY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dino_embedder import create_dino_embedder  # noqa: E402
from novelty_detector.calibration import ensure_calibrated  # noqa: E402
from prototype_classifier.gallery import HierarchicalPrototypeGallery  # noqa: E402
from Teaching.teaching_store import TeachingStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gallery", default="Models/Prototype_Classifier/galleries/default_gallery.npz")
    parser.add_argument("--base-known-embeddings", default="Models/Novelty_Detector/artifacts/embeddings/known.npz")
    parser.add_argument("--teaching-dir", default="data/teaching")
    parser.add_argument("--output", default="Models/Prototype_Classifier/galleries/active_gallery.npz")
    parser.add_argument("--known-embeddings-output", default="Models/Novelty_Detector/artifacts/embeddings/active_known.npz")
    parser.add_argument("--calibration-output", default="Models/Novelty_Detector/artifacts/calibration/active_novelty_mahalanobis.npz")
    parser.add_argument("--dino-model", default="dinov2-small", choices=["dinov2-small", "dinov3"])
    parser.add_argument("--dino-backend", default="torch", choices=["torch", "pytorch", "onnx"])
    parser.add_argument("--dino-onnx-path", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default="replace", choices=["replace", "append_average"])
    parser.add_argument("--update-config", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    return normalize(np.mean(np.stack(vectors), axis=0))


@dataclass
class EmbeddedTeachingData:
    grouped: dict[tuple[str, str], list[np.ndarray]]
    embeddings: list[np.ndarray]
    labels: list[str]


def load_base_subclasses(path: Path) -> dict[tuple[str, str], np.ndarray]:
    gallery = HierarchicalPrototypeGallery.load(path)
    return {
        (class_name, subclass_name): normalize(vector)
        for class_name, subclass_name, vector in zip(
            gallery.subclass_class_names,
            gallery.subclass_names,
            gallery.subclass_vectors,
        )
    }


def embed_teaching_images(teaching_dir: Path, embedder) -> EmbeddedTeachingData:
    store = TeachingStore(teaching_dir)
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    for class_name, object_name, image_path in store.iter_labeled_images():
        embedding = embedder.embed_image(image_path)
        normalized = normalize(embedding)
        grouped[(class_name, object_name)].append(normalized)
        embeddings.append(normalized)
        labels.append(f"{class_name}/{object_name}")
    return EmbeddedTeachingData(grouped=grouped, embeddings=embeddings, labels=labels)


def load_known_embeddings(path: Path | None) -> tuple[list[np.ndarray], list[str]]:
    if path is None or not path.is_file():
        return [], []
    data = np.load(path, allow_pickle=True)
    label_key = "labels" if "labels" in data.files else "subclass_labels"
    embeddings = [normalize(vector) for vector in np.asarray(data["embeddings"], dtype=np.float32)]
    labels = [str(label) for label in data[label_key].tolist()]
    return embeddings, labels


def write_known_embeddings(path: Path, embeddings: list[np.ndarray], labels: list[str], *, meta: dict) -> None:
    if not embeddings:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.stack(embeddings).astype(np.float32),
        labels=np.asarray(labels, dtype=object),
        meta_json=np.asarray(json.dumps(meta)),
    )
    path.with_suffix(".json").write_text(json.dumps({**meta, "num_embeddings": len(embeddings)}, indent=2), encoding="utf-8")


def build_gallery(
    *,
    base_gallery: Path,
    base_known_embeddings: Path | None,
    teaching_dir: Path,
    output_path: Path,
    known_embeddings_output: Path | None,
    calibration_output: Path | None,
    dino_model: str,
    dino_backend: str,
    dino_onnx_path: Path | None,
    device: str,
    mode: str,
) -> dict:
    subclass_vectors = load_base_subclasses(base_gallery)
    embedder = create_dino_embedder(
        backend=dino_backend,
        model_name=dino_model,
        device=device,
        onnx_path=dino_onnx_path,
    )
    taught = embed_teaching_images(teaching_dir, embedder)

    for label, embeddings in taught.grouped.items():
        taught_vector = mean_embedding(embeddings)
        if label in subclass_vectors and mode == "append_average":
            subclass_vectors[label] = mean_embedding([subclass_vectors[label], taught_vector])
        else:
            subclass_vectors[label] = taught_vector

    class_to_subclasses: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for (class_name, subclass_name), vector in subclass_vectors.items():
        class_to_subclasses[class_name].append((subclass_name, vector))

    class_names = sorted(class_to_subclasses)
    class_vectors = []
    subclass_names = []
    subclass_class_names = []
    ordered_subclass_vectors = []
    for class_name in class_names:
        pairs = sorted(class_to_subclasses[class_name], key=lambda item: item[0])
        class_vectors.append(mean_embedding([vector for _, vector in pairs]))
        for subclass_name, vector in pairs:
            subclass_class_names.append(class_name)
            subclass_names.append(subclass_name)
            ordered_subclass_vectors.append(vector)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "gallery_type": "hierarchical",
        "source": "base_gallery_plus_teaching_images",
        "base_gallery": str(base_gallery),
        "teaching_dir": str(teaching_dir),
        "dino_model": dino_model,
        "dino_backend": dino_backend,
        "dino_onnx_path": str(dino_onnx_path) if dino_onnx_path else None,
        "mode": mode,
        "num_classes": len(class_names),
        "num_subclasses": len(subclass_names),
        "taught_labels": [f"{class_name}/{object_name}" for class_name, object_name in sorted(taught.grouped)],
    }
    np.savez_compressed(
        output_path,
        class_vectors=np.stack(class_vectors).astype(np.float32),
        class_names=np.asarray(class_names, dtype=object),
        subclass_vectors=np.stack(ordered_subclass_vectors).astype(np.float32),
        subclass_names=np.asarray(subclass_names, dtype=object),
        subclass_class_names=np.asarray(subclass_class_names, dtype=object),
        meta_json=np.asarray(json.dumps(meta)),
    )
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if known_embeddings_output is not None:
        base_embeddings, base_labels = load_known_embeddings(base_known_embeddings)
        known_meta = {
            "source": "base_known_embeddings_plus_teaching_images",
            "base_known_embeddings": str(base_known_embeddings) if base_known_embeddings else None,
            "teaching_dir": str(teaching_dir),
            "gallery_path": str(output_path),
            "dino_model": dino_model,
            "dino_backend": dino_backend,
        }
        write_known_embeddings(
            known_embeddings_output,
            base_embeddings + taught.embeddings,
            base_labels + taught.labels,
            meta=known_meta,
        )
        if calibration_output is not None and (base_embeddings or taught.embeddings):
            ensure_calibrated(
                gallery_path=output_path,
                known_embeddings_path=known_embeddings_output,
                calibration_path=calibration_output,
                force=True,
            )
    return meta


def update_config_gallery(
    config_path: Path,
    gallery_path: Path,
    known_embeddings_path: Path | None = None,
    calibration_path: Path | None = None,
) -> None:
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    classifier = data.setdefault("classifier", {})
    classifier["gallery_path"] = str(gallery_path.relative_to(PROJECT_ROOT))
    novelty = data.setdefault("novelty", {})
    novelty["gallery_path"] = str(gallery_path.relative_to(PROJECT_ROOT))
    if known_embeddings_path is not None:
        novelty["known_embeddings_path"] = str(known_embeddings_path.relative_to(PROJECT_ROOT))
    if calibration_path is not None:
        novelty["calibration_path"] = str(calibration_path.relative_to(PROJECT_ROOT))
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_path = resolve(args.output)
    known_embeddings_output = resolve(args.known_embeddings_output) if args.known_embeddings_output else None
    calibration_output = resolve(args.calibration_output) if args.calibration_output else None
    meta = build_gallery(
        base_gallery=resolve(args.base_gallery),
        base_known_embeddings=resolve(args.base_known_embeddings) if args.base_known_embeddings else None,
        teaching_dir=resolve(args.teaching_dir),
        output_path=output_path,
        known_embeddings_output=known_embeddings_output,
        calibration_output=calibration_output,
        dino_model=args.dino_model,
        dino_backend=args.dino_backend,
        dino_onnx_path=resolve(args.dino_onnx_path) if args.dino_onnx_path else None,
        device=args.device,
        mode=args.mode,
    )
    if args.update_config:
        update_config_gallery(PROJECT_ROOT / "config.yaml", output_path, known_embeddings_output, calibration_output)
    print(f"Wrote active gallery: {output_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
