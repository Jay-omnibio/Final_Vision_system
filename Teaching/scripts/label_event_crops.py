#!/usr/bin/env python
"""Copy saved event crops into the teaching dataset with an operator label."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Teaching.teaching_store import TeachingStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="data/operator_events/events.jsonl")
    parser.add_argument("--teaching-dir", default="data/teaching")
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--label-filter", default=None, help="Only import events whose current label matches this")
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def iter_events(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Event file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    store = TeachingStore(resolve(args.teaching_dir))
    imported = 0
    for event in iter_events(resolve(args.events)):
        if args.label_filter is not None and event.get("label") != args.label_filter:
            continue
        if args.track_id is not None and int(event.get("track_id", -1)) != args.track_id:
            continue
        crop_path = event.get("crop_path")
        if not crop_path:
            continue
        samples = store.add_images(
            [crop_path],
            class_name=args.class_name,
            object_name=args.object_name,
            source_event=event,
        )
        imported += len(samples)
        if args.limit is not None and imported >= args.limit:
            break
    print(f"Imported {imported} crop(s) into {resolve(args.teaching_dir)}")


if __name__ == "__main__":
    main()
