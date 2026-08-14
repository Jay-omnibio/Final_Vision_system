#!/usr/bin/env python
"""Export DINOv2-small to ONNX as a normalized 384-d embedding model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dino_embedder.model_registry import resolve_model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="Models/DINO/onnx/dinov2-small.onnx")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--model-id", default=None, help="Optional local/Hugging Face model ID override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModel

    class DinoEmbeddingWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            outputs = self.model(pixel_values=pixel_values)
            cls_token = outputs.last_hidden_state[:, 0, :]
            return torch.nn.functional.normalize(cls_token, p=2, dim=1)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id("dinov2-small", args.model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(args.device)
    wrapped = DinoEmbeddingWrapper(model).eval()
    dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=args.device)

    torch.onnx.export(
        wrapped,
        (dummy,),
        str(output),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"exported {output} model_id={model_id} output_shape=(batch, 384)")


if __name__ == "__main__":
    main()
