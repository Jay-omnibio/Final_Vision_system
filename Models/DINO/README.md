# DINO Embedding Module

Standalone image embedding code for the final vision system.

This module only creates embeddings. It does not classify images, build
prototype galleries, detect objects, track objects, or call the robotics API.

## Supported Models

| Name | Hugging Face model |
| --- | --- |
| `dinov2-small` | `facebook/dinov2-small` |
| `dinov3` | `facebook/dinov3-vits16-pretrain-lvd1689m` |

`dinov3` requires approved Hugging Face access to Meta's gated DINOv3 weights.

## Setup

```powershell
cd D:\Coding\Omnibio\Final_Model_train\Final_Vision_system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For DINOv3 gated access, login before the first run:

```powershell
huggingface-cli login
```

## Python Usage

```python
from dino_embedder import DinoEmbedder

embedder = DinoEmbedder(model_name="dinov2-small", device="auto")
vector = embedder.embed_image("sample.jpg")
print(vector.shape, vector.dtype)
```

## Examples

Embed one image:

```powershell
python Models\DINO\examples\embed_image.py path\to\image.jpg --model dinov2-small --output outputs\image_embedding.npy
```

Embed a folder:

```powershell
python Models\DINO\examples\embed_folder.py path\to\images --model dinov2-small --output outputs\folder_embeddings.npz
```

Outputs are L2-normalized `float32` NumPy vectors.

## DINOv2-Small ONNX

Export only `dinov2-small` first:

```powershell
python Models\DINO\scripts\export_dinov2_onnx.py --output Models\DINO\onnx\dinov2-small.onnx --device cpu
```

Compare PyTorch and ONNX embeddings before using ONNX in runtime:

```powershell
python Models\DINO\scripts\compare_onnx_embeddings.py path\to\test_images --onnx Models\DINO\onnx\dinov2-small.onnx --limit 20 --details
```

The ONNX model outputs one L2-normalized `float32` embedding per image with
shape `(batch, 384)`. A good export should produce cosine similarities very
close to `1.0` compared with the PyTorch `DinoEmbedder` on the same crops.
The main `requirements.txt` installs CPU ONNX Runtime; install
`onnxruntime-gpu` on the VM if CUDA ONNX inference is needed.

Use ONNX directly from Python:

```python
from dino_embedder import OnnxDinoEmbedder

embedder = OnnxDinoEmbedder("Models/DINO/onnx/dinov2-small.onnx", device="auto")
vector = embedder.embed_image("sample.jpg")
print(vector.shape, vector.dtype)
```
