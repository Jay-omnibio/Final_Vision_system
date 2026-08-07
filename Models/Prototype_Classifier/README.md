# Prototype Classifier Module

Standalone runtime classifier for existing DINO prototype-gallery `.npz` files.

This module only loads an existing gallery and classifies images or crops. It
does not build galleries, import datasets, save embedding caches, train models,
detect boxes, track objects, or call the robotics API.

## Default Gallery

The default gallery path is:

```text
Models/Prototype_Classifier/galleries/default_gallery.npz
```

The migrated default is a hierarchical DINO-small bbox-crop gallery from the old
robotics demo artifacts.

## Python Usage

```python
from prototype_classifier import PrototypeClassifier

classifier = PrototypeClassifier(
    gallery_path="Models/Prototype_Classifier/galleries/default_gallery.npz",
    dino_model="dinov2-small",
    device="auto",
)

result = classifier.classify_image("crop.jpg")
print(result.label, result.score)
```

You can also classify a crop array directly:

```python
result = classifier.classify_image(crop_bgr, array_format="bgr")
```

## Example

```powershell
python Models\Prototype_Classifier\examples\classify_image.py path\to\crop.jpg
```

For DINOv3, use a gallery built with DINOv3 embeddings and run:

```powershell
python Models\Prototype_Classifier\examples\classify_image.py crop.jpg --dino-model dinov3
```

`dinov3` may require Hugging Face access through `huggingface-cli login` or
`HF_TOKEN`.

