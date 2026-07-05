# Test Tooth Detection Model by Dataset 15

Trial run of tooth/pathology detection on panoramic dental X-rays.

## Background

The original goal was to use [clemkoa/tooth-detection](https://github.com/clemkoa/tooth-detection)
(a Faster R-CNN / TensorFlow Object Detection API project) on a private dataset of 48 panoramic
dental X-rays. That repository does not ship trained weights or a dataset — both are withheld by
the author for patient privacy reasons, so it cannot run out of the box.

As a substitute, this project uses a publicly available pretrained YOLOv8 model instead:
[nsitnov/8024-yolov8-model](https://huggingface.co/nsitnov/8024-yolov8-model) (Hugging Face),
which detects 8 classes on dental X-rays:

`Caries, Crown, Filling, Implant, Missing teeth, Periapical lesion, Root Piece, Root canal obturation`

## Setup

```
pip install -r requirements.txt
python download_model.py   # downloads model/8024.pt from Hugging Face (~144 MB)
```

Place your own panoramic X-ray images (`.png`) in a `dataset/` folder next to `run_inference.py`.

## Run

```
python run_inference.py
```

Annotated images (bounding boxes + class + confidence) are written to `results/`.

## Notes / limitations

- `dataset/`, `results/`, and `model/` are excluded from version control (see `.gitignore`):
  the dataset contains real patient X-rays and is not published here for privacy reasons; the
  model weights are fetched on demand instead of being vendored (also over GitHub's 100 MB file
  limit).
- The model is used as-is, not fine-tuned on this dataset, so confidence scores are modest
  (roughly 0.25-0.75) and results should be treated as a preliminary screening demo, not a
  diagnostic tool.
- Unlike the original clemkoa project, this model does not perform per-tooth ISO/FDI numbering -
  it only classifies restorations/pathology regions.
