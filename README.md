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

## Second experiment: tooth numbering (ISO/FDI notation)

To also cover the original clemkoa project's other stated goal - detecting individual teeth and
labeling them with ISO/FDI notation - `run_inference_fdi.py` runs a second, separate model:
[Mobe1/argos-dentsight-stage1-fdi-v5](https://huggingface.co/Mobe1/argos-dentsight-stage1-fdi-v5)
(D-FINE object detector, Hugging Face `transformers`, CC-BY-NC-SA-4.0), which predicts the 32
permanent-tooth FDI positions (11-48).

```
python run_inference_fdi.py
```

Annotated images are written to `result2/` (also gitignored, same patient-privacy reason as above).

**Known model limitation:** the model card documents a classification-head calibration issue -
raw confidence tops out around ~0.07 on real radiographs, so a normal confidence threshold
returns zero detections. Per the model card's own recommendation, this script instead takes the
highest-scoring box per FDI class ("argmax-per-class") rather than filtering by absolute
confidence. Box positions are reasonably accurate in practice, but the displayed scores are not
calibrated probabilities and per-tooth numbering should be treated as approximate, not
diagnostic-grade.

## Evaluation against ground truth

`fdi_ground_truth_template.csv` records, for a 15-image subset, which of the 32 FDI tooth
positions each patient actually has. `evaluate_fdi.py` re-runs the FDI model on those 15 images,
compares against the CSV, and writes:

- `fdi_evaluation_report.csv` - per-class precision/recall
- `result4/` - annotated images (green box = correct, red box = phantom tooth, caption lists
  missed teeth) - gitignored, same patient-privacy reason as the other result folders

```
python evaluate_fdi.py
```

Two real issues were found and fixed here:

1. **A fixed-position artifact was being detected as a tooth.** Inspecting raw per-query boxes
   across several unrelated patients turned up the exact same pixel box (around x=48-54%,
   y=60-69% of the image) competing for different tooth labels in each image - only possible if
   it's a stationary object in the scan (a bite block / positioning peg), not a tooth, since a
   real tooth's position varies between patients. `evaluate_fdi.py` now excludes any detection
   whose box center falls in that zone (`ARTIFACT_ZONE`).
2. **Recall was too low (0.75).** The model's classification head is uncalibrated (see above), so
   the original confidence floor (0.02) was cutting off many real teeth along with the noise. A
   threshold sweep (see `tune_fdi_postprocessing.py`) found 0.012 recovers recall to 0.97 with
   essentially no precision cost (0.970 vs 0.969 before). A fancier class-agnostic-NMS +
   per-box-argmax pipeline was also tried but made recall *worse* (down to ~0.58), so it was
   dropped in favor of this simpler threshold retune.

Result: **precision 0.970, recall 0.972** (was 0.969 / 0.752), on the 15 ground-truth images.
A handful of false positives remain (14, spread over 6 images) - inspecting them shows these are
mostly real teeth assigned the *wrong* FDI code (e.g. crowding/rotation confusing the quadrant or
tooth-type guess), not the artifact from issue 1, which is a separate and harder problem: the
model's tooth-type classification itself is unreliable, and fixing that would need real
bounding-box-level training data (this dataset only has image-level tooth-presence labels, not
box annotations, so proper re-training of the detector isn't possible here). Also note the
threshold was tuned on the same 15 images used to report these numbers (no separate validation
set was available), so real-world performance on new, unlabeled images is likely somewhat lower
than these numbers suggest.
