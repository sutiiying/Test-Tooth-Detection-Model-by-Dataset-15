import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
OUTPUT_DIR = BASE / "result2"
MODEL_ID = "Mobe1/argos-dentsight-stage1-fdi-v5"

# NOTE: this model has a documented classification-head calibration issue
# (raw confidence maxes out around ~0.07 on real radiographs - see the model
# card). A normal confidence threshold yields zero detections, so instead we
# take the highest-scoring box per FDI class ("argmax-per-class"), which is
# the usage the model card itself recommends. Absolute scores are shown for
# reference only and should not be read as calibrated confidence.
MIN_SCORE = 0.02

OUTPUT_DIR.mkdir(exist_ok=True)

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
model.eval()

images = sorted(DATASET_DIR.glob("*.png"))
print(f"Found {len(images)} images")

if "--test" in sys.argv:
    images = images[:1]

for img_path in images:
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.0, target_sizes=target_sizes
    )[0]

    scores, labels, boxes = results["scores"], results["labels"], results["boxes"]
    order = torch.argsort(scores, descending=True)

    draw = ImageDraw.Draw(image)
    seen_labels = set()
    n_drawn = 0
    for i in order:
        label = labels[i].item()
        if label in seen_labels:
            continue
        seen_labels.add(label)
        score = scores[i].item()
        if score < MIN_SCORE:
            continue
        x0, y0, x1, y1 = boxes[i].tolist()
        fdi_number = model.config.id2label[label]
        draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
        draw.text((x0, max(0, y0 - 12)), f"{fdi_number} {score:.2f}", fill="red")
        n_drawn += 1

    out_path = OUTPUT_DIR / img_path.name
    image.save(out_path)
    print(f"{img_path.name}: {n_drawn} teeth detected -> {out_path}")

print("Done.")
