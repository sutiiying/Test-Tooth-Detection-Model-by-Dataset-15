import sys
from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "8024.pt"
DATASET_DIR = BASE / "dataset"
OUTPUT_DIR = BASE / "results"

OUTPUT_DIR.mkdir(exist_ok=True)

model = YOLO(str(MODEL_PATH))
print("Model classes:", model.names)

images = sorted(DATASET_DIR.glob("*.png"))
print(f"Found {len(images)} images")

if "--test" in sys.argv:
    images = images[:1]

results = model.predict(
    source=[str(p) for p in images],
    conf=0.25,
    save=False,
    verbose=False,
)

for img_path, result in zip(images, results):
    out_path = OUTPUT_DIR / img_path.name
    result.save(filename=str(out_path))
    n_detections = len(result.boxes) if result.boxes is not None else 0
    print(f"{img_path.name}: {n_detections} detections -> {out_path}")

print("Done.")
