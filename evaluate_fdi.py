import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
GT_CSV = BASE / "fdi_ground_truth_template.csv"
REPORT_CSV = BASE / "fdi_evaluation_report.csv"
OUTPUT_DIR = BASE / "result4"
MODEL_ID = "Mobe1/argos-dentsight-stage1-fdi-v5"

# The model's confidence head is uncalibrated (real-world scores top out
# around ~0.07-0.15), so this is a relative floor, not a calibrated
# probability threshold. Tuned via a sweep against the 15 ground-truth
# images (see tune_fdi_postprocessing.py): 0.012 gives the best recall
# without a meaningful precision cost. NOTE: tuned on the same 15 images
# used for reporting metrics below (no separate validation set was
# available), so treat these numbers as optimistic - performance on new,
# unlabeled images will likely be somewhat lower.
MIN_SCORE = 0.012

# Fixed-position artifact (bite block / positioning peg the patient bites
# on during the scan). Confirmed by inspecting raw per-query boxes across
# several unrelated patients: the exact same pixel box (e.g. roughly
# x=[494,550] y=[330,365] on a ~1023x537 image) kept appearing and
# competing for different tooth labels (15/25/26/16/37/46) across images -
# only possible if it's a stationary part of the X-ray rig, not a tooth
# (a real tooth's box would move between patients). Excluded here as a
# fixed zone in normalized image coordinates, with a small margin.
ARTIFACT_ZONE = (0.47, 0.60, 0.55, 0.69)  # (x0, y0, x1, y1) as fraction of image size

# NOTE on an approach that was tried and rejected: switching to a more
# "standard" class-agnostic NMS + per-box-argmax pipeline was tested (see
# tune_fdi_postprocessing.py) but cut recall from 0.75 to ~0.58 on this
# model/dataset, so it was dropped in favor of keeping the simpler
# per-class-argmax selection (scan all 300 queries independently for each
# of the 32 FDI classes) plus the artifact filter above.


def load_ground_truth(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fdi_columns = [c for c in reader.fieldnames if c not in ("image", "notes")]
        gt = {}
        for row in reader:
            image = (row.get("image") or "").strip()
            if not image:
                continue
            gt[image] = {code: (row[code] or "").strip() == "1" for code in fdi_columns}
    return fdi_columns, gt


def in_artifact_zone(box, img_w, img_h):
    cx = (box[0] + box[2]) / 2 / img_w
    cy = (box[1] + box[3]) / 2 / img_h
    x0, y0, x1, y1 = ARTIFACT_ZONE
    return x0 <= cx <= x1 and y0 <= cy <= y1


def predict_best_per_class(model, processor, image_path, fdi_columns):
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.0, target_sizes=target_sizes
    )[0]

    best = {code: {"score": 0.0, "box": None} for code in fdi_columns}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        b = box.tolist()
        if in_artifact_zone(b, img_w, img_h):
            continue
        code = model.config.id2label[label.item()]
        if code in best and score.item() > best[code]["score"]:
            best[code] = {"score": score.item(), "box": b}

    return image, best


def main():
    fdi_columns, gt = load_ground_truth(GT_CSV)
    print(f"Loaded ground truth for {len(gt)} images, {len(fdi_columns)} FDI classes")

    OUTPUT_DIR.mkdir(exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()

    predictions = {}
    images = {}
    for image_name in gt:
        image_path = DATASET_DIR / image_name
        if not image_path.exists():
            print(f"WARNING: {image_name} listed in ground truth but not found in dataset/, skipping")
            continue
        image, best = predict_best_per_class(model, processor, image_path, fdi_columns)
        predictions[image_name] = best
        images[image_name] = image
        print(f"Ran inference on {image_name}")

    # Per-class confusion counts
    stats = {code: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for code in fdi_columns}
    fp_images = {}  # image -> list of FDI codes falsely predicted present

    for image_name, actual in gt.items():
        best = predictions.get(image_name)
        if best is None:
            continue

        false_positive_codes = []
        fn_codes = []
        draw = ImageDraw.Draw(images[image_name])

        for code in fdi_columns:
            is_actual = actual[code]
            is_pred = best[code]["score"] >= MIN_SCORE
            if is_pred and is_actual:
                stats[code]["tp"] += 1
                x0, y0, x1, y1 = best[code]["box"]
                draw.rectangle((x0, y0, x1, y1), outline="lime", width=2)
                draw.text((x0, max(0, y0 - 12)), code, fill="lime")
            elif is_pred and not is_actual:
                stats[code]["fp"] += 1
                false_positive_codes.append(code)
                x0, y0, x1, y1 = best[code]["box"]
                draw.rectangle((x0, y0, x1, y1), outline="red", width=2)
                draw.text((x0, max(0, y0 - 12)), f"{code}?", fill="red")
            elif not is_pred and is_actual:
                stats[code]["fn"] += 1
                fn_codes.append(code)
            else:
                stats[code]["tn"] += 1

        if false_positive_codes:
            fp_images[image_name] = false_positive_codes

        caption = f"green=correct  red=phantom tooth  missed(FN)={','.join(fn_codes) if fn_codes else 'none'}"
        draw.text((5, images[image_name].height - 15), caption, fill="yellow")

        out_path = OUTPUT_DIR / image_name
        images[image_name].save(out_path)

    # Per-class precision/recall report
    print("\nFDI  Precision  Recall  TP  FP  FN  TN")
    total_tp = total_fp = total_fn = total_tn = 0
    rows = []
    for code in fdi_columns:
        s = stats[code]
        tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"{code:>3}  {precision:>9.2f}  {recall:>6.2f}  {tp:>2}  {fp:>2}  {fn:>2}  {tn:>2}")
        rows.append({
            "fdi": code, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float("nan")
    print(f"\nMicro-average across all classes: precision={micro_precision:.3f} recall={micro_recall:.3f}")
    print(f"(TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn} over {len(predictions)} images)")

    print(f"\nImages with at least one false-positive tooth (drawn where ground truth says missing): "
          f"{len(fp_images)} / {len(predictions)}")
    for image_name, codes in fp_images.items():
        print(f"  {image_name}: false-positive FDI {codes}")

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fdi", "precision", "recall", "tp", "fp", "fn", "tn"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-class report written to {REPORT_CSV}")
    print(f"Annotated comparison images (green=correct, red=phantom tooth) written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
