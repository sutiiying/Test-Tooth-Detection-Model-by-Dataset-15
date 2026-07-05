import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
GT_CSV = BASE / "fdi_ground_truth_template.csv"
MODEL_ID = "Mobe1/argos-dentsight-stage1-fdi-v5"
MIN_SCORE = 0.02
NMS_IOU_THRESHOLD = 0.5
ARTIFACT_ZONE = (0.47, 0.60, 0.55, 0.69)


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


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def in_artifact_zone(box, img_w, img_h):
    cx = (box[0] + box[2]) / 2 / img_w
    cy = (box[1] + box[3]) / 2 / img_h
    x0, y0, x1, y1 = ARTIFACT_ZONE
    return x0 <= cx <= x1 and y0 <= cy <= y1


def raw_candidates(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)[0]
    out = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        code = model.config.id2label[label.item()]
        out.append({"score": score.item(), "code": code, "box": box.tolist()})
    return image.size, out


def strategy_per_class_argmax(all_candidates, img_size, fdi_columns, use_artifact_filter, min_score=MIN_SCORE):
    img_w, img_h = img_size
    best = {code: 0.0 for code in fdi_columns}
    for c in all_candidates:
        if use_artifact_filter and in_artifact_zone(c["box"], img_w, img_h):
            continue
        if c["code"] in best and c["score"] > best[c["code"]]:
            best[c["code"]] = c["score"]
    return {code: (best[code] >= min_score) for code in fdi_columns}


def strategy_nms_argbox(all_candidates, img_size, fdi_columns, use_artifact_filter):
    img_w, img_h = img_size
    cands = [c for c in all_candidates if c["score"] >= MIN_SCORE]
    if use_artifact_filter:
        cands = [c for c in cands if not in_artifact_zone(c["box"], img_w, img_h)]
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if all(box_iou(c["box"], k["box"]) < NMS_IOU_THRESHOLD for k in kept):
            kept.append(c)
    present = {code: False for code in fdi_columns}
    for c in kept:
        if c["code"] in present:
            present[c["code"]] = True
    return present


def evaluate(gt, predictions_by_image, fdi_columns):
    tp = fp = fn = tn = 0
    fp_image_count = 0
    for image_name, actual in gt.items():
        pred = predictions_by_image.get(image_name)
        if pred is None:
            continue
        has_fp = False
        for code in fdi_columns:
            a, p = actual[code], pred[code]
            if p and a:
                tp += 1
            elif p and not a:
                fp += 1
                has_fp = True
            elif not p and a:
                fn += 1
            else:
                tn += 1
        if has_fp:
            fp_image_count += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall, tp, fp, fn, tn, fp_image_count


def main():
    """Diagnostic sweep used to choose evaluate_fdi.py's MIN_SCORE and to
    decide between per-class-argmax vs NMS+per-box-argmax postprocessing.
    Not part of the regular pipeline - rerun manually if the model,
    ground-truth CSV, or artifact zone change."""
    fdi_columns, gt = load_ground_truth(GT_CSV)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
    model.eval()

    raw = {}
    for image_name in gt:
        image_path = DATASET_DIR / image_name
        if not image_path.exists():
            continue
        img_size, cands = raw_candidates(model, processor, image_path)
        raw[image_name] = (img_size, cands)
        print(f"Ran inference on {image_name}: {len(cands)} raw candidates")

    variants = {
        "A: per-class-argmax, no filter (old baseline)": lambda name: strategy_per_class_argmax(raw[name][1], raw[name][0], fdi_columns, False),
        "B: per-class-argmax + artifact filter": lambda name: strategy_per_class_argmax(raw[name][1], raw[name][0], fdi_columns, True),
        "C: NMS+argbox, no filter": lambda name: strategy_nms_argbox(raw[name][1], raw[name][0], fdi_columns, False),
        "D: NMS+argbox + artifact filter": lambda name: strategy_nms_argbox(raw[name][1], raw[name][0], fdi_columns, True),
    }

    print("\n=== Strategy comparison (micro-averaged over 15 GT images) ===")
    for name, fn_ in variants.items():
        preds = {image_name: fn_(image_name) for image_name in raw}
        precision, recall, tp, fp, fn, tn, fp_img = evaluate(gt, preds, fdi_columns)
        print(f"{name}")
        print(f"    precision={precision:.3f} recall={recall:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}  images_with_FP={fp_img}/{len(raw)}")

    print("\n=== MIN_SCORE sweep: per-class-argmax + artifact filter ===")
    for thresh in [0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.040]:
        preds = {
            image_name: strategy_per_class_argmax(raw[image_name][1], raw[image_name][0], fdi_columns, True, min_score=thresh)
            for image_name in raw
        }
        precision, recall, tp, fp, fn, tn, fp_img = evaluate(gt, preds, fdi_columns)
        print(f"  MIN_SCORE={thresh:.3f}  precision={precision:.3f} recall={recall:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}  images_with_FP={fp_img}/{len(raw)}")

    print("\n=== Same sweep WITHOUT artifact filter (isolate its effect at low threshold) ===")
    for thresh in [0.010, 0.012, 0.015, 0.020]:
        preds = {
            image_name: strategy_per_class_argmax(raw[image_name][1], raw[image_name][0], fdi_columns, False, min_score=thresh)
            for image_name in raw
        }
        precision, recall, tp, fp, fn, tn, fp_img = evaluate(gt, preds, fdi_columns)
        print(f"  MIN_SCORE={thresh:.3f}  precision={precision:.3f} recall={recall:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}  images_with_FP={fp_img}/{len(raw)}")


if __name__ == "__main__":
    main()
