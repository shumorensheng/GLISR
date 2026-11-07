"""CPU-only scoring of CFC native-coordinate detections, COCO category 1.

Both public scorers consume the original COCO dictionary and an ordered list of
``image_id, bbox, score, category_id`` dictionaries. Bboxes are native xywh;
image population comes from COCO images, including images with no detections.
``read_predictions`` reads one structured NPY with the fixed dtype below. The
runner writes rows in inference order, preserving within-image score ties.

Only AP50 and its matching support are independently computed. The canonical
12-element ``stats`` array remains pycocotools-only. The command-line entry
writes the independent result from --gt JSON and --pred NPY to --out JSON.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np


PREDICTION_DTYPE = np.dtype([
    ("image_id", "<i8"),
    ("bbox", "<f4", (4,)),
    ("score", "<f4"),
    ("category_id", "<i8"),
])


def canonical_evaluate(coco_gt, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO()
    gt.dataset = copy.deepcopy(coco_gt)
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(copy.deepcopy(predictions))
    else:
        # loadRes indexes its first annotation, so an empty result needs its own COCO.
        dt = COCO()
        dt.dataset = {
            "images": copy.deepcopy(coco_gt["images"]),
            "categories": copy.deepcopy(coco_gt["categories"]),
            "annotations": [],
        }
        dt.createIndex()
    evaluation = COCOeval(gt, dt, "bbox")
    evaluation.params.catIds = [1]
    evaluation.params.imgIds = sorted(image["id"] for image in coco_gt["images"])
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()

    image_ids = evaluation.params.imgIds
    # One category; the default all-area range is the first block of image results.
    entries = [e for e in evaluation.evalImgs[:len(image_ids)] if e is not None]
    scores = np.array([s for e in entries for s in e["dtScores"]], dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    indices = np.array([i - 1 for e in entries for i in e["dtIds"]], dtype=np.int64)[order]
    matches = np.array([i for e in entries for i in e["dtMatches"][0]], dtype=np.int64)[order]
    ignored = np.array([i for e in entries for i in e["dtIgnore"][0]], dtype=bool)[order]
    return {
        "AP50": float(evaluation.stats[1]),
        "stats": evaluation.stats.tolist(),
        "category_id": 1,
        "raw_prediction_count": len(predictions),
        "gt_count": sum(len(e["gtIds"]) for e in entries),
        "gt_support": sum(int(np.count_nonzero(e["gtIgnore"] == 0)) for e in entries),
        "matched_gt": sum(int(np.count_nonzero(
            (e["gtMatches"][0] != 0) & (e["gtIgnore"] == 0)
        )) for e in entries),
        "detections": int(len(matches)),
        "true_positives": int(np.count_nonzero((matches != 0) & ~ignored)),
        "false_positives": int(np.count_nonzero((matches == 0) & ~ignored)),
        "ignored_detections": int(np.count_nonzero(ignored)),
        "ordering": {
            "image_ids": [int(i) for i in image_ids],
            "prediction_indices": indices.tolist(),
            "scores": scores[order].tolist(),
            "gt_match_ids": matches.tolist(),
            "ignored": ignored.tolist(),
        },
    }


def independent_ap50(coco_gt, predictions):
    """Recompute the category-1, all-area, maxDet-100 AP50 directly in NumPy FP64."""
    image_ids = sorted(image["id"] for image in coco_gt["images"])
    gts = {i: [] for i in image_ids}
    dts = {i: [] for i in image_ids}
    for annotation in coco_gt["annotations"]:
        if annotation["category_id"] == 1:
            gts[annotation["image_id"]].append(annotation)
    for index, prediction in enumerate(predictions):
        if prediction["category_id"] == 1:
            dts[prediction["image_id"]].append((index, prediction))

    gt_count = sum(len(rows) for rows in gts.values())
    gt_support = 0
    matched_gt = 0
    all_indices, all_scores, all_matches, all_ignored = [], [], [], []
    for image_id in image_ids:
        gt = gts[image_id]
        # Canonical bbox _prepare replaces raw ignore with iscrowd, then applies area.
        gt_ignore = np.array([
            bool(g["iscrowd"]) or g["area"] < 0 or g["area"] > 1e10 for g in gt
        ], dtype=bool)
        gt_order = np.argsort(gt_ignore, kind="mergesort")
        gt = [gt[i] for i in gt_order]
        gt_ignore = gt_ignore[gt_order]
        crowd = np.array([bool(g["iscrowd"]) for g in gt], dtype=bool)
        gt_matched = np.zeros(len(gt), dtype=bool)
        gt_support += int(np.count_nonzero(~gt_ignore))

        dt = dts[image_id]
        dt_order = np.argsort([-p["score"] for _, p in dt], kind="mergesort")[:100]
        dt = [dt[i] for i in dt_order]
        dt_boxes = np.array([p["bbox"] for _, p in dt], dtype=np.float64).reshape(-1, 4)
        gt_boxes = np.array([g["bbox"] for g in gt], dtype=np.float64).reshape(-1, 4)
        dt_area = dt_boxes[:, 2] * dt_boxes[:, 3]
        gt_box_area = gt_boxes[:, 2] * gt_boxes[:, 3]
        top_left = np.maximum(dt_boxes[:, None, :2], gt_boxes[None, :, :2])
        bottom_right = np.minimum(
            dt_boxes[:, None, :2] + dt_boxes[:, None, 2:],
            gt_boxes[None, :, :2] + gt_boxes[None, :, 2:],
        )
        overlap = np.maximum(bottom_right - top_left, 0.)
        intersection = overlap[:, :, 0] * overlap[:, :, 1]
        denominator = np.where(
            crowd[None, :], dt_area[:, None],
            dt_area[:, None] + gt_box_area[None, :] - intersection,
        )
        # COCO's box kernel returns zero without division when intersection is zero.
        ious = np.divide(intersection, denominator, out=np.zeros_like(intersection), where=intersection > 0)
        matches = np.zeros(len(dt), dtype=np.int64)
        ignored = np.zeros(len(dt), dtype=bool)
        for di in range(len(dt)):
            best_iou = .5
            match = -1
            for gi in range(len(gt)):
                if gt_matched[gi] and not crowd[gi]:
                    continue
                if match != -1 and not gt_ignore[match] and gt_ignore[gi]:
                    break
                if ious[di, gi] < best_iou:
                    continue
                best_iou = ious[di, gi]
                match = gi
            if match != -1:
                matches[di] = gt[match]["id"]
                ignored[di] = gt_ignore[match]
                gt_matched[match] = True
        ignored |= (matches == 0) & ((dt_area < 0) | (dt_area > 1e10))
        matched_gt += int(np.count_nonzero(gt_matched & ~gt_ignore))
        all_indices.extend(index for index, _ in dt)
        all_scores.extend(p["score"] for _, p in dt)
        all_matches.extend(matches.tolist())
        all_ignored.extend(ignored.tolist())

    scores = np.array(all_scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    matches = np.array(all_matches, dtype=np.int64)[order]
    ignored = np.array(all_ignored, dtype=bool)[order]
    true_positives = (matches != 0) & ~ignored
    false_positives = (matches == 0) & ~ignored
    ap50 = -1.
    if gt_support:
        tp = np.cumsum(true_positives).astype(np.float64)
        fp = np.cumsum(false_positives).astype(np.float64)
        recall = tp / gt_support
        precision = tp / (tp + fp + np.spacing(1))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        positions = np.searchsorted(recall, np.linspace(0., 1., 101), side="left")
        sampled_precision = np.zeros(101, dtype=np.float64)
        covered = positions < len(precision)
        sampled_precision[covered] = precision[positions[covered]]
        ap50 = float(sampled_precision.mean())
    return {
        "AP50": ap50,
        "category_id": 1,
        "raw_prediction_count": len(predictions),
        "gt_count": gt_count,
        "gt_support": gt_support,
        "matched_gt": matched_gt,
        "detections": int(len(matches)),
        "true_positives": int(true_positives.sum()),
        "false_positives": int(false_positives.sum()),
        "ignored_detections": int(ignored.sum()),
        "ordering": {
            "image_ids": image_ids,
            "prediction_indices": np.array(all_indices, dtype=np.int64)[order].tolist(),
            "scores": scores[order].tolist(),
            "gt_match_ids": matches.tolist(),
            "ignored": ignored.tolist(),
        },
    }


def read_predictions(path):
    """Load ordered FP32 native xywh/score rows; image/category IDs remain int64."""
    rows = np.load(path, allow_pickle=False, mmap_mode="r")
    if rows.dtype != PREDICTION_DTYPE:
        raise ValueError(f"Prediction dtype must be {PREDICTION_DTYPE}, got {rows.dtype}")
    return [
        {"image_id": int(row["image_id"]), "bbox": row["bbox"].tolist(),
         "score": float(row["score"]), "category_id": int(row["category_id"])}
        for row in rows
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = independent_ap50(json.loads(args.gt.read_text()), read_predictions(args.pred))
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
