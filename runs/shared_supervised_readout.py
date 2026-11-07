"""Fixed FSOD and NKSID train/validation readouts for the supervised reference."""

import copy
import hashlib
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from runs.cfc_independent_ap50 import PREDICTION_DTYPE, read_predictions
from sonar_encoder.training.readout import (
    confusion_matrix, grayscale, load_targets, metrics_from_confusion, state_digest, write_json,
)


STATES = ("T0", "E_U", "E_M", "E_UM")
CATEGORIES = list(range(1, 11))
COUNTS = ("raw_prediction_count", "gt_count", "gt_support", "matched_gt", "detections",
          "true_positives", "false_positives", "ignored_detections")


def canonical_evaluate(coco_gt, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO()
    gt.dataset = copy.deepcopy(coco_gt)
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(copy.deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = {"images": copy.deepcopy(coco_gt["images"]),
                      "categories": copy.deepcopy(coco_gt["categories"]), "annotations": []}
        dt.createIndex()
    evaluation = COCOeval(gt, dt, "bbox")
    evaluation.params.catIds = CATEGORIES.copy()
    evaluation.params.imgIds = sorted(image["id"] for image in coco_gt["images"])
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    image_ids = evaluation.params.imgIds
    per_class = {}
    for category_index, category_id in enumerate(CATEGORIES):
        start = category_index * len(evaluation.params.areaRng) * len(image_ids)
        entries = [e for e in evaluation.evalImgs[start:start + len(image_ids)] if e is not None]
        scores = np.array([s for e in entries for s in e["dtScores"]], dtype=np.float64)
        order = np.argsort(-scores, kind="mergesort")
        indices = np.array([i - 1 for e in entries for i in e["dtIds"]], dtype=np.int64)[order]
        matches = np.array([i for e in entries for i in e["dtMatches"][0]], dtype=np.int64)[order]
        ignored = np.array([i for e in entries for i in e["dtIgnore"][0]], dtype=bool)[order]
        precision = evaluation.eval["precision"][0, :, category_index, 0, 2]
        per_class[str(category_id)] = {
            "AP50": float(precision.mean()), "category_id": category_id,
            "raw_prediction_count": sum(p["category_id"] == category_id for p in predictions),
            "gt_count": sum(len(e["gtIds"]) for e in entries),
            "gt_support": sum(int(np.count_nonzero(e["gtIgnore"] == 0)) for e in entries),
            "matched_gt": sum(int(np.count_nonzero(
                (e["gtMatches"][0] != 0) & (e["gtIgnore"] == 0))) for e in entries),
            "detections": len(matches),
            "true_positives": int(np.count_nonzero((matches != 0) & ~ignored)),
            "false_positives": int(np.count_nonzero((matches == 0) & ~ignored)),
            "ignored_detections": int(np.count_nonzero(ignored)),
            "ordering": {"image_ids": image_ids, "prediction_indices": indices.tolist(),
                         "scores": scores[order].tolist(), "gt_match_ids": matches.tolist(),
                         "ignored": ignored.tolist()},
        }
    return {"AP50": float(evaluation.stats[1]), "stats": evaluation.stats.tolist(),
            "category_ids": CATEGORIES.copy(), "per_class": per_class,
            **{key: sum(v[key] for v in per_class.values()) for key in COUNTS}}


def independent_ap50(coco_gt, predictions):
    """NumPy FP64 per-class COCO 101-point AP50, all area, maxDet100."""
    image_ids = sorted(image["id"] for image in coco_gt["images"])
    per_class = {}
    for category_id in CATEGORIES:
        gts, dts = {i: [] for i in image_ids}, {i: [] for i in image_ids}
        for annotation in coco_gt["annotations"]:
            if annotation["category_id"] == category_id:
                gts[annotation["image_id"]].append(annotation)
        for index, prediction in enumerate(predictions):
            if prediction["category_id"] == category_id:
                dts[prediction["image_id"]].append((index, prediction))
        gt_support, matched_gt = 0, 0
        all_indices, all_scores, all_matches, all_ignored = [], [], [], []
        for image_id in image_ids:
            gt = gts[image_id]
            # The canonical bbox evaluator replaces raw ignore with iscrowd.
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
            dt_area, gt_area = dt_boxes[:, 2] * dt_boxes[:, 3], gt_boxes[:, 2] * gt_boxes[:, 3]
            top_left = np.maximum(dt_boxes[:, None, :2], gt_boxes[None, :, :2])
            bottom_right = np.minimum(dt_boxes[:, None, :2] + dt_boxes[:, None, 2:],
                                      gt_boxes[None, :, :2] + gt_boxes[None, :, 2:])
            overlap = np.maximum(bottom_right - top_left, 0.)
            intersection = overlap[:, :, 0] * overlap[:, :, 1]
            denominator = np.where(crowd[None, :], dt_area[:, None],
                                   dt_area[:, None] + gt_area[None, :] - intersection)
            ious = np.divide(intersection, denominator, out=np.zeros_like(intersection), where=intersection > 0)
            matches, ignored = np.zeros(len(dt), dtype=np.int64), np.zeros(len(dt), dtype=bool)
            for di in range(len(dt)):
                best_iou, match = .5, -1
                for gi in range(len(gt)):
                    if gt_matched[gi] and not crowd[gi]:
                        continue
                    if match != -1 and not gt_ignore[match] and gt_ignore[gi]:
                        break
                    if ious[di, gi] < best_iou:
                        continue
                    best_iou, match = ious[di, gi], gi
                if match != -1:
                    matches[di], ignored[di] = gt[match]["id"], gt_ignore[match]
                    gt_matched[match] = True
            ignored |= (matches == 0) & ((dt_area < 0) | (dt_area > 1e10))
            matched_gt += int(np.count_nonzero(gt_matched & ~gt_ignore))
            all_indices.extend(index for index, _ in dt)
            all_scores.extend(p["score"] for _, p in dt)
            all_matches.extend(matches.tolist())
            all_ignored.extend(ignored.tolist())
        scores = np.array(all_scores, dtype=np.float64)
        order = np.argsort(-scores, kind="mergesort")
        matches, ignored = np.array(all_matches, dtype=np.int64)[order], np.array(all_ignored, dtype=bool)[order]
        true_positives, false_positives = (matches != 0) & ~ignored, (matches == 0) & ~ignored
        if gt_support == 0:
            raise ValueError(f"FSOD fixed macro category {category_id} has no positive support")
        tp, fp = np.cumsum(true_positives).astype(np.float64), np.cumsum(false_positives).astype(np.float64)
        recall, precision = tp / gt_support, tp / (tp + fp + np.spacing(1))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        positions = np.searchsorted(recall, np.linspace(0., 1., 101), side="left")
        sampled = np.zeros(101, dtype=np.float64)
        covered = positions < len(precision)
        sampled[covered] = precision[positions[covered]]
        per_class[str(category_id)] = {
            "AP50": float(sampled.mean()), "category_id": category_id,
            "raw_prediction_count": sum(len(v) for v in dts.values()),
            "gt_count": sum(len(v) for v in gts.values()), "gt_support": gt_support,
            "matched_gt": matched_gt, "detections": len(matches),
            "true_positives": int(true_positives.sum()), "false_positives": int(false_positives.sum()),
            "ignored_detections": int(ignored.sum()),
            "ordering": {"image_ids": image_ids,
                         "prediction_indices": np.array(all_indices, dtype=np.int64)[order].tolist(),
                         "scores": scores[order].tolist(), "gt_match_ids": matches.tolist(),
                         "ignored": ignored.tolist()},
        }
    return {"AP50": sum(v["AP50"] for v in per_class.values()) / 10,
            "category_ids": CATEGORIES.copy(), "per_class": per_class,
            **{key: sum(v[key] for v in per_class.values()) for key in COUNTS}}


def make_head(task):
    if task == "nksid":
        return nn.Linear(384, 8)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from runs.cfc_frozen_detector import FrozenDetectorHead

    head = FrozenDetectorHead()
    head.roi_heads.box_predictor = FastRCNNPredictor(1024, 11)
    return head


def make_fsod_head():
    return make_head("fsod")


def prediction_rows(detections, rows):
    from runs.cfc_frozen_detector import scale_boxes

    pieces = []
    for detection, row in zip(detections, rows):
        boxes = scale_boxes(detection["boxes"], (224, 224), (row["height"], row["width"])).cpu().numpy()
        boxes[:, 2:] -= boxes[:, :2]
        values = np.empty(len(boxes), dtype=PREDICTION_DTYPE)
        values["image_id"], values["bbox"] = row["image_id"], boxes
        values["score"], values["category_id"] = detection["scores"].cpu().numpy(), detection["labels"].cpu().numpy()
        assert np.isfinite(boxes).all() and np.isfinite(values["score"]).all()
        assert ((values["category_id"] >= 1) & (values["category_id"] <= 10)).all()
        pieces.append(values)
    return np.concatenate(pieces)


def prediction_list(array):
    return [{"image_id": int(row["image_id"]), "bbox": row["bbox"].tolist(),
             "score": float(row["score"]), "category_id": int(row["category_id"])} for row in array]


@torch.no_grad()
def targetextract(state, task):
    from runs import shared_supervised_reference as core
    from sonar_encoder.training.model import prepare_images

    core.configure_runtime(task)
    protocol = core.load_protocol()
    target = protocol["targets"][task]
    folder = core.RUN / "readouts" / state / task
    folder.mkdir(parents=True)
    encoder = core.load_frozen_encoder(state).to("cuda:0").eval().requires_grad_(False)
    before = state_digest(encoder.state_dict())
    caches = {}
    begun = time.monotonic()
    for split in ("train", "validation"):
        rows = target["rows"][split]
        path = folder / f"{split}.npy"
        temporary = path.with_suffix(".npy.partial")
        shape = (len(rows), 384, 16, 16) if task == "fsod" else (len(rows), 384)
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        batch_count = 0
        for offset in range(0, len(rows), 16):
            batch = rows[offset:offset + 16]
            samples = []
            for row in batch:
                assert core.digest(Path(row["image"])) == row["image_sha256"]
                values = torch.from_numpy(grayscale(row["image"])).float().unsqueeze(0) / 255
                samples.append({"intensity": values, "valid": torch.ones(values.shape[-2:], dtype=torch.bool)})
            tokens = encoder(prepare_images(samples, target["normalization"], None).to("cuda:0"))["patch_tokens"]
            output = tokens.transpose(1, 2).reshape(len(batch), 384, 16, 16) if task == "fsod" else tokens.mean(dim=1)
            assert output.dtype == torch.float32 and tuple(output.shape) == (len(batch), *shape[1:])
            if not torch.isfinite(output).all().item():
                raise FloatingPointError("Nonfinite frozen target features")
            array[offset:offset + len(batch)] = output.cpu().numpy()
            batch_count += 1
            core.resources()
            if batch_count % 20 == 0:
                print(json.dumps({"state": state, "task": task, "phase": "extract", "split": split,
                                  "samples": offset + len(batch), "seconds": time.monotonic() - begun}), flush=True)
        array.flush()
        del array
        temporary.replace(path)
        caches[split] = {"path": str(path), "sha256": core.digest(path), "shape": list(shape),
                         "row_ids": [row["image_id" if task == "fsod" else "row_id"] for row in rows],
                         "samples": len(rows), "batches": batch_count}
    after = state_digest(encoder.state_dict())
    assert before == after
    write_json(folder / "extraction.json", {"state": state, "task": task, "encoder_before_sha256": before,
               "encoder_after_sha256": after, "caches": caches, "feature_dtype": "float32",
               "mean_reduction": "GPU FP32 before CPU copy" if task == "nksid" else None,
               "protocol_sha256": core.digest(core.RUN / "protocol.json"), "resources": core.resources()})


@torch.no_grad()
def evaluate(head, arrays, rows, task, coco):
    from runs import shared_supervised_reference as core

    head.eval()
    outputs = []
    batch_size = 8 if task == "fsod" else 64
    for offset in range(0, len(rows), batch_size):
        features = torch.from_numpy(np.array(arrays[offset:offset + batch_size], copy=True)).to("cuda:0")
        output = head(features)
        outputs.append(prediction_rows(output, rows[offset:offset + batch_size]) if task == "fsod" else output.cpu())
        core.resources()
    if task == "fsod":
        predictions = np.concatenate(outputs)
        return canonical_evaluate(coco, prediction_list(predictions)), predictions
    logits = torch.cat(outputs)
    predictions = logits.argmax(dim=1)
    metrics = metrics_from_confusion(confusion_matrix(load_targets(rows, "nksid"), predictions, 8), "nksid")
    return metrics, {"logits": logits, "predictions": predictions, "row_ids": [r["row_id"] for r in rows]}


def fit_head(state, task):
    from runs import shared_supervised_reference as core
    from runs.cfc_frozen_detector import scale_boxes

    core.configure_runtime(task)
    protocol = core.load_protocol()
    target = protocol["targets"][task]
    folder = core.RUN / "readouts" / state / task
    extracted = json.loads((folder / "extraction.json").read_text())
    protocol_sha = core.digest(core.RUN / "protocol.json")
    assert (extracted["state"], extracted["task"]) == (state, task)
    assert extracted["protocol_sha256"] == protocol_sha
    assert extracted["encoder_before_sha256"] == extracted["encoder_after_sha256"]
    rows, arrays = target["rows"], {}
    for split in ("train", "validation"):
        cache = extracted["caches"][split]
        path = Path(cache["path"])
        assert core.digest(path) == cache["sha256"]
        assert cache["row_ids"] == [r["image_id" if task == "fsod" else "row_id"] for r in rows[split]]
        arrays[split] = np.load(path, mmap_mode="r", allow_pickle=False)
        assert list(arrays[split].shape) == cache["shape"] and arrays[split].dtype == np.float32
    if task == "fsod":
        coco = json.loads(Path(target["validation_coco"]).read_text())
        assert core.digest(Path(target["validation_coco"])) == protocol["inputs_sha256"][target["validation_coco"]]
        targets = []
        for row in rows["train"]:
            annotations = row["annotations"]
            boxes = torch.tensor([a["bbox"] for a in annotations], dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            targets.append({"boxes": scale_boxes(boxes, (row["height"], row["width"]), (224, 224)),
                            "labels": torch.tensor([a["category_id"] for a in annotations], dtype=torch.int64)})
        epochs, batch_size, updates_required = 60, 8, 17700
    else:
        coco = None
        targets = load_targets(rows["train"], "nksid")
        epochs, batch_size, updates_required = 100, 64, 2700
    torch.manual_seed(0)
    head = make_head(task)
    initial = state_digest(head.state_dict())
    assert initial == target["initial_head_sha256"]
    head.to("cuda:0")
    optimizer = (torch.optim.SGD(head.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
                 if task == "fsod" else torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001,
                                                         betas=(.9, .999), eps=1e-8))
    generator = torch.Generator().manual_seed(0)
    order_hash = hashlib.sha256()
    metric = "AP50" if task == "fsod" else "macro_f1"
    extension = ".npy" if task == "fsod" else ".pt"
    best_score, best_epoch, updates = -1., 0, 0
    begun = time.monotonic()
    selector = {"best_score": -1., "best_epoch": 0}
    stream = {"epoch": 0, "cursor": 0, "order": torch.empty(0, dtype=torch.int64), "updates": 0}

    def save_boundary(failure):
        payload = {"head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "stream": stream, "selector": selector, "private_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(0),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "head_state_sha256": state_digest(head.state_dict()), "protocol_sha256": protocol_sha,
            "failure_not_safe_for_automatic_resume": failure is not None}
        if failure is not None:
            payload["failure"] = failure
            payload["gradients"] = {name: parameter.grad for name, parameter in head.named_parameters()}
        core.atomic_torch(folder / "last_head.pt", payload)

    def interrupted(signum, frame):
        raise InterruptedError("Target head received SIGTERM; stop without automatic resume")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    event = {"stage": "initial_boundary_save", "epoch": 0}
    try:
        save_boundary(None)
        with (folder / "metrics.jsonl").open("w") as log:
            for epoch in range(1, epochs + 1):
                event = {"stage": "epoch_start", "epoch": epoch}
                head.train()
                order = torch.randperm(len(rows["train"]), generator=generator)
                order_hash.update(order.numpy().tobytes())
                stream.update(epoch=epoch, cursor=0, order=order,
                              epoch_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest())
                lr = (.005 if epoch <= 40 else (.0005 if epoch <= 55 else .00005)) if task == "fsod" else .001
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss_sum = 0.
                for indices in order.split(batch_size):
                    event = {"stage": "batch_forward", "epoch": epoch, "indices": indices.tolist(), "update_committed": False}
                    features = torch.from_numpy(np.array(arrays["train"][indices.numpy()], copy=True)).to("cuda:0")
                    optimizer.zero_grad(set_to_none=True)
                    if task == "fsod":
                        current_targets = [{key: value.to("cuda:0") for key, value in targets[i].items()} for i in indices.tolist()]
                        loss = sum(head(features, current_targets).values())
                    else:
                        loss = F.cross_entropy(head(features), targets[indices].to("cuda:0"))
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Nonfinite target head loss")
                    event["stage"] = "backward"
                    loss.backward()
                    if not all(torch.isfinite(p.grad).all().item() for p in head.parameters()):
                        raise FloatingPointError("Nonfinite target head gradient")
                    event["stage"] = "optimizer_step"
                    optimizer.step()
                    if not all(torch.isfinite(p).all().item() for p in head.parameters()):
                        raise FloatingPointError("Nonfinite target head parameter")
                    updates += 1
                    loss_sum += loss.item() * len(indices)
                    stream.update(cursor=stream["cursor"] + len(indices), updates=updates, loss_sum=loss_sum)
                    event.update(stage="accepted_update", update_committed=True)
                    core.resources()
                    if updates % 100 == 0:
                        print(json.dumps({"state": state, "task": task, "phase": "head", "epoch": epoch,
                                          "updates": updates, "seconds": time.monotonic() - begun}), flush=True)
                entry = {"epoch": epoch, "updates": updates, "LR": lr, "train_loss": loss_sum / len(rows["train"])}
                if task == "nksid" or epoch % 5 == 0:
                    event = {"stage": "validation", "epoch": epoch}
                    metrics, predictions = evaluate(head, arrays["validation"], rows["validation"], task, coco)
                    entry["validation"] = metrics
                    if metrics[metric] > best_score:
                        best_score, best_epoch, best_metrics = metrics[metric], epoch, metrics
                        best_head_state = state_digest(head.state_dict())
                        selector.update(best_score=best_score, best_epoch=best_epoch, best_metrics=best_metrics,
                                        best_head_state_sha256=best_head_state)
                        event["stage"] = "best_artifact_write"
                        core.atomic_torch(folder / "best_head.pt", {"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                            "epoch": epoch, "validation": metrics, "head_state_sha256": best_head_state,
                            "initial_head_sha256": initial, "encoder_state_sha256": extracted["encoder_after_sha256"],
                            "protocol_sha256": protocol_sha})
                        best_path = folder / ("best_validation" + extension)
                        if task == "fsod":
                            core.atomic_predictions(best_path, predictions)
                        else:
                            core.atomic_torch(best_path, {**predictions, "epoch": epoch, "protocol_sha256": protocol_sha})
                    if epoch == epochs:
                        event["stage"] = "terminal_evaluation_output_write"
                        terminal_metrics, terminal_head_state = metrics, state_digest(head.state_dict())
                        terminal_path = best_path if best_epoch == epochs else folder / ("terminal_validation" + extension)
                        if best_epoch != epochs:
                            if task == "fsod":
                                core.atomic_predictions(terminal_path, predictions)
                            else:
                                core.atomic_torch(terminal_path, {**predictions, "epoch": epoch, "protocol_sha256": protocol_sha})
                    del predictions
                log.write(json.dumps(entry, allow_nan=False) + "\n")
                log.flush()
                event = {"stage": "epoch_boundary_save", "epoch": epoch}
                save_boundary(None)
        event = {"stage": "final_report", "epoch": epochs}
        assert updates == updates_required
        write_json(folder / "report.json", {"state": state, "task": task, "epochs": epochs, "updates": updates,
            "initial_head_sha256": initial, "private_order_sha256": order_hash.hexdigest(),
            "best_epoch": best_epoch, "best": best_metrics, "terminal": terminal_metrics,
            "best_head_state_sha256": best_head_state, "terminal_head_state_sha256": terminal_head_state,
            "best_head_file_sha256": core.digest(folder / "best_head.pt"),
            "best_prediction": str(best_path), "terminal_prediction": str(terminal_path),
            "best_prediction_sha256": core.digest(best_path), "terminal_prediction_sha256": core.digest(terminal_path),
            "logical_endpoints": 2, "unique_prediction_artifacts": 1 if terminal_path == best_path else 2,
            "terminal_weight_retained": best_epoch == epochs, "protocol_sha256": protocol_sha,
            "resources": core.resources()})
    except Exception as error:
        failure = {"state": state, "task": task, "event": event,
                   "exception_type": type(error).__name__, "message": str(error),
                   "actual_epoch": stream["epoch"], "actual_cursor": stream["cursor"],
                   "completed_updates": stream["updates"],
                   "failure_not_safe_for_automatic_resume": True, "checkpoint_saved": False,
                   "protocol_sha256": protocol_sha}
        try:
            failure["checkpoint_saved"] = True
            save_boundary(failure)
            failure["checkpoint_sha256"] = core.digest(folder / "last_head.pt")
        except Exception as save_error:
            failure["checkpoint_saved"] = False
            failure["save_error"] = {"exception_type": type(save_error).__name__, "message": str(save_error)}
        try:
            write_json(folder / "failure.json", failure)
        except Exception as record_error:
            print(json.dumps({**failure, "failure_record_error": str(record_error)}), flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)



def independent_nksid(target, payload):
    logits = payload["logits"].numpy()
    assert logits.shape == (len(target), 8) and logits.dtype == np.float32 and np.isfinite(logits).all()
    prediction = logits.argmax(axis=1)
    assert np.array_equal(prediction, payload["predictions"].numpy())
    matrix = np.bincount(target * 8 + prediction, minlength=64).reshape(8, 8)
    tp = matrix.diagonal().astype(np.float64)
    actual, guessed = matrix.sum(1), matrix.sum(0)
    return {"confusion_matrix": matrix.tolist(), "support": actual.tolist(),
            "macro_f1": float((2 * tp / np.maximum(actual + guessed, 1)).mean()),
            "accuracy": float(tp.sum() / actual.sum()),
            "per_class_f1": (2 * tp / np.maximum(actual + guessed, 1)).tolist(),
            "per_class_precision": (tp / np.maximum(guessed, 1)).tolist(),
            "per_class_recall": (tp / np.maximum(actual, 1)).tolist()}


def score_readout(state, task):
    from runs import shared_supervised_reference as core

    protocol = core.load_protocol()
    protocol_sha = core.digest(core.RUN / "protocol.json")
    checks = []
    target = protocol["targets"][task]
    folder = core.RUN / "readouts" / state / task
    report = json.loads((folder / "report.json").read_text())
    assert (report["state"], report["task"]) == (state, task)
    assert report["protocol_sha256"] == protocol_sha
    assert report["initial_head_sha256"] == target["initial_head_sha256"]
    extracted = json.loads((folder / "extraction.json").read_text())
    assert (extracted["state"], extracted["task"]) == (state, task)
    assert extracted["protocol_sha256"] == protocol_sha
    assert extracted["encoder_before_sha256"] == extracted["encoder_after_sha256"]
    for cache in extracted["caches"].values():
        assert core.digest(Path(cache["path"])) == cache["sha256"]
    assert core.digest(folder / "best_head.pt") == report["best_head_file_sha256"]
    best = torch.load(folder / "best_head.pt", map_location="cpu", weights_only=True)
    assert best["protocol_sha256"] == protocol_sha and best["epoch"] == report["best_epoch"]
    assert state_digest(best["head"]) == best["head_state_sha256"] == report["best_head_state_sha256"]
    assert best["encoder_state_sha256"] == extracted["encoder_after_sha256"]
    history = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    expected_epochs, expected_updates = (60, 17700) if task == "fsod" else (100, 2700)
    assert len(history) == expected_epochs and history[-1]["updates"] == expected_updates
    evaluated = [entry for entry in history if "validation" in entry]
    assert [e["epoch"] for e in evaluated] == (list(range(5, 61, 5)) if task == "fsod" else list(range(1, 101)))
    metric = "AP50" if task == "fsod" else "macro_f1"
    earliest = max(evaluated, key=lambda entry: entry["validation"][metric])
    assert earliest["epoch"] == report["best_epoch"] and earliest["validation"] == report["best"] == best["validation"]
    assert history[-1]["validation"] == report["terminal"]
    task_result = {"best_epoch": report["best_epoch"], "terminal_epoch": expected_epochs,
                   "logical_endpoints": 2, "unique_prediction_artifacts": report["unique_prediction_artifacts"]}
    for when in ("terminal", "best"):
        path = Path(report[when + "_prediction"])
        assert core.digest(path) == report[when + "_prediction_sha256"]
        expected = report[when]
        if task == "fsod":
            coco_path = Path(target["validation_coco"])
            assert core.digest(coco_path) == protocol["inputs_sha256"][str(coco_path)]
            independent = independent_ap50(json.loads(coco_path.read_text()), read_predictions(path))
            assert independent["category_ids"] == expected["category_ids"] == CATEGORIES
            assert abs(independent["AP50"] - expected["AP50"]) <= 1e-12
            for key in COUNTS:
                assert independent[key] == expected[key]
            for category in map(str, CATEGORIES):
                actual_class, expected_class = independent["per_class"][category], expected["per_class"][category]
                assert abs(actual_class["AP50"] - expected_class["AP50"]) <= 1e-12
                for key in (*COUNTS, "category_id", "ordering"):
                    assert actual_class[key] == expected_class[key]
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert payload["protocol_sha256"] == protocol_sha
            assert payload["epoch"] == (expected_epochs if when == "terminal" else report["best_epoch"])
            assert payload["row_ids"] == [r["row_id"] for r in target["rows"]["validation"]]
            truth = np.array([r["classID"] for r in target["rows"]["validation"]], dtype=np.int64)
            independent = independent_nksid(truth, payload)
            for key in ("confusion_matrix", "support"):
                assert independent[key] == expected[key]
            for key in ("macro_f1", "accuracy", "per_class_f1", "per_class_precision", "per_class_recall"):
                assert np.max(np.abs(np.asarray(independent[key]) - np.asarray(expected[key]))) <= 1e-12
        task_result[when] = independent
        checks.append({"state": state, "task": task, "endpoint": when,
                       "prediction_path": str(path), "prediction_sha256": core.digest(path),
                       "status": "PASS_INDEPENDENT_NUMPY_COUNTS_AND_METRICS"})
        core.resources()
    last = folder / "last_head.pt"
    last_sha = core.digest(last)
    saved_last = torch.load(last, map_location="cpu", weights_only=False)
    assert saved_last["protocol_sha256"] == protocol_sha and saved_last["stream"]["updates"] == expected_updates
    assert saved_last["head_state_sha256"] == report["terminal_head_state_sha256"]
    assert state_digest(saved_last["head"]) == report["terminal_head_state_sha256"]
    del saved_last
    task_result["removed_rolling_head_sha256"] = last_sha
    receipt = {"state": state, "task": task, "result": task_result, "checks": checks,
               "status": "PASS_INDEPENDENT_NUMPY_COUNTS_AND_METRICS",
               "protocol_sha256": protocol_sha, "report_sha256": core.digest(folder / "report.json"),
               "extraction_sha256": core.digest(folder / "extraction.json"),
               "best_head_file_sha256": report["best_head_file_sha256"],
               "rolling_removal_after_this_receipt": str(last), "resources": core.resources()}
    write_json(folder / "score_receipt.json", receipt)
    last.unlink()
    return receipt


def score_and_summarize():
    from runs import shared_supervised_reference as core

    protocol_sha = core.digest(core.RUN / "protocol.json")
    results, checks, unique_predictions = {}, [], set()
    for state in STATES:
        results[state] = {}
        for task in ("fsod", "nksid"):
            folder = core.RUN / "readouts" / state / task
            receipt = json.loads((folder / "score_receipt.json").read_text())
            assert (receipt["state"], receipt["task"]) == (state, task)
            assert receipt["status"] == "PASS_INDEPENDENT_NUMPY_COUNTS_AND_METRICS"
            assert receipt["protocol_sha256"] == protocol_sha
            assert core.digest(folder / "report.json") == receipt["report_sha256"]
            assert core.digest(folder / "extraction.json") == receipt["extraction_sha256"]
            assert core.digest(folder / "best_head.pt") == receipt["best_head_file_sha256"]
            assert not Path(receipt["rolling_removal_after_this_receipt"]).exists()
            for check in receipt["checks"]:
                path = Path(check["prediction_path"])
                assert core.digest(path) == check["prediction_sha256"]
                unique_predictions.add(str(path))
            checks.extend(receipt["checks"])
            results[state][task] = receipt["result"]
    comparisons, nondominated = {}, {}
    for when in ("terminal", "best"):
        vectors = {state: [results[state]["fsod"][when]["AP50"], results[state]["nksid"][when]["macro_f1"]]
                   for state in STATES}
        comparisons[when] = {f"E_UM-{base}": [a - b for a, b in zip(vectors["E_UM"], vectors[base])]
                             for base in ("E_U", "E_M", "T0")}
        comparisons[when].update({f"{state}-T0": [a - b for a, b in zip(vectors[state], vectors["T0"])]
                                  for state in ("E_U", "E_M")})
        nondominated[when] = [state for state in STATES if not any(
            all(a >= b for a, b in zip(vectors[other], vectors[state]))
            and any(a > b for a, b in zip(vectors[other], vectors[state]))
            for other in STATES if other != state)]
    assert len(checks) == 16
    summary = {"states": results, "comparison_order": ["FSOD_AP50", "NKSID_macro_f1"],
               "primary": "fixed terminal FSOD epoch60 / NKSID epoch100", "auxiliary": "validation-selected best",
               "differences": comparisons, "nondominated_states": nondominated,
               "interpretation": "Fixed two-task ordinary reference; no aggregate score or significance claim.",
               "logical_endpoints": len(checks), "unique_prediction_artifacts": len(unique_predictions),
               "checks": checks, "protocol_sha256": protocol_sha, "resources": core.resources()}
    write_json(core.RUN / "target_results.json", summary)
    return summary
