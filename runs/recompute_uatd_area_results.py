"""Independently score the two frozen UATD development readouts after completion."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "4"
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
CONTRACT = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))/"uatd_task_contract_v1"
AP_RECALL_TOLERANCE = 1e-12
F1_TOLERANCE = 1e-6


def digest(path):
    hashed = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            hashed.update(block)
    return hashed.hexdigest()


def average_precision(scores, correct, support):
    ordered = correct[np.argsort(-scores, kind="stable")].astype(np.float64)
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1, dtype=np.float64)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(envelope * ordered) / support)


def global_metrics(scores, labels):
    truth = np.zeros((len(labels), 10), dtype=bool)
    for index, row_labels in enumerate(labels):
        truth[index, row_labels] = True
    support = truth.sum(0)
    ap = [average_precision(scores[:, k], truth[:, k], support[k]) for k in range(10)]
    predicted = scores >= .5
    correct = (predicted & truth).sum(0)
    f1 = 2 * correct / (predicted.sum(0) + support)
    return {"map": float(np.mean(ap)), "per_class_ap": ap,
            "macro_f1_at_0_5": float(np.mean(f1)),
            "micro_f1_at_0_5": float(2 * correct.sum() / (predicted.sum() + support.sum())),
            "per_class_support": support.tolist()}


def dense_metrics(predictions, boxes, labels):
    per_class, supports, recalls = [], [], []
    for label in range(10):
        truth = [b[y == label] for b, y in zip(boxes, labels)]
        matched = [np.zeros(len(b), dtype=bool) for b in truth]
        entries = [(float(score), image, box)
                   for image, prediction in enumerate(predictions)
                   for score, box in zip(prediction["scores"][prediction["labels"] == label],
                                         prediction["boxes"][prediction["labels"] == label])]
        entries.sort(key=lambda item: -item[0])
        correct = np.zeros(len(entries), dtype=bool)
        for index, (_, image, box) in enumerate(entries):
            available = np.flatnonzero(~matched[image])
            if len(available):
                candidates = truth[image][available]
                overlap_sides = np.maximum(np.minimum(box[2:], candidates[:, 2:])
                                           - np.maximum(box[:2], candidates[:, :2]), np.float32(0))
                intersection = np.prod(overlap_sides, axis=1, dtype=np.float32)
                area = np.prod(box[2:] - box[:2], dtype=np.float32)
                areas = np.prod(candidates[:, 2:] - candidates[:, :2], axis=1, dtype=np.float32)
                iou = intersection / (area + areas - intersection)
                best = int(np.argmax(iou))
                if iou[best] >= .5:
                    matched[image][available[best]] = True
                    correct[index] = True
        support = sum(len(b) for b in truth)
        supports.append(support)
        recalls.append(float(correct.sum() / support))
        scores = np.array([entry[0] for entry in entries], dtype=np.float32)
        per_class.append(average_precision(scores, correct, support))
    return {"map50": float(np.mean(per_class)), "per_class_ap50": per_class,
            "per_class_gt_count": supports, "per_class_recall_at_0_05": recalls,
            "macro_recall_at_0_05": float(np.mean(recalls)),
            "predictions": sum(len(p["scores"]) for p in predictions)}


def compare_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    comparisons = {}
    for key in actual:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        assert a.shape == b.shape
        tolerance = F1_TOLERANCE if "f1" in key else AP_RECALL_TOLERANCE
        if key in ("per_class_support", "per_class_gt_count", "predictions"):
            tolerance = 0
        error = float(np.max(np.abs(a - b)))
        comparisons[key] = {"max_abs_error": error, "tolerance": tolerance, "pass": error <= tolerance}
    return comparisons


def main(run):
    started = time.monotonic()
    run = run.resolve()
    bindings = {}

    def bind(path, expected):
        actual = digest(path)
        assert actual == expected, f"Changed input: {path}"
        bindings[str(path)] = actual

    protocol_path, cache_path = run / "protocol.json", run / "cache_manifest.json"
    protocol = json.loads(protocol_path.read_text())
    extracted = json.loads(cache_path.read_text())
    protocol_sha, cache_sha = digest(protocol_path), digest(cache_path)
    bindings.update({str(protocol_path): protocol_sha, str(cache_path): cache_sha})
    assert extracted["status"] == "COMPLETE_FULL_IMAGE_FROZEN_FEATURES"
    assert extracted["protocol_sha256"] == protocol_sha
    assert extracted["model_state_sha256_before"] == extracted["model_state_sha256_after"]
    for name, expected in protocol["inputs_sha256"].items():
        bind(Path(name), expected)
    dino = Path(protocol["encoder"]["code_root"])
    for path in sorted((run / "code").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run / "code")
        original = (dino / relative.relative_to("dinov2") if relative.parts[0] == "dinov2"
                    else Path("/tmp/pmca_uatd_contract_astra_guidance.txt") if relative.name == "astra_guidance.txt"
                    else PROJECT / relative)
        bind(path, protocol["inputs_sha256"][str(original)])
    prepared_path = Path(protocol["prepared_manifest"])
    prepared = json.loads(prepared_path.read_text())
    assert protocol["normalization"]["source_manifest_sha256"] == bindings[str(prepared_path)]
    bind(Path(prepared["protocol_path"]), prepared["protocol_sha256"])
    raw_path = Path(prepared["source_manifest"])
    bind(raw_path, prepared["source_manifest_sha256"])
    raw = {row["id"]: row for row in map(json.loads, raw_path.read_text().splitlines())}
    all_ids = {row["row_id"] for split in ("train", "validation") for row in prepared[split]}
    assert all_ids == raw.keys() - set(prepared["excluded_row_ids"])
    assert len(all_ids) == 7556 and len(prepared["train"]) == 6051 and len(prepared["validation"]) == 1505
    for split in ("train", "validation"):
        rows = prepared[split]
        assert protocol["split"][split] == {"count": len(rows), "row_ids": [row["row_id"] for row in rows]}
        assert extracted["area_token_counts_by_row"][split] == [16 * math.ceil(16 * row["range_m"] / 25) for row in rows]
        for row in rows:
            source = raw[row["row_id"]]
            height, width = source["shape"][:2]
            original_boxes = np.array([obj["bbox_xyxy"] for obj in source["objects"]], dtype=np.float64)
            assert row["labels"] == [prepared["class_names"].index(obj["name"]) for obj in source["objects"]]
            assert np.array_equal(np.array(row["boxes"]), original_boxes / [width, height, width, height])
            assert np.array_equal(np.array(row["original_boxes_xyxy"]), original_boxes)
            assert row["shape"] == source["shape"] and row["image_sha256"] == source["image_sha256"]
            assert row["pixel_sha256"] == source["pixel_sha256"]
        for arm in ("aa", "area"):
            path = run / f"{arm}_{split}.npy"
            bind(path, extracted["cache_files_sha256"][path.name])
            with path.open("rb") as handle:
                assert np.lib.format.read_magic(handle) == (1, 0)
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            assert shape == (len(rows), 384, 16, 16) and dtype == np.float32 and not fortran
    assert extracted["new_network_forwards"] == 2 * 7556
    settings = protocol["heads"]
    assert settings["epochs"] == 100 and settings["batch_size"] == 32 and settings["evaluate_every_epochs"] == 5
    per_epoch = math.ceil(len(prepared["train"]) / settings["batch_size"])
    assert per_epoch == 190
    validation_ids = [row["row_id"] for row in prepared["validation"]]
    boxes = [np.asarray(row["boxes"], dtype=np.float32) for row in prepared["validation"]]
    labels = [np.asarray(row["labels"], dtype=np.int64) for row in prepared["validation"]]
    arms, reports, checks, dense_size_bounds = {}, {}, {}, {}
    for arm in ("aa", "area"):
        output = run / f"heads_{arm}"
        report_path, log_path = output / "report.json", output / "metrics.jsonl"
        report = json.loads(report_path.read_text())
        log = list(map(json.loads, log_path.read_text().splitlines()))
        bindings.update({str(report_path): digest(report_path), str(log_path): digest(log_path)})
        reports[arm] = report
        assert report["status"] == "COMPLETE_FIXED_DEVELOPMENT_HEADS" and report["arm"] == arm
        assert report["epochs"] == len(log) == 100 and report["updates"] == 19000
        assert report["class_names"] == prepared["class_names"] and report["final_test_evaluated"] is False
        provenance = {"arm": arm, "protocol_sha256": protocol_sha, "cache_manifest_sha256": cache_sha,
                      "encoder_state_sha256": extracted["model_state_sha256_after"], "encoder_parameters_in_optimizer": False}
        assert report["provenance"] == provenance
        for epoch, row in enumerate(log, 1):
            assert row["epoch"] == epoch and row["updates"] == epoch * per_epoch
            assert ("validation" in row) == (epoch % 5 == 0)
        evaluated = [row for row in log if "validation" in row]
        assert len(evaluated) == 20 and report["epoch100_metrics"] == log[-1]["validation"]
        selected, metrics, head_states = {}, {}, {}
        checks[arm] = {}
        for task, score_key in (("global", "map"), ("dense", "map50")):
            best = {"score": -1., "epoch": 0}
            for row in evaluated:
                if row["validation"][task][score_key] > best["score"]:
                    best = {"score": row["validation"][task][score_key], "epoch": row["epoch"]}
            assert report["selected"][task] == best
            selected[task] = best
            head_path = output / f"best_{task}.pt"
            prediction_path = output / f"best_{task}_validation_predictions.pt"
            bind(head_path, report["best_heads_sha256"][task])
            bind(prediction_path, report["best_predictions_sha256"][task])
            checkpoint = torch.load(head_path, map_location="cpu", weights_only=True)
            prediction = torch.load(prediction_path, map_location="cpu", weights_only=True)
            chosen = log[best["epoch"] - 1]["validation"][task]
            assert checkpoint["epoch"] == best["epoch"] and checkpoint["metrics"] == chosen
            assert checkpoint["provenance"] == provenance and prediction["row_ids"] == validation_ids
            expected_shapes = ({"weight": (10, 384), "bias": (10,)} if task == "global"
                               else {"weight": (50, 384, 1, 1), "bias": (50,)})
            assert checkpoint["head"].keys() == expected_shapes.keys()
            state_hash = hashlib.sha256()
            for name, value in sorted(checkpoint["head"].items()):
                array = value.numpy()
                assert array.shape == expected_shapes[name] and array.dtype == np.float32
                state_hash.update(name.encode())
                state_hash.update(str(array.dtype).encode())
                state_hash.update(str(array.shape).encode())
                state_hash.update(array.tobytes())
            head_states[task] = state_hash.hexdigest()
            if task == "global":
                scores = prediction["predictions"].numpy()
                assert scores.shape == (1505, 10) and scores.dtype == np.float32
                metrics[task] = global_metrics(scores, labels)
            else:
                predictions = [{key: value.numpy() for key, value in p.items()} for p in prediction["predictions"]]
                assert len(predictions) == 1505
                for p in predictions:
                    count = len(p["scores"])
                    assert p["boxes"].shape == (count, 4) and p["labels"].shape == (count,) and count <= 100
                    assert p["boxes"].dtype == p["scores"].dtype == np.float32 and p["labels"].dtype == np.int64
                metrics[task] = dense_metrics(predictions, boxes, labels)
                with zipfile.ZipFile(prediction_path) as archive:
                    tensor_bytes = sum(entry.file_size for entry in archive.infolist() if "/data/" in entry.filename)
                overhead = prediction_path.stat().st_size - tensor_bytes
                # Three tensors per image; reserve 1 KiB each for varying pickle/ZIP layout.
                dense_size_bounds[arm] = {"current_bytes": prediction_path.stat().st_size,
                                          "maximum_tensor_payload_bytes": 1505 * 100 * (4 * 4 + 4 + 8),
                                          "current_serialization_overhead_bytes": overhead,
                                          "serialization_growth_allowance_bytes": 1505 * 3 * 1024}
            checks[arm][task] = compare_metrics(metrics[task], chosen)
        arms[arm] = {"selected": selected, "best": metrics, "best_head_state_sha256": head_states,
                     "epoch100": report["epoch100_metrics"], "epochs": 100, "updates": 19000, "evaluations": 20}
    assert reports["aa"]["initial_head_sha256"] == reports["area"]["initial_head_sha256"]
    differences = {}
    for endpoint in ("best", "epoch100"):
        signed, absolute = {}, {}
        for task in ("global", "dense"):
            signed[task], absolute[task] = {}, {}
            for key in arms["area"][endpoint][task]:
                delta = np.asarray(arms["area"][endpoint][task][key]) - np.asarray(arms["aa"][endpoint][task][key])
                signed[task][key], absolute[task][key] = delta.tolist(), np.abs(delta).tolist()
        differences[endpoint] = {"raw_area_minus_aa": signed, "absolute_area_minus_aa": absolute}
    reader_path = Path(__file__).resolve()
    bindings[str(reader_path)] = digest(reader_path)
    passed = all(item["pass"] for arm in checks.values() for task in arm.values() for item in task.values())
    summary = {"status": "VERIFIED_SAVED_BEST_DEVELOPMENT_METRICS" if passed else "UNVERIFIED_METRIC_MISMATCH",
               "scope": protocol["scope"], "class_names": prepared["class_names"], "arms": arms, "differences": differences,
               "epoch100_scope": "Reported metrics checked against epoch100 log; independently scored only if this task selected epoch100.",
               "protocol_sha256": protocol_sha, "reader_sha256": bindings[str(reader_path)], "final_test_evaluated": False}
    result_path, verification_path = run / "results.json", run / "verification.json"
    result_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    logs = [run.parent / f"{run.name}.{stage}.log" for stage in ("extract", "aa_heads", "area_heads")]
    logs.append(CONTRACT / "smoke.log")
    for path in logs:
        bindings[str(path)] = digest(path)
    run_files = [p for p in run.rglob("*") if p.is_file() and p not in (result_path, verification_path)]
    contract_files = [p for p in CONTRACT.rglob("*") if p.is_file()]
    current_fixed = sum(p.stat().st_size for p in run_files + contract_files + logs[:3]) + reader_path.stat().st_size
    dense_growth = sum(max(0, b["maximum_tensor_payload_bytes"] + b["current_serialization_overhead_bytes"]
                           + b["serialization_growth_allowance_bytes"] - b["current_bytes"]) for b in dense_size_bounds.values())
    json_reserve, head_metadata_growth = 1024**2, 4 * 1024
    estimated_bound = current_fixed + dense_growth + json_reserve + head_metadata_growth
    storage = {"kind": "Current completed-file sizes plus estimated finite serialization upper bound; historical instantaneous peak not measured.",
               "run_current_bytes_excluding_assembly_json": sum(p.stat().st_size for p in run_files),
               "contract_current_bytes": sum(p.stat().st_size for p in contract_files),
               "external_logs": {str(p): p.stat().st_size for p in logs[:3]}, "reader_bytes": reader_path.stat().st_size,
               "dense_prediction_bounds": dense_size_bounds, "head_metadata_growth_allowance_bytes": head_metadata_growth,
               "assembly_json_reserved_bytes": json_reserve, "estimated_peak_upper_bound_bytes": estimated_bound,
               "budget_bytes": protocol["budgets"]["new_peak_output_bytes"],
               "pass": estimated_bound < protocol["budgets"]["new_peak_output_bytes"],
               "assumptions": "Four persistent caches; two arms each overwrite two best heads and two best prediction files; no epoch archives or extra temporary feature caches. Dense files allow up to100 detections/image and explicit serialization allowance."}
    assert {path: digest(Path(path)) for path in bindings} == bindings
    verification = {"status": "PASS_SAVED_BEST_METRICS_AND_BINDINGS" if passed and storage["pass"] else "FAIL",
                    "bindings_sha256": bindings, "results_sha256": digest(result_path), "metric_checks": checks,
                    "tolerances": {"ap_recall": AP_RECALL_TOLERANCE, "f1": F1_TOLERANCE, "counts": 0},
                    "shared_initial_head_sha256": reports["aa"]["initial_head_sha256"],
                    "gt_rows_checked_against_raw_training_metadata": 7556, "validation_rows_per_endpoint": 1505,
                    "training_checks": "Both arms:100 consecutive epochs,19000 logged updates,20 evaluations; separate earliest strict-max selections; epoch100 report/log exact equality.",
                    "reported_encoder_state_sha256": extracted["model_state_sha256_after"], "storage": storage,
                    "reader_wall_seconds": time.monotonic() - started,
                    "limits": ["Only Torch CPU deserialization and tensor-to-NumPy conversion; no model creation, head/encoder forward, optimizer, raw image decoding, downloads or Test1/Test2 reads.",
                               "AP/F1/dense IoU matching are independently recomputed from saved best outputs and original row GT. Existing threshold/NMS/decoder and prediction-from-head are not independently recomputed.",
                               "Unselected epoch100 and historical evaluations have no saved predictions; their numerical metrics and training updates/losses were not independently reproduced.",
                               "Initial-head equality and unchanged encoder state are checked through saved hashes/provenance, not independently rebuilt states. Raw normalization/attention and feature-cache contents are not recomputed.",
                               "Whole encoding path comparison on validation-selected image split; no independent test, scene generalization, single-factor causal or new SSL claim."]}
    verification_path.write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n")
    assert result_path.stat().st_size + verification_path.stat().st_size <= json_reserve
    print(json.dumps({"status": verification["status"], "results": str(result_path), "verification": str(verification_path),
                      "verification_sha256": digest(verification_path), "storage_upper_bound_bytes": estimated_bound,
                      "wall_seconds": verification["reader_wall_seconds"]}), flush=True)
    assert verification["status"] == "PASS_SAVED_BEST_METRICS_AND_BINDINGS", "See verification.json; tolerances were not changed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run)
