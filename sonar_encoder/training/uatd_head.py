"""Fixed UATD readouts from frozen image features, metadata, or their paired inputs."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import create_artifact_directory
from .metadata_probe import (load_paired_cache, load_paired_final_cache, verify_metadata_provenance,
                             verify_paired_provenance, verify_source_receipt)
from .readout import file_digest, state_digest, write_json


class MetadataGlobalHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.phi = nn.Sequential(nn.Conv2d(8, 16, 1), nn.GELU(), nn.Conv2d(16, 384, 1))
        self.head = head

    def forward(self, features):
        return self.head(self.phi(features).mean((-2, -1)))


class ImageMetadataHead(nn.Module):
    def __init__(self, head, global_readout):
        super().__init__()
        self.phi = nn.Sequential(nn.Conv2d(8, 16, 1), nn.GELU(), nn.Conv2d(16, 384, 1))
        nn.init.zeros_(self.phi[-1].weight)
        nn.init.zeros_(self.phi[-1].bias)
        self.head = head
        self.global_readout = global_readout

    def forward(self, features):
        features = features[:, :384] + self.phi(features[:, 384:])
        if self.global_readout:
            features = features.mean((-2, -1))
        return self.head(features)


def global_targets(targets, device):
    result = torch.zeros(len(targets), 10, device=device)
    for index, target in enumerate(targets):
        result[index, target["labels"].to(device)] = 1
    return result


def box_targets(targets, device):
    heatmap = torch.zeros(len(targets), 10, 16, 16, device=device)
    regression = torch.zeros(len(targets), 10, 4, 16, 16, device=device)
    for index, target in enumerate(targets):
        boxes = target["boxes"].to(device)
        labels = target["labels"].to(device)
        centers = (boxes[:, :2] + boxes[:, 2:]) * 8
        cells = centers.floor().long()
        sizes = (boxes[:, 2:] - boxes[:, :2]) * 16
        heatmap[index, labels, cells[:, 1], cells[:, 0]] = 1
        regression[index, labels, :, cells[:, 1], cells[:, 0]] = torch.cat(
            [centers - cells, sizes.log()], dim=1
        )
    return heatmap, regression, heatmap.bool()


def dense_loss(output, targets):
    heatmap, regression, positive = targets
    logits = output[:, :10]
    probability = logits.sigmoid()
    p_target = torch.where(positive, probability, 1 - probability)
    alpha = torch.where(positive, .25, .75)
    focal = (alpha * (1 - p_target).square() * F.binary_cross_entropy_with_logits(
        logits, heatmap, reduction="none"
    )).sum() / positive.sum()
    raw = output[:, 10:].reshape(-1, 10, 4, 16, 16)
    prediction = torch.cat([raw[:, :, :2].sigmoid(), raw[:, :, 2:]], dim=2)
    boxes = F.smooth_l1_loss(prediction.permute(0, 1, 3, 4, 2)[positive],
                             regression.permute(0, 1, 3, 4, 2)[positive], beta=1.)
    return focal + boxes


def box_iou(box, boxes):
    intersection = np.maximum(np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]), 0).prod(1)
    area = (box[2:] - box[:2]).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    return intersection / (area + areas - intersection)


def decode_detections(output):
    output = output.detach().float().cpu()
    probabilities = output[:, :10].sigmoid().numpy()
    regression = output[:, 10:].reshape(-1, 10, 4, 16, 16)
    offsets = regression[:, :, :2].sigmoid().numpy()
    sizes = (regression[:, :, 2:].exp() / 16).numpy()
    predictions = []
    for batch_index in range(len(output)):
        kept_boxes, kept_scores, kept_labels = [], [], []
        for label in range(10):
            scores = probabilities[batch_index, label].reshape(-1)
            cells = np.flatnonzero(scores >= .05)
            cells = cells[np.argsort(-scores[cells], kind="stable")[:100]]
            rows, cols = cells // 16, cells % 16
            centers = (np.stack([cols, rows], 1) + offsets[batch_index, label, :, rows, cols]) / 16
            wh = sizes[batch_index, label, :, rows, cols]
            boxes = np.concatenate([centers - wh / 2, centers + wh / 2], axis=1).clip(0, 1).astype(np.float32)
            order = np.flatnonzero((boxes[:, 2:] > boxes[:, :2]).all(1))
            while len(order):
                current = order[0]
                kept_boxes.append(boxes[current])
                kept_scores.append(scores[cells[current]])
                kept_labels.append(label)
                remaining = order[1:]
                order = remaining[box_iou(boxes[current], boxes[remaining]) <= .5]
        scores = np.array(kept_scores, dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:100]
        predictions.append({"boxes": torch.from_numpy(np.array(kept_boxes, dtype=np.float32).reshape(-1, 4)[order]),
                            "scores": torch.from_numpy(scores[order]),
                            "labels": torch.tensor(kept_labels, dtype=torch.long)[order]})
    return predictions


def average_precision(scores, true_positive, positives):
    order = torch.argsort(scores, descending=True, stable=True)
    correct = true_positive[order].double()
    precision = correct.cumsum(0) / torch.arange(1, len(correct) + 1, dtype=torch.float64)
    envelope = precision.flip(0).cummax(0).values.flip(0)
    return float((envelope * correct).sum() / positives)


def multilabel_metrics(scores, targets):
    per_class = [average_precision(scores[:, k], targets[:, k].bool(), int(targets[:, k].sum())) for k in range(10)]
    predictions = scores >= .5
    truth = targets.bool()
    tp = (predictions & truth).sum(0)
    f1 = 2 * tp / (predictions.sum(0) + truth.sum(0))
    return {"map": float(np.mean(per_class)), "per_class_ap": per_class,
            "macro_f1_at_0_5": float(f1.mean()),
            "micro_f1_at_0_5": float(2 * tp.sum() / (predictions.sum() + truth.sum())),
            "per_class_support": targets.sum(0).long().tolist()}


def detection_metrics(predictions, targets):
    per_class, supports, recalls = [], [], []
    for label in range(10):
        truth = [target["boxes"][target["labels"] == label].numpy() for target in targets]
        matched = [np.zeros(len(boxes), dtype=bool) for boxes in truth]
        entries = [(float(score), image, box.numpy())
                   for image, prediction in enumerate(predictions)
                   for box, score in zip(prediction["boxes"][prediction["labels"] == label],
                                         prediction["scores"][prediction["labels"] == label])]
        entries.sort(key=lambda item: -item[0])
        correct = []
        for score, image, box in entries:
            available = np.flatnonzero(~matched[image])
            hit = False
            if len(available):
                overlaps = box_iou(box, truth[image][available])
                best = int(overlaps.argmax())
                if overlaps[best] >= .5:
                    matched[image][available[best]] = True
                    hit = True
            correct.append(hit)
        support = sum(len(boxes) for boxes in truth)
        supports.append(support)
        recalls.append(sum(correct) / support)
        per_class.append(average_precision(torch.tensor([item[0] for item in entries]), torch.tensor(correct), support))
    return {"map50": float(np.mean(per_class)), "per_class_ap50": per_class,
            "per_class_gt_count": supports, "per_class_recall_at_0_05": recalls,
            "macro_recall_at_0_05": float(np.mean(recalls)),
            "predictions": sum(len(p["scores"]) for p in predictions)}


@torch.no_grad()
def evaluate_heads(global_head, dense_head, cache, device, metadata_only=False, image_metadata=False):
    global_head.eval()
    dense_head.eval()
    scores, detections = [], []
    for start in range(0, len(cache["features"]), 32):
        features = cache["features"][start:start + 32].to(device)
        if image_metadata:
            features = torch.cat([features, cache["metadata"][start:start + 32].to(device)], dim=1)
        scores.append(global_head(features if metadata_only or image_metadata else features.mean((-2, -1))).sigmoid().cpu())
        detections.extend(decode_detections(dense_head(features)))
    scores = torch.cat(scores)
    metrics = {"global": multilabel_metrics(scores, global_targets(cache["targets"], "cpu")),
               "dense": detection_metrics(detections, cache["targets"])}
    return metrics, {"global_scores": scores, "detections": detections, "row_ids": cache["row_ids"]}


def train_head(cache_dir, output, device):
    started = time.monotonic()
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    provenance = manifest["provenance"]
    metadata_only = provenance["task"] == "uatd_metadata_only"
    image_metadata = provenance["task"] == "uatd_image_metadata"
    if image_metadata:
        train, validation = load_paired_cache(manifest)
    else:
        for name, digest in manifest["cache_files"].items():
            if file_digest(cache_dir / name) != digest:
                raise ValueError(f"feature cache changed: {name}")
        train = torch.load(cache_dir / "train.pt", map_location="cpu", weights_only=True)
        validation = torch.load(cache_dir / "validation.pt", map_location="cpu", weights_only=True)
        if train["provenance"] != provenance or validation["provenance"] != provenance:
            raise ValueError("train/validation feature provenance differs")
    if metadata_only:
        verify_metadata_provenance(provenance, Path(__file__).parent)
    elif not image_metadata:
        checkpoint = Path(provenance["checkpoint"])
        if file_digest(checkpoint) != provenance["checkpoint_sha256"]:
            raise ValueError("source checkpoint changed")
        encoder_state = torch.load(checkpoint, map_location="cpu", weights_only=False)["student"]
        before = state_digest(encoder_state)
        if before != provenance["encoder_state_sha256"]:
            raise ValueError("cache and encoder state differ")
    create_artifact_directory(output)
    if (output / "best_global.pt").exists():
        raise FileExistsError("head run already exists; choose a new output directory")
    torch.manual_seed(0)
    global_head = nn.Linear(384, 10).to(device)
    dense_head = nn.Conv2d(384, 50, 1).to(device)
    # Labels and augmentation are fixed; avoid per-image GPU work in every update.
    train_boxes = tuple(value.to(device) for value in box_targets(train["targets"], "cpu"))
    train_presence = global_targets(train["targets"], "cpu").to(device)
    with torch.no_grad():
        dense_head.bias[:10].fill_(math.log(.01 / .99))
    if metadata_only or image_metadata:
        initial_base_head_sha256 = {"global": state_digest(global_head.state_dict()),
                                    "dense": state_digest(dense_head.state_dict())}
        # Construct both common heads first and preserve their RNG consumption.
        with torch.random.fork_rng(devices=[]):
            if image_metadata:
                global_head = ImageMetadataHead(global_head, global_readout=True).to(device)
                dense_head = ImageMetadataHead(dense_head, global_readout=False).to(device)
            else:
                global_head = MetadataGlobalHead(global_head).to(device)
                dense_head = nn.Sequential(
                    nn.Sequential(nn.Conv2d(8, 16, 1), nn.GELU(), nn.Conv2d(16, 384, 1)), dense_head
                ).to(device)
        global_phi = global_head.phi
        dense_phi = dense_head.phi if image_metadata else dense_head[0]
        initial_phi_sha256 = {"global": state_digest(global_phi.state_dict()),
                              "dense": state_digest(dense_phi.state_dict())}
        best_phi_sha256 = {}
    initial_head_sha256 = {"global": state_digest(global_head.state_dict()),
                           "dense": state_digest(dense_head.state_dict())}
    optimizer = torch.optim.AdamW(list(global_head.parameters()) + list(dense_head.parameters()), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(0)
    best = {"global": {"score": -1., "epoch": 0}, "dense": {"score": -1., "epoch": 0}}
    updates = 0
    with (output / "metrics.jsonl").open("w") as log:
        for epoch in range(1, 101):
            global_head.train()
            dense_head.train()
            order = torch.randperm(len(train["features"]), generator=generator)
            loss_sum = 0.
            for indices in order.split(32):
                features = train["features"][indices].to(device)
                if image_metadata:
                    features = torch.cat([features, train["metadata"][indices].to(device)], dim=1)
                device_indices = indices.to(device)
                optimizer.zero_grad(set_to_none=True)
                classification = F.binary_cross_entropy_with_logits(
                    global_head(features if metadata_only or image_metadata else features.mean((-2, -1))), train_presence[device_indices])
                loss = classification + dense_loss(dense_head(features), tuple(value[device_indices] for value in train_boxes))
                loss.backward()
                optimizer.step()
                updates += 1
                loss_sum += float(loss.detach()) * len(indices)
            row = {"epoch": epoch, "updates": updates, "train_loss": loss_sum / len(order)}
            if epoch % 5 == 0:
                metrics, predictions = evaluate_heads(global_head, dense_head, validation, device, metadata_only, image_metadata)
                row["validation"] = metrics
                for name, head, key in (("global", global_head, "map"), ("dense", dense_head, "map50")):
                    score = metrics[name][key]
                    if score > best[name]["score"]:
                        best[name] = {"score": score, "epoch": epoch}
                        if metadata_only or image_metadata:
                            phi = global_phi if name == "global" else dense_phi
                            best_phi_sha256[name] = state_digest(phi.state_dict())
                        torch.save({"head": head.state_dict(), "epoch": epoch, "provenance": provenance,
                                    "metrics": metrics[name]}, output / f"best_{name}.pt")
                        torch.save({"row_ids": predictions["row_ids"],
                                    "predictions": predictions["global_scores" if name == "global" else "detections"]},
                                   output / f"best_{name}_validation_predictions.pt")
                print(json.dumps({"epoch": epoch, "global_map": metrics["global"]["map"], "dense_map50": metrics["dense"]["map50"]}), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
    if metadata_only:
        verify_metadata_provenance(provenance, Path(__file__).parent)
    elif image_metadata:
        verify_paired_provenance(provenance, Path(__file__).parent)
    else:
        after = state_digest(encoder_state)
        if before != after or file_digest(checkpoint) != provenance["checkpoint_sha256"]:
            raise RuntimeError("encoder checkpoint changed during frozen head training")
    report = {"task": provenance["task"] if metadata_only or image_metadata else "uatd_approximate_uniform_polar",
              "epochs": 100, "updates": updates, "selected": best, "elapsed_seconds": time.monotonic() - started,
              "initial_head_sha256": initial_head_sha256,
              "cache_manifest_sha256": file_digest(cache_dir / "cache_manifest.json"),
              "head_source_sha256": file_digest(Path(__file__)),
              "artifacts_source_sha256": file_digest(Path(__file__).with_name("artifacts.py")),
              "class_names": manifest["class_names"],
              "best_heads_sha256": {name: file_digest(output / f"best_{name}.pt") for name in best},
              "final_test_evaluated": False}
    if metadata_only or image_metadata:
        report.update(initial_base_head_sha256=initial_base_head_sha256, initial_phi_sha256=initial_phi_sha256,
                      best_phi_sha256=best_phi_sha256,
                      final_phi_sha256={"global": state_digest(global_phi.state_dict()),
                                        "dense": state_digest(dense_phi.state_dict())},
                      final_head_state_sha256={"global": state_digest(global_head.state_dict()),
                                               "dense": state_digest(dense_head.state_dict())},
                      phi_parameters={"global": sum(p.numel() for p in global_phi.parameters()),
                                      "dense": sum(p.numel() for p in dense_phi.parameters())},
                      head_parameters={"global": sum(p.numel() for p in global_head.parameters()),
                                       "dense": sum(p.numel() for p in dense_head.parameters())},
                      scope=provenance["scope"], provenance=provenance)
        if image_metadata:
            report["encoder_parameters_in_optimizer"] = False
    else:
        report.update(encoder_hash_before=before, encoder_hash_after=after, encoder_parameters_in_optimizer=False)
    write_json(output / "report.json", report)
    return report


def final_test(final_cache_dir, head_dir, output, device):
    if (output / "final_metrics.json").exists():
        raise FileExistsError("final evaluation already exists")
    manifest = json.loads((final_cache_dir / "cache_manifest.json").read_text())
    selection = json.loads((head_dir / "report.json").read_text())
    if manifest["parent_cache_manifest_sha256"] != selection["cache_manifest_sha256"]:
        raise ValueError("final cache and selected heads have different training caches")
    provenance = manifest["provenance"]
    metadata_only = provenance["task"] == "uatd_metadata_only"
    image_metadata = provenance["task"] == "uatd_image_metadata"
    if image_metadata:
        cache = load_paired_final_cache(manifest)
    else:
        for name, digest in manifest["cache_files"].items():
            if file_digest(final_cache_dir / name) != digest:
                raise ValueError(f"final feature cache changed: {name}")
        cache = torch.load(final_cache_dir / "final.pt", map_location="cpu", weights_only=True, mmap=True)
        if cache["provenance"] != provenance:
            raise ValueError("final cache provenance differs from its manifest")
    if metadata_only or image_metadata:
        receipt_path = Path(manifest["source_receipt"])
        if file_digest(receipt_path) != manifest["source_receipt_sha256"]:
            raise ValueError("original readout source receipt changed")
        receipt = json.loads(receipt_path.read_text())
        if selection["head_source_sha256"] != receipt["source_sha256"]["uatd_head.py"]:
            raise ValueError("selected head source differs from the original source receipt")
    if metadata_only:
        parent_cache = Path(manifest["parent_cache"])
        parent_path = parent_cache / "cache_manifest.json"
        if file_digest(parent_path) != manifest["parent_cache_manifest_sha256"]:
            raise ValueError("original metadata training cache changed")
        parent = json.loads(parent_path.read_text())
        if parent["provenance"] != provenance or parent["class_names"] != manifest["class_names"]:
            raise ValueError("final metadata differs from the original training cache")
        source_root = verify_source_receipt(provenance, parent_cache, receipt_path)
        if str(source_root) != manifest["source_root"]:
            raise ValueError("original metadata source root differs from receipt")
        verify_metadata_provenance(provenance, source_root)
        if file_digest(source_root / "uatd_head.py") != selection["head_source_sha256"]:
            raise ValueError("original selected metadata head source changed")
        prepared = json.loads((Path(provenance["prepared"]) / "manifest.json").read_text())
        source = Path(prepared["final_manifests"][manifest["test_split"]])
        if (source.resolve() != Path(manifest["source_manifest"]).resolve()
                or file_digest(source) != manifest["source_manifest_sha256"]
                or manifest["source_manifest_sha256"] != prepared["final_manifest_sha256"][manifest["test_split"]]):
            raise ValueError("final metadata source differs from the frozen test split")
    elif not image_metadata:
        checkpoint = Path(provenance["checkpoint"])
        if file_digest(checkpoint) != provenance["checkpoint_sha256"]:
            raise ValueError("source checkpoint changed")
    global_head = nn.Linear(384, 10).to(device)
    dense_head = nn.Conv2d(384, 50, 1).to(device)
    if image_metadata:
        global_head = ImageMetadataHead(global_head, global_readout=True).to(device)
        dense_head = ImageMetadataHead(dense_head, global_readout=False).to(device)
    elif metadata_only:
        global_head = MetadataGlobalHead(global_head).to(device)
        dense_head = nn.Sequential(
            nn.Sequential(nn.Conv2d(8, 16, 1), nn.GELU(), nn.Conv2d(16, 384, 1)), dense_head
        ).to(device)
    for name, head in (("global", global_head), ("dense", dense_head)):
        path = head_dir / f"best_{name}.pt"
        if file_digest(path) != selection["best_heads_sha256"][name]:
            raise ValueError(f"selected {name} head changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["provenance"] != provenance:
            raise ValueError("selected head and final cache provenance differ")
        head.load_state_dict(saved["head"])
    metrics, predictions = evaluate_heads(global_head, dense_head, cache, device, metadata_only, image_metadata)
    create_artifact_directory(output)
    torch.save(predictions, output / "predictions.pt")
    report = {"task": provenance["task"] if metadata_only or image_metadata else "uatd_approximate_uniform_polar",
              "test_split": manifest["test_split"],
              "count": len(cache["row_ids"]), "row_ids": cache["row_ids"], "metrics": metrics,
              "selected": selection["selected"], "class_names": manifest["class_names"],
              "provenance": provenance, "final_cache_manifest_sha256": file_digest(final_cache_dir / "cache_manifest.json"),
              "head_selection_sha256": file_digest(head_dir / "report.json"),
              "predictions_sha256": file_digest(output / "predictions.pt"),
              "head_source_sha256": file_digest(Path(__file__)),
              "selected_head_source_sha256": selection["head_source_sha256"],
              "best_heads_sha256": selection["best_heads_sha256"], "final_test_evaluated": True,
              "scope": provenance["scope"] if metadata_only or image_metadata else
              "Full encoder transfer under declared approximate calibration; no scene-independent or exact metric geometry claim."}
    if metadata_only or image_metadata:
        report.update(source_receipt=str(receipt_path.resolve()), source_receipt_sha256=file_digest(receipt_path),
                      final_execution_source_sha256={name: file_digest(Path(__file__).with_name(name))
                                                     for name in ("uatd_head.py", "metadata_probe.py", "readout.py", "artifacts.py")})
    write_json(output / "final_metrics.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    final = commands.add_parser("finaltest")
    final.add_argument("--head-dir", type=Path, required=True)
    for command in (train, final):
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "train":
        result = train_head(args.cache, args.output, args.device)
    else:
        result = final_test(args.cache, args.head_dir, args.output, args.device)
    print(json.dumps(result, indent=2))
