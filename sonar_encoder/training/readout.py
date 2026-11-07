"""Frozen R010 display-image readouts: NKSID p0/k0 and Marine segmentation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import create_artifact_directory
from .model import build_encoder, prepare_images
from .train import file_digest, manifest_digest


SOURCE_PREFIX = Path("/media/hit/大磁盘/pmca_real_dataset")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def relocated_rows(path, task, real_root):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["row_id"] = row["index"] if task == "nksid" else row["id"]
        row["image"] = str(real_root / Path(row["image"]).relative_to(SOURCE_PREFIX))
        if task == "marine":
            row["mask"] = str(real_root / Path(row["mask"]).relative_to(SOURCE_PREFIX))
    return sorted(rows, key=lambda row: row["row_id"])


def grayscale(path):
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.uint8)


def prepare_task(task, protocol_path, real_root, output):
    protocol = json.loads(protocol_path.read_text())
    if task == "nksid":
        source = Path(protocol[task]["outer_partition"]["train_manifest"])
        held_out = Path(protocol[task]["outer_partition"]["evaluation_manifest"])
        sources = [source, held_out]
    else:
        source = Path(protocol[task]["train_manifest"])
        validation_source = Path(protocol[task]["val_manifest"])
        held_out = source.with_name("test.jsonl")
        sources = [source, validation_source]
    for path in sources:
        if file_digest(path) != protocol["source_sha256"][str(path)]:
            raise ValueError(f"source manifest changed: {path}")
    rows = relocated_rows(source, task, real_root)
    if task == "nksid":
        train_ids = set(protocol[task]["inner_split"]["train_indices"])
        val_ids = set(protocol[task]["inner_split"]["val_indices"])
        if train_ids & val_ids or train_ids | val_ids != {row["row_id"] for row in rows}:
            raise ValueError("inner partition does not match the outer training membership")
        train = [row for row in rows if row["row_id"] in train_ids]
        validation = [row for row in rows if row["row_id"] in val_ids]
        names = {row["classID"]: row["className"] for row in rows}
        classes = [names[index] for index in sorted(names)]
    else:
        train = rows
        validation = relocated_rows(validation_source, task, real_root)
        classes = [protocol[task]["classes"][str(index)] for index in range(12)]
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"task": task, "protocol_sha256": file_digest(protocol_path), "protocol_path": str(protocol_path.resolve()),
                "source_manifest_hashes": {str(path): file_digest(path) for path in sources},
                "real_root": str(real_root.resolve()), "class_names": classes, "train": train, "validation": validation,
                "held_out_manifest": str(held_out), "held_out_manifest_sha256": file_digest(held_out),
                "held_out_role": "outer_evaluation" if task == "nksid" else "local_test"}
    write_json(output / "manifest.json", manifest)
    total, squared, count = 0.0, 0.0, 0
    for row in train:
        values = np.log1p(grayscale(row["image"]).astype(np.float64) / 255)
        total += values.sum()
        squared += np.square(values).sum()
        count += values.size
    mean = total / count
    normalization = {"mean": mean, "std": float(np.sqrt(squared / count - mean * mean)), "pixel_count": count,
                     "domain": "log1p(uint8 grayscale / 255); native head-training pixels only",
                     "source_manifest_sha256": file_digest(output / "manifest.json"),
                     "train_row_ids_sha256": manifest_digest([row["row_id"] for row in train])}
    write_json(output / "normalization.json", normalization)
    return manifest


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def checkpoint_encoder(checkpoint_path, device):
    checkpoint_sha = file_digest(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if file_digest(checkpoint_path) != checkpoint_sha:
        raise ValueError("encoder checkpoint changed while loading")
    config = saved["resolved_config"]
    encoder = build_encoder(Path(config["code_dir"]), Path(config["weights_path"]), "R010")
    encoder.load_state_dict(saved["student"])
    encoder.eval().requires_grad_(False).to(device)
    return encoder, checkpoint_sha, saved["step"]


@torch.inference_mode()
def encode_rows(encoder, rows, normalization, task, device, batch_size):
    batches = []
    for start in range(0, len(rows), batch_size):
        samples = []
        for row in rows[start:start + batch_size]:
            values = torch.from_numpy(grayscale(row["image"])).float().unsqueeze(0) / 255
            samples.append({"intensity": values, "valid": torch.ones(values.shape[-2:], dtype=torch.bool)})
        images = prepare_images(samples, normalization, None).to(device)
        output = encoder(images)
        if task == "nksid":
            features = output["global_features"]
        else:
            features = output["patch_tokens"].transpose(1, 2).reshape(-1, 384, 16, 16)
        batches.append(features.float().cpu())
    return torch.cat(batches)


def load_targets(rows, task):
    if task == "nksid":
        return torch.tensor([row["classID"] for row in rows], dtype=torch.long)
    masks = []
    for row in rows:
        with Image.open(row["mask"]) as image:
            masks.append(torch.from_numpy(np.array(image, dtype=np.uint8)))
    return torch.stack(masks)


def extract_features(prepared, checkpoint_path, output, device, batch_size):
    manifest = json.loads((prepared / "manifest.json").read_text())
    normalization = json.loads((prepared / "normalization.json").read_text())
    if normalization["source_manifest_sha256"] != file_digest(prepared / "manifest.json"):
        raise ValueError("normalization does not match task manifest")
    encoder, checkpoint_sha, step = checkpoint_encoder(checkpoint_path, device)
    before = state_digest(encoder.state_dict())
    create_artifact_directory(output)
    provenance = {"task": manifest["task"], "prepared": str(prepared.resolve()),
                  "task_manifest_sha256": file_digest(prepared / "manifest.json"),
                  "normalization_sha256": file_digest(prepared / "normalization.json"),
                  "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha,
                  "encoder_state_sha256": before, "checkpoint_step": step, "encoder_weights": "student",
                  "protocol_sha256": manifest["protocol_sha256"], "feature_dtype": "float32",
                  "feature_normalization": "none; final LayerNorm patch tokens / their mean",
                  "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__, "pillow": pillow_version,
                  "readout_source_sha256": file_digest(Path(__file__)), "model_source_sha256": file_digest(Path(__file__).with_name("model.py")),
                  "artifacts_source_sha256": file_digest(Path(__file__).with_name("artifacts.py"))}
    cache_files = {}
    for split in ("train", "validation"):
        rows = manifest[split]
        features = encode_rows(encoder, rows, normalization, manifest["task"], device, batch_size)
        path = output / f"{split}.pt"
        torch.save({"features": features, "targets": load_targets(rows, manifest["task"]),
                    "row_ids": [row["row_id"] for row in rows], "provenance": provenance}, path)
        cache_files[path.name] = file_digest(path)
        print(f"cached {manifest['task']} {split}: {list(features.shape)} FP32", flush=True)
    after = state_digest(encoder.state_dict())
    if before != after:
        raise RuntimeError("encoder changed during frozen feature extraction")
    cache_manifest = {"provenance": provenance, "cache_files": cache_files, "encoder_hash_after_extraction": after,
                      "class_names": manifest["class_names"]}
    write_json(output / "cache_manifest.json", cache_manifest)
    return cache_manifest


def confusion_matrix(target, prediction, classes):
    indices = target.long().reshape(-1) * classes + prediction.long().reshape(-1)
    return torch.bincount(indices, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(confusion, task):
    matrix = confusion.double()
    tp = matrix.diag()
    actual, predicted = matrix.sum(1), matrix.sum(0)
    result = {"confusion_matrix": confusion.tolist(), "support": actual.long().tolist()}
    if task == "nksid":
        f1 = 2 * tp / (actual + predicted).clamp_min(1)
        result.update(macro_f1=float(f1.mean()), accuracy=float(tp.sum() / actual.sum()),
                      per_class_f1=f1.tolist(), per_class_precision=(tp / predicted.clamp_min(1)).tolist(),
                      per_class_recall=(tp / actual.clamp_min(1)).tolist())
    else:
        iou = tp / (actual + predicted - tp).clamp_min(1)
        result.update(miou12=float(iou.mean()), debris_miou10=float(iou[1:11].mean()), per_class_iou=iou.tolist())
    return result


def head_logits(head, features, targets, task):
    logits = head(features)
    if task == "marine":
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def evaluate_head(head, cache, task, device, batch_size):
    classes = 8 if task == "nksid" else 12
    confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)
    head.eval()
    for start in range(0, len(cache["features"]), batch_size):
        features = cache["features"][start:start + batch_size].to(device)
        targets = cache["targets"][start:start + batch_size].to(device)
        prediction = head_logits(head, features, targets, task).argmax(dim=1)
        confusion += confusion_matrix(targets, prediction, classes)
    return metrics_from_confusion(confusion.cpu(), task)


def train_head(cache_dir, output, device):
    started = time.monotonic()
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    provenance = cache_manifest["provenance"]
    task = provenance["task"]
    for name, digest in cache_manifest["cache_files"].items():
        if file_digest(cache_dir / name) != digest:
            raise ValueError(f"feature cache changed: {name}")
    train = torch.load(cache_dir / "train.pt", map_location="cpu", weights_only=True)
    validation = torch.load(cache_dir / "validation.pt", map_location="cpu", weights_only=True)
    if train["provenance"] != provenance or validation["provenance"] != provenance:
        raise ValueError("train/validation cache provenance differs")
    checkpoint_path = Path(provenance["checkpoint"])
    if file_digest(checkpoint_path) != provenance["checkpoint_sha256"]:
        raise ValueError("source encoder checkpoint changed")
    encoder_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["student"]
    before = state_digest(encoder_state)
    if before != provenance["encoder_state_sha256"]:
        raise ValueError("cached features and encoder state differ")
    torch.manual_seed(0)
    head = (nn.Linear(384, 8, bias=True) if task == "nksid" else nn.Conv2d(384, 12, 1, bias=True)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    generator = torch.Generator().manual_seed(0)
    batch_size = 64 if task == "nksid" else 16
    primary = "macro_f1" if task == "nksid" else "miou12"
    create_artifact_directory(output)
    if (output / "best.pt").exists():
        raise FileExistsError("head run already exists; choose a new output directory")
    initial_head_sha256 = state_digest(head.state_dict())
    best_score, best_epoch, updates = -1.0, 0, 0
    with (output / "metrics.jsonl").open("w") as log:
        for epoch in range(1, 101):
            head.train()
            order = torch.randperm(len(train["features"]), generator=generator)
            loss_sum = 0.0
            for indices in order.split(batch_size):
                features = train["features"][indices].to(device)
                targets = train["targets"][indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head_logits(head, features, targets, task), targets.long())
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
                updates += 1
            metrics = evaluate_head(head, validation, task, device, batch_size)
            entry = {"epoch": epoch, "updates": updates, "train_loss": loss_sum / len(order), "validation": metrics}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            saved = {"head": head.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "shuffle_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                     "provenance": provenance, "validation": metrics}
            if metrics[primary] > best_score:
                best_score, best_epoch = metrics[primary], epoch
                torch.save(saved, output / "best.pt")
            if epoch % 10 == 0:
                print(f"{task} epoch {epoch}/100 {primary}={metrics[primary]:.6f}", flush=True)
    after = state_digest(encoder_state)
    if before != after or file_digest(checkpoint_path) != provenance["checkpoint_sha256"]:
        raise RuntimeError("encoder state/checkpoint changed during head training")
    report = {"task": task, "epochs": 100, "optimizer_updates": updates, "selected_epoch": best_epoch,
              "selection_metric": primary, "best_validation_score": best_score, "encoder_hash_before": before,
              "encoder_hash_after": after, "encoder_parameters_in_optimizer": False,
              "initial_head_sha256": initial_head_sha256,
              "cache_manifest_sha256": file_digest(cache_dir / "cache_manifest.json"),
              "best_head_sha256": file_digest(output / "best.pt"), "elapsed_seconds": time.monotonic() - started,
              "final_test_evaluated": False}
    write_json(output / "report.json", report)
    return report


def final_test(cache_dir, head_path, output, device, batch_size):
    if (output / "final_metrics.json").exists():
        raise FileExistsError("final evaluation already exists")
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    provenance = cache_manifest["provenance"]
    prepared = Path(provenance["prepared"])
    for name, key in (("manifest.json", "task_manifest_sha256"), ("normalization.json", "normalization_sha256")):
        if file_digest(prepared / name) != provenance[key]:
            raise ValueError(f"prepared task file changed: {name}")
    manifest = json.loads((prepared / "manifest.json").read_text())
    normalization = json.loads((prepared / "normalization.json").read_text())
    held_out = Path(manifest["held_out_manifest"])
    if file_digest(held_out) != manifest["held_out_manifest_sha256"]:
        raise ValueError("held-out manifest changed")
    selected = torch.load(head_path, map_location="cpu", weights_only=True)
    selection_report = json.loads(head_path.with_name("report.json").read_text())
    if file_digest(head_path) != selection_report["best_head_sha256"] or selected["provenance"] != provenance:
        raise ValueError("final evaluation requires the selected validation head from this cache")
    task = manifest["task"]
    rows = relocated_rows(held_out, task, Path(manifest["real_root"]))
    encoder, checkpoint_sha, step = checkpoint_encoder(Path(provenance["checkpoint"]), device)
    before = state_digest(encoder.state_dict())
    if checkpoint_sha != provenance["checkpoint_sha256"] or before != provenance["encoder_state_sha256"]:
        raise ValueError("held-out encoder differs from the train/validation encoder")
    features = encode_rows(encoder, rows, normalization, task, device, batch_size)
    targets = load_targets(rows, task)
    head = (nn.Linear(384, 8, bias=True) if task == "nksid" else nn.Conv2d(384, 12, 1, bias=True)).to(device)
    head.load_state_dict(selected["head"])
    metrics = evaluate_head(head, {"features": features, "targets": targets}, task, device, batch_size)
    after = state_digest(encoder.state_dict())
    if before != after:
        raise RuntimeError("encoder changed during final evaluation")
    output.mkdir(parents=True, exist_ok=True)
    report = {"task": task, "role": manifest["held_out_role"], "count": len(rows), "row_ids": [row["row_id"] for row in rows],
              "checkpoint_step": step, "head_epoch": selected["epoch"], "class_names": manifest["class_names"],
              "metrics": metrics, "provenance": provenance, "head_sha256": file_digest(head_path),
              "encoder_hash_before": before, "encoder_hash_after": after}
    write_json(output / "final_metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="prepare train/val rows and train-only display statistics")
    prepare.add_argument("--task", choices=("nksid", "marine"), required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--real-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract", help="cache only train/val frozen student features")
    extract.add_argument("--prepared", type=Path, required=True)
    extract.add_argument("--checkpoint", type=Path, required=True)
    train = commands.add_parser("train", help="fit a head for the fixed 100 epochs, selecting on validation")
    train.add_argument("--cache", type=Path, required=True)
    final = commands.add_parser("finaltest", help="explicitly open held-out pixels and evaluate the already selected head")
    final.add_argument("--cache", type=Path, required=True)
    final.add_argument("--head", type=Path, required=True)
    for command in (extract, train, final):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cpu")
    for command in (extract, final):
        command.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "prepare":
        result = prepare_task(args.task, args.protocol, args.real_root, args.output)
        print(json.dumps({"task": result["task"], "train": len(result["train"]), "validation": len(result["validation"])}))
    elif args.command == "extract":
        extract_features(args.prepared, args.checkpoint, args.output, args.device, args.batch_size)
    elif args.command == "train":
        print(json.dumps(train_head(args.cache, args.output, args.device), indent=2))
    else:
        print(json.dumps(final_test(args.cache, args.head, args.output, args.device, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
