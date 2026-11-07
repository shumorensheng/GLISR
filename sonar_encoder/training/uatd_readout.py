"""UATD complete-native frozen features under the fixed approximate polar mapping."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import __version__ as pillow_version
import torch

from .artifacts import create_artifact_directory
from .model import build_encoder
from .readout import file_digest, grayscale, state_digest, write_json
from .train import manifest_digest


def prepared_rows(rows, real_root, source_prefix, class_names):
    prepared = []
    for row in rows:
        height, width = row["shape"][:2]
        boxes = np.array([obj["bbox_xyxy"] for obj in row["objects"]], dtype=np.float64)
        prepared.append({"row_id": row["id"],
                         "image": str(real_root / Path(row["image_path"]).relative_to(source_prefix)),
                         "image_sha256": row["image_sha256"], "pixel_sha256": row["pixel_sha256"],
                         "shape": row["shape"], "range_m": row["range_m"],
                         "horizontal_fov_deg": row["horizontal_fov_deg"], "frequency": row["frequency"],
                         "original_boxes_xyxy": boxes.tolist(),
                         "boxes": (boxes / [width, height, width, height]).tolist(),
                         "labels": [class_names.index(obj["name"]) for obj in row["objects"]]})
    return prepared


def prepare_task(protocol_path, real_root, output):
    protocol = json.loads(protocol_path.read_text())
    source = Path(protocol["input_contract"]["source_manifest"])
    if file_digest(source) != protocol["source_sha256"][str(source)]:
        raise ValueError(f"source manifest changed: {source}")
    rows = {row["id"]: row for row in map(json.loads, source.read_text().splitlines())}
    train_ids = protocol["split"]["train_ids"]
    val_ids = protocol["split"]["validation_ids"]
    excluded = {row_id for group in protocol["duplicate_conflict_policy"]["excluded_groups"]
                for row_id in group["row_ids"]}
    if set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != rows.keys() - excluded:
        raise ValueError("frozen memberships do not partition the non-conflicting Training rows")
    class_names = protocol["labels"]["class_names"]
    prefix = Path(protocol["input_contract"]["source_prefix"])
    real_root = real_root.resolve()
    train = prepared_rows([rows[row_id] for row_id in train_ids], real_root, prefix, class_names)
    validation = prepared_rows([rows[row_id] for row_id in val_ids], real_root, prefix, class_names)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"task": "uatd", "protocol_path": str(protocol_path.resolve()),
                "protocol_sha256": file_digest(protocol_path), "source_manifest": str(source),
                "source_manifest_sha256": file_digest(source), "source_prefix": str(prefix),
                "real_root": str(real_root), "class_names": class_names,
                "geometry_assumption_set": protocol["approximate_geometry"]["assumption_set_id"],
                "train": train, "validation": validation, "excluded_row_ids": sorted(excluded),
                "final_manifests": {split: str(source.with_name(f"{split}.jsonl"))
                                    for split in ("test1", "test2")},
                "final_manifest_sha256": {split: file_digest(source.with_name(f"{split}.jsonl"))
                                          for split in ("test1", "test2")}}
    write_json(output / "manifest.json", manifest)
    total, squared, count = 0., 0., 0
    for row in train:
        if file_digest(Path(row["image"])) != row["image_sha256"]:
            raise ValueError(f"source image changed: {row['image']}")
        values = np.log1p(grayscale(row["image"]).astype(np.float64) / 255)
        total += values.sum()
        squared += np.square(values).sum()
        count += values.size
    mean = total / count
    write_json(output / "normalization.json", {
        "mean": mean, "std": float(np.sqrt(squared / count - mean * mean)), "pixel_count": count,
        "domain": "log1p(uint8 grayscale / 255); inner-training native pixels only",
        "source_manifest_sha256": file_digest(output / "manifest.json"),
        "train_row_ids_sha256": manifest_digest(train_ids)})
    return manifest


def native_sample(row):
    if file_digest(Path(row["image"])) != row["image_sha256"]:
        raise ValueError(f"source image changed: {row['image']}")
    intensity = torch.from_numpy(grayscale(row["image"])).float().unsqueeze(0) / 255
    height, width = intensity.shape[-2:]
    angular_span = math.radians(row["horizontal_fov_deg"])
    return {"intensity": intensity, "valid": torch.ones((height, width), dtype=torch.bool),
            "range_m": (torch.arange(height, dtype=torch.float32) + .5) * row["range_m"] / height,
            "bearing_rad": angular_span / 2 - (torch.arange(width, dtype=torch.float32) + .5) * angular_span / width,
            "configuration": {"RangeMin": 0., "RangeMax": row["range_m"]}, "sample_id": row["row_id"]}


def checkpoint_encoder(checkpoint_path, variant, device):
    checkpoint_sha = file_digest(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if file_digest(checkpoint_path) != checkpoint_sha:
        raise ValueError("encoder checkpoint changed while loading")
    config = saved["resolved_config"]
    # The already-frozen R010 v1 checkpoint predates the explicit variant field.
    declared = config["variant"] if "variant" in config else "R010"
    if declared != variant:
        raise ValueError(f"checkpoint variant {declared} differs from requested {variant}")
    encoder = build_encoder(Path(config["code_dir"]), Path(config["weights_path"]), variant)
    encoder.load_state_dict(saved["student"])
    encoder.eval().requires_grad_(False).to(device)
    return encoder, checkpoint_sha, saved["step"]


@torch.inference_mode()
def encode_rows(encoder, rows, normalization, batch_size):
    features = torch.empty((len(rows), 384, 16, 16), dtype=torch.float32)
    for start in range(0, len(rows), batch_size):
        samples = [native_sample(row) for row in rows[start:start + batch_size]]
        patches = encoder.forward_native(samples, normalization, None)["patch_tokens"]
        features[start:start + len(samples)] = patches.transpose(1, 2).reshape(-1, 384, 16, 16).float().cpu()
    return features


def cache_split(encoder, rows, normalization, batch_size, provenance, path):
    features = encode_rows(encoder, rows, normalization, batch_size)
    targets = [{"labels": torch.tensor(row["labels"], dtype=torch.long),
                "boxes": torch.tensor(row["boxes"], dtype=torch.float32)} for row in rows]
    torch.save({"features": features, "targets": targets,
                "row_ids": [row["row_id"] for row in rows], "provenance": provenance}, path)
    print(f"cached UATD {path.stem}: {list(features.shape)} FP32", flush=True)
    return file_digest(path)


def extract_features(prepared, checkpoint_path, variant, output, device, batch_size):
    manifest = json.loads((prepared / "manifest.json").read_text())
    normalization = json.loads((prepared / "normalization.json").read_text())
    if normalization["source_manifest_sha256"] != file_digest(prepared / "manifest.json"):
        raise ValueError("normalization does not match task manifest")
    if file_digest(Path(manifest["protocol_path"])) != manifest["protocol_sha256"]:
        raise ValueError("readout protocol changed")
    encoder, checkpoint_sha, step = checkpoint_encoder(checkpoint_path, variant, device)
    before = state_digest(encoder.state_dict())
    create_artifact_directory(output)
    provenance = {"task": "uatd", "variant": variant, "prepared": str(prepared.resolve()),
                  "task_manifest_sha256": file_digest(prepared / "manifest.json"),
                  "normalization_sha256": file_digest(prepared / "normalization.json"),
                  "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha,
                  "encoder_state_sha256": before, "checkpoint_step": step, "encoder_weights": "student",
                  "protocol_sha256": manifest["protocol_sha256"], "feature_dtype": "float32",
                  "geometry_assumption_set": manifest["geometry_assumption_set"],
                  "encoder_scope": "complete native encoder, including its trained stem and PolarPE",
                  "feature_normalization": "none; final LayerNorm patch tokens",
                  "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__, "pillow": pillow_version,
                  "source_sha256": {name: file_digest(Path(__file__).with_name(name))
                                    for name in ("uatd_readout.py", "model.py", "frontends.py", "artifacts.py")}}
    cache_files = {}
    for split in ("train", "validation"):
        path = output / f"{split}.pt"
        cache_files[path.name] = cache_split(encoder, manifest[split], normalization, batch_size, provenance, path)
    after = state_digest(encoder.state_dict())
    if before != after:
        raise RuntimeError("encoder changed during frozen feature extraction")
    result = {"provenance": provenance, "cache_files": cache_files, "class_names": manifest["class_names"],
              "encoder_hash_after_extraction": after}
    write_json(output / "cache_manifest.json", result)
    return result


def extract_final(cache_dir, test_split, output, device, batch_size):
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
    provenance = cache_manifest["provenance"]
    prepared = Path(provenance["prepared"])
    for name, key in (("manifest.json", "task_manifest_sha256"), ("normalization.json", "normalization_sha256")):
        if file_digest(prepared / name) != provenance[key]:
            raise ValueError(f"prepared task file changed: {name}")
    for name, digest in provenance["source_sha256"].items():
        if file_digest(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"native feature source changed: {name}")
    manifest = json.loads((prepared / "manifest.json").read_text())
    normalization = json.loads((prepared / "normalization.json").read_text())
    source = Path(manifest["final_manifests"][test_split])
    if file_digest(source) != manifest["final_manifest_sha256"][test_split]:
        raise ValueError(f"final manifest changed: {source}")
    rows = prepared_rows([json.loads(line) for line in source.read_text().splitlines()],
                         Path(manifest["real_root"]), Path(manifest["source_prefix"]), manifest["class_names"])
    encoder, checkpoint_sha, _ = checkpoint_encoder(Path(provenance["checkpoint"]), provenance["variant"], device)
    before = state_digest(encoder.state_dict())
    if checkpoint_sha != provenance["checkpoint_sha256"] or before != provenance["encoder_state_sha256"]:
        raise ValueError("final encoder differs from the train/validation encoder")
    create_artifact_directory(output)
    digest = cache_split(encoder, rows, normalization, batch_size, provenance, output / "final.pt")
    after = state_digest(encoder.state_dict())
    if before != after:
        raise RuntimeError("encoder changed during final feature extraction")
    result = {"provenance": provenance,
              "parent_cache_manifest_sha256": file_digest(cache_dir / "cache_manifest.json"),
              "cache_files": {"final.pt": digest}, "class_names": manifest["class_names"],
              "test_split": test_split, "source_manifest": str(source), "source_manifest_sha256": file_digest(source),
              "encoder_hash_after_extraction": after}
    write_json(output / "cache_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="prepare fixed Training partitions and training-only statistics")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--real-root", type=Path, required=True)
    extract = commands.add_parser("extract", help="cache complete-native train/validation features")
    extract.add_argument("--prepared", type=Path, required=True)
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--variant", choices=("R010", "R011", "R012", "R013", "R020", "R021", "R030", "R031", "R032", "R036", "R037", "R038", "R039"), required=True)
    final = commands.add_parser("extract-final", help="explicitly open one final test split using frozen train settings")
    final.add_argument("--cache", type=Path, required=True)
    final.add_argument("--test-split", choices=("test1", "test2"), required=True)
    for command in (prepare, extract, final):
        command.add_argument("--output", type=Path, required=True)
    for command in (extract, final):
        command.add_argument("--device", default="cpu")
        command.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "prepare":
        result = prepare_task(args.protocol, args.real_root, args.output)
        print(json.dumps({"train": len(result["train"]), "validation": len(result["validation"])}))
    elif args.command == "extract":
        extract_features(args.prepared, args.checkpoint, args.variant, args.output, args.device, args.batch_size)
    else:
        extract_final(args.cache, args.test_split, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
