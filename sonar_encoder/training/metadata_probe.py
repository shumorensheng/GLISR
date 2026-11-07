"""Prepare calibration-only descriptors or paired cache references without opening pixels."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from .artifacts import create_artifact_directory
from .readout import file_digest, write_json
from .uatd_readout import prepared_rows


def metadata_features(rows):
    features = torch.empty((len(rows), 8, 16, 16), dtype=torch.float32)
    centers = (torch.arange(16, dtype=torch.float32) + .5) / 16
    for index, row in enumerate(rows):
        height, width = row["shape"][:2]
        maximum_range = row["range_m"]
        angular_span = math.radians(row["horizontal_fov_deg"])
        ranges = centers * maximum_range
        bearings = angular_span / 2 - centers * angular_span
        # Tokens use nominal centers; sampling-scale logs retain native H/W.
        x = torch.full((16, 16), math.log(maximum_range / height))
        y = (ranges[:, None] * angular_span / width).log().expand(16, 16)
        features[index] = torch.stack([
            centers[:, None].expand(16, 16), bearings.sin()[None].expand(16, 16),
            bearings.cos()[None].expand(16, 16), x, y, x + y, y - x,
            torch.ones((16, 16)),
        ])
    return features


def verify_metadata_provenance(provenance, source_root):
    for path, digest in ((Path(provenance["prepared"]) / "manifest.json", provenance["task_manifest_sha256"]),
                         (Path(provenance["protocol_path"]), provenance["protocol_sha256"]),
                         (Path(provenance["image_readout_protocol"]), provenance["image_readout_protocol_sha256"])):
        if file_digest(path) != digest:
            raise ValueError(f"metadata input/protocol changed: {path}")
    for name, digest in provenance["source_sha256"].items():
        if file_digest(source_root / name) != digest:
            raise ValueError(f"metadata cache source changed: {name}")


def prepare_cache(protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    prepared = Path(protocol["prepared"])
    if file_digest(prepared / "manifest.json") != protocol["prepared_manifest_sha256"]:
        raise ValueError("prepared UATD manifest changed")
    image_protocol = Path(protocol["image_readout_protocol"])
    if file_digest(image_protocol) != protocol["image_readout_protocol_sha256"]:
        raise ValueError("image readout protocol changed")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["task"] != "uatd" or manifest["protocol_sha256"] != protocol["image_readout_protocol_sha256"]:
        raise ValueError("prepared manifest differs from the frozen UATD task")
    for split in ("train", "validation"):
        if len(manifest[split]) != protocol["counts"][split]:
            raise ValueError(f"prepared {split} count differs from metadata protocol")
    provenance = {"task": "uatd_metadata_only", "prepared": str(prepared.resolve()),
                  "task_manifest_sha256": protocol["prepared_manifest_sha256"],
                  "protocol_path": str(protocol_path.resolve()), "protocol_sha256": file_digest(protocol_path),
                  "image_readout_protocol": str(image_protocol.resolve()),
                  "image_readout_protocol_sha256": protocol["image_readout_protocol_sha256"],
                  "geometry_assumption_set": manifest["geometry_assumption_set"], "feature_dtype": "float32",
                  "input_8": protocol["input_8"], "token_coordinates": protocol["token_coordinates"],
                  "network": protocol["network"], "scope": protocol["scope"],
                  "python": sys.version, "torch": str(torch.__version__),
                  "source_sha256": {name: file_digest(Path(__file__).with_name(name))
                                    for name in ("metadata_probe.py", "artifacts.py")}}
    create_artifact_directory(output)
    if (output / "cache_manifest.json").exists():
        raise FileExistsError("metadata cache already exists; choose a new output directory")
    cache_files = {}
    for split in ("train", "validation"):
        rows = manifest[split]
        features = metadata_features(rows)
        targets = [{"labels": torch.tensor(row["labels"], dtype=torch.long),
                    "boxes": torch.tensor(row["boxes"], dtype=torch.float32)} for row in rows]
        path = output / f"{split}.pt"
        torch.save({"features": features, "targets": targets, "row_ids": [row["row_id"] for row in rows],
                    "provenance": provenance}, path)
        cache_files[path.name] = file_digest(path)
        print(f"cached metadata-only UATD {split}: {list(features.shape)} FP32", flush=True)
    verify_metadata_provenance(provenance, Path(__file__).parent)
    result = {"provenance": provenance, "cache_files": cache_files, "class_names": manifest["class_names"]}
    write_json(output / "cache_manifest.json", result)
    return result


def verify_paired_provenance(provenance, source_root):
    protocol_path = Path(provenance["protocol_path"])
    if file_digest(protocol_path) != provenance["protocol_sha256"]:
        raise ValueError("combined readout protocol changed")
    protocol = json.loads(protocol_path.read_text())
    for name in ("image", "metadata"):
        directory = Path(provenance[f"{name}_cache"])
        digest = provenance[f"{name}_cache_manifest_sha256"]
        if directory.resolve() != Path(protocol[f"{name}_cache"]).resolve() or digest != protocol[f"{name}_cache_manifest_sha256"]:
            raise ValueError(f"paired {name} cache differs from combined protocol")
        if file_digest(directory / "cache_manifest.json") != digest:
            raise ValueError(f"paired {name} cache manifest changed")
        parent = json.loads((directory / "cache_manifest.json").read_text())
        if parent["provenance"] != provenance[f"{name}_provenance"]:
            raise ValueError(f"paired {name} provenance differs from parent cache")
    metadata = provenance["metadata_provenance"]
    receipt_path = Path(provenance["metadata_source_receipt"])
    if file_digest(receipt_path) != provenance["metadata_source_receipt_sha256"]:
        raise ValueError("original metadata source receipt changed")
    receipt = json.loads(receipt_path.read_text())
    metadata_source_root = Path(receipt["source_root"]).resolve()
    if str(metadata_source_root) != provenance["metadata_source_root"]:
        raise ValueError("original metadata source root differs from receipt")
    if Path(receipt["cache_output"]).resolve() != Path(provenance["metadata_cache"]).resolve():
        raise ValueError("original metadata receipt belongs to a different cache")
    if receipt["protocol_sha256"] != metadata["protocol_sha256"]:
        raise ValueError("original metadata receipt protocol differs from cache")
    for name, digest in metadata["source_sha256"].items():
        if receipt["source_sha256"][name] != digest or file_digest(metadata_source_root / name) != digest:
            raise ValueError(f"original metadata source changed: {name}")
    for path, digest in ((Path(metadata["prepared"]) / "manifest.json", metadata["task_manifest_sha256"]),
                         (Path(metadata["protocol_path"]), metadata["protocol_sha256"]),
                         (Path(metadata["image_readout_protocol"]), metadata["image_readout_protocol_sha256"])):
        if file_digest(path) != digest:
            raise ValueError(f"paired metadata input/protocol changed: {path}")
    image = provenance["image_provenance"]
    if (image["task"] != "uatd" or image["variant"] != "R021" or image["checkpoint_step"] != 10000
            or metadata["task"] != "uatd_metadata_only"
            or image["task_manifest_sha256"] != metadata["task_manifest_sha256"]
            or Path(image["prepared"]).resolve() != Path(metadata["prepared"]).resolve()
            or image["protocol_sha256"] != metadata["image_readout_protocol_sha256"]):
        raise ValueError("paired parents differ from the frozen R021 UATD task")
    if file_digest(Path(image["checkpoint"])) != image["checkpoint_sha256"]:
        raise ValueError("paired image source checkpoint changed")
    for name, digest in provenance["source_sha256"].items():
        if file_digest(source_root / name) != digest:
            raise ValueError(f"combined readout source changed: {name}")


def load_paired_cache(manifest):
    provenance = manifest["provenance"]
    verify_paired_provenance(provenance, Path(__file__).parent)
    image_dir, metadata_dir = Path(provenance["image_cache"]), Path(provenance["metadata_cache"])
    image_manifest = json.loads((image_dir / "cache_manifest.json").read_text())
    metadata_manifest = json.loads((metadata_dir / "cache_manifest.json").read_text())
    if image_manifest["class_names"] != metadata_manifest["class_names"] or manifest["class_names"] != image_manifest["class_names"]:
        raise ValueError("paired cache class names differ")
    prepared = json.loads((Path(provenance["metadata_provenance"]["prepared"]) / "manifest.json").read_text())
    result = []
    for split in ("train", "validation"):
        name = f"{split}.pt"
        for directory, parent in ((image_dir, image_manifest), (metadata_dir, metadata_manifest)):
            if file_digest(directory / name) != parent["cache_files"][name]:
                raise ValueError(f"paired cache tensor changed: {directory / name}")
        image = torch.load(image_dir / name, map_location="cpu", weights_only=True, mmap=True)
        metadata = torch.load(metadata_dir / name, map_location="cpu", weights_only=True, mmap=True)
        if image["provenance"] != provenance["image_provenance"] or metadata["provenance"] != provenance["metadata_provenance"]:
            raise ValueError(f"paired {split} tensor provenance differs")
        if image["row_ids"] != metadata["row_ids"] or image["row_ids"] != [row["row_id"] for row in prepared[split]]:
            raise ValueError(f"paired {split} row IDs differ")
        count = len(image["row_ids"])
        if (image["features"].shape != (count, 384, 16, 16) or metadata["features"].shape != (count, 8, 16, 16)
                or image["features"].dtype != torch.float32 or metadata["features"].dtype != torch.float32):
            raise ValueError(f"paired {split} feature shape/dtype differs from frozen inputs")
        if len(image["targets"]) != count or len(metadata["targets"]) != count:
            raise ValueError(f"paired {split} target count differs")
        for image_target, metadata_target in zip(image["targets"], metadata["targets"]):
            if any(not torch.equal(image_target[key], metadata_target[key]) for key in ("labels", "boxes")):
                raise ValueError(f"paired {split} targets differ")
        image["metadata"] = metadata["features"]
        result.append(image)
    return tuple(result)


def prepare_paired_cache(protocol_path, metadata_source_receipt, output):
    protocol = json.loads(protocol_path.read_text())
    image_dir, metadata_dir = Path(protocol["image_cache"]), Path(protocol["metadata_cache"])
    image_manifest = json.loads((image_dir / "cache_manifest.json").read_text())
    metadata_manifest = json.loads((metadata_dir / "cache_manifest.json").read_text())
    receipt = json.loads(metadata_source_receipt.read_text())
    provenance = {"task": "uatd_image_metadata", "protocol_path": str(protocol_path.resolve()),
                  "protocol_sha256": file_digest(protocol_path),
                  "image_cache": str(image_dir.resolve()), "metadata_cache": str(metadata_dir.resolve()),
                  "image_cache_manifest_sha256": protocol["image_cache_manifest_sha256"],
                  "metadata_cache_manifest_sha256": protocol["metadata_cache_manifest_sha256"],
                  "image_provenance": image_manifest["provenance"], "metadata_provenance": metadata_manifest["provenance"],
                  "metadata_source_receipt": str(metadata_source_receipt.resolve()),
                  "metadata_source_receipt_sha256": file_digest(metadata_source_receipt),
                  "metadata_source_root": str(Path(receipt["source_root"]).resolve()),
                  "network": protocol["network"], "scope": protocol["scope"],
                  "source_sha256": {name: file_digest(Path(__file__).with_name(name))
                                    for name in ("metadata_probe.py", "uatd_head.py", "readout.py", "artifacts.py")}}
    result = {"provenance": provenance, "class_names": image_manifest["class_names"]}
    train, validation = load_paired_cache(result)
    result["counts"] = {"train": len(train["row_ids"]), "validation": len(validation["row_ids"])}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "cache_manifest.json").exists():
        raise FileExistsError("paired cache already exists; choose a new output directory")
    write_json(output / "cache_manifest.json", result)
    return result


def verify_source_receipt(provenance, cache_dir, receipt_path):
    receipt = json.loads(receipt_path.read_text())
    if Path(receipt["cache_output"]).resolve() != cache_dir.resolve():
        raise ValueError("original source receipt belongs to a different cache")
    if receipt["protocol_sha256"] != provenance["protocol_sha256"]:
        raise ValueError("original source receipt protocol differs from cache")
    source_root = Path(receipt["source_root"]).resolve()
    for name, digest in provenance["source_sha256"].items():
        if receipt["source_sha256"][name] != digest:
            raise ValueError(f"original source receipt differs from cache: {name}")
    return source_root


def extract_final(cache_dir, source_receipt, test_split, output):
    parent_path = cache_dir / "cache_manifest.json"
    parent = json.loads(parent_path.read_text())
    provenance = parent["provenance"]
    if provenance["task"] != "uatd_metadata_only":
        raise ValueError("final metadata extraction requires the metadata-only training cache")
    source_root = verify_source_receipt(provenance, cache_dir, source_receipt)
    verify_metadata_provenance(provenance, source_root)
    if (output / "cache_manifest.json").exists():
        raise FileExistsError("final metadata cache already exists")
    manifest = json.loads((Path(provenance["prepared"]) / "manifest.json").read_text())
    source = Path(manifest["final_manifests"][test_split])
    if file_digest(source) != manifest["final_manifest_sha256"][test_split]:
        raise ValueError("final UATD source manifest changed")
    rows = prepared_rows([json.loads(line) for line in source.read_text().splitlines()],
                         Path(manifest["real_root"]), Path(manifest["source_prefix"]), manifest["class_names"])
    targets = [{"labels": torch.tensor(row["labels"], dtype=torch.long),
                "boxes": torch.tensor(row["boxes"], dtype=torch.float32)} for row in rows]
    create_artifact_directory(output)
    torch.save({"features": metadata_features(rows), "targets": targets,
                "row_ids": [row["row_id"] for row in rows], "provenance": provenance}, output / "final.pt")
    result = {"provenance": provenance, "parent_cache": str(cache_dir.resolve()),
              "parent_cache_manifest_sha256": file_digest(parent_path),
              "source_receipt": str(source_receipt.resolve()), "source_receipt_sha256": file_digest(source_receipt),
              "source_root": str(source_root), "test_split": test_split,
              "source_manifest": str(source.resolve()), "source_manifest_sha256": file_digest(source),
              "class_names": manifest["class_names"], "cache_files": {"final.pt": file_digest(output / "final.pt")},
              "preparation_source_sha256": {name: file_digest(Path(__file__).with_name(name))
                                            for name in ("metadata_probe.py", "uatd_readout.py", "readout.py", "artifacts.py")}}
    write_json(output / "cache_manifest.json", result)
    return result


def load_paired_final_cache(manifest):
    parent_cache = Path(manifest["parent_cache"])
    parent_path = parent_cache / "cache_manifest.json"
    if file_digest(parent_path) != manifest["parent_cache_manifest_sha256"]:
        raise ValueError("final parent combined cache changed")
    parent = json.loads(parent_path.read_text())
    provenance = manifest["provenance"]
    if provenance != parent["provenance"] or manifest["class_names"] != parent["class_names"]:
        raise ValueError("final parent combined provenance/classes differ")
    receipt_path = Path(manifest["source_receipt"])
    if file_digest(receipt_path) != manifest["source_receipt_sha256"]:
        raise ValueError("original combined source receipt changed")
    source_root = verify_source_receipt(provenance, parent_cache, receipt_path)
    if str(source_root) != manifest["source_root"]:
        raise ValueError("original combined source root differs from receipt")
    verify_paired_provenance(provenance, source_root)
    prepared = json.loads((Path(provenance["metadata_provenance"]["prepared"]) / "manifest.json").read_text())
    source = Path(prepared["final_manifests"][manifest["test_split"]])
    if (source.resolve() != Path(manifest["source_manifest"]).resolve()
            or file_digest(source) != manifest["source_manifest_sha256"]
            or manifest["source_manifest_sha256"] != prepared["final_manifest_sha256"][manifest["test_split"]]):
        raise ValueError("final parent source differs from the frozen test split")
    caches = {}
    for name in ("image", "metadata"):
        directory = Path(manifest["parents"][name]["cache"])
        path = directory / "cache_manifest.json"
        if file_digest(path) != manifest["parents"][name]["cache_manifest_sha256"]:
            raise ValueError(f"final parent {name} manifest changed")
        final = json.loads(path.read_text())
        if (final["parent_cache_manifest_sha256"] != provenance[f"{name}_cache_manifest_sha256"]
                or final["provenance"] != provenance[f"{name}_provenance"]
                or final["test_split"] != manifest["test_split"]
                or final["class_names"] != manifest["class_names"]
                or Path(final["source_manifest"]).resolve() != source.resolve()
                or final["source_manifest_sha256"] != manifest["source_manifest_sha256"]):
            raise ValueError(f"final parent {name} differs from the frozen task/split/cache")
        if name == "metadata" and (
                Path(final["source_receipt"]).resolve() != Path(provenance["metadata_source_receipt"]).resolve()
                or final["source_receipt_sha256"] != provenance["metadata_source_receipt_sha256"]
                or final["source_root"] != provenance["metadata_source_root"]):
            raise ValueError("final parent metadata source receipt differs")
        if file_digest(directory / "final.pt") != final["cache_files"]["final.pt"]:
            raise ValueError(f"final parent {name} tensor changed")
        cache = torch.load(directory / "final.pt", map_location="cpu", weights_only=True, mmap=True)
        if cache["provenance"] != final["provenance"]:
            raise ValueError(f"final parent {name} tensor provenance differs")
        caches[name] = cache
    image, metadata = caches["image"], caches["metadata"]
    if image["row_ids"] != metadata["row_ids"]:
        raise ValueError("final row IDs differ between image and metadata")
    count = len(image["row_ids"])
    if (image["features"].shape != (count, 384, 16, 16) or metadata["features"].shape != (count, 8, 16, 16)
            or image["features"].dtype != torch.float32 or metadata["features"].dtype != torch.float32):
        raise ValueError("final parent feature shape/dtype differs from frozen inputs")
    if len(image["targets"]) != count or len(metadata["targets"]) != count:
        raise ValueError("final targets count differs")
    for image_target, metadata_target in zip(image["targets"], metadata["targets"]):
        if any(not torch.equal(image_target[key], metadata_target[key]) for key in ("labels", "boxes")):
            raise ValueError("final targets differ between image and metadata")
    image["metadata"] = metadata["features"]
    return image


def prepare_paired_final(parent_cache, source_receipt, image_final, metadata_final, output):
    if (output / "cache_manifest.json").exists():
        raise FileExistsError("paired final cache already exists")
    parent = json.loads((parent_cache / "cache_manifest.json").read_text())
    image = json.loads((image_final / "cache_manifest.json").read_text())
    receipt = json.loads(source_receipt.read_text())
    result = {"provenance": parent["provenance"], "parent_cache": str(parent_cache.resolve()),
              "parent_cache_manifest_sha256": file_digest(parent_cache / "cache_manifest.json"),
              "source_receipt": str(source_receipt.resolve()), "source_receipt_sha256": file_digest(source_receipt),
              "source_root": str(Path(receipt["source_root"]).resolve()),
              "test_split": image["test_split"], "source_manifest": image["source_manifest"],
              "source_manifest_sha256": image["source_manifest_sha256"], "class_names": parent["class_names"],
              "parents": {name: {"cache": str(directory.resolve()),
                                  "cache_manifest_sha256": file_digest(directory / "cache_manifest.json")}
                          for name, directory in (("image", image_final), ("metadata", metadata_final))},
              "preparation_source_sha256": {name: file_digest(Path(__file__).with_name(name))
                                            for name in ("metadata_probe.py", "readout.py", "artifacts.py")}}
    result["count"] = len(load_paired_final_cache(result)["row_ids"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cache_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="cache only prepared train/validation calibration descriptors")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    pair = commands.add_parser("pair", help="reference frozen image/metadata train/validation caches")
    pair.add_argument("--protocol", type=Path, required=True)
    pair.add_argument("--metadata-source-receipt", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("extract-final", help="explicitly open one test annotation list for calibration descriptors; no pixels")
    final.add_argument("--cache", type=Path, required=True)
    final.add_argument("--source-receipt", type=Path, required=True)
    final.add_argument("--test-split", choices=("test1", "test2"), required=True)
    final.add_argument("--output", type=Path, required=True)
    paired_final = commands.add_parser("pair-final", help="reference aligned image/metadata final caches for the original combined heads")
    paired_final.add_argument("--parent-cache", type=Path, required=True)
    paired_final.add_argument("--source-receipt", type=Path, required=True)
    paired_final.add_argument("--image-final", type=Path, required=True)
    paired_final.add_argument("--metadata-final", type=Path, required=True)
    paired_final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "prepare":
        result = prepare_cache(args.protocol, args.output)
    elif args.command == "pair":
        result = prepare_paired_cache(args.protocol, args.metadata_source_receipt, args.output)
    elif args.command == "extract-final":
        result = extract_final(args.cache, args.source_receipt, args.test_split, args.output)
    else:
        result = prepare_paired_final(args.parent_cache, args.source_receipt, args.image_final, args.metadata_final, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
