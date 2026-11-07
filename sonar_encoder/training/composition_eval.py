"""Empirical B5 feature correspondence composition on a shared physical domain."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from .config_eval import bilinear_sample, interpolation_stencil, select_frames, token_geometry
from .data import SonarDataset
from .model import build_encoder
from .readout import state_digest, write_json
from .train import file_digest, manifest_digest


DEVELOPMENT_TRIADS = (
    ("C00", "C01", "C02"), ("C00", "C03", "C04"), ("C00", "C05", "C06"),
    ("C00", "C02", "C10"), ("C00", "C04", "C10"), ("C00", "C05", "C10"), ("C00", "C07", "C10"),
    ("C00", "C02", "C12"),
)
FINAL_TRIADS = DEVELOPMENT_TRIADS[:7] + (
    ("C00", "C07", "C08"), ("C00", "C02", "C09"), ("C00", "C04", "C09"),
    ("C00", "C06", "C09"), ("C00", "C08", "C09"),
    ("C00", "C02", "C12"), ("C00", "C12", "C13"),
)
SOURCE_FILES = ("composition_eval.py", "config_eval.py", "model.py", "frontends.py",
                "data.py", "readout.py", "train.py")
MEAN_METRICS = ("composition_error_m", "direct_error_m", "composed_error_m", "coverage",
                "ab_unique_target_fraction", "bc_unique_target_fraction",
                "ac_unique_target_fraction", "composed_unique_target_fraction")
MAX_METRICS = ("composition_max_error_m", "direct_max_error_m", "composed_max_error_m")


def composition_metrics(fields, geometries, native_samples):
    bounds = [(max(float(geometry[axis].min()) for geometry in geometries),
               min(float(geometry[axis].max()) for geometry in geometries)) for axis in (0, 1)]
    if any(lower >= upper for lower, upper in bounds):
        raise ValueError("triad has no two-dimensional token-center overlap")
    r, a = torch.meshgrid(*(torch.linspace(lower, upper, 16, dtype=torch.double)
                           for lower, upper in bounds), indexing="ij")
    points = torch.stack([r.flatten(), a.flatten()], dim=1)
    usable = torch.ones(256, dtype=torch.bool)
    sampled = []
    for field, geometry, sample in zip(fields, geometries, native_samples):
        values, token_valid = bilinear_sample(field, *geometry, points)
        sampled.append(values)
        usable &= token_valid
        usable &= interpolation_stencil(sample["range_m"].double(), sample["bearing_rad"].double(),
                                         sample["valid"], points)[-1]
    count = int(usable.sum())
    if count == 0:
        raise ValueError("no common triad points have complete token and native interpolation support")
    features = [F.normalize(values[usable].double(), dim=-1) for values in sampled]
    points = points[usable]
    xy = points[:, :1] * torch.stack([points[:, 1].cos(), points[:, 1].sin()], dim=1)
    ab = (features[0] @ features[1].T).argmax(dim=1)
    bc = (features[1] @ features[2].T).argmax(dim=1)
    ac = (features[0] @ features[2].T).argmax(dim=1)
    composed = bc[ab]
    composition_error = (xy[composed] - xy[ac]).norm(dim=1)
    direct_error = (xy[ac] - xy).norm(dim=1)
    composed_error = (xy[composed] - xy).norm(dim=1)
    return {
        "candidate_points": 256, "valid_points": count, "coverage": count / 256,
        "range_bounds_m": list(bounds[0]), "bearing_bounds_rad": list(bounds[1]),
        "composition_error_m": float(composition_error.mean()),
        "composition_max_error_m": float(composition_error.max()),
        "direct_error_m": float(direct_error.mean()), "direct_max_error_m": float(direct_error.max()),
        "composed_error_m": float(composed_error.mean()), "composed_max_error_m": float(composed_error.max()),
        "ab_unique_target_fraction": ab.unique().numel() / count,
        "bc_unique_target_fraction": bc.unique().numel() / count,
        "ac_unique_target_fraction": ac.unique().numel() / count,
        "composed_unique_target_fraction": composed.unique().numel() / count,
    }


def _aggregate(rows):
    return {
        **{key: sum(row[key] for row in rows) / len(rows) for key in MEAN_METRICS},
        **{key: max(row[key] for row in rows) for key in MAX_METRICS},
        "pose_triads": len(rows), "poses": len({row["pose_id"] for row in rows}),
        "valid_points": sum(row["valid_points"] for row in rows), "candidate_points": 256 * len(rows),
    }


def summarize(per_pose, triads):
    per_triad = {"->".join(triad): _aggregate([row for row in per_pose if row["triad"] == "->".join(triad)])
                 for triad in triads}
    return {
        "per_triad": per_triad,
        "per_group": {group: _aggregate([row for row in per_pose if row["group_id"] == group])
                      for group in sorted({row["group_id"] for row in per_pose})},
        "macro": {key: sum(row[key] for row in per_triad.values()) / len(per_triad) for key in MEAN_METRICS},
        "worst_triad": {key: (max if key.endswith("_error_m") else min)(row[key] for row in per_triad.values())
                        for key in MEAN_METRICS},
        "maximum": {key: max(row[key] for row in per_triad.values()) for key in MAX_METRICS},
        "aggregation": "Equal pose weight within each triad; equal triad weight in macro. Groups are acquisition groups. Unique-target fractions describe many-to-one matching, not correctness.",
    }


@torch.inference_mode()
def run_evaluation(protocol_path, variant, device, output_path):
    protocol_hash = file_digest(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol["status"] != "FROZEN":
        raise ValueError("composition evaluation requires a FROZEN protocol")
    population = protocol["population"]
    split = population["split"]
    if split not in ("development", "test"):
        raise ValueError("composition population must be development or test")
    triads = protocol["triads"]
    expected = DEVELOPMENT_TRIADS if split == "development" else FINAL_TRIADS
    if triads != [list(triad) for triad in expected]:
        raise ValueError("composition triads differ from the fixed B5 definition")
    if set(protocol["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("composition protocol must lock all evaluator source files")
    source_dir = Path(__file__).parent
    for name, expected_hash in protocol["source_sha256"].items():
        if file_digest(source_dir / name) != expected_hash:
            raise ValueError(f"evaluator source changed: {name}")
    for record in protocol["input_files"].values():
        if file_digest(record["path"]) != record["sha256"]:
            raise ValueError(f"composition input changed: {record['path']}")
    manifest_path = Path(protocol["input_files"]["manifest"]["path"])
    normalization_path = Path(protocol["input_files"]["normalization"]["path"])
    manifest = json.loads(manifest_path.read_text())
    normalization = json.loads(normalization_path.read_text())
    role = json.loads(Path(protocol["input_files"]["role_protocol"]["path"]).read_text())
    if normalization["source_manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("normalization does not match composition manifest")
    if role["source_manifest_sha256"] != file_digest(manifest_path):
        raise ValueError("role protocol does not match composition manifest")
    if (set(role["development_exposed_configs"]) != {"C02", "C04", "C06", "C10"} or
            set(role["development_unseen_configs"]) != {"C08", "C09", "C13"}):
        raise ValueError("composition role protocol differs from approved configuration roles")
    frames = select_frames(manifest, split, population["configs"], None, role)
    membership = set(population["pose_ids"])
    frames = [row for row in frames if row["pose_id"] in membership]
    poses = sorted({row["pose_id"] for row in frames})
    counts = dict(Counter(row["config_id"] for row in frames))
    if (poses != population["pose_ids"] or len(poses) != population["pose_count"] or
            len(frames) != population["frame_count"] or counts != population["per_configuration_counts"] or
            [row["sample_id"] for row in frames] != population["sample_ids"] or
            len(frames) != len(poses) * len(population["configs"])):
        raise ValueError("composition population differs from the exact declared membership")
    needed = sorted({config for triad in triads for config in triad})
    if not set(needed) <= set(population["configs"]):
        raise ValueError("composition population omits a required triad configuration")
    if output_path.exists():
        raise FileExistsError(output_path)
    dataset = SonarDataset(manifest_path, Path(protocol["data_root"]),
                           "ssl_validation" if split == "development" else "test_seen")
    dataset.frames = [row for row in frames if row["config_id"] in needed]
    stages = {}
    for step in (0, 10000):
        checkpoint = protocol["checkpoints"][variant][str(step)]
        path = Path(checkpoint["path"])
        if file_digest(path) != checkpoint["sha256"]:
            raise ValueError(f"composition checkpoint changed: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        config = saved["resolved_config"]
        if (saved["step"] != step or config["variant"] != variant or
                config["seed"] != 0 or config["steps"] != 10000):
            raise ValueError("composition requires each own seed0 initial and fixed10000 student")
        for key, input_path in (("manifest", manifest_path), ("normalization", normalization_path)):
            if saved["provenance"][key] != file_digest(input_path):
                raise ValueError(f"checkpoint {key} differs from composition input")
        if file_digest(config["weights_path"]) != saved["provenance"]["weights"]:
            raise ValueError("foundation weights differ from checkpoint provenance")
        if step == 0:
            initial_config = config
            encoder = build_encoder(Path(config["code_dir"]), Path(config["weights_path"]), variant)
            encoder.eval().requires_grad_(False).to(device)
        elif config != initial_config:
            raise ValueError("composition initial and fixed10000 configurations differ")
        encoder.load_state_dict(saved["student"])
        before = state_digest(encoder.state_dict())
        training_provenance = saved["provenance"]
        del saved
        per_pose, metadata_hashes = [], {}
        for start in range(0, len(dataset), len(needed)):
            samples = [dataset[index] for index in range(start, start + len(needed))]
            for sample in samples:
                native_path = Path(protocol["data_root"]) / sample["npz_relpath"]
                if file_digest(native_path) != sample["sha256"]:
                    raise ValueError(f"native composition frame changed: {native_path}")
                metadata_path = Path(protocol["data_root"]) / sample["metadata_relpath"]
                metadata_hashes[str(metadata_path)] = file_digest(metadata_path)
            patches = encoder.forward_native(samples, normalization, None)["patch_tokens"].float().cpu()
            fields = dict(zip(needed, patches.reshape(len(needed), 16, 16, -1)))
            native = dict(zip(needed, samples))
            geometry = {key: token_geometry(native[key]) for key in needed}
            for triad in triads:
                metrics = composition_metrics([fields[key] for key in triad], [geometry[key] for key in triad],
                                              [native[key] for key in triad])
                per_pose.append({"pose_id": samples[0]["pose_id"], "group_id": samples[0]["group_id"],
                                 "triad": "->".join(triad), **metrics})
        after = state_digest(encoder.state_dict())
        if before != after or file_digest(path) != checkpoint["sha256"]:
            raise RuntimeError("encoder or checkpoint changed during composition evaluation")
        if step == 10000 and metadata_hashes != stages["0"]["provenance"]["metadata_sha256"]:
            raise RuntimeError("native metadata changed between composition stages")
        stages[str(step)] = {
            "per_pose": per_pose, "summary": summarize(per_pose, triads),
            "provenance": {"checkpoint": str(path.resolve()), "checkpoint_sha256": checkpoint["sha256"],
                           "checkpoint_step": step, "encoder_weights": "student",
                           "encoder_hash_before": before, "encoder_hash_after": after,
                           "checkpoint_training_provenance": training_provenance,
                           "metadata_sha256": metadata_hashes},
        }
    if file_digest(protocol_path) != protocol_hash:
        raise RuntimeError("composition protocol changed during evaluation")
    for name, expected_hash in protocol["source_sha256"].items():
        if file_digest(source_dir / name) != expected_hash:
            raise RuntimeError(f"evaluator source changed during composition evaluation: {name}")
    for record in protocol["input_files"].values():
        if file_digest(record["path"]) != record["sha256"]:
            raise RuntimeError(f"composition input changed during evaluation: {record['path']}")
    report = {
        "variant": variant, "split": split, "triads": triads, "stages": stages,
        "population": population, "encoded_configs": needed, "encoded_frames_per_stage": len(dataset),
        "final_test_evaluated": split == "test",
        "provenance": {"protocol": str(protocol_path.resolve()), "protocol_sha256": protocol_hash,
                       "input_files": protocol["input_files"], "source_sha256": protocol["source_sha256"],
                       "feature_dtype": "float32", "geometry_and_bilinear_weights_dtype": "float64",
                       "interpolated_feature_and_metric_dtype": "float64", "argmax_ties": "first candidate",
                       "device": str(device), "python": sys.version, "torch": str(torch.__version__)},
        "limitations": ["Empirical feature correspondence diagnostic, not an exact group-equivariance theorem.",
                        "Composition zero alone does not establish localization: report direct/composed truth error and unique-target fractions together.",
                        "The shared domain is uniform in clean r/theta, not equal-area XY; nominal token centers do not localize transformer receptive fields.",
                        "Long-range triads measure only the common near-range bounds reported per pose; they do not establish localization across the full 120 m domain.",
                        "Groups are acquisition groups; no independent-scene confidence interval is claimed."],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", choices=("R011", "R012", "R020", "R021"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = run_evaluation(args.protocol, args.variant, args.device, args.output)
    print(json.dumps({"output": str(args.output), "variant": report["variant"], "split": report["split"],
                      "stages": {step: stage["summary"] for step, stage in report["stages"].items()}}, indent=2))


if __name__ == "__main__":
    main()
