"""Frozen same-pose configuration diagnostics on calibrated polar token fields."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from .data import SonarDataset
from .model import build_encoder
from .readout import state_digest, write_json
from .train import file_digest, manifest_digest


def select_frames(manifest, split, configs, limit_poses, role_protocol=None):
    if split == "development":
        seen = {"C00", "C01", "C03", "C05", "C07", "C12"}
        exposed = set(role_protocol["development_exposed_configs"])
        rows = [row for row in manifest["frames"] if row["split"] == "validation" and
                ((row["config_id"] in seen and row["usage"] == "ssl_validation") or
                 (row["config_id"] in exposed and row["usage"] == "unused"))]
        declared = role_protocol["development_sample_ids"]
        if len(declared) != len(rows) or set(declared) != {row["sample_id"] for row in rows}:
            raise ValueError("development membership differs from the approved validation overlay")
    else:
        usages = {"ssl_validation"} if split == "validation" else {
            "test_seen", "test_unseen_single", "test_unseen_combination", "test_reference"}
        rows = [row for row in manifest["frames"] if row["split"] == split and row["usage"] in usages]
    available = {row["config_id"] for row in rows}
    if not set(configs) <= available:
        raise ValueError(f"configurations are not allowed in {split}: {set(configs) - available}")
    poses = sorted({row["pose_id"] for row in rows})
    if limit_poses is not None:
        if limit_poses < 1:
            raise ValueError("limit-poses must be positive")
        poses = poses[:limit_poses]
    membership = set(poses)
    return sorted([row for row in rows if row["pose_id"] in membership and row["config_id"] in configs],
                  key=lambda row: (row["pose_id"], row["config_id"]))


def token_geometry(sample):
    axes = [F.interpolate(sample[key].double()[None, None], size=16, mode="linear",
                          align_corners=False).flatten() for key in ("range_m", "bearing_rad")]
    # Track positive interpolation support of invalid pixels through resize and projection.
    invalid = F.interpolate((~sample["valid"])[None, None].float(), size=(224, 224),
                            mode="bilinear", align_corners=False, antialias=True)
    valid = F.max_pool2d(invalid, 14, 14)[0, 0] == 0
    return *axes, valid


def axis_cell(axis, values):
    sign = 1 if axis[-1] > axis[0] else -1
    ordered = axis * sign
    index = (torch.searchsorted(ordered, (values * sign).contiguous(), right=True) - 1).clamp(0, len(axis) - 2)
    weight = (values - axis[index]) / (axis[index + 1] - axis[index])
    inside = (values * sign >= ordered[0]) & (values * sign <= ordered[-1])
    return index, weight, inside


def interpolation_stencil(ranges, bearings, valid, points):
    i, u, inside_r = axis_cell(ranges, points[:, 0])
    j, v, inside_a = axis_cell(bearings, points[:, 1])
    usable = inside_r & inside_a & valid[i, j] & valid[i + 1, j] & valid[i, j + 1] & valid[i + 1, j + 1]
    return i, j, u, v, usable


def bilinear_sample(field, ranges, bearings, valid, points):
    i, j, u, v, usable = interpolation_stencil(ranges, bearings, valid, points)
    u, v = u[:, None], v[:, None]
    values = ((1 - u) * (1 - v) * field[i, j] + u * (1 - v) * field[i + 1, j]
              + (1 - u) * v * field[i, j + 1] + u * v * field[i + 1, j + 1])
    return values, usable


def retrieval_metrics(query, gallery, query_poses, gallery_poses):
    similarity = F.normalize(query.double(), dim=-1) @ F.normalize(gallery.double(), dim=-1).T
    order = similarity.argsort(dim=1, descending=True, stable=True)
    lookup = {pose: index for index, pose in enumerate(gallery_poses)}
    result = []
    for index, pose in enumerate(query_poses):
        positive = lookup[pose]
        rank = int((order[index] == positive).nonzero()[0, 0]) + 1
        result.append({"pose_id": pose, "rank": rank, "top1": int(rank == 1),
                       "top5": int(rank <= min(5, len(gallery_poses))),
                       "reciprocal_rank": 1 / rank, "positive_cosine": float(similarity[index, positive])})
    return result


def dense_metrics(query_tokens, query_geometry, query_native, reference_tokens, reference_geometry, reference_native):
    bounds = [(max(float(q.min()), float(r.min())), min(float(q.max()), float(r.max())))
              for q, r in zip(query_geometry[:2], reference_geometry[:2])]
    if any(lower >= upper for lower, upper in bounds):
        raise ValueError("configurations have no two-dimensional token-center overlap")
    r, a = torch.meshgrid(*(torch.linspace(lower, upper, 16, dtype=torch.double) for lower, upper in bounds), indexing="ij")
    points = torch.stack([r.flatten(), a.flatten()], dim=1)
    query, query_valid = bilinear_sample(query_tokens, *query_geometry, points)
    reference, reference_valid = bilinear_sample(reference_tokens, *reference_geometry, points)
    usable = query_valid & reference_valid
    for sample in (query_native, reference_native):
        usable &= interpolation_stencil(sample["range_m"].double(), sample["bearing_rad"].double(),
                                         sample["valid"], points)[-1]
    count = int(usable.sum())
    if count == 0:
        raise ValueError("no common query points have complete valid interpolation support")
    query = F.normalize(query[usable].double(), dim=-1)
    reference = F.normalize(reference[usable].double(), dim=-1)
    points = points[usable]
    xy = points[:, :1] * torch.stack([points[:, 1].cos(), points[:, 1].sin()], dim=1)
    similarities = query @ reference.T
    forward = (xy - xy[similarities.argmax(dim=1)]).norm(dim=1)
    reverse = (xy - xy[similarities.argmax(dim=0)]).norm(dim=1)
    return {"candidate_points": 256, "valid_points": count, "coverage": count / 256,
            "range_bounds_m": list(bounds[0]), "bearing_bounds_rad": list(bounds[1]),
            "cosine": float((query * reference).sum(-1).mean()),
            "forward_error_m": float(forward.mean()), "reverse_error_m": float(reverse.mean()),
            "bidirectional_error_m": float((forward.mean() + reverse.mean()) / 2)}


def summarize(per_pose, configs, split, role_protocol):
    per_config = {}
    for config in configs:
        rows = [row for row in per_pose if row["config_id"] == config]
        metrics = {"top1": [row["global"]["top1"] for row in rows],
                   "top5": [row["global"]["top5"] for row in rows],
                   "mrr": [row["global"]["reciprocal_rank"] for row in rows]}
        metrics.update({key: [row["dense"][key] for row in rows]
                        for key in ("cosine", "forward_error_m", "reverse_error_m", "bidirectional_error_m", "coverage")})
        per_config[config] = {key: sum(values) / len(values) for key, values in metrics.items()}
        per_config[config].update(poses=len(rows), valid_points=sum(row["dense"]["valid_points"] for row in rows),
                                  candidate_points=256 * len(rows))
    keys = ("top1", "top5", "mrr", "cosine", "forward_error_m", "reverse_error_m", "bidirectional_error_m", "coverage")
    groups = {}
    roles = [("seen", {"C01", "C03", "C05", "C07", "C12"})]
    single, combination = {"C02", "C04", "C06", "C08", "C13"}, {"C09", "C10"}
    if split == "development":
        exposed = set(role_protocol["development_exposed_configs"])
        roles.extend([("development_single", exposed & single), ("development_combination", exposed & combination)])
    elif split == "test":
        exposed = set(role_protocol["development_exposed_configs"])
        unseen = set(role_protocol["development_unseen_configs"])
        roles.extend([("development_exposed_single", exposed & single),
                      ("development_exposed_combination", exposed & combination),
                      ("development_unseen_single", unseen & single),
                      ("development_unseen_combination", unseen & combination)])
    for role, membership in roles:
        compared = sorted(set(configs) & membership)
        if compared:
            groups[role] = {
                "configs": compared,
                "macro": {key: sum(per_config[config][key] for config in compared) / len(compared) for key in keys},
                "worst": {key: (max if key.endswith("error_m") else min)(per_config[config][key] for config in compared) for key in keys}}
    return {"per_config": per_config, "configuration_groups": groups,
            "references": "C00 self-check and C11 external reference are only reported individually, excluded from all macro/worst groups."}


@torch.inference_mode()
def run_evaluation(checkpoint_path, manifest_path, data_root, normalization_path, protocol_path,
                   output_path, variant, split, configs, device, limit_poses=None, role_protocol_path=None):
    configs = sorted(set(configs) | {"C00"})
    if len(configs) < 2:
        raise ValueError("include at least one query configuration other than C00")
    manifest = json.loads(manifest_path.read_text())
    normalization = json.loads(normalization_path.read_text())
    if normalization["source_manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("normalization provenance does not match manifest")
    if split in ("development", "test") and role_protocol_path is None:
        raise ValueError("a role protocol is required for development and final test")
    role_protocol = None
    if role_protocol_path is not None:
        role_protocol = json.loads(role_protocol_path.read_text())
        if role_protocol["source_manifest_sha256"] != file_digest(manifest_path):
            raise ValueError("role protocol source manifest differs from the evaluation manifest")
        if (set(role_protocol["development_exposed_configs"]) != {"C02", "C04", "C06", "C10"} or
                set(role_protocol["development_unseen_configs"]) != {"C08", "C09", "C13"}):
            raise ValueError("role protocol differs from the approved configuration roles")
    frames = select_frames(manifest, split, configs, limit_poses, role_protocol)
    poses = sorted({row["pose_id"] for row in frames})
    if not frames or len(frames) != len(poses) * len(configs):
        raise ValueError("selected poses do not have exactly one frame for every selected configuration")
    checkpoint_sha = file_digest(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if file_digest(checkpoint_path) != checkpoint_sha:
        raise ValueError("checkpoint changed while loading")
    config = saved["resolved_config"]
    if "variant" in config:
        if config["variant"] != variant:
            raise ValueError("explicit variant differs from checkpoint variant")
    elif variant != "R010":
        raise ValueError("only the original R010 checkpoint permits a missing variant field")
    for key, path in (("manifest", manifest_path), ("normalization", normalization_path)):
        if saved["provenance"][key] != file_digest(path):
            raise ValueError(f"checkpoint {key} provenance differs from evaluation input")
    if file_digest(config["weights_path"]) != saved["provenance"]["weights"]:
        raise ValueError("foundation weights differ from checkpoint provenance")
    encoder = build_encoder(Path(config["code_dir"]), Path(config["weights_path"]), variant)
    encoder.load_state_dict(saved["student"])
    encoder.eval().requires_grad_(False).to(device)
    before = state_digest(encoder.state_dict())
    dataset = SonarDataset(manifest_path, data_root, "ssl_validation" if split in ("validation", "development") else "test_seen")
    dataset.frames = frames
    features = {config_id: [] for config_id in configs}
    per_pose, sources = [], []
    metadata_hashes = {}
    for start in range(0, len(frames), len(configs)):
        samples = [dataset[index] for index in range(start, start + len(configs))]
        for sample in samples:
            path = data_root / sample["npz_relpath"]
            digest = file_digest(path)
            if digest != sample["sha256"]:
                raise ValueError(f"native frame differs from frozen manifest: {path}")
            sources.append({"sample_id": sample["sample_id"], "npz_relpath": sample["npz_relpath"], "sha256": digest})
            metadata_path = data_root / sample["metadata_relpath"]
            metadata_hashes[str(metadata_path)] = file_digest(metadata_path)
        output = encoder.forward_native(samples, normalization, None)
        patches = output["patch_tokens"].float().cpu().reshape(len(samples), 16, 16, -1)
        geometry = [token_geometry(sample) for sample in samples]
        reference = configs.index("C00")
        for index, sample in enumerate(samples):
            features[sample["config_id"]].append(patches[index].mean((0, 1)))
            per_pose.append({"pose_id": sample["pose_id"], "group_id": sample["group_id"],
                             "scene_id": sample["scene_id"], "config_id": sample["config_id"],
                             "dense": dense_metrics(patches[index], geometry[index], sample,
                                                    patches[reference], geometry[reference], samples[reference])})
    gallery = torch.stack(features["C00"])
    global_results = {}
    for config_id in configs:
        for row in retrieval_metrics(torch.stack(features[config_id]), gallery, poses, poses):
            global_results[(row["pose_id"], config_id)] = row
    for row in per_pose:
        row["global"] = global_results[(row["pose_id"], row["config_id"])]
    after = state_digest(encoder.state_dict())
    if before != after:
        raise RuntimeError("encoder changed during frozen evaluation")
    source_dir = Path(__file__).parent
    report = {"split": split, "configs": configs, "gallery_config": "C00", "gallery_poses": poses,
              "gallery_size": len(poses), "limit_poses": limit_poses, "final_test_evaluated": split == "test",
              "summary": summarize(per_pose, configs, split, role_protocol), "per_pose": per_pose,
              "limitations": ["Common domain is a rectangle in calibrated r/theta coordinates, not an XY convex hull.",
                              "Dense metrics measure same-pose representation consistency, not a semantic dense task.",
                              "Same-map groups are acquisition labels, not independent scene identities.",
                              "Nominal token coordinates do not localize transformer receptive fields; uniform r/theta sampling is not equal-area XY."],
              "provenance": {"variant": variant, "encoder_weights": "student", "checkpoint_step": saved["step"],
                             "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha,
                             "checkpoint_training_provenance": saved["provenance"],
                             "encoder_hash_before": before, "encoder_hash_after": after,
                             "manifest": str(manifest_path.resolve()), "manifest_sha256": file_digest(manifest_path),
                             "normalization": str(normalization_path.resolve()), "normalization_sha256": file_digest(normalization_path),
                             "protocol": str(protocol_path.resolve()), "protocol_sha256": file_digest(protocol_path),
                             "role_protocol": str(role_protocol_path.resolve()) if role_protocol_path is not None else None,
                             "role_protocol_sha256": file_digest(role_protocol_path) if role_protocol_path is not None else None,
                             "data_root": str(data_root.resolve()), "selected_sources": sources, "metadata_sha256": metadata_hashes,
                             "source_sha256": {name: file_digest(source_dir / name) for name in
                                               ("config_eval.py", "data.py", "model.py", "frontends.py", "readout.py", "train.py", "artifacts.py")},
                             "feature_dtype": "float32", "metric_dtype": "float64", "device": str(device),
                             "python": sys.version, "torch": str(torch.__version__)}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "data-root", "normalization", "protocol", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=("R010", "R011", "R012", "R013", "R020", "R021", "R030", "R031", "R032", "R036", "R037", "R038", "R039"))
    parser.add_argument("--split", required=True, choices=("validation", "development", "test"))
    parser.add_argument("--role-protocol", type=Path, help="approved membership/exposure overlay; required for development and test")
    parser.add_argument("--configs", required=True, nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-poses", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = run_evaluation(args.checkpoint, args.manifest, args.data_root, args.normalization,
                            args.protocol, args.output, args.variant, args.split, args.configs,
                            args.device, args.limit_poses, args.role_protocol)
    print(json.dumps({"output": str(args.output), "gallery_size": result["gallery_size"],
                      "split": result["split"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
