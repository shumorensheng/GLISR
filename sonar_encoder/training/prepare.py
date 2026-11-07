"""Portable references and train-only statistics for the fixed real-data snapshot."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


CONFIG_USAGE = {
    "C00": "seen", "C01": "seen", "C02": "unseen_single", "C03": "seen",
    "C04": "unseen_single", "C05": "seen", "C06": "unseen_single", "C07": "seen",
    "C08": "unseen_single", "C09": "unseen_combination", "C10": "unseen_combination",
    "C11": "reference", "C12": "seen", "C13": "unseen_single",
}


def build_manifest(index_path: Path, assignment: dict[str, str], data_root: Path) -> dict:
    source_bytes = index_path.read_bytes()
    records = json.loads(source_bytes)
    groups = {record["group_id"] for record in records}
    if groups != set(assignment):
        raise ValueError(f"Index and assignment group mismatch: {groups ^ set(assignment)}")
    if not set(assignment.values()) <= {"train", "validation", "test"}:
        raise ValueError("Assignment split must be train, validation, or test")

    pose_assignments = {}
    frames = []
    for record in records:
        source_pose_file = record["source_pose_file"]
        source_index = record["source_pose_index_zero_based"]
        pose_id = hashlib.sha256(f"{source_pose_file}\n{source_index}".encode("utf-8")).hexdigest()
        split = assignment[record["group_id"]]
        pose_assignments[pose_id] = {
            "source_pose_file": source_pose_file,
            "source_pose_index_zero_based": source_index,
            "group_id": record["group_id"],
            "scene_id": record["scene_id"],
            "split": split,
        }
        anchor = Path(record["anchor"]).relative_to(Path(source_pose_file).parent)
        metadata_relpath = anchor / "metadata.json"
        metadata = json.loads((data_root / metadata_relpath).read_text())
        configurations = {frame["config_id"].removeprefix("CPU_") for frame in metadata["frames"].values()}
        if configurations != set(CONFIG_USAGE) or len(metadata["frames"]) != 14:
            raise ValueError(f"Expected 14 configurations at {metadata_relpath}")
        for frame_key, frame in sorted(metadata["frames"].items()):
            config_id = frame["config_id"].removeprefix("CPU_")
            role = CONFIG_USAGE[config_id]
            if split == "test":
                usage = f"test_{role}"
            elif role == "seen":
                usage = f"ssl_{split}"
            else:
                usage = "unused"
            frames.append({
                "sample_id": f"{pose_id}/{frame_key}",
                "pose_id": pose_id,
                "group_id": record["group_id"],
                "scene_id": record["scene_id"],
                "split": split,
                "usage": usage,
                "config_id": config_id,
                "npz_relpath": (anchor / frame["file"]).as_posix(),
                "sha256": frame["sha256"],
                "metadata_relpath": metadata_relpath.as_posix(),
            })
    return {
        "source_index_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "pose_assignments": pose_assignments,
        "frames": frames,
        "counts": {
            "poses": len(records), "frames": len(frames), "groups": len(groups),
            "poses_by_split": dict(Counter(pose["split"] for pose in pose_assignments.values())),
            "groups_by_split": dict(Counter(assignment.values())),
            "frames_by_usage": dict(Counter(frame["usage"] for frame in frames)),
        },
    }


def fit_normalization(manifest: dict, data_root: Path) -> dict:
    count, mean, m2 = 0, 0.0, 0.0
    for frame in manifest["frames"]:
        if frame["usage"] != "ssl_train":
            continue
        with np.load(data_root / frame["npz_relpath"], allow_pickle=False) as arrays:
            values = np.log1p(arrays["intensity"][arrays["intensity_valid_region_mask"]].astype(np.float64))
        batch_count = values.size
        batch_mean = values.mean()
        delta = batch_mean - mean
        combined_count = count + batch_count
        # Merge pixel moments without retaining previous frames or averaging per-frame statistics.
        m2 += np.square(values - batch_mean).sum() + delta * delta * count * batch_count / combined_count
        mean += delta * batch_count / combined_count
        count = combined_count
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "count": count, "mean": float(mean), "std": float(np.sqrt(m2 / count)),
        "source_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.assignment.read_text())
    if hashlib.sha256(args.index.read_bytes()).hexdigest() != config["source_index_sha256"]:
        raise ValueError("Canonical source index SHA-256 differs from the configured snapshot")
    manifest = build_manifest(args.index, config["assignment"], args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
