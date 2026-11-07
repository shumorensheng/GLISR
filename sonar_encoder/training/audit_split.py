"""Read-only exact/near-duplicate evidence for the fixed real-data split."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw


AUDIT_CONFIGS = ("C00", "C12", "C13")
NEAR_CONSTANT_STD = 1e-6


def area_descriptor(intensity):
    height, width = intensity.shape
    return np.log1p(intensity).reshape(64, height // 64, 64, width // 64).mean(axis=(1, 3)).ravel()


def pose_difference(left, right):
    distance = np.linalg.norm(left[:3, 3] - right[:3, 3])
    cosine = (np.trace(left[:3, :3].T @ right[:3, :3]) - 1) / 2
    return float(distance), float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def cross_split_candidates(rows, descriptors, poses):
    centered = descriptors - descriptors.mean(axis=1, keepdims=True)
    informative = descriptors.std(axis=1) > NEAR_CONSTANT_STD
    normalized = np.zeros_like(centered)
    normalized[informative] = centered[informative] / np.linalg.norm(centered[informative], axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    splits = np.array([row["split"] for row in rows])
    cross_split = splits[:, None] != splits[None, :]
    usable = cross_split & informative[:, None] & informative[None, :]
    similarity[~usable] = -np.inf
    ranking = np.argsort(-similarity, axis=1, kind="stable")[:, :3]
    retrieval = []
    for source_index, neighbors in enumerate(ranking):
        for rank, candidate_index in enumerate(neighbors, 1):
            if not usable[source_index, candidate_index]:
                continue
            distance, angle = pose_difference(poses[source_index], poses[candidate_index])
            retrieval.append({"source_index": source_index, "candidate_index": int(candidate_index), "rank": rank,
                              "cosine_similarity": float(similarity[source_index, candidate_index]),
                              "translation_m": distance, "rotation_deg": angle})
    translations = poses[:, :3, 3]
    distances = np.linalg.norm(translations[:, None] - translations[None, :], axis=2)
    close_pairs = []
    for source_index, candidate_index in np.argwhere(np.triu(cross_split & (distances < 2), k=1)):
        distance, angle = pose_difference(poses[source_index], poses[candidate_index])
        if angle < 20:
            score = float(similarity[source_index, candidate_index]) if usable[source_index, candidate_index] else None
            close_pairs.append({"source_index": int(source_index), "candidate_index": int(candidate_index),
                                "cosine_similarity": score, "translation_m": distance, "rotation_deg": angle})
    return retrieval, close_pairs


def exact_duplicate_groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return [{"sha256": digest, "sample_ids": [row["sample_id"] for row in group],
             "splits": sorted({row["split"] for row in group})}
            for digest, group in groups.items() if len({row["split"] for row in group}) > 1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def render_contact_sheets(queue, data_root, output, config):
    sheet_names = []
    for start in range(0, len(queue), 10):
        page = queue[start:start + 10]
        canvas = Image.new("RGB", (1320, 330 * len(page)), "white")
        draw = ImageDraw.Draw(canvas)
        for line, pair in enumerate(page):
            y = line * 330
            score = pair["cosine_similarity"]
            score_text = "undefined" if score is None else f"{score:.6f}"
            draw.text((10, y + 3), f'{pair["pair_id"]} cosine={score_text} distance={pair["translation_m"]:.3f}m rotation={pair["rotation_deg"]:.2f}deg', fill="black")
            for side, x in (("left", 10), ("right", 665)):
                row = pair[side]
                with np.load(data_root / row["npz_relpath"]) as archive:
                    intensity = archive["intensity"]
                # Shared display scale preserves weak/empty returns across both frames.
                pixels = (np.clip(np.log1p(intensity) / np.log(3), 0, 1) * 255).astype(np.uint8)
                picture = Image.fromarray(pixels)
                picture.thumbnail((260, 260), Image.Resampling.BOX)
                canvas.paste(picture, (x, y + 56))
                draw.text((x, y + 20), f'{row["group_id"]} / {row["split"]}', fill="black")
                draw.text((x, y + 36), f'{Path(row["npz_relpath"]).parent.name} std={row["descriptor_std"]:.6f} hits={row["hit_count"]}', fill="black")
                draw.text((x + 270, y + 65), row["pose_id"][:24], fill="black")
                draw.text((x + 270, y + 84), ", ".join(pair["review_reasons"]), fill="black")
                draw.text((x + 270, y + 103), "Original range/bearing matrix", fill="black")
                draw.text((x + 270, y + 122), "Shared log1p scale: [0, log(3)]", fill="black")
        name = f"{config}_review_{start // 10 + 1:03d}.png"
        canvas.save(output / name)
        sheet_names.append(name)
    return sheet_names


def run_audit(manifest_path, data_root, output):
    started = time.monotonic()
    manifest = json.loads(manifest_path.read_text())
    frames = manifest["frames"]
    output.mkdir(parents=True, exist_ok=True)
    pixel_rows = []
    selected = {config: [] for config in AUDIT_CONFIGS}
    descriptors = {config: [] for config in AUDIT_CONFIGS}
    previous_metadata_path = None
    for index, frame in enumerate(frames):
        path = data_root / frame["npz_relpath"]
        with np.load(path) as archive:
            intensity = archive["intensity"]
        digest = hashlib.sha256()
        digest.update(str(intensity.dtype).encode())
        digest.update(json.dumps(list(intensity.shape)).encode())
        digest.update(intensity.tobytes(order="C"))
        pixel_rows.append({"sample_id": frame["sample_id"], "split": frame["split"], "sha256": frame["sha256"],
                           "pixel_sha256": digest.hexdigest(), "shape": list(intensity.shape), "dtype": str(intensity.dtype)})
        config = frame["config_id"]
        if config in AUDIT_CONFIGS:
            metadata_path = frame["metadata_relpath"]
            if metadata_path != previous_metadata_path:
                metadata_frames = json.loads((data_root / metadata_path).read_text())["frames"]
                previous_metadata_path = metadata_path
            metadata = metadata_frames[path.stem]
            descriptor = area_descriptor(intensity)
            selected[config].append({**frame, "sonar_pose_client_4x4": metadata["sonar_pose_client_4x4"],
                                     "hit_count": metadata["quality"]["hit_count"], "descriptor_std": float(descriptor.std())})
            descriptors[config].append(descriptor)
        if (index + 1) % 1000 == 0:
            print(f"intensity hashes {index + 1}/{len(frames)}, elapsed {time.monotonic() - started:.1f}s", flush=True)
    with (output / "pixel_hashes.jsonl").open("w") as stream:
        for row in pixel_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    exact = {"file": exact_duplicate_groups(pixel_rows, "sha256"), "pixel": exact_duplicate_groups(pixel_rows, "pixel_sha256")}
    write_json(output / "exact_duplicates.json", exact)
    write_json(output / "descriptor_frames.json", selected)
    np.savez_compressed(output / "descriptors.npz", **{config: np.asarray(values) for config, values in descriptors.items()})
    report = {"manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "manifest_hash": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
              "frame_count": len(frames), "exact_file_cross_split_groups": len(exact["file"]),
              "exact_pixel_cross_split_groups": len(exact["pixel"]), "manual_review_status": "pending",
              "file_hash_source": "manifest sha256; reuse input integrity verification", "configurations": {},
              "protocol": {"descriptor": "log1p intensity; disjoint 64x64 area means; centered cosine",
                           "pixel_hash": "SHA256(dtype string + JSON shape + C-order unmodified intensity bytes)",
                           "retrieval": "top 3 other-split frames in the same configuration; undefined constant cosines excluded",
                           "near_constant_std": NEAR_CONSTANT_STD,
                           "empty_return_flag": "metadata quality.hit_count == 0; audit annotation only",
                           "low_variance_listing": "bottom 10% descriptor std per configuration; descriptive, not a removal or binding rule",
                           "review_queue": "unique top 50 similarity pairs per configuration plus all translation <2 m and full rotation <20 deg pairs",
                           "rotation": "acos((trace(R_left.T @ R_right)-1)/2), clipped for floating-point roundoff",
                           "display": "original range/bearing rectangle, log1p intensity / log(3), shared [0,1] grayscale",
                           "decision": "No samples removed or groups reassigned; no manual review has been performed"}}
    for config in AUDIT_CONFIGS:
        rows = selected[config]
        values = np.asarray(descriptors[config])
        poses = np.asarray([row["sonar_pose_client_4x4"] for row in rows])
        retrieval, close_pairs = cross_split_candidates(rows, values, poses)
        columns = ["source_index", "candidate_index", "rank", "cosine_similarity", "translation_m", "rotation_deg"]
        with (output / f"{config}_top3.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(retrieval)
        unique = {}
        for pair in sorted(retrieval, key=lambda pair: -pair["cosine_similarity"]):
            key = tuple(sorted((pair["source_index"], pair["candidate_index"])))
            if key not in unique:
                unique[key] = {**pair, "source_index": key[0], "candidate_index": key[1]}
        queue_pairs = {key: {**pair, "review_reasons": ["similarity_top50"]} for key, pair in list(unique.items())[:50]}
        for pair in close_pairs:
            key = (pair["source_index"], pair["candidate_index"])
            if key in queue_pairs:
                queue_pairs[key]["review_reasons"].append("pose_close")
            else:
                queue_pairs[key] = {**pair, "review_reasons": ["pose_close"]}
        queue = []
        for number, pair in enumerate(queue_pairs.values(), 1):
            queue.append({**pair, "pair_id": f"{config}-{number:04d}", "left": rows[pair["source_index"]],
                          "right": rows[pair["candidate_index"]], "review_status": "pending", "decision": "", "reason": ""})
        write_json(output / f"{config}_review_queue.json", queue)
        standard_deviations = np.array([row["descriptor_std"] for row in rows])
        low_cutoff = float(np.quantile(standard_deviations, 0.1))
        empty = [row for row in rows if row["hit_count"] == 0]
        constant = [row for row in rows if row["descriptor_std"] <= NEAR_CONSTANT_STD]
        low = [row for row in rows if row["descriptor_std"] <= low_cutoff]
        write_json(output / f"{config}_low_information.json", {"empty_return": empty, "near_constant": constant,
                                                               "low_variance_cutoff": low_cutoff, "low_variance": low})
        sheets = render_contact_sheets(queue, data_root, output, config)
        report["configurations"][config] = {"frames": len(rows), "directed_top3_candidates": len(retrieval),
                                           "unique_retrieved_pairs": len(unique), "pose_close_pairs": len(close_pairs),
                                           "review_pairs": len(queue), "empty_return_frames": len(empty),
                                           "near_constant_frames": len(constant), "low_variance_frames": len(low),
                                           "low_variance_cutoff": low_cutoff, "contact_sheets": sheets}
        print(f"{config}: {len(retrieval)} candidates, {len(close_pairs)} close-pose pairs, {len(queue)} review pairs", flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "split_audit.json", report)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", hashes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_audit(args.manifest, args.data_root, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
