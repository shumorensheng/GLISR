"""Convert normalized COCO annotations to YOLO format for all locations.

Expected output under YOLO_ROOT
────────────────────────────────
  images/
    klamath/{clip}/{frame}.jpg

  labels/                            (written by this script)
    klamath/{clip}/{frame}.txt

  train.txt / val.txt / test.txt     (written by this script)
  dataset.yaml                       (written by this script)

Usage
─────
    python converters/coco_to_yolo.py \\
        --coco_dir  /path/to/coco \\
        --yolo_root /path/to/yolo \\
        --splits_dir /path/to/splits

    # Image paths in train/val/test.txt must contain "/images/" so that
    # YOLOv5's label-discovery can replace /images/ → /labels/.
"""

import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import yaml

NC = 1
CLASS_NAMES = ["fish"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_loc(loc: str) -> str:
    """Map Kenai sub-location names to their shared directory names."""
    if loc.startswith("kenai-channel"):
        return "channel"
    if loc.startswith("kenai-rightbank"):
        return "rightbank"
    if loc.startswith("kenai-leftbank"):
        return "kenai"
    return loc


def load_split_assignments(splits_loc_dir: Path) -> dict[str, str]:
    """Load {coco_json_stem: split} from splits/{loc}/{train,val,test}.txt."""
    assignments: dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_file = splits_loc_dir / f"{split}.txt"
        if not split_file.exists():
            continue
        for line in split_file.read_text().splitlines():
            stem = line.strip()
            if stem:
                assignments[Path(stem).stem] = split
    return assignments


def coco_to_yolo_line(ann: dict, img_w: int, img_h: int, class_idx: int) -> str:
    x, y, w, h = ann["bbox"]
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    return f"{class_idx} {x_c:.6f} {y_c:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))

    except UnicodeDecodeError:
        print(f"  [re-encoded] {path.name}")
        return json.loads(path.read_text(encoding="latin-1"))


def convert_loc(
    coco_dir: Path,
    labels_dir: Path,
    image_base_dir: Path,
    loc: str,
    split_assignments: dict[str, str],
    exclude_stems: set[str],
) -> dict[str, list[str]]:
    """Write YOLO label files and return per-split image path lists."""
    split_images: dict[str, list[str]] = defaultdict(list)
    missing_clips: set[str] = set()

    for json_file in sorted(f for f in coco_dir.glob("*.json") if not f.name.startswith(".")):
        json_stem = json_file.stem
        if json_stem in exclude_stems:
            print(f"  [skip] {json_stem}")
            continue

        split = split_assignments.get(json_stem)
        if split is None:
            print(f"  [warn] {json_stem} — no split assignment, defaulting to train")
            split = "train"

        coco_data = read_json(json_file)

        cat_to_idx = {
            cat["id"]: idx
            for idx, cat in enumerate(coco_data.get("categories", []))
        }

        images = {
            img["id"]: (img["file_name"].replace("_frame_", "_"), img["width"], img["height"])
            for img in coco_data.get("images", [])
        }

        anns_by_image: dict[int, list] = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            anns_by_image[ann["image_id"]].append(ann)

        loc_dir_name = normalize_loc(loc)

        for img_id, (filename, img_w, img_h) in images.items():
            stem = Path(filename).stem
            clip_dir = stem.rsplit("_", 1)[0]

            img_path = image_base_dir / loc_dir_name / clip_dir / filename
            if not img_path.exists():
                for ext in (".jpg", ".png", ".jpeg"):
                    candidate = img_path.with_suffix(ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                else:
                    missing_clips.add(clip_dir)

            label_path = labels_dir / loc_dir_name / clip_dir / (stem + ".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)

            lines = [
                coco_to_yolo_line(ann, img_w, img_h, cat_to_idx.get(ann["category_id"], 0))
                for ann in anns_by_image.get(img_id, [])
            ]
            label_path.write_text("\n".join(lines))

            split_images[split].append(str(img_path))

    if missing_clips:
        print(f"  [missing clips] {len(missing_clips)} clip(s) with images not found on disk:")
        for clip in sorted(missing_clips):
            print(f"    {clip}")

    return split_images


def main():
    parser = ArgumentParser(description="Convert COCO annotations to YOLO format.")
    parser.add_argument("--coco_dir",   required=True, help="Root dir containing {location}/*.json COCO files")
    parser.add_argument("--yolo_root",  required=True, help="Output root for labels/, train.txt, dataset.yaml, etc.")
    parser.add_argument("--splits_dir", required=True, help="Dir containing {location}/{train,val,test}.txt split files")
    parser.add_argument("--image_base", default=None,  help="Base dir for image paths (default: {yolo_root}/images)")
    parser.add_argument("--exclude",    nargs="*", default=[], metavar="STEM", help="COCO JSON stems to skip")
    args = parser.parse_args()

    coco_base   = Path(args.coco_dir)
    yolo_root   = Path(args.yolo_root)
    splits_dir  = Path(args.splits_dir)
    image_base  = Path(args.image_base) if args.image_base else yolo_root / "images"
    exclude_stems = set(args.exclude)

    labels_dir = yolo_root / "labels"
    yolo_root.mkdir(parents=True, exist_ok=True)

    all_split_images: dict[str, list[str]] = defaultdict(list)
    loc_val_images: dict[str, list[str]] = {}

    loc_dirs = sorted(d for d in coco_base.iterdir() if d.is_dir())
    print(f"Found {len(loc_dirs)} location(s): {[d.name for d in loc_dirs]}\n")

    for loc_dir in loc_dirs:
        loc = loc_dir.name
        print(f"[{loc}] Converting...")

        split_assignments = load_split_assignments(splits_dir / loc)
        if not split_assignments:
            print(f"  [warn] no split files found at {splits_dir / loc}")

        split_images = convert_loc(
            coco_dir=loc_dir,
            labels_dir=labels_dir,
            image_base_dir=image_base,
            loc=loc,
            split_assignments=split_assignments,
            exclude_stems=exclude_stems,
        )

        for split, paths in split_images.items():
            all_split_images[split].extend(paths)
            print(f"  {split}: {len(paths)} images")

        if split_images.get("val"):
            loc_val_images[loc] = split_images["val"]

    # Write train/val/test.txt
    print("\nWriting split file lists...")
    split_files: dict[str, str] = {}
    for split in ("train", "val", "test"):
        paths = all_split_images.get(split, [])
        if not paths:
            continue
        out_file = yolo_root / f"{split}.txt"
        out_file.write_text("\n".join(paths) + "\n")
        split_files[split] = f"{split}.txt"
        print(f"  {split}.txt — {len(paths)} images")

    # Write dataset.yaml
    dataset_yaml = {
        "path":  str(yolo_root),
        "train": split_files.get("train", ""),
        "val":   split_files.get("val", ""),
        "test":  split_files.get("test", ""),
        "nc":    NC,
        "names": CLASS_NAMES,
    }
    yaml_path = yolo_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    # Write per-location val_{loc}.txt and dataset_val_{loc}.yaml
    print("\nWriting per-location val files...")
    for loc, paths in sorted(loc_val_images.items()):
        val_txt = yolo_root / f"val_{loc}.txt"
        val_txt.write_text("\n".join(paths) + "\n")

        loc_yaml = {
            "path":  str(yolo_root),
            "train": split_files.get("train", ""),
            "val":   f"val_{loc}.txt",
            "nc":    NC,
            "names": CLASS_NAMES,
        }
        loc_yaml_path = yolo_root / f"dataset_val_{loc}.yaml"
        with open(loc_yaml_path, "w") as f:
            yaml.dump(loc_yaml, f, default_flow_style=False, sort_keys=False)
        print(f"  {loc}: {len(paths)} val images → dataset_val_{loc}.yaml")

    print(f"\nDone. dataset.yaml → {yaml_path}")


if __name__ == "__main__":
    main()