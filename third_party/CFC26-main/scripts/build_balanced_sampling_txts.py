"""
Pre-generate fish-only, balanced sampling train and val txt files for YOLO experiments.

Reads train.txt and per-location val yamls from a source YOLO data directory,
filters to frames with at least one fish annotation (non-empty label file), then
applies balanced sampling with a fixed seed. Writes outputs for both
upsample and no-upsample variants in one pass so that large locations (n >= target)
draw identical frames across both experiments.

Sampling logic:
  n >= target : random.sample(paths, target)  — same result in both variants
  n <  target : no_upsample keeps all n; upsample repeats to target

Label path convention: replace /images/ -> /labels/ in the image path, ext -> .txt

Input layout:
  {data_dir}/dataset.yaml               (path + train fields)
  {data_dir}/dataset_val_{loc}.yaml     (path + val fields, one per location)

Output layout:
  {output_dir}/upsample/
      train.txt
      dataset.yaml
      dataset_val_{loc}.yaml
      val_{loc}.txt
  {output_dir}/no_upsample/
      (same structure)

After running this script:
  - Point each experiment's DATA_YAML to the relevant dataset.yaml above
  - Remove --location_weighted_sampling and --val_location_weighted_sampling
    from train.sh (sampling is pre-done)

Usage:
  python scripts/build_balanced_sampling_txts.py \\
      --data_dir yolo_data/ \\
      --output_dir /balanced-sampling/yolo_dataset \\
      --frames_per_location 10000 \\
      --frames_per_location_val 3000 \\
      --seed 42
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml as pyyaml


def get_label_path(img_path: str) -> Path:
    parts = list(Path(img_path).parts)
    try:
        idx = next(i for i, p in enumerate(parts) if p == 'images')
        parts[idx] = 'labels'
    except StopIteration:
        pass
    return Path(*parts).with_suffix('.txt')


def has_fish(img_path: str) -> bool:
    label_path = get_label_path(img_path)
    if not label_path.exists():
        return False
    return bool(label_path.read_text().strip())


def _location_from_img_path(img_path: str) -> str:
    parts = Path(img_path).parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == 'images')
        return parts[idx + 1]
    except (StopIteration, IndexError):
        return Path(img_path).parent.name


def normalize_location(loc: str) -> str:
    """Collapse Kenai stratum sub-locations into their bank/channel group.

    kenai-channel-stratum* → kenai-channel
    kenai-rightbank-stratum* → kenai-rightbank
    kenai-leftbank-stratum* → kenai-leftbank
    All other locations are returned unchanged.
    """
    for group in ("kenai-channel", "kenai-rightbank", "kenai-leftbank"):
        if loc.startswith(group):
            return group
    return loc


def sample_location_weighted(loc_to_paths: dict, target: int, seed: int):
    """
    Returns (no_upsample_samples, upsample_samples).

    Both variants share one Random instance so large locations (n >= target)
    call rng.sample once and produce identical results in both outputs.

    no_upsample: large locations floored to target; small locations kept as-is.
    upsample:    large locations floored to target (same frames); small locations
                 repeated with replacement to reach target.
    """
    rng = random.Random(seed)

    # Pass 1: floor large locations (shared between both variants)
    base = {}
    for loc in sorted(loc_to_paths):
        paths = sorted(loc_to_paths[loc])
        n = len(paths)
        if n > target:
            base[loc] = rng.sample(paths, target)
        else:
            base[loc] = list(paths)

    # Pass 2: upsample small locations (only affects upsample variant)
    upsampled = {}
    for loc in sorted(base):
        paths = base[loc]
        if len(paths) < target:
            upsampled[loc] = paths + rng.choices(paths, k=target - len(paths))
        else:
            upsampled[loc] = paths

    return base, upsampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Source YOLO data dir with dataset.yaml and dataset_val_*.yaml')
    parser.add_argument('--output_dir', required=True,
                        help='Root output dir; upsample/ and no_upsample/ subdirs will be created')
    parser.add_argument('--frames_per_location', type=int, default=10_000,
                        help='Target frames per location for train (default: 10000)')
    parser.add_argument('--frames_per_location_val', type=int, default=3_000,
                        help='Target frames per location for val (default: 3000)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # ── Source dataset yaml ────────────────────────────────────────────────────
    source_yaml_path = data_dir / 'dataset.yaml'
    if not source_yaml_path.exists():
        print(f"ERROR: {source_yaml_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(source_yaml_path) as f:
        source_yaml = pyyaml.safe_load(f)

    train_txt = Path(source_yaml['path']) / source_yaml['train']
    if not train_txt.exists():
        print(f"ERROR: train.txt not found at {train_txt}", file=sys.stderr)
        sys.exit(1)

    # ── Per-location val yamls ─────────────────────────────────────────────────
    val_yamls = sorted(data_dir.glob('dataset_val_*.yaml'))
    if not val_yamls:
        print(f"ERROR: no dataset_val_*.yaml files in {data_dir}", file=sys.stderr)
        sys.exit(1)
    val_sets = {f.stem.split('dataset_val_', 1)[1]: f for f in val_yamls}
    print(f"Found {len(val_sets)} val locations: {sorted(val_sets)}")

    # ── Filter train to fish-only ──────────────────────────────────────────────
    all_train = [l.strip() for l in train_txt.read_text().splitlines() if l.strip()]
    print(f"\nTrain: {len(all_train)} total images, filtering to fish-only...")
    fish_train = [p for p in all_train if has_fish(p)]
    print(f"  {len(fish_train)} images with fish ({len(all_train) - len(fish_train)} dropped)")

    loc_to_train = defaultdict(list)
    for p in fish_train:
        loc = normalize_location(_location_from_img_path(p))
        loc_to_train[loc].append(p)
    for loc, paths in sorted(loc_to_train.items()):
        print(f"  {loc}: {len(paths)} fish frames")

    # ── Sample train (both variants in one pass) ───────────────────────────────
    no_up_train, up_train = sample_location_weighted(
        loc_to_train, args.frames_per_location, args.seed
    )

    # ── Filter and sample val per location ────────────────────────────────────
    # Group val yamls by normalized location so Kenai stratums are combined.
    val_groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for loc, yaml_path in sorted(val_sets.items()):
        with open(yaml_path) as f:
            val_yaml = pyyaml.safe_load(f)
        val_groups[normalize_location(loc)].append((loc, val_yaml))

    no_up_val = {}
    up_val = {}

    for norm_loc, members in sorted(val_groups.items()):
        fish_val = []
        for raw_loc, val_yaml in members:
            val_txt_path = Path(val_yaml['path']) / val_yaml['val']
            all_val = [l.strip() for l in val_txt_path.read_text().splitlines() if l.strip()]
            loc_fish = [p for p in all_val if has_fish(p)]
            print(f"\n[val/{raw_loc}] {len(all_val)} total → {len(loc_fish)} fish frames")
            fish_val.extend(loc_fish)
        if len(members) > 1:
            print(f"  → combined into [{norm_loc}]: {len(fish_val)} fish frames")

        # Use first member's yaml as template for the combined output yaml
        _, template_yaml = members[0]
        no_up, up = sample_location_weighted(
            {norm_loc: fish_val}, args.frames_per_location_val, args.seed
        )
        no_up_val[norm_loc] = (no_up[norm_loc], template_yaml)
        up_val[norm_loc] = (up[norm_loc], template_yaml)

    # ── Write outputs ──────────────────────────────────────────────────────────
    shuffle_rng = random.Random(args.seed)

    for variant, train_samples, val_samples in [
        ('no_upsample', no_up_train, no_up_val),
        ('upsample',    up_train,    up_val),
    ]:
        out = output_dir / variant
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n── {variant} ──")

        # Train txt
        combined = []
        for loc in sorted(train_samples):
            lines = train_samples[loc]
            n_orig = len(loc_to_train[loc])
            n_out = len(lines)
            if n_out == n_orig:
                tag = 'unchanged'
            elif n_out > n_orig:
                tag = f'upsampled from {n_orig}'
            else:
                tag = f'floored from {n_orig}'
            print(f"  train/{loc}: {n_orig} → {n_out} ({tag})")
            combined.extend(lines)

        shuffle_rng.shuffle(combined)
        train_out = out / 'train.txt'
        train_out.write_text('\n'.join(combined) + '\n')
        print(f"  train.txt: {len(combined)} lines → {train_out}")

        # Dataset yaml (copy source, redirect train path)
        out_dataset_yaml = dict(source_yaml)
        out_dataset_yaml['path'] = str(out)
        out_dataset_yaml['train'] = 'train.txt'
        (out / 'dataset.yaml').write_text(pyyaml.dump(out_dataset_yaml))

        # Val txts + yamls (per-location and combined)
        all_val_lines = []
        for loc, (lines, val_yaml) in sorted(val_samples.items()):
            shuffled = list(lines)
            shuffle_rng.shuffle(shuffled)
            val_txt_out = out / f'val_{loc}.txt'
            val_txt_out.write_text('\n'.join(shuffled) + '\n')
            all_val_lines.extend(shuffled)

            out_val_yaml = dict(val_yaml)
            out_val_yaml['path'] = str(out)
            out_val_yaml['val'] = f'val_{loc}.txt'
            (out / f'dataset_val_{loc}.yaml').write_text(pyyaml.dump(out_val_yaml))

        shuffle_rng.shuffle(all_val_lines)
        val_all_out = out / 'val.txt'
        val_all_out.write_text('\n'.join(all_val_lines) + '\n')
        out_dataset_yaml['val'] = 'val.txt'
        (out / 'dataset.yaml').write_text(pyyaml.dump(out_dataset_yaml))
        print(f"  val.txt: {len(all_val_lines)} lines → {val_all_out}")

        print(f"  Written to {out}")

    print(f"\nDone. Point each experiment's DATA_YAML to:")
    print(f"  upsample:    {output_dir}/upsample/dataset.yaml")
    print(f"  no_upsample: {output_dir}/no_upsample/dataset.yaml")
    print(f"\nRemove --location_weighted_sampling and --val_location_weighted_sampling")
    print(f"from train.sh — sampling is pre-done.")


if __name__ == '__main__':
    main()