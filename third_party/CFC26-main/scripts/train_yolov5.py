"""Per-location validation wrapper for YOLOv5 (separate repo).

YOLOv5 exposes a Callbacks object that receives events during training.
We hook two events:

  on_train_batch_end  — fires every step; log metrics at every 10% of an epoch.
                        Uses the EMA model so sub-epoch metrics are comparable
                        to epoch-end metrics. No checkpoint is written here.

  on_model_save       — fires after last.pt is written at the end of each epoch.
                        Loads last.pt (which contains the EMA weights) and runs
                        per-location val. Saves per-location best checkpoints by
                        copying last.pt when a location improves.

Usage
─────
python scripts/train_yolov5.py \\
    --yolov5_path /path/to/yolov5 \\
    --weights yolov5m.pt \\
    --data yolo_data/ \\
    --project runs/ \\
    --name my_run \\
    --epochs 2 --imgsz 896 --batch 16 --patience 50 --workers 6 \\
    --val_by_river

Assumptions
───────────
- Standard YOLOv5 repo (github.com/ultralytics/yolov5) at any recent release.
- train.txt in the dataset yaml lists one image path per line.
- val sets produced by coco_to_yolo.py (dataset_val_{loc}.yaml files).
- Single-GPU / CPU training (DDP not tested).
"""

import argparse
import csv
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path


def resolve_dataset_yaml(data_arg):
    p = Path(data_arg)
    if p.suffix in {".yaml", ".yml"}:
        return p
    return p / "dataset.yaml"


def count_lines(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _location_from_img_path(img_path: str) -> str:
    """Extract location name from image path (.../images/<location>/...)."""
    parts = Path(img_path).parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "images")
        return parts[idx + 1]
    except (StopIteration, IndexError):
        return Path(img_path).parent.name


def build_location_weighted_train_txt(original_train_txt: Path, output_path: Path, frames_per_location: int = 10_000, upsample: bool = True) -> int:
    """Write a new train.txt sampling frames_per_location frames per location.

    upsample=True:  locations below target are upsampled with replacement; locations above are kept as-is.
    upsample=False: locations are capped at frames_per_location (sampled without replacement); locations
                    below target keep all their frames (no upsampling).
    Returns the total number of lines written.
    """
    from collections import defaultdict
    import random

    lines = [l.strip() for l in original_train_txt.read_text().splitlines() if l.strip()]

    loc_to_paths: dict[str, list[str]] = defaultdict(list)
    for p in lines:
        loc_to_paths[_location_from_img_path(p)].append(p)

    mode = "upsample" if upsample else "floor"
    print(f"\n[LocationWeightedSampler] {len(loc_to_paths)} locations in {original_train_txt.name}, target={frames_per_location}/location, mode={mode}")
    balanced: list[str] = []
    for loc, paths in sorted(loc_to_paths.items()):
        n = len(paths)
        if upsample:
            if n >= frames_per_location:
                loc_balanced = paths
                tag = "unchanged"
            else:
                loc_balanced = random.choices(paths, k=frames_per_location)
                tag = f"upsampled from {n}"
        else:
            k = min(n, frames_per_location)
            loc_balanced = random.sample(paths, k)
            tag = "unchanged" if k == n else f"floored from {n}"
        print(f"  {loc}: {n} images → {len(loc_balanced)} lines ({tag})")
        balanced.extend(loc_balanced)

    random.shuffle(balanced)
    output_path.write_text("\n".join(balanced) + "\n")
    print(f"  Total: {len(balanced)} lines → {output_path}")
    return len(balanced)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv5 with per-location validation every 10% of an epoch"
    )
    parser.add_argument("--yolov5_path", required=True, help="Absolute path to the YOLOv5 repo root")
    parser.add_argument("--weights", required=True, help="Initial weights (.pt or .yaml)")
    parser.add_argument("--data", required=True, help="dataset.yaml or yolo root dir")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience in epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", type=int, default="", help="cuda device (e.g. 0) or cpu")
    parser.add_argument("--hyp", default="", help="Path to hyp yaml; defaults to data/hyps/hyp.scratch-low.yaml in the yolov5 repo")
    parser.add_argument(
        "--val_by_river",
        action="store_true",
        help="Validate separately per location using dataset_val_{loc}.yaml files",
    )
    parser.add_argument(
        "--frames_per_location",
        type=int,
        default=10_000,
        help="Target frames per location when --location_weighted_sampling is set (default: 10000).",
    )
    parser.add_argument(
        "--no_upsample_train",
        action="store_true",
        help="When set, do not upsample small train locations — floor at frames_per_location instead.",
    )
    parser.add_argument(
        "--val_location_weighted_sampling",
        action="store_true",
        help="Sample --frames_per_location_val frames per location for each val set. "
             "Requires --val_by_river.",
    )
    parser.add_argument(
        "--frames_per_location_val",
        type=int,
        default=3_000,
        help="Target frames per location for val when --val_location_weighted_sampling is set (default: 2000).",
    )
    parser.add_argument(
        "--no_upsample_val",
        action="store_true",
        help="When set, do not upsample small val locations — floor at frames_per_location_val instead.",
    )
    args = parser.parse_args()

    yolov5_path = Path(args.yolov5_path).resolve()
    if not yolov5_path.exists():
        raise FileNotFoundError(f"YOLOv5 repo not found at {yolov5_path}")
    sys.path.insert(0, str(yolov5_path))

    import torch
    import yaml as pyyaml
    from utils.callbacks import Callbacks
    import val as yolov5_val
    import train as yolov5_train

    # ── Dataset setup ─────────────────────────────────────────────────────────
    training_yaml = resolve_dataset_yaml(args.data)
    if not training_yaml.exists():
        raise FileNotFoundError(f"{training_yaml} does not exist")
    data_dir = training_yaml.parent

    if args.val_by_river:
        val_yamls = sorted(
            f for f in data_dir.iterdir()
            if f.name.startswith("dataset_val_") and f.suffix == ".yaml"
        )
        if not val_yamls:
            raise FileNotFoundError(
                f"No dataset_val_*.yaml files in {data_dir}. "
                "Run coco_to_yolo.py first."
            )
        val_sets = {
            f.stem.split("dataset_val_", 1)[1]: str(f) for f in val_yamls
        }
    else:
        val_sets = {"all": str(training_yaml)}

    print(f"Val sets ({len(val_sets)}): {sorted(val_sets)}")

    # ── Compute batches-per-epoch (nb) ────────────────────────────────────────
    with open(training_yaml) as f:
        train_data = pyyaml.safe_load(f)
    train_txt = Path(train_data["path"]) / train_data["train"]
    num_train_images = count_lines(train_txt)

    nb = math.ceil(num_train_images / args.batch)
    val_interval = max(1, int(nb * 0.1))
    print(f"Batches per epoch: {nb}, val every {val_interval} steps (10%)")

    # ── Save dir / CSV setup ───────────────────────────────────────────────────
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "val_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch_frac", "global_step", "time"]
        header += [f"{r}_mAP50" for r in sorted(val_sets)]
        header += ["avg_mAP50"]
        writer.writerow(header)

    # ── Shared mutable state (closures can't rebind plain variables) ───────────
    state = {"global_step": 0}

    def run_per_location_val(weights_path, epoch_frac):
        """Run val for every location, log to CSV, return river_map50 dict."""
        river_map50 = {}
        for river, data_yaml in val_sets.items():
            results, _, _ = yolov5_val.run(
                data=data_yaml,
                weights=str(weights_path),
                batch_size=args.batch * 2,
                imgsz=args.imgsz,
                device=args.device,
                workers=args.workers,
                verbose=False,
                plots=False,
                save_json=False,
                save_txt=False,
                save_dir=save_dir / "_val_tmp",
            )
            map50 = float(results[2])
            river_map50[river] = map50
            print(f"  [{river}] mAP50={map50:.4f}")

        avg_map50 = sum(river_map50.values()) / len(river_map50)
        print(f"  [avg] mAP50={avg_map50:.4f}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [epoch_frac, state["global_step"], datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            row += [round(river_map50[r], 4) for r in sorted(val_sets)]
            row += [round(avg_map50, 4)]
            writer.writerow(row)

        return river_map50, avg_map50

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_train_batch_end(model, ni, imgs, targets, paths, mloss):
        state["global_step"] += 1
        step_in_epoch = (ni % nb) + 1

        if step_in_epoch >= nb:
            return
        if step_in_epoch % val_interval != 0:
            return

        epoch = ni // nb
        epoch_frac = round(epoch + step_in_epoch / nb, 3)
        print(f"\n[Step val] epoch_frac={epoch_frac}, global_step={state['global_step']}")

        ema_model = model.ema.ema if hasattr(model, "ema") and model.ema else model

        tmp_pt = save_dir / "_step_val_tmp.pt"
        torch.save({"model": deepcopy(ema_model).float()}, tmp_pt)
        try:
            run_per_location_val(tmp_pt, epoch_frac)
        finally:
            tmp_pt.unlink(missing_ok=True)

    def on_model_save(last, epoch, final_epoch, best_fitness, fi):
        epoch_frac = float(epoch + 1)
        print(f"\n[Epoch val] epoch={epoch + 1}, global_step={state['global_step']}")
        run_per_location_val(last, epoch_frac)

    callbacks = Callbacks()
    callbacks.register_action("on_train_batch_end", name="river_step_val", callback=on_train_batch_end)
    callbacks.register_action("on_model_save", name="river_epoch_val", callback=on_model_save)

    # Build opt and run training
    hyp = args.hyp or str(yolov5_path / "data" / "hyps" / "hyp.scratch-low.yaml")

    import argparse as _ap
    opt = _ap.Namespace(
        weights=args.weights,
        cfg="",
        data=str(training_yaml),
        hyp=hyp,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        rect=False,
        resume=False,
        nosave=False,
        noval=False,
        noautoanchor=False,
        noplots=False,
        evolve=None,
        evolve_population=str(yolov5_path / "data" / "hyps"),
        resume_evolve=None,
        bucket="",
        cache=None,
        image_weights=False,
        device=args.device,
        multi_scale=False,
        single_cls=False,
        optimizer="SGD",
        sync_bn=False,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        quad=False,
        cos_lr=False,
        label_smoothing=0.0,
        patience=args.patience,
        freeze=[0],
        save_period=-1,
        seed=0,
        local_rank=-1,
        entity=None,
        upload_dataset=False,
        bbox_interval=-1,
        artifact_alias="latest",
        ndjson_console=False,
        ndjson_file=False,
    )

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    yolov5_train.main(opt, callbacks=callbacks)


if __name__ == "__main__":
    main()