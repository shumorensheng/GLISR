"""Train YOLO (Ultralytics) with per-location validation and configurable selection metric.

Usage
─────
    python scripts/train_yolov26.py \\
        --model    yolo26m.pt \\
        --data     /path/to/yolo/dataset.yaml \\
        --project  /path/to/output \\
        --name     my_experiment \\
        --epochs   2 \\
        --imgsz    896 \\
        --batch    16 \\
        --workers  6 \\
        --selection_metric mAP50 \\
        --val_by_river
"""

import io
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import argparse
import csv
import gc

import numpy as np
import torch
import random as _random
from torch.utils.data import DataLoader, SubsetRandomSampler
import yaml

from ultralytics import YOLO, __version__
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, GIT
from ultralytics.utils.torch_utils import (
    convert_optimizer_state_dict_to_fp16,
    unwrap_model,
)


def load_gt_counts(label_path):
    gt_pos = 0
    gt_neg = 0
    if not Path(label_path).exists():
        print(f"ERROR: Label file {label_path} not found. Returning 0 counts.")
        return 0, 0
    with open(label_path) as f:
        for line in f:
            cls = int(line.split()[0])
            if cls == 0:
                gt_pos += 1
            elif cls == 1:
                gt_neg += 1
    return gt_pos, gt_neg


def compute_crossing_nmae_from_predict(
    model,
    image_paths,
    label_paths,
    conf=0.25,
    imgsz=1280,
    device=0,
    verbose=True,
):
    if verbose:
        print(f"Running model.predict on {len(image_paths)} images with conf={conf}, imgsz={imgsz}, device={device}")

    total_error = 0
    total_gt = 0

    for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        results = model.predict(
            source=image_path,
            conf=conf,
            verbose=False,
            batch=1,
            device=device,
            imgsz=imgsz,
            stream=False,
            save=False,
        )

        result = results[0].cpu()
        gt_pos, gt_neg = load_gt_counts(label_path)

        pred_pos = 0
        pred_neg = 0

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            if -1 in cls_ids:
                print(f"  [-1 class detected, skipping] {Path(label_path).stem}")
                del result, results
                continue
            pred_pos = int((cls_ids == 0).sum())
            pred_neg = int((cls_ids == 1).sum())

        denom = gt_pos + gt_neg
        error = abs(pred_pos - gt_pos) + abs(pred_neg - gt_neg)
        total_error += error
        total_gt += denom

        if verbose and error > 0:
            nmae = error / denom if denom > 0 else -1
            print(
                f"NMAE: {nmae:.4f} Pred: +ve_-ve {pred_pos}_{pred_neg} GT: +ve_-ve {gt_pos}_{gt_neg} Image: {Path(label_path).stem}"
            )

        del result, results

        if i % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if verbose:
        print(f"mae calc = {total_error}/{total_gt}")
    return total_error / total_gt if total_gt > 0 else None


def compute_f1_from_val_results(val_results):
    box = getattr(val_results, "box", None) or getattr(val_results, "boxes", None)

    if box is not None and hasattr(box, "mp") and hasattr(box, "mr"):
        try:
            p, r = float(box.mp), float(box.mr)
            if p + r > 0:
                return 2.0 * p * r / (p + r)
        except Exception:
            pass

    for p_attr, r_attr in (("p", "r"), ("precision", "recall")):
        if box is not None and hasattr(box, p_attr) and hasattr(box, r_attr):
            try:
                pairs = [
                    (float(p_i), float(r_i))
                    for p_i, r_i in zip(getattr(box, p_attr), getattr(box, r_attr))
                    if float(p_i) + float(r_i) > 0
                ]
                if pairs:
                    return sum(2 * p * r / (p + r) for p, r in pairs) / len(pairs)
            except Exception:
                pass

    return 0.0


def _location_from_img_path(img_path: str) -> str:
    """Extract location name from image path.

    Expects paths of the form .../images/<location>/...
    Falls back to the immediate parent directory name.
    """
    parts = Path(img_path).parts
    try:
        idx = next(i for i, p in enumerate(parts) if p == "images")
        return parts[idx + 1]
    except (StopIteration, IndexError):
        return Path(img_path).parent.name


class RiverDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})

        self.val_sets = overrides.pop("val_sets", {}) or {}
        self.selection_metric = overrides.pop("selection_metric", "mAP50")
        self.val_by_river = overrides.pop("val_by_river", False)
        self.location_weighted_sampling = overrides.pop("location_weighted_sampling", False)
        self.frames_per_location = overrides.pop("frames_per_location", 10_000)
        self.session = overrides.pop("session", None)

        self.criteria_less_is_better = self.selection_metric == "crossing_nmae"

        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        self._global_step = 0
        self._step_batch_i = 0
        self.add_callback("on_train_epoch_start", self._on_epoch_start)
        self.add_callback("on_train_batch_end", self._on_batch_end)

        csv_path = Path(self.save_dir) / "val_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["epoch_frac", "global_step", "time"]
            header += [f"{r}_{self.selection_metric}" for r in sorted(self.val_sets)]
            if self.selection_metric != "mAP50":
                header += [f"{r}_mAP50" for r in sorted(self.val_sets)]
            header += [f"avg_{self.selection_metric}"]
            if self.selection_metric != "mAP50":
                header += ["avg_mAP50"]
            writer.writerow(header)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if not self.location_weighted_sampling or mode != "train":
            return loader

        dataset = loader.dataset
        im_files = getattr(dataset, "im_files", None) or getattr(dataset, "img_files", [])
        if not im_files:
            print("[LocationWeightedSampler] WARNING: could not find image file list on dataset — falling back to unweighted loader")
            return loader

        location_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, path in enumerate(im_files):
            location_to_indices[_location_from_img_path(path)].append(i)

        fpl = self.frames_per_location
        print(f"\n[LocationWeightedSampler] {len(location_to_indices)} locations, target={fpl}/location")
        sampled_indices: list[int] = []
        for loc, indices in sorted(location_to_indices.items()):
            if len(indices) >= fpl:
                chosen = indices
                tag = "unchanged"
            else:
                chosen = _random.choices(indices, k=fpl)
                tag = f"upsampled from {len(indices)}"
            print(f"  {loc}: {len(indices)} images → {len(chosen)} samples/epoch ({tag})")
            sampled_indices.extend(chosen)

        _random.shuffle(sampled_indices)
        sampler = SubsetRandomSampler(sampled_indices)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
        )

    def _on_epoch_start(self, trainer):
        self._step_batch_i = 0

    def _on_batch_end(self, trainer):
        self._global_step += 1
        self._step_batch_i += 1
        nb = getattr(self, "nb", 0) or len(self.train_loader)
        if not nb:
            return
        val_interval = max(1, int(nb * 0.1))
        if self._step_batch_i < nb and self._step_batch_i % val_interval == 0:
            self.validate()

    def _serialize_checkpoint(self):
        buffer = io.BytesIO()
        ema_model = self.ema.ema if self.ema else self.model
        train_metrics = dict(getattr(self, "metrics", {}) or {})
        train_metrics["fitness"] = self.fitness
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,
                "ema": deepcopy(unwrap_model(ema_model)).half(),
                "updates": self.ema.updates if self.ema else None,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": train_metrics,
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        return buffer.getvalue()

    def validate(self):
        river_scores = {}
        river_map50 = {}

        ckpt_bytes = self._serialize_checkpoint()
        temp_path = Path(self.wdir) / "_current_val.pt"
        temp_path.write_bytes(ckpt_bytes)

        try:
            current_model = YOLO(str(temp_path))

            for river, data_yaml in self.val_sets.items():
                print(f"\nValidating on '{river}' with {data_yaml}")

                val_results = current_model.val(
                    data=data_yaml,
                    split="val",
                    imgsz=self.args.imgsz,
                    batch=self.args.batch,
                    device=self.args.device,
                    workers=self.args.workers,
                    plots=False,
                    save_json=True,
                    verbose=False,
                )

                map50 = float(val_results.box.map50)
                river_map50[river] = map50

                if self.selection_metric == "mAP50":
                    metric_value = map50
                elif self.selection_metric == "f1@0.3":
                    metric_value = compute_f1_from_val_results(val_results)
                elif self.selection_metric == "crossing_nmae":
                    with open(data_yaml) as f:
                        data = yaml.safe_load(f)
                    base = Path(data["path"])
                    val_img_dir = data["val"]
                    val_lbl_dir = val_img_dir.replace("images", "labels")
                    image_paths = sorted((base / val_img_dir).glob("*.png"))
                    label_paths = [base / val_lbl_dir / (p.stem + ".txt") for p in image_paths]
                    metric_value = compute_crossing_nmae_from_predict(
                        current_model,
                        image_paths,
                        label_paths,
                        conf=0.25,
                        imgsz=self.args.imgsz,
                        device=self.args.device,
                        verbose=False,
                    )
                else:
                    raise ValueError(f"Unknown selection_metric: {self.selection_metric!r}")

                river_scores[river] = metric_value
                print(f"  {river}: {self.selection_metric}={metric_value:.4f}  mAP50={map50:.4f}")
        finally:
            temp_path.unlink(missing_ok=True)

        macro_metric = sum(river_scores.values()) / len(river_scores)
        macro_map50 = sum(river_map50.values()) / len(river_map50)

        self.fitness = -macro_metric if self.criteria_less_is_better else macro_metric
        if self.best_fitness is None or self.fitness > self.best_fitness:
            self.best_fitness = self.fitness

        nb = getattr(self, "nb", 0) or len(self.train_loader)
        epoch_frac = round(self._global_step / nb, 3) if nb else self._global_step
        print(f"\nEpoch {epoch_frac} (global step {self._global_step}) summary:")
        print(f"  Per-river {self.selection_metric}: {river_scores}")
        print(f"  Macro {self.selection_metric}: {macro_metric:.4f}  Macro mAP50: {macro_map50:.4f}")

        with open(Path(self.save_dir) / "val_metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            row = [epoch_frac, self._global_step, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            row += [round(river_scores[r], 4) for r in sorted(self.val_sets)]
            if self.selection_metric != "mAP50":
                row += [round(river_map50[r], 4) for r in sorted(self.val_sets)]
            row += [round(macro_metric, 4)]
            if self.selection_metric != "mAP50":
                row += [round(macro_map50, 4)]
            writer.writerow(row)

        # Run one global val pass over the combined dataset so results.csv gets
        # the same global metrics that YOLOv5 writes (not macro-averaged per location).
        # Only at epoch-end (when ultralytics reads the return value to write results.csv);
        # mid-epoch calls from _on_batch_end don't need it.
        nb = getattr(self, "nb", 0) or len(self.train_loader)
        is_epoch_end = nb and self._step_batch_i >= nb
        global_metrics = {}
        if not is_epoch_end:
            return global_metrics, self.fitness
        try:
            ckpt_bytes = self._serialize_checkpoint()
            temp_path = Path(self.wdir) / "_global_val.pt"
            temp_path.write_bytes(ckpt_bytes)
            global_model = YOLO(str(temp_path))
            global_results = global_model.val(
                data=self.args.data,
                split="val",
                imgsz=self.args.imgsz,
                batch=self.args.batch,
                device=self.args.device,
                workers=self.args.workers,
                plots=False,
                save_json=False,
                verbose=False,
            )
            global_metrics = {
                "metrics/precision(B)":   float(global_results.box.mp),
                "metrics/recall(B)":      float(global_results.box.mr),
                "metrics/mAP50(B)":       float(global_results.box.map50),
                "metrics/mAP50-95(B)":    float(global_results.box.map),
            }
            print(f"  [global] mAP50={global_metrics['metrics/mAP50(B)']:.4f}  mAP50-95={global_metrics['metrics/mAP50-95(B)']:.4f}")
        except Exception as e:
            print(f"  [global val] failed: {e}")
        finally:
            temp_path.unlink(missing_ok=True)

        return global_metrics, self.fitness


def resolve_dataset_yaml(data_arg):
    p = Path(data_arg)
    if p.suffix in {".yaml", ".yml"}:
        return p
    return p / "dataset.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26 detection with per-location validation")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml or yolo root dir")
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val_by_river",
        action="store_true",
        help="Validate separately on each location using dataset_val_{loc}.yaml files in the data dir",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="mAP50",
        choices=["mAP50", "f1@0.3", "crossing_nmae"],
    )
    parser.add_argument(
        "--location_weighted_sampling",
        action="store_true",
        help="Oversample minority locations so each location contributes equally per epoch",
    )

    args = parser.parse_args()

    training_yaml = resolve_dataset_yaml(args.data)
    if not training_yaml.exists():
        raise FileNotFoundError(f"{training_yaml} does not exist")
    data_dir = training_yaml.parent

    if args.val_by_river:
        val_filenames = [
            x for x in os.listdir(data_dir)
            if x.startswith("dataset_val_") and x.endswith(".yaml")
        ]
        if not val_filenames:
            raise FileNotFoundError(
                f"No dataset_val_*.yaml files found in {data_dir}. "
                "Run coco_to_yolo.py first to generate per-location val yamls."
            )
        val_sets = {
            x.split("dataset_val_")[1].split(".yaml")[0]: str(data_dir / x)
            for x in val_filenames
        }
        print(f"Val sets ({len(val_sets)}): {sorted(val_sets)}")
        for path in val_sets.values():
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} does not exist")
    else:
        val_sets = {"all": str(training_yaml)}

    model = YOLO(args.model)
    model.train(
        trainer=RiverDetectionTrainer,
        model=args.model,
        data=str(training_yaml),
        epochs=args.epochs,
        project=args.project,
        name=args.name,
        exist_ok=True,
        val_sets=val_sets,
        val_by_river=args.val_by_river,
        selection_metric=args.selection_metric,
        location_weighted_sampling=args.location_weighted_sampling,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
    )
