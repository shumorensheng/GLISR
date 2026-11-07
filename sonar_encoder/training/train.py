"""Shared masked latent training; checkpoints are taken at optimizer boundaries."""

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .artifacts import create_artifact_directory
from .data import SonarDataset
from .model import build_encoder, make_patch_mask, masked_latent_loss, update_teacher


def manifest_digest(manifest):
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optimizer_update(student, teacher, optimizer, scaler, tau, clip_grad):
    scaler.unscale_(optimizer)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    norm = nn.utils.clip_grad_norm_(parameters, clip_grad)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= previous_scale
    if updated:
        update_teacher(student, teacher, tau)
    optimizer.zero_grad(set_to_none=True)
    return updated, float(norm)


def feature_statistics(features):
    features = features.float().cpu()
    centered = features - features.mean(0)
    singular = torch.linalg.svdvals(centered.reshape(-1, centered.shape[-1]))
    total = singular.sum()
    if total == 0:
        rank = 0.0
    else:
        probabilities = singular[singular > 0] / total
        rank = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    normalized = nn.functional.normalize(features, dim=-1)
    cosine = (normalized * normalized.roll(1, 0)).sum(-1).flatten()
    return {"mean_std": float(features.std(0, unbiased=False).mean()),
            "effective_rank": rank, "singular_values": singular.tolist(),
            "pairwise_cosine_quantiles": torch.quantile(cosine, torch.tensor([0., .25, .5, .75, 1.])).tolist()}


def validation_masks(samples):
    masks = []
    for sample in samples:
        digest = hashlib.sha256((sample["sample_id"] + ":val-mask-v1").encode()).digest()
        seed = int.from_bytes(digest[:8], "big")
        masks.append(make_patch_mask(1, torch.Generator().manual_seed(seed))[0])
    return torch.stack(masks)


@torch.no_grad()
def validate(student, teacher, predictor, dataset, normalization, config):
    student.eval()
    losses = defaultdict(list)
    batch_size = config["micro_batch"]
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        masks = validation_masks(samples)
        with torch.autocast(torch.device(config["device"]).type, dtype=torch.float16, enabled=config["amp"]):
            target = teacher.forward_native(samples, normalization, None)["patch_tokens"]
            prediction = predictor(student.forward_native(samples, normalization, masks)["patch_tokens"])
        masks = masks.to(config["device"])
        for i, sample in enumerate(samples):
            losses[sample["config_id"]].append(float(masked_latent_loss(prediction[i], target[i], masks[i])))
    student.train()
    per_config = {key: sum(values) / len(values) for key, values in losses.items()}
    return {"per_config": per_config, "macro_loss": sum(per_config.values()) / len(per_config)}


@torch.no_grad()
def diagnose(student, teacher, predictor, train_data, val_data, normalization, config):
    """Compare readout health and predictors on fixed seen-configuration subsets."""
    student.eval()
    subsets = []
    for dataset in (train_data, val_data):
        by_config = defaultdict(list)
        for i, row in enumerate(dataset.frames):
            by_config[row["config_id"]].append(i)
        indices = []
        for key in sorted(by_config):
            indices.extend(sorted(by_config[key], key=lambda i: hashlib.sha256(
                dataset.frames[i]["sample_id"].encode()).hexdigest())[:config["diagnostics_per_config"]])
        subsets.append(indices)
    target_sum = torch.zeros(student.embed_dim, device=config["device"])
    count = 0
    for i in subsets[0]:
        target = teacher.forward_native([train_data[i]], normalization, None)["patch_tokens"].float()
        target_sum += target.sum((0, 1))
        count += target.shape[0] * target.shape[1]
    constant = target_sum / count
    global_features, patch_features, predictions, targets, masks_list = [], [], [], [], []
    ids = []
    for i in subsets[1]:
        sample = val_data[i]
        masks = validation_masks([sample])
        output = student.forward_native([sample], normalization, None)
        global_features.append(output["global_features"].cpu())
        patch_features.append(output["patch_tokens"].cpu())
        predictions.append(predictor(student.forward_native([sample], normalization, masks)["patch_tokens"]).cpu())
        targets.append(teacher.forward_native([sample], normalization, None)["patch_tokens"].cpu())
        masks_list.append(masks)
        ids.append(sample["sample_id"])
    prediction, target, masks = torch.cat(predictions), torch.cat(targets), torch.cat(masks_list)
    student.train()
    return {"sample_ids": ids,
            "constant_target_train_ids": [train_data.frames[i]["sample_id"] for i in subsets[0]],
            "global": feature_statistics(torch.cat(global_features)),
            "patch": feature_statistics(torch.cat(patch_features)),
            "masked_loss": float(masked_latent_loss(prediction, target, masks)),
            "constant_loss": float(masked_latent_loss(constant.cpu().expand_as(target), target, masks)),
            "shuffled_target_loss": float(masked_latent_loss(prediction, target.roll(1, 0), masks))}


def run_training(config_path: Path, manifest_path: Path, data_root: Path,
                 normalization_path: Path, output_dir: Path, stop_after: int,
                 resume_path: Path | None) -> dict:
    config = json.loads(config_path.read_text())
    normalization = json.loads(normalization_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if normalization["source_manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("normalization provenance does not match manifest")
    if not 0 < stop_after <= config["steps"]:
        raise ValueError("stop-after must be in the configured training schedule")
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    train_data = SonarDataset(manifest_path, data_root, "ssl_train")
    val_data = SonarDataset(manifest_path, data_root, "ssl_validation")
    student = build_encoder(Path(config["code_dir"]), Path(config["weights_path"]), config["variant"]).to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    predictor = nn.Sequential(nn.Linear(student.embed_dim, student.embed_dim * 2), nn.GELU(),
                              nn.Linear(student.embed_dim * 2, student.embed_dim)).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameter_groups = defaultdict(list)
    for name, parameter in student.named_parameters():
        rate = config["frontend_lr"] if "patch_embed" in name or "mask_token" in name or name.startswith("stem.") else config["trunk_lr"]
        decay = config["weight_decay"] if parameter.ndim > 1 else 0.
        parameter_groups[(rate, decay)].append(parameter)
    for parameter in predictor.parameters():
        decay = config["weight_decay"] if parameter.ndim > 1 else 0.
        parameter_groups[(config["frontend_lr"], decay)].append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": parameters, "lr": rate, "initial_lr": rate, "weight_decay": decay}
        for (rate, decay), parameters in parameter_groups.items()], betas=(.9, .999))
    scaler = torch.amp.GradScaler(device.type, enabled=config["amp"])
    sample_rng = torch.Generator().manual_seed(config["seed"] + 1)
    mask_rng = torch.Generator().manual_seed(config["seed"] + 2)
    source_dir = Path(__file__).parent
    training_sources = [source_dir / name for name in ("data.py", "model.py", "frontends.py", "train.py", "artifacts.py")]
    provenance = {"manifest": file_digest(manifest_path), "normalization": file_digest(normalization_path),
                  "weights": file_digest(config["weights_path"]), "config": manifest_digest(config),
                  "source": {p.name: file_digest(p) for p in training_sources},
                  "python": sys.version, "torch": torch.__version__, "numpy": np.__version__}
    step, epoch, sample_cursor, best_val = 0, 0, 0, math.inf
    epoch_order = torch.randperm(len(train_data), generator=sample_rng)
    checkpoint_dir = output_dir / "checkpoints"
    create_artifact_directory(checkpoint_dir)
    if resume_path is not None:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("checkpoint provenance changed; start a new run for changed inputs/code")
        student.load_state_dict(saved["student"])
        teacher.load_state_dict(saved["teacher"])
        predictor.load_state_dict(saved["predictor"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step, epoch, sample_cursor, best_val = (saved[k] for k in ("step", "epoch", "sample_cursor", "best_val"))
        epoch_order = saved["epoch_order"]
        sample_rng.set_state(saved["sample_rng"])
        mask_rng.set_state(saved["mask_rng"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        if (checkpoint_dir / "initial.pt").exists():
            raise FileExistsError("run already initialized; use --resume or a new output directory")
        (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        snapshot = output_dir / "code_snapshot"
        snapshot.mkdir()
        for path in source_dir.glob("*.py"):
            shutil.copy2(path, snapshot / path.name)
        shutil.copy2(source_dir / "pyproject.toml", snapshot / "pyproject.toml")
        (output_dir / "git_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
        (output_dir / "git_diff.patch").write_text(subprocess.check_output(["git", "diff", "--", "sonar_encoder", "docs/TRAINING.md"], text=True))

    def save_checkpoint(path):
        state = {"student": student.state_dict(), "teacher": teacher.state_dict(),
                 "predictor": predictor.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "step": step, "epoch": epoch,
                 "sample_cursor": sample_cursor, "epoch_order": epoch_order, "best_val": best_val,
                 "sample_rng": sample_rng.get_state(), "mask_rng": mask_rng.get_state(),
                 "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(), "resolved_config": config, "provenance": provenance}
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    if resume_path is None:
        save_checkpoint(checkpoint_dir / "initial.pt")
        initial = diagnose(student, teacher, predictor, train_data, val_data, normalization, config)
        (output_dir / "diagnostics_initial.json").write_text(json.dumps(initial, indent=2) + "\n")
    student.train()
    skipped = 0
    resume_step = step
    window_seconds, window_data_seconds, window_frames, window_updates = 0., 0., 0, 0
    with (output_dir / "metrics.jsonl").open("a", buffering=1) as log:
        while step < stop_after:
            start = time.perf_counter()
            data_seconds, loss_value = 0., 0.
            sample_ids = []
            resume_masks = hashlib.sha256()
            if step < config["warmup_steps"]:
                factor = (step + 1) / config["warmup_steps"]
            else:
                progress = (step - config["warmup_steps"]) / (config["steps"] - config["warmup_steps"])
                factor = .1 + .9 * (1 + math.cos(math.pi * progress)) / 2
            tau = 1 - (1 - config["ema_start"]) * (1 + math.cos(math.pi * step / config["steps"])) / 2
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            for _ in range(config["accumulation"]):
                data_start = time.perf_counter()
                samples = []
                for _ in range(config["micro_batch"]):
                    if sample_cursor == len(epoch_order):
                        epoch += 1
                        epoch_order = torch.randperm(len(train_data), generator=sample_rng)
                        sample_cursor = 0
                    samples.append(train_data[int(epoch_order[sample_cursor])])
                    sample_cursor += 1
                sample_ids.extend(sample["sample_id"] for sample in samples)
                masks = make_patch_mask(len(samples), mask_rng)
                if resume_path is not None and step == resume_step:
                    resume_masks.update(masks.numpy().tobytes())
                data_seconds += time.perf_counter() - data_start
                with torch.autocast(device.type, dtype=torch.float16, enabled=config["amp"]):
                    with torch.no_grad():
                        target = teacher.forward_native(samples, normalization, None)["patch_tokens"]
                    prediction = predictor(student.forward_native(samples, normalization, masks)["patch_tokens"])
                loss = masked_latent_loss(prediction, target, masks.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at update {step}: {sample_ids}")
                loss_value += float(loss.detach()) / config["accumulation"]
                scaler.scale(loss / config["accumulation"]).backward()
            if resume_path is not None and step == resume_step:
                log.write(json.dumps({"kind": "resume_batch", "step": step,
                                      "sample_ids": sample_ids,
                                      "mask_sha256": resume_masks.hexdigest()}) + "\n")
            updated, grad_norm = optimizer_update(student, teacher, optimizer, scaler, tau, config["clip_grad"])
            if not updated:
                skipped += 1
                log.write(json.dumps({"kind": "amp_skip", "step": step, "sample_ids": sample_ids,
                                      "amp_scale": scaler.get_scale()}) + "\n")
                if skipped >= 3:
                    raise FloatingPointError("three consecutive AMP overflows; inspect input and gradients")
                continue
            skipped = 0
            step += 1
            elapsed = time.perf_counter() - start
            window_seconds += elapsed
            window_data_seconds += data_seconds
            window_frames += len(sample_ids)
            window_updates += 1
            if step % config["log_every"] == 0 or step == stop_after:
                row = {"kind": "train", "step": step, "loss": loss_value, "lr_factor": factor,
                       "tau": tau, "grad_norm": grad_norm, "amp_scale": scaler.get_scale(),
                       "seconds": window_seconds, "data_seconds": window_data_seconds,
                       "data_measurement": "native sample loading and mask generation; model time includes preprocessing",
                       "updates": window_updates, "frames_per_second": window_frames / window_seconds,
                       "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                window_seconds, window_data_seconds, window_frames, window_updates = 0., 0., 0, 0
            if step % config["validation_every"] == 0 or step == stop_after:
                val_start = time.perf_counter()
                metrics = validate(student, teacher, predictor, val_data, normalization, config)
                log.write(json.dumps({"kind": "validation", "step": step,
                                      "seconds": time.perf_counter() - val_start, **metrics}) + "\n")
                if metrics["macro_loss"] < best_val:
                    best_val = metrics["macro_loss"]
                save_checkpoint(checkpoint_dir / "last.pt")
    final = diagnose(student, teacher, predictor, train_data, val_data, normalization, config)
    (output_dir / f"diagnostics_step{step}.json").write_text(json.dumps(final, indent=2) + "\n")
    return {"step": step, "best_val": best_val, "output_dir": str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "data-root", "normalization", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--stop-after", required=True, type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_training(args.config, args.manifest, args.data_root, args.normalization,
                                  args.output, args.stop_after, args.resume)))


if __name__ == "__main__":
    main()
