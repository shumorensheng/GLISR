"""Fixed source-supervised head initialization and body adaptation."""

import copy
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sonar_encoder.training.readout import load_targets, state_digest
from sonar_encoder.training.uatd_head import box_targets, global_targets


class SourceSampler:
    def __init__(self, population, seed):
        self.population = population
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.randperm(population, generator=self.generator)
        self.cursor = self.epoch = self.ordinal = 0

    def draw(self, count):
        indices = []
        while len(indices) < count:
            if self.cursor == self.population:
                self.permutation = torch.randperm(self.population, generator=self.generator)
                self.cursor = 0
                self.epoch += 1
            take = min(count - len(indices), self.population - self.cursor)
            indices.extend(self.permutation[self.cursor:self.cursor + take].tolist())
            self.cursor += take
            self.ordinal += take
        return indices

    def state_dict(self):
        return {"permutation": self.permutation.clone(), "cursor": self.cursor, "epoch": self.epoch,
                "ordinal": self.ordinal, "rng": self.generator.get_state()}

    def load_state_dict(self, state):
        self.permutation = state["permutation"]
        self.cursor, self.epoch, self.ordinal = state["cursor"], state["epoch"], state["ordinal"]
        self.generator.set_state(state["rng"])


def init_source_heads(arm):
    heads = {}
    if arm in ("E_U", "E_UM"):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            heads["U_global"] = nn.Linear(384, 10)
            heads["U_dense"] = nn.Conv2d(384, 50, 1)
            with torch.no_grad():
                heads["U_dense"].bias[:10].fill_(math.log(.01 / .99))
    if arm in ("E_M", "E_UM"):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            heads["M"] = nn.Conv2d(384, 12, 1)
    return nn.ModuleDict(heads)


def u_loss(global_logits, dense_output, targets, source_count, positive_count):
    heatmap, regression, positive = box_targets(targets, dense_output.device)
    logits = dense_output[:, :10]
    probability = logits.sigmoid()
    p_target = torch.where(positive, probability, 1 - probability)
    focal = (torch.where(positive, .25, .75) * (1 - p_target).square()
             * F.binary_cross_entropy_with_logits(logits, heatmap, reduction="none")).sum()
    raw = dense_output[:, 10:].reshape(-1, 10, 4, 16, 16)
    prediction = torch.cat((raw[:, :, :2].sigmoid(), raw[:, :, 2:]), 2)
    # Empty-positive micro batches contribute an actual zero sum; the macro still has P > 0.
    regression_sum = F.smooth_l1_loss(prediction.permute(0, 1, 3, 4, 2)[positive],
                                     regression.permute(0, 1, 3, 4, 2)[positive], beta=1., reduction="sum")
    return {"presence": F.binary_cross_entropy_with_logits(
                global_logits, global_targets(targets, global_logits.device), reduction="sum") / (source_count * 10),
            "focal": focal / positive_count, "regression": regression_sum / (4 * positive_count)}


def m_loss(output, targets, source_count):
    logits = F.interpolate(output, (480, 320), mode="bilinear", align_corners=False)
    return F.cross_entropy(logits, targets.long(), reduction="sum") / (source_count * 480 * 320)


def set_stage(encoder, heads, stage):
    heads.train().requires_grad_(True)
    if stage == "HEAD_INIT":
        encoder.eval().requires_grad_(False)
        groups = [{"params": list(heads.parameters()), "lr": .001, "weight_decay": .0001,
                   "role": "head"}]
    else:
        encoder.train().requires_grad_(True)
        decay, no_decay = [], []
        for name, parameter in encoder.named_parameters():
            if name.endswith("mask_token"):
                parameter.requires_grad_(False)
                continue
            if parameter.ndim == 1 or name.endswith("bias") or name.endswith(("cls_token", "pos_embed")):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = [{"params": decay, "lr": .000125, "weight_decay": .05, "role": "body"},
                  {"params": no_decay, "lr": .000125, "weight_decay": 0., "role": "body"},
                  {"params": list(heads.parameters()), "lr": .001, "weight_decay": .0001, "role": "head"}]
    return torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)


def source_field(encoder, inputs, stage):
    if stage == "HEAD_INIT":
        with torch.no_grad():
            output = encoder(inputs)
    else:
        output = encoder(inputs)
    return output["patch_tokens"].transpose(1, 2).reshape(-1, 384, 16, 16)


def body_learning_rates(optimizer, success):
    for group in optimizer.param_groups:
        peak, minimum = ((.000125, .00000125) if group["role"] == "body" else (.001, .00001))
        group["lr"] = (peak * success / 200 if success <= 200 else
                       minimum + (peak - minimum) * .5 * (1 + math.cos(math.pi * (success - 200) / 5800)))


def training_state(encoder, heads, optimizer, samplers, clock, exposures):
    current = optimizer.state_dict()
    optimizer_state = {"state": {index: {name: value.detach().cpu().clone() if torch.is_tensor(value) else value
                                         for name, value in values.items()}
                                  for index, values in current["state"].items()},
                       "param_groups": copy.deepcopy(current["param_groups"])}
    return {"encoder": {name: value.detach().cpu().clone() for name, value in encoder.state_dict().items()},
            "heads": {name: value.detach().cpu().clone() for name, value in heads.state_dict().items()},
            "optimizer": optimizer_state, "samplers": {name: sampler.state_dict() for name, sampler in samplers.items()},
            "clock": copy.deepcopy(clock), "exposures": copy.deepcopy(exposures),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}


def restore_training(encoder, heads, samplers, saved):
    if "failure_not_safe_for_automatic_resume" in saved and saved["failure_not_safe_for_automatic_resume"]:
        raise ValueError("failure checkpoint forbids automatic resume")
    encoder.load_state_dict(saved["encoder"])
    heads.load_state_dict(saved["heads"])
    optimizer = set_stage(encoder, heads, saved["clock"]["stage"])
    optimizer.load_state_dict(saved["optimizer"])
    for name, sampler in samplers.items():
        sampler.load_state_dict(saved["samplers"][name])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    return optimizer, saved["clock"], saved["exposures"]


def failure_state(encoder, heads, optimizer, samplers, clock, exposures, pending, error, last_complete):
    saved = training_state(encoder, heads, optimizer, samplers, clock, exposures)
    saved.update({"failure_not_safe_for_automatic_resume": True,
                  "failure_error": {"type": type(error).__name__, "message": str(error)},
                  "pending": copy.deepcopy(pending),
                  "last_complete_boundary": {**copy.deepcopy(last_complete), "binary_retained": False},
                  "gradients": {kind: {name: parameter.grad.detach().cpu().clone()
                                       for name, parameter in model.named_parameters() if parameter.grad is not None}
                                for kind, model in (("encoder", encoder), ("heads", heads))}})
    return saved


def train_arm(arm: str, body_stop: int):
    from runs.shared_supervised_reference import (
        RUN, atomic_torch, configure_runtime, digest, load_initial_encoder, load_protocol, resources,
    )

    configure_runtime("source")
    protocol = load_protocol()
    protocol_sha = digest(RUN / "protocol.json")
    directory = RUN / "arms" / arm
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "last.pt"
    order = {"E_U": ["U"] * 4, "E_M": ["M"] * 4, "E_UM": ["U", "M", "U", "M"]}[arm]
    sources = list(dict.fromkeys(order))
    rows = {source: protocol["source_inputs"][source]["rows"] for source in sources}
    views = {source: np.load(protocol["source_inputs"][source]["view_path"], mmap_mode="r") for source in sources}
    for source, view in views.items():
        specification = protocol["source_inputs"][source]
        if view.shape != (len(rows[source]), 1, 224, 224) or view.dtype != np.float32:
            raise ValueError(f"source input view shape/dtype differs from protocol: {source}")
        if digest(Path(specification["view_path"])) != specification["view_sha256"]:
            raise ValueError(f"source input view SHA differs from protocol: {source}")
    targets = {}
    if "U" in sources:
        targets["U"] = [{"boxes": torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4),
                         "labels": torch.tensor(row["labels"], dtype=torch.long)} for row in rows["U"]]
    if "M" in sources:
        for row in rows["M"]:
            if digest(Path(row["mask"])) != row["mask_sha256"]:
                raise ValueError(f"native Marine mask SHA differs from protocol: {row['row_id']}")
        targets["M"] = load_targets(rows["M"], "marine")
        if (targets["M"].dtype != torch.uint8 or tuple(targets["M"].shape) != (len(rows["M"]), 480, 320)
                or int(targets["M"].max()) > 11):
            raise ValueError("native Marine labels differ from the bound uint8 N480x320 class0..11 interface")
    samplers = {source: SourceSampler(len(rows[source]), 3 if source == "U" else 4) for source in sources}
    heads = init_source_heads(arm)
    for name, head in heads.items():
        if state_digest(head.state_dict()) != protocol["initial_source_head_sha256"][name]:
            raise ValueError(f"source head initialization differs from bound protocol: {name}")
    encoder = load_initial_encoder()
    initial_encoder_sha = state_digest(encoder.state_dict())
    initial_mask_sha = hashlib.sha256(encoder.backbone.mask_token.detach().contiguous().numpy().tobytes()).hexdigest()
    encoder.to("cuda")
    heads.to("cuda")
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["protocol_sha256"] != protocol_sha:
            raise ValueError("source checkpoint was produced under a different protocol")
        optimizer, clock, exposures = restore_training(encoder, heads, samplers, saved)
        if state_digest(encoder.state_dict()) != saved["encoder_state_sha256"]:
            raise ValueError("restored source encoder does not match the saved state SHA")
        body_witness = saved["body_gradient_witness"]
    else:
        clock = {"stage": "HEAD_INIT", "head_success": 0, "body_success": 0,
                 "attempts": 0, "elapsed_seconds": 0.}
        exposures = {stage: {source: {kind: torch.zeros(len(rows[source]), dtype=torch.long)
                                     for kind in ("attempted", "completed", "effective")}
                             for source in sources} for stage in ("HEAD_INIT", "BODY_FT")}
        optimizer = set_stage(encoder, heads, "HEAD_INIT")
        body_witness = {}
    started, inherited_elapsed = time.monotonic(), clock["elapsed_seconds"]
    trace = (directory / "source_trace.jsonl").open("a", buffering=1)

    def save_boundary():
        torch.cuda.synchronize()
        clock["elapsed_seconds"] = inherited_elapsed + time.monotonic() - started
        boundary = training_state(encoder, heads, optimizer, samplers, clock, exposures)
        boundary.update({"protocol_sha256": protocol_sha, "encoder_state_sha256": state_digest(boundary["encoder"]),
                         "body_gradient_witness": body_witness})
        atomic_torch(checkpoint_path, boundary)

    def report(endpoint):
        exposure_report = {}
        for stage, stage_sources in exposures.items():
            exposure_report[stage] = {}
            for source, counts in stage_sources.items():
                exposure_report[stage][source] = {}
                for kind, histogram in counts.items():
                    bins, bin_counts = histogram.unique(return_counts=True)
                    exposure_report[stage][source][kind] = {
                        "visits": int(histogram.sum()), "unique": int((histogram > 0).sum()),
                        "repeat_histogram": {str(int(visits)): int(count) for visits, count in zip(bins, bin_counts)},
                    }
        full = {}
        for source in sources:
            counts = exposures["HEAD_INIT"][source]["completed"] + exposures["BODY_FT"][source]["completed"]
            bins, bin_counts = counts.unique(return_counts=True)
            full[source] = {"visits": int(counts.sum()), "unique": int((counts > 0).sum()),
                            "repeat_histogram": {str(int(visits)): int(count) for visits, count in zip(bins, bin_counts)}}
        result = {"arm": arm, "endpoint": endpoint, "clock": copy.deepcopy(clock),
                  "protocol_sha256": protocol_sha, "body_gradient_witness": body_witness,
                  "initial_mask_token_sha256": initial_mask_sha,
                  "mask_token_sha256": hashlib.sha256(encoder.backbone.mask_token.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                  "encoder_training": encoder.training,
                  "body_requires_grad_except_mask": all(parameter.requires_grad for name, parameter in encoder.named_parameters()
                                                        if not name.endswith("mask_token")),
                  "mask_token_requires_grad": encoder.backbone.mask_token.requires_grad,
                  "mask_token_in_optimizer": any(parameter is encoder.backbone.mask_token
                                                  for group in optimizer.param_groups for parameter in group["params"]),
                  "encoder_state_sha256": state_digest(encoder.state_dict()),
                  "head_state_sha256": {name: state_digest(head.state_dict()) for name, head in heads.items()},
                  "source_exposure": exposure_report, "source_exposure_full": full,
                  "samplers": {source: {"ordinal": sampler.ordinal, "epoch": sampler.epoch,
                                        "cursor": sampler.cursor,
                                        "rng_sha256": hashlib.sha256(sampler.generator.get_state().numpy().tobytes()).hexdigest()}
                               for source, sampler in samplers.items()},
                  "resources": resources(), "source_trace": str(directory / "source_trace.jsonl")}
        (directory / f"report_{endpoint}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)
        return result

    pending = {"micro_order": [], "micro_index": None, "event": "before_macro"}

    def terminate(signum, frame):
        raise InterruptedError("source process received SIGTERM from its resource guard")

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        while clock["head_success"] < 200 or clock["body_success"] < body_stop:
            pending = {"micro_order": [], "micro_index": None, "event": "stage_transition"}
            if clock["stage"] == "HEAD_INIT" and clock["head_success"] == 200:
                if state_digest(encoder.state_dict()) != initial_encoder_sha:
                    raise ValueError("head initialization changed the frozen encoder")
                clock["stage"] = "BODY_FT"
                optimizer = set_stage(encoder, heads, "BODY_FT")
            stage = clock["stage"]
            pending = {"micro_order": [], "micro_index": None, "event": "sampling"}
            planned = pending["micro_order"]
            for source in order:
                sampler = samplers[source]
                ordinal = sampler.ordinal
                indices = sampler.draw(8)
                planned.append({"source": source, "indices": indices, "ordinal_start": ordinal,
                                "ordinal_stop": sampler.ordinal, "epoch_after": sampler.epoch,
                                "cursor_after": sampler.cursor,
                                "rng_sha256": hashlib.sha256(sampler.generator.get_state().numpy().tobytes()).hexdigest(),
                                "actual_ids": [rows[source][index]["row_id"] for index in indices]})
            pending["event"] = "loss_denominators"
            source_counts = {source: order.count(source) * 8 for source in sources}
            positive_count = 0
            if "U" in sources:
                whole_targets = [targets["U"][index] for micro in planned if micro["source"] == "U"
                                 for index in micro["indices"]]
                positive_count = int(box_targets(whole_targets, "cpu")[2].sum())
                if positive_count <= 0:
                    raise ValueError("U source macro has no positive cells; the bound loss is undefined")
            clock["attempts"] += 1
            trace.write(json.dumps({"event": "macro_planned", "clock": clock, "micro_order": planned,
                                    "source_counts": source_counts, "U_positive_cells": positive_count}) + "\n")
            if stage == "BODY_FT":
                body_learning_rates(optimizer, clock["body_success"] + 1)
            if stage == "BODY_FT" and clock["body_success"] == 0:
                named_parameters = dict(encoder.named_parameters())
                witness_parameters = {name: named_parameters[name] for name in (
                    "backbone.patch_embed.proj.weight", "backbone.blocks.0.attn.qkv.weight", "backbone.norm.weight")}
                witness_before = {name: parameter.detach().cpu().clone() for name, parameter in witness_parameters.items()}
            pending["event"] = "macro_planned"
            optimizer.zero_grad(set_to_none=True)
            terms_total, macro_total = {}, 0.
            for micro_index, micro in enumerate(planned):
                pending.update({"micro_index": micro_index, "event": "input_transfer"})
                source, indices = micro["source"], micro["indices"]
                inputs = torch.from_numpy(views[source][indices]).to("cuda")
                exposures[stage][source]["attempted"].index_add_(0, torch.tensor(indices), torch.ones(8, dtype=torch.long))
                trace.write(json.dumps({"event": "forward_started", "attempt": clock["attempts"], "stage": stage,
                                        "micro": micro_index, "source": source, "actual_ids": micro["actual_ids"]}) + "\n")
                pending["event"] = "forward_started"
                field = source_field(encoder, inputs, stage)
                pending["event"] = "forward_returned"
                exposures[stage][source]["completed"].index_add_(0, torch.tensor(indices), torch.ones(8, dtype=torch.long))
                if source == "U":
                    terms = u_loss(heads["U_global"](field.mean((-2, -1))), heads["U_dense"](field),
                                   [targets["U"][index] for index in indices], source_counts["U"], positive_count)
                else:
                    terms = {"ce": m_loss(heads["M"](field), targets["M"][indices].to("cuda"), source_counts["M"])}
                coefficient = .5 if arm == "E_UM" else 1.
                loss = coefficient * sum(terms.values())
                if not all(bool(torch.isfinite(value)) for value in terms.values()) or not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite {source} source loss at {stage} attempt {clock['attempts']}")
                pending["event"] = "backward_started"
                loss.backward()
                pending["event"] = "backward_returned"
                for name, value in terms.items():
                    key = f"{source}_{name}"
                    terms_total[key] = terms_total.get(key, 0.) + float(value.detach())
                macro_total += float(loss.detach())
                torch.cuda.synchronize()
                trace.write(json.dumps({"event": "micro_completed", "attempt": clock["attempts"], "stage": stage,
                                        "micro": micro_index, "source": source, "loss_terms": {
                                            name: float(value.detach()) for name, value in terms.items()}}) + "\n")
                pending["event"] = "micro_completed"
                resources()
            parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError("an active source optimizer parameter has no gradient")
            pending["event"] = "clip_started"
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if stage == "BODY_FT" and clock["body_success"] == 0:
                body_witness = {name: {"requires_grad": parameter.requires_grad,
                                       "in_optimizer": any(parameter is active for active in parameters),
                                       "gradient_finite": bool(torch.isfinite(parameter.grad).all()),
                                       "gradient_l2_norm": float(parameter.grad.norm()),
                                       "gradient_abs_max": float(parameter.grad.abs().max()),
                                       "weight_before_sha256": hashlib.sha256(witness_before[name].numpy().tobytes()).hexdigest()}
                                for name, parameter in witness_parameters.items()}
            pending["event"] = "optimizer_step_started"
            optimizer.step()
            pending["event"] = "optimizer_step_returned"
            if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
                raise FloatingPointError(f"nonfinite source parameters at {stage} attempt {clock['attempts']}")
            if stage == "BODY_FT" and clock["body_success"] == 0:
                for name, parameter in witness_parameters.items():
                    after = parameter.detach().cpu()
                    body_witness[name]["weight_movement_abs_max"] = float((after - witness_before[name]).abs().max())
                    body_witness[name]["weight_after_sha256"] = hashlib.sha256(after.contiguous().numpy().tobytes()).hexdigest()
                del witness_before
            clock["head_success" if stage == "HEAD_INIT" else "body_success"] += 1
            for micro in planned:
                exposures[stage][micro["source"]]["effective"].index_add_(
                    0, torch.tensor(micro["indices"]), torch.ones(8, dtype=torch.long))
            trace.write(json.dumps({"event": "macro_success", "clock": clock, "loss_terms": terms_total,
                                    "arm_loss": macro_total, "whole_grad_norm_before_clip": float(norm),
                                    "learning_rates": [group["lr"] for group in optimizer.param_groups]}) + "\n")
            success = clock["head_success"] if stage == "HEAD_INIT" else clock["body_success"]
            pending["event"] = "macro_success"
            if success % 50 == 0:
                pending["event"] = "saving_boundary"
                save_boundary()
                pending["event"] = "complete_boundary_saved"
            if stage == "HEAD_INIT" and success == 200:
                if state_digest(encoder.state_dict()) != initial_encoder_sha:
                    raise ValueError("head initialization changed the frozen encoder")
                report("head200")
            if stage == "BODY_FT" and success in (50, 6000):
                report(f"body{success}")
    except (FloatingPointError, RuntimeError, ValueError, AssertionError, InterruptedError) as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        clock["elapsed_seconds"] = inherited_elapsed + time.monotonic() - started
        previous = {"checkpoint_sha256": None, "clock": None, "binary_retained": False}
        receipt = {"failure_error": {"type": type(error).__name__, "message": str(error)},
                   "failure_not_safe_for_automatic_resume": True, "clock": copy.deepcopy(clock),
                   "pending": copy.deepcopy(pending), "last_complete_boundary": previous,
                   "forensic_checkpoint_saved": False}
        try:
            if checkpoint_path.exists():
                previous.update({"checkpoint_sha256": digest(checkpoint_path),
                                 "clock": torch.load(checkpoint_path, map_location="cpu", weights_only=False)["clock"],
                                 "binary_retained": True})
            failed = failure_state(encoder, heads, optimizer, samplers, clock, exposures, pending, error, previous)
            failed.update({"protocol_sha256": protocol_sha, "encoder_state_sha256": state_digest(failed["encoder"]),
                           "body_gradient_witness": body_witness})
            atomic_torch(checkpoint_path, failed)
            receipt["forensic_checkpoint_saved"] = True
            previous["binary_retained"] = False
        except (OSError, RuntimeError, ValueError, AssertionError, MemoryError) as save_error:
            receipt["failure_save_error"] = f"{type(save_error).__name__}: {save_error}"
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError as cleanup_error:
                    receipt["failure_temporary_cleanup_error"] = str(cleanup_error)
        try:
            (directory / "failure.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        except OSError as receipt_error:
            receipt["failure_receipt_write_error"] = str(receipt_error)
        print(json.dumps(receipt, allow_nan=False), flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        trace.close()
    return json.loads((directory / f"report_body{body_stop}.json").read_text())
