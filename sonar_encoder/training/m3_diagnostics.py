"""Frozen inference-only M3 sensitivity diagnostics; no efficacy or gate decision."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from .config_eval import dense_metrics, retrieval_metrics, select_frames, summarize, token_geometry
from .data import SonarDataset
from .frontends import _metric_update, footprint_descriptor
from .model import build_encoder, prepare_native
from .readout import state_digest, write_json
from .train import file_digest, manifest_digest


def wrong_calibration(sample, factor):
    configuration = sample["configuration"]
    minimum = configuration["RangeMin"]
    bearings = sample["bearing_rad"]
    center = (bearings[0] + bearings[-1]) / 2
    return {**sample,
            "range_m": minimum + factor * (sample["range_m"] - minimum),
            "bearing_rad": center + factor * (bearings - center),
            "configuration": {**configuration,
                              "RangeMax": minimum + factor * (configuration["RangeMax"] - minimum)}}


def metadata_calibration(sample, donor):
    configuration = sample["configuration"]
    fraction = ((sample["range_m"] - configuration["RangeMin"]) /
                (configuration["RangeMax"] - configuration["RangeMin"]))
    bearings = sample["bearing_rad"]
    center = (bearings[0] + bearings[-1]) / 2
    return {**sample,
            "range_m": donor["RangeMin"] + fraction * (donor["RangeMax"] - donor["RangeMin"]),
            "bearing_rad": center + (bearings - center) * donor["Azimuth"] / configuration["Azimuth"],
            "configuration": {**configuration, **donor}}


def constant_descriptor(descriptor, constants):
    changed = descriptor.clone()
    changed[..., 3] = constants["descriptor_x"]
    changed[..., 4] = constants["descriptor_y"]
    changed[..., 5] = changed[..., 3] + changed[..., 4]
    changed[..., 6] = changed[..., 4] - changed[..., 3]
    return changed


@contextmanager
def stem_condition(stem, condition, constants, branch_masks):
    scales = stem.residual_scale.detach().clone()
    hooks = []
    try:
        if condition in branch_masks:
            with torch.no_grad():
                stem.residual_scale.mul_(stem.residual_scale.new_tensor(branch_masks[condition]))
        elif condition == "constant_footprint":
            for gate in stem.gates:
                hooks.append(gate.register_forward_pre_hook(
                    lambda module, arguments: (constant_descriptor(arguments[0], constants),)))
        elif condition == "flat_gate":
            for branch, gate in enumerate(stem.gates):
                value = constants["gate_mean"][branch]
                hooks.append(gate.register_forward_hook(
                    lambda module, arguments, output, value=value: torch.full_like(output, value)))
        yield
    finally:
        for hook in hooks:
            hook.remove()
        with torch.no_grad():
            stem.residual_scale.copy_(scales)


def condition_patches(encoder, truth_samples, normalization, condition, constants, protocol,
                      signs, donors, clean_patches):
    model_samples = truth_samples
    query_only = condition.startswith("wrong_") or condition == "metadata_reassignment"
    if condition.startswith("wrong_"):
        level = protocol["wrong_calibration"]["levels"][condition.removeprefix("wrong_")]
        model_samples = [sample if sample["config_id"] == "C00" else
                         wrong_calibration(sample, level ** signs[sample["sample_id"]])
                         for sample in truth_samples]
    elif condition == "metadata_reassignment":
        model_samples = [sample if sample["config_id"] == "C00" else
                         metadata_calibration(sample, donors[sample["sample_id"]]) for sample in truth_samples]
    with stem_condition(encoder.stem, condition, constants, protocol["branch_masks"]):
        patches = encoder.forward_native(model_samples, normalization, None)["patch_tokens"]
    patches = patches.float().cpu().reshape(len(truth_samples), 16, 16, -1)
    if query_only:
        reference = next(index for index, sample in enumerate(truth_samples) if sample["config_id"] == "C00")
        patches[reference] = clean_patches[reference]
    return patches


def branch_fields(stem, sample, normalization):
    """Use the exact sampler for contributions and independent unit weights for support."""
    device = stem.weights.device
    z = prepare_native(sample, normalization, None).to(device)
    valid = sample["valid"].to(device)
    ranges, bearings = (sample[key].to(device) for key in ("range_m", "bearing_rad"))
    descriptor = footprint_descriptor(ranges, bearings, valid,
                                      (sample["configuration"]["RangeMin"], sample["configuration"]["RangeMax"]))
    result = []
    for branch in range(2):
        if stem.variant == "R013":
            shape, padding = ((5, 1), (2, 0)) if branch == 0 else ((1, 5), (0, 2))
            kernel = stem.weights[branch].reshape(1, 1, *shape)
            support = (F.conv2d(valid.float()[None, None], torch.ones_like(kernel), padding=padding)[0, 0] == 5) & valid
            update = F.conv2d(z[None], kernel, padding=padding)[0, 0].masked_fill(~support, 0)
        else:
            support = _metric_update(torch.ones_like(z), valid, ranges, bearings,
                                     torch.ones_like(stem.weights[branch]), branch == 0) != 0
            update = _metric_update(z, valid, ranges, bearings, stem.weights[branch], branch == 0)
        if stem.variant in ("R013", "R021"):
            gate = stem.gates[branch](descriptor).squeeze(-1)
            contribution = stem.residual_scale[branch] * update * gate
            gate = gate[support].double().cpu()
        else:
            gate = None
            contribution = stem.residual_scale[branch] * update
        result.append({"support_cells": int(support.sum()), "valid_cells": int(valid.sum()),
                       "gate": gate, "anisotropy": descriptor[..., 6][support].double().cpu(),
                       "contribution_rms": float(contribution[valid].double().square().mean().sqrt()),
                       "input_rms": float(z[0, valid].double().square().mean().sqrt())})
    return descriptor[valid].double().cpu(), result


def gate_statistics(gate, anisotropy):
    gate_centered, anisotropy_centered = gate - gate.mean(), anisotropy - anisotropy.mean()
    denominator = (gate_centered.square().sum() * anisotropy_centered.square().sum()).sqrt()
    association = float((gate_centered * anisotropy_centered).sum() / denominator) if denominator != 0 else None
    quantiles = gate.quantile(torch.tensor([.05, .5, .95], dtype=torch.double))
    return {"cells": len(gate), "mean": float(gate.mean()), "std": float(gate.std(correction=0)),
            "q05": float(quantiles[0]), "q50": float(quantiles[1]), "q95": float(quantiles[2]),
            "pearson_log_r_dtheta_over_dr": association,
            "pearson_defined": association is not None}


def audit_metadata(frames, data_root, mapping):
    headers, metadata_hashes = {}, {}
    for relative in sorted({frame["metadata_relpath"] for frame in frames}):
        path = data_root / relative
        metadata_hashes[str(path)] = file_digest(path)
        metadata = json.loads(path.read_text())
        for frame in (frame for frame in frames if frame["metadata_relpath"] == relative):
            configuration = metadata["frames"][frame["sample_id"].rsplit("/", 1)[1]]["configuration"]
            headers[frame["sample_id"]] = {key: configuration[key] for key in ("RangeMin", "RangeMax", "Azimuth")}
    header_hashes = {sample_id: hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()
                     for sample_id, header in headers.items()}
    lookup = {(frame["pose_id"], frame["config_id"]): frame for frame in frames}
    groups = sorted({frame["group_id"] for frame in frames})
    group_poses = {group: sorted({frame["pose_id"] for frame in frames if frame["group_id"] == group}) for group in groups}
    signs = {}
    for group in groups:
        for config in sorted({frame["config_id"] for frame in frames}):
            selected = sorted((frame for frame in frames if frame["group_id"] == group and frame["config_id"] == config),
                              key=lambda frame: frame["pose_id"])
            signs.update({frame["sample_id"]: 1 if rank % 2 == 0 else -1 for rank, frame in enumerate(selected)})
    donors, pairs = {}, []
    for frame in frames:
        if frame["config_id"] == "C00":
            continue
        sample_id = frame["sample_id"]
        config = mapping[frame["config_id"]]
        within = lookup[(frame["pose_id"], config)]["sample_id"]
        next_group = groups[(groups.index(frame["group_id"]) + 1) % len(groups)]
        rank = group_poses[frame["group_id"]].index(frame["pose_id"])
        across_pose = group_poses[next_group][rank % len(group_poses[next_group])]
        across = lookup[(across_pose, config)]["sample_id"]
        if headers[sample_id] == headers[within] or headers[sample_id] == headers[across]:
            raise ValueError(f"metadata reassignment is a model-readable no-op: {sample_id}")
        if headers[within] != headers[across]:
            raise ValueError(f"within/across calibration changed; frozen collapsed condition is no longer applicable: {sample_id}")
        donors[sample_id] = headers[within]
        pairs.append({"target_sample_id": sample_id, "within_donor_sample_id": within,
                      "across_donor_sample_id": across, "target_header_sha256": header_hashes[sample_id],
                      "within_header_sha256": header_hashes[within], "across_header_sha256": header_hashes[across],
                      "within_across_equal": True, "query_calibration_changed": True})
    return donors, signs, {"query_pairs": len(pairs), "all_within_across_equal": True,
                          "all_query_calibration_changed": True, "group_controls_collapsed": True,
                          "executed_condition": "metadata_reassignment", "headers": headers,
                          "metadata_sha256": metadata_hashes, "pairs": pairs,
                          "meaning": "Target-shaped bounds/FOV reassignment; identical derived within/across controls; no acquisition-group causal contrast."}


def training_constants(stem, dataset, normalization):
    descriptor_means, gates, sources, metadata_hashes = [], [[], []], [], {}
    support_counts = [0, 0]
    for index in range(len(dataset)):
        sample = dataset[index]
        path = dataset.data_root / sample["npz_relpath"]
        digest = file_digest(path)
        if digest != sample["sha256"]:
            raise ValueError(f"train constant source differs from frozen manifest: {path}")
        metadata_path = dataset.data_root / sample["metadata_relpath"]
        metadata_hashes[str(metadata_path)] = file_digest(metadata_path)
        descriptor, fields = branch_fields(stem, sample, normalization)
        descriptor_means.append(descriptor[:, 3:5].mean(0))
        sources.append({"sample_id": sample["sample_id"], "npz_relpath": sample["npz_relpath"], "sha256": digest})
        for branch, field in enumerate(fields):
            support_counts[branch] += field["support_cells"]
            if field["gate"] is not None:
                gates[branch].append(float(field["gate"].mean()))
    descriptor_mean = torch.stack(descriptor_means).mean(0)
    return {"descriptor_x": float(descriptor_mean[0]), "descriptor_y": float(descriptor_mean[1]),
            "gate_mean": [sum(values) / len(values) for values in gates] if stem.variant != "R020" else None,
            "frames": len(dataset), "branch_support_cells": support_counts,
            "aggregation": "equal frame mean; descriptor over valid cells, actual sigmoid gates over branch-active cells",
            "selected_sources": sources, "metadata_sha256": metadata_hashes}


def clean_branch_statistics(stem, sample, normalization):
    _, fields = branch_fields(stem, sample, normalization)
    result = []
    for branch, field in enumerate(fields):
        result.append({"branch": "radial" if branch == 0 else "tangential",
                       "support_cells": field["support_cells"], "valid_cells": field["valid_cells"],
                       "scaled_gated_contribution_rms": field["contribution_rms"],
                       "valid_input_rms": field["input_rms"],
                       "rms_population": "valid native cells, including zero contributions outside branch support",
                       "gate": gate_statistics(field["gate"], field["anisotropy"]) if field["gate"] is not None else None})
    return result


def diagnostic_summary(rows, configs, clean_rows):
    sensitivity, branches = {}, {}
    for config in configs:
        selected = [row for row in rows if row["config_id"] == config]
        sensitivity[config] = {key: sum(row["same_input"][key] for row in selected) / len(selected)
                               for key in ("global_cosine", "patch_cosine", "patch_mean_abs_difference")}
        sensitivity[config]["patch_max_abs_difference"] = max(row["same_input"]["patch_max_abs_difference"] for row in selected)
        branches[config] = []
        for branch in range(2):
            values = [row["clean_branches"][branch] for row in clean_rows if row["config_id"] == config]
            stats = {"branch": values[0]["branch"], "frames": len(values),
                     "support_cells": sum(value["support_cells"] for value in values),
                     "valid_cells": sum(value["valid_cells"] for value in values),
                     "mean_frame_scaled_gated_contribution_rms": sum(value["scaled_gated_contribution_rms"] for value in values) / len(values),
                     "mean_frame_valid_input_rms": sum(value["valid_input_rms"] for value in values) / len(values),
                     "gate": None}
            if values[0]["gate"] is not None:
                gates = [value["gate"] for value in values]
                associations = [gate["pearson_log_r_dtheta_over_dr"] for gate in gates if gate["pearson_defined"]]
                stats["gate"] = {key: sum(gate[key] for gate in gates) / len(gates) for key in ("mean", "std", "q05", "q50", "q95")}
                stats["gate"].update(pearson_defined_frames=len(associations),
                                      pearson_undefined_frames=len(gates) - len(associations),
                                      mean_defined_frame_pearson=sum(associations) / len(associations) if associations else None,
                                      aggregation="equal frame mean of support statistics; quantiles are mean frame quantiles")
            branches[config].append(stats)
    return {"same_input_per_config": sensitivity, "clean_branches_per_config": branches,
            "support": "Same-index patch cosine/mean abs use clean valid token support; maximum abs uses all patch tokens. Dense metrics and coverage use untouched clean native geometry."}


def evaluate_conditions(encoder, dataset, normalization, protocol, conditions, constants,
                        donors, signs, metadata_audit, exact_noop, role_protocol, split):
    configs = protocol["population"]["configs"]
    poses = sorted({frame["pose_id"] for frame in dataset.frames})
    if exact_noop and torch.count_nonzero(encoder.stem.residual_scale) != 0:
        raise ValueError("step0 residual scales are not exactly zero")
    analytical_noop = split == "test" and exact_noop
    executed_conditions = ["clean"] if analytical_noop else conditions
    features = {condition: {config: [] for config in configs} for condition in executed_conditions}
    rows = {condition: [] for condition in executed_conditions}
    across_donors = {pair["target_sample_id"]: metadata_audit["headers"][pair["across_donor_sample_id"]]
                     for pair in metadata_audit["pairs"]}
    sources, derived_audit = [], []
    for start in range(0, len(dataset), len(configs)):
        truth_samples = [dataset[index] for index in range(start, start + len(configs))]
        geometry = [token_geometry(sample) for sample in truth_samples]
        reference = configs.index("C00")
        for sample in truth_samples:
            path = dataset.data_root / sample["npz_relpath"]
            digest = file_digest(path)
            if digest != sample["sha256"]:
                raise ValueError(f"native frame differs from frozen manifest: {path}")
            metadata_path = dataset.data_root / sample["metadata_relpath"]
            if file_digest(metadata_path) != metadata_audit["metadata_sha256"][str(metadata_path)]:
                raise ValueError(f"metadata changed after complete header audit: {metadata_path}")
            sources.append({"sample_id": sample["sample_id"], "npz_relpath": sample["npz_relpath"], "sha256": digest})
            if sample["config_id"] != "C00":
                changed = metadata_calibration(sample, donors[sample["sample_id"]])
                across = metadata_calibration(sample, across_donors[sample["sample_id"]])
                if any(not torch.equal(changed[key], across[key]) for key in ("range_m", "bearing_rad")):
                    raise ValueError(f"within/across derived inputs differ: {sample['sample_id']}")
                range_changed = not torch.equal(changed["range_m"], sample["range_m"])
                bearing_changed = not torch.equal(changed["bearing_rad"], sample["bearing_rad"])
                if not range_changed and not bearing_changed:
                    raise ValueError(f"derived metadata reassignment is a no-op: {sample['sample_id']}")
                derived_audit.append({"sample_id": sample["sample_id"], "range_changed": range_changed,
                                      "bearing_changed": bearing_changed,
                                      "intensity_shape_valid_preserved": True,
                                      "within_across_derived_equal": True})
        clean = condition_patches(encoder, truth_samples, normalization, "clean", constants,
                                  protocol, signs, donors, None)
        clean_branches = [clean_branch_statistics(encoder.stem, sample, normalization) for sample in truth_samples]
        for condition in executed_conditions:
            gallery_condition = "clean" if condition.startswith("wrong_") or condition == "metadata_reassignment" else condition
            patches = clean if condition == "clean" else condition_patches(
                encoder, truth_samples, normalization, condition, constants, protocol, signs, donors, clean)
            if exact_noop and not torch.equal(patches, clean):
                raise RuntimeError(f"step0 condition is not bit-identical: {condition}, {truth_samples[0]['pose_id']}")
            for index, sample in enumerate(truth_samples):
                pooled, clean_pooled = patches[index].mean((0, 1)), clean[index].mean((0, 1))
                features[condition][sample["config_id"]].append(pooled)
                valid_tokens = geometry[index][2]
                difference = patches[index].double() - clean[index].double()
                sensitivity = {"global_cosine": float(F.cosine_similarity(pooled.double(), clean_pooled.double(), dim=0)),
                               "patch_cosine": float(F.cosine_similarity(patches[index][valid_tokens].double(),
                                                                         clean[index][valid_tokens].double(), dim=-1).mean()),
                               "patch_mean_abs_difference": float(difference[valid_tokens].abs().mean()),
                               "patch_max_abs_difference": float(difference.abs().max()),
                               "valid_patch_tokens": int(valid_tokens.sum()), "candidate_patch_tokens": 256,
                               "patch_bit_identical": bool(torch.equal(patches[index], clean[index]))}
                row = {"sample_id": sample["sample_id"], "condition": condition, "gallery_condition": gallery_condition,
                       "split": split, "final_test_evaluated": split == "test",
                       "pose_id": sample["pose_id"], "group_id": sample["group_id"],
                       "scene_id": sample["scene_id"], "config_id": sample["config_id"],
                       "dense": dense_metrics(patches[index], geometry[index], sample,
                                              patches[reference], geometry[reference], truth_samples[reference]),
                       "same_input": sensitivity}
                if condition.startswith("wrong_"):
                    sign = signs[sample["sample_id"]]
                    row["stress_stratum"] = "clean_reference" if sample["config_id"] == "C00" else "expansion" if sign == 1 else "contraction"
                    row["stress_factor"] = 1. if sample["config_id"] == "C00" else protocol["wrong_calibration"]["levels"][condition.removeprefix("wrong_")] ** sign
                if condition == "clean":
                    row["clean_branches"] = clean_branches[index]
                rows[condition].append(row)
        completed = start // len(configs) + 1
        if completed == 1 or completed % 20 == 0 or completed == len(poses):
            print(json.dumps({"progress": "inference", "pose": completed, "poses": len(poses),
                              "exact_step0_noop": exact_noop}), flush=True)
    reports = {}
    for condition in executed_conditions:
        gallery = torch.stack(features[condition]["C00"])
        global_results = {}
        for config in configs:
            for row in retrieval_metrics(torch.stack(features[condition][config]), gallery, poses, poses):
                global_results[(row["pose_id"], config)] = row
        for row in rows[condition]:
            row["global"] = global_results[(row["pose_id"], row["config_id"])]
        report = {"condition": condition, "gallery_condition": "clean" if condition.startswith("wrong_") or condition == "metadata_reassignment" else condition,
                  "split": split, "final_test_evaluated": split == "test", "execution": "computed",
                  "summary": summarize(rows[condition], configs, split, role_protocol),
                  "diagnostic_statistics": diagnostic_summary(rows[condition], configs, rows["clean"]),
                  "per_pose": rows[condition], "all_patch_tokens_bit_identical_to_clean": all(row["same_input"]["patch_bit_identical"] for row in rows[condition])}
        reports[condition] = report
    if analytical_noop:
        # Every intervention acts only through branches whose actual scalar multiplier is exactly zero.
        for condition in conditions:
            if condition == "clean":
                continue
            gallery_condition = "clean" if condition.startswith("wrong_") or condition == "metadata_reassignment" else condition
            shared_rows = [{**row, "condition": condition, "gallery_condition": gallery_condition} for row in rows["clean"]]
            if condition.startswith("wrong_"):
                for row in shared_rows:
                    sign = signs[row["sample_id"]]
                    row["stress_stratum"] = "clean_reference" if row["config_id"] == "C00" else "expansion" if sign == 1 else "contraction"
                    row["stress_factor"] = 1. if row["config_id"] == "C00" else protocol["wrong_calibration"]["levels"][condition.removeprefix("wrong_")] ** sign
            reports[condition] = {**reports["clean"], "condition": condition, "gallery_condition": gallery_condition,
                                  "per_pose": shared_rows, "execution": "analytical_zero_residual_noop",
                                  "shared_computed_condition": "clean"}
    for condition in conditions:
        report = reports[condition]
        if condition.startswith("wrong_"):
            report["stress_strata"] = {}
            for stratum in protocol["wrong_calibration"]["strata"]:
                selected = [row for row in report["per_pose"] if row["stress_stratum"] == stratum]
                report["stress_strata"][stratum] = {
                    "frames": len(selected),
                    "mean_global_cosine": sum(row["same_input"]["global_cosine"] for row in selected) / len(selected) if selected else None,
                    "mean_patch_cosine": sum(row["same_input"]["patch_cosine"] for row in selected) / len(selected) if selected else None}
    return {"split": split, "configs": configs, "gallery_config": "C00", "gallery_poses": poses,
            "gallery_size": len(poses), "final_test_evaluated": split == "test", "conditions": reports,
            "step0_noop_basis": {"residual_scale": encoder.stem.residual_scale.detach().cpu().tolist(),
                                  "execution": "clean_computed_interventions_shared_analytically" if analytical_noop else "all_conditions_computed"} if exact_noop else None,
            "derived_input_audit": {"query_frames": len(derived_audit), "all_query_inputs_changed": True,
                                    "range_changed_queries": sum(row["range_changed"] for row in derived_audit),
                                    "bearing_changed_queries": sum(row["bearing_changed"] for row in derived_audit),
                                    "both_changed_queries": sum(row["range_changed"] and row["bearing_changed"] for row in derived_audit),
                                    "all_truth_intensity_shape_valid_preserved": True,
                                    "within_across_derived_equal": True, "per_query": derived_audit}}, sources


@torch.inference_mode()
def _run_diagnostics(protocol_path, variant, device, output_path, smoke, final):
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if final:
        if (protocol["status"] != "FROZEN_BEFORE_FINAL_DIAGNOSTIC_METRICS" and
                not (smoke and protocol["status"] == "PROSPECTIVE_FINAL_DIAGNOSTICS_BEFORE_FREEZE")):
            raise ValueError("a frozen final diagnostic protocol is required before final data access")
    elif protocol["status"] != "FROZEN_BEFORE_NEW_DIAGNOSTIC_METRICS":
        raise ValueError("development diagnostic protocol is not frozen")
    if (not protocol["inference_only"] or protocol["trained_m3_controls_complete"] or
            protocol["m2_gate_unchanged"] != "NO_GO"):
        raise ValueError("diagnostic protocol does not preserve frozen inference-only scope")
    manifest_path, normalization_path = Path(protocol["source_manifest"]), Path(protocol["normalization"])
    role_path, data_root = Path(protocol["role_protocol"]), Path(protocol["data_root"])
    if file_digest(manifest_path) != protocol["source_manifest_sha256"]:
        raise ValueError("manifest differs from frozen diagnostic protocol")
    if file_digest(role_path) != protocol["role_protocol_sha256"]:
        raise ValueError("configuration role protocol differs from frozen diagnostic protocol")
    source_dir = Path(__file__).parent
    source_initial = {name: file_digest(source_dir / name) for name in
                      ("m3_diagnostics.py", "config_eval.py", "data.py", "model.py", "frontends.py", "readout.py", "train.py", "artifacts.py")}
    for name, expected in protocol["source_sha256"].items():
        if source_initial[name] != expected:
            raise ValueError(f"encoder/evaluation source differs from frozen protocol: {name}")
    manifest, normalization = json.loads(manifest_path.read_text()), json.loads(normalization_path.read_text())
    role_protocol = json.loads(role_path.read_text())
    if normalization["source_manifest_sha256"] != manifest_digest(manifest):
        raise ValueError("normalization provenance does not match manifest")
    if role_protocol["source_manifest_sha256"] != file_digest(manifest_path):
        raise ValueError("role protocol source manifest differs from frozen manifest")
    if final:
        final_population = protocol["population"]
        final_configs = [f"C{index:02d}" for index in range(14)]
        if (final_population["split"] != "test" or final_population["configs"] != final_configs or
                final_population["poses"] != 442 or len(final_population["sample_ids"]) != 442 * 14):
            raise ValueError("final population must declare all 14 configurations and 442 sealed test poses")
        if set(protocol["metadata_reassignment"]["mapping"]) != set(final_configs) - {"C00"}:
            raise ValueError("final metadata reassignment must declare all 13 query donors")
        variants = {"R013", "R020", "R021"}
        if set(protocol["checkpoints"]) != variants or set(protocol["checkpoint_sha256"]) != variants:
            raise ValueError("final checkpoints must contain only R013/R020/R021")
        if any(set(protocol[key][name]) != {"step0", "step10000"}
               for key in ("checkpoints", "checkpoint_sha256") for name in variants):
            raise ValueError("final checkpoints must be fixed at steps 0 and 10000")
        if (protocol["constant_training_population"]["frames"] != 192 or
                protocol["constant_training_population"]["usage"] != "ssl_train"):
            raise ValueError("final constants must use the frozen 192 train-only frames")
        population = protocol["smoke_population"] if smoke else final_population
        split = population["split"]
        if smoke and (split != "development" or population["poses"] != 2):
            raise ValueError("final smoke must use two approved development poses")
    else:
        population, split = protocol["population"], "development"
    configs = population["configs"]
    frames = select_frames(manifest, split, configs, None, role_protocol)
    audit_frames = frames
    if final and smoke:
        membership = set(population["sample_ids"])
        frames = [frame for frame in frames if frame["sample_id"] in membership]
    if ([frame["sample_id"] for frame in frames] != population["sample_ids"] or
            len({frame["pose_id"] for frame in frames}) != population["poses"] or
            len(frames) != population["poses"] * len(configs)):
        raise ValueError(f"{split} population differs from frozen diagnostic protocol")
    donors, signs, metadata_audit = audit_metadata(audit_frames, data_root, protocol["metadata_reassignment"]["mapping"])
    train_lookup = {frame["sample_id"]: frame for frame in manifest["frames"]}
    train_frames = [train_lookup[sample_id] for sample_id in protocol["constant_training_population"]["sample_ids"]]
    if (len(train_frames) != protocol["constant_training_population"]["frames"] or
            any(frame["usage"] != protocol["constant_training_population"]["usage"] for frame in train_frames)):
        raise ValueError("constant population is not the frozen train-only selection")
    train_dataset = SonarDataset(manifest_path, data_root, "ssl_train")
    train_dataset.frames = train_frames
    dataset = SonarDataset(manifest_path, data_root, "test_seen" if split == "test" else "ssl_validation")
    dataset.frames = ([frame for frame in frames if frame["pose_id"] in protocol["step0"]["smoke_poses"]]
                      if smoke and not final else frames)
    if not final:
        baseline_path = Path(protocol["step0"]["full_baseline_reports"][variant])
        baseline = json.loads(baseline_path.read_text())
        if (baseline["provenance"]["checkpoint_sha256"] != protocol["checkpoint_sha256"][variant]["step0"] or
                baseline["provenance"]["checkpoint_step"] != 0 or baseline["gallery_size"] != population["poses"] or
                baseline["configs"] != configs or baseline["split"] != "development" or baseline["final_test_evaluated"] or
                baseline["gallery_poses"] != sorted({frame["pose_id"] for frame in frames}) or
                baseline["provenance"]["manifest_sha256"] != protocol["source_manifest_sha256"] or
                baseline["provenance"]["role_protocol_sha256"] != protocol["role_protocol_sha256"]):
            raise ValueError("existing full step0 baseline differs from frozen scope/checkpoint")
        full_step0 = {"path": str(baseline_path.resolve()), "sha256": file_digest(baseline_path),
                      "summary": baseline["summary"], "gallery_size": baseline["gallery_size"],
                      "usage": "existing complete step0 baseline retained; full step0 interventions not rerun"}
    conditions = list(protocol["conditions"]["all"])
    if variant in ("R013", "R021"):
        conditions.extend(protocol["conditions"]["gated_only"])
    reports = {}
    for step_name in (("step0", "step10000") if smoke or final else ("step10000",)):
        source_before = {name: file_digest(source_dir / name) for name in source_initial}
        if source_before != source_initial:
            raise RuntimeError("source changed after frozen protocol verification")
        checkpoint_path = Path(protocol["checkpoints"][variant][step_name])
        checkpoint_sha = file_digest(checkpoint_path)
        if checkpoint_sha != protocol["checkpoint_sha256"][variant][step_name]:
            raise ValueError(f"checkpoint differs from frozen protocol: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if file_digest(checkpoint_path) != checkpoint_sha:
            raise ValueError("checkpoint changed while loading")
        config = saved["resolved_config"]
        if config["variant"] != variant or saved["step"] != int(step_name.removeprefix("step")):
            raise ValueError("checkpoint variant/step differs from diagnostic selection")
        for key, path in (("manifest", manifest_path), ("normalization", normalization_path)):
            if saved["provenance"][key] != file_digest(path):
                raise ValueError(f"checkpoint {key} provenance differs from diagnostic input")
        if file_digest(config["weights_path"]) != saved["provenance"]["weights"]:
            raise ValueError("foundation weights differ from checkpoint provenance")
        encoder = build_encoder(Path(config["code_dir"]), Path(config["weights_path"]), variant)
        encoder.load_state_dict(saved["student"])
        encoder.eval().requires_grad_(False).to(device)
        checkpoint_step, training_provenance = saved["step"], saved["provenance"]
        del saved
        before = state_digest(encoder.state_dict())
        exact_noop = step_name == "step0"
        constants = training_constants(encoder.stem, train_dataset, normalization)
        print(json.dumps({"variant": variant, "checkpoint": step_name, "train_constants_frames": constants["frames"]}), flush=True)
        report, sources = evaluate_conditions(encoder, dataset, normalization, {**protocol, "population": population}, conditions, constants,
                                              donors, signs, metadata_audit, exact_noop, role_protocol, split)
        after = state_digest(encoder.state_dict())
        if before != after:
            raise RuntimeError("encoder changed during inference-only diagnostics")
        checkpoint_after = file_digest(checkpoint_path)
        if checkpoint_after != checkpoint_sha:
            raise RuntimeError("checkpoint changed during inference-only diagnostics")
        source_after = {name: file_digest(source_dir / name) for name in source_before}
        if source_before != source_after:
            raise RuntimeError("source changed during inference-only diagnostics")
        report.update(train_constants=constants, step0_noop_verified=exact_noop,
                      provenance={"variant": variant, "encoder_weights": "student", "checkpoint_step": checkpoint_step,
                                  "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha,
                                  "checkpoint_sha256_before": checkpoint_sha, "checkpoint_sha256_after": checkpoint_after,
                                  "checkpoint_training_provenance": training_provenance,
                                  "encoder_hash_before": before, "encoder_hash_after": after,
                                  "source_sha256_before": source_before, "source_sha256_after": source_after,
                                  "selected_sources": sources, "feature_dtype": "float32", "metric_dtype": "float64"})
        reports[step_name] = report
        del encoder
    result = {"inference_only": True, "trained_m3_controls_complete": False, "m2_gate_unchanged": "NO_GO",
              "mode": ("final_validation_smoke" if smoke else "full_final_step0_and_step10000") if final else ("smoke" if smoke else "full_development_step10000"),
              "split": split, "configs": configs, "final_test_evaluated": split == "test",
              "steps": reports, "metadata_audit": metadata_audit,
              "interpretation": protocol["interpretation"], "remaining_original_requirements": protocol["remaining_original_requirements"],
              "limitations": ["Exact same-map single-seed proxy; acquisition groups are not independent scene identities.",
                              "Continuous descriptive sensitivity has no pass threshold or unlock and does not establish beneficial PMCA efficacy.",
                              "Wrong calibration is joint radial/angular algorithmic stress, alternating expansion q and contraction 1/q within group/configuration.",
                              "Dense metrics use clean r/theta common-domain geometry and nominal token centers, not localized receptive fields or equal-area XY.",
                              "Gate associations are descriptive; zero gate/anisotropy variance makes Pearson undefined (null)."] +
                             (["Smoke retrieval uses only the two frozen development poses and is a path/no-op check, not the full diagnostic result."] if smoke else []),
              "provenance": {"manifest": str(manifest_path.resolve()), "manifest_sha256": file_digest(manifest_path),
                             "normalization": str(normalization_path.resolve()), "normalization_sha256": file_digest(normalization_path),
                             "protocol": str(protocol_path.resolve()), "protocol_sha256": protocol_sha,
                             "role_protocol": str(role_path.resolve()), "role_protocol_sha256": file_digest(role_path),
                             "data_root": str(data_root.resolve()), "metadata_sha256": metadata_audit["metadata_sha256"],
                             "source_sha256": source_after,
                             "device": str(device), "python": sys.version, "torch": str(torch.__version__)}}
    if not final:
        result["existing_full_step0_baseline"] = full_step0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def run_diagnostics(protocol_path, variant, device, output_path, smoke=False):
    return _run_diagnostics(protocol_path, variant, device, output_path, smoke, False)


def run_final_diagnostics(protocol_path, variant, device, output_path, smoke=False):
    return _run_diagnostics(protocol_path, variant, device, output_path, smoke, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", choices=("R013", "R020", "R021"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.final:
        result = run_final_diagnostics(args.protocol, args.variant, args.device, args.output, args.smoke)
    else:
        result = run_diagnostics(args.protocol, args.variant, args.device, args.output, args.smoke)
    print(json.dumps({"output": str(args.output), "variant": args.variant, "mode": result["mode"],
                      "gallery_sizes": {name: step["gallery_size"] for name, step in result["steps"].items()}}))


if __name__ == "__main__":
    main()
