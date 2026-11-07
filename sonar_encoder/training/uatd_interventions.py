"""Frozen UATD image-head diagnostics under recipient-preserving calibration stress."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import create_artifact_directory
from .m3_diagnostics import metadata_calibration, stem_condition, wrong_calibration
from .model import build_encoder
from .readout import file_digest, state_digest, write_json
from .uatd_head import box_targets, dense_loss, evaluate_heads, global_targets
from .uatd_readout import native_sample as clean_native_sample, prepared_rows


def native_sample(row):
    sample = clean_native_sample(row)
    sample["configuration"]["Azimuth"] = row["horizontal_fov_deg"]
    return sample


def intervention_assignments(rows):
    ordered = sorted(rows, key=lambda row: row["row_id"])
    groups = {}
    for row in ordered:
        key = (*row["shape"][:2], row["range_m"], row["horizontal_fov_deg"])
        groups.setdefault(key, []).append(row["row_id"])
    signs = {row_id: 1 if rank % 2 == 0 else -1 for ids in groups.values() for rank, row_id in enumerate(ids)}
    donors, pairs = {}, []
    for index, row in enumerate(ordered):
        donor = ordered[(index + 1) % len(ordered)]
        header = {"RangeMin": 0., "RangeMax": donor["range_m"], "Azimuth": donor["horizontal_fov_deg"]}
        donors[row["row_id"]] = header
        pairs.append({"row_id": row["row_id"], "donor_row_id": donor["row_id"], "donor_header": header,
                      "header_noop": row["range_m"] == donor["range_m"] and
                                     row["horizontal_fov_deg"] == donor["horizontal_fov_deg"]})
    return signs, donors, {"wrong_calibration_groups": [
        {"native_calibration_tuple": list(key), "tuple_fields": ["height", "width", "range_m", "horizontal_fov_deg"],
         "sorted_row_ids": ids, "signs": [signs[row_id] for row_id in ids]} for key, ids in groups.items()],
        "metadata_reassignment": pairs, "metadata_header_noops": sum(pair["header_noop"] for pair in pairs),
        "donor_rule": "cyclic next sorted row_id within the actual evaluation split; Range/FOV only, no group identity"}


def zero_residual_state(encoder):
    if torch.count_nonzero(encoder.stem.residual_scale) != 0:
        raise ValueError("step0 residual scales are not exactly zero")
    return {"residual_scale": encoder.stem.residual_scale.detach().cpu().tolist(),
            "polar_pe": "not_present_in_R013_R020_R021"}


@torch.inference_mode()
def encode_condition(encoder, rows, normalization, condition, constants, protocol, signs, donors):
    features = torch.empty((len(rows), 384, 16, 16), dtype=torch.float32)
    for start in range(0, len(rows), 4):
        samples = [native_sample(row) for row in rows[start:start + 4]]
        if condition.startswith("wrong_"):
            level = protocol["wrong_calibration"]["levels"][condition.removeprefix("wrong_")]
            samples = [wrong_calibration(sample, level ** signs[sample["sample_id"]]) for sample in samples]
        elif condition == "metadata_reassignment":
            samples = [sample if donors[sample["sample_id"]] ==
                       {key: sample["configuration"][key] for key in ("RangeMin", "RangeMax", "Azimuth")} else
                       metadata_calibration(sample, donors[sample["sample_id"]]) for sample in samples]
        with stem_condition(encoder.stem, condition, constants, protocol["branch_masks"]):
            patches = encoder.forward_native(samples, normalization, None)["patch_tokens"]
        features[start:start + len(samples)] = patches.transpose(1, 2).reshape(-1, 384, 16, 16).float().cpu()
    return features


@torch.inference_mode()
def evaluate_fixed_heads(global_head, dense_head, cache, device):
    metrics, predictions = evaluate_heads(global_head, dense_head, cache, device)
    classification, detection = 0., 0.
    for start in range(0, len(cache["features"]), 32):
        features = cache["features"][start:start + 32].to(device)
        targets = cache["targets"][start:start + 32]
        classification += float(F.binary_cross_entropy_with_logits(
            global_head(features.mean((-2, -1))), global_targets(targets, device))) * len(features)
        detection += float(dense_loss(dense_head(features), box_targets(targets, device))) * len(features)
    count = len(cache["features"])
    metrics["loss"] = {"global_bce": classification / count, "dense_focal_box": detection / count,
                       "total": (classification + detection) / count,
                       "aggregation": "row-weighted mean of original batch32 losses; original AP/IoU/NMS unchanged"}
    return metrics, predictions


def same_input_metrics(features, clean, row_ids):
    rows = []
    for start in range(0, len(features), 32):
        changed, reference = features[start:start + 32], clean[start:start + 32]
        pooled, clean_pooled = changed.mean((-2, -1)).double(), reference.mean((-2, -1)).double()
        cosine = F.cosine_similarity(pooled, clean_pooled, dim=1)
        patch_cosine = F.cosine_similarity(changed.double(), reference.double(), dim=1).mean((-2, -1))
        difference = (changed.double() - reference.double()).abs().flatten(1)
        for index in range(len(changed)):
            rows.append({"row_id": row_ids[start + index], "global_cosine": float(cosine[index]),
                         "patch_cosine": float(patch_cosine[index]), "patch_mean_abs_difference": float(difference[index].mean()),
                         "patch_max_abs_difference": float(difference[index].max()),
                         "patch_bit_identical": bool(torch.equal(changed[index], reference[index]))})
    return {"per_row": rows, "mean": {key: sum(row[key] for row in rows) / len(rows) for key in
            ("global_cosine", "patch_cosine", "patch_mean_abs_difference")},
            "all_patch_tokens_bit_identical": all(row["patch_bit_identical"] for row in rows),
            "meaning": "Same recipient and same token-index semantic feature sensitivity; no physical localization claim."}


@torch.inference_mode()
def run_interventions(protocol_path, variant, step, test_split, device, output, smoke=False):
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if (protocol["status"] != "FROZEN_BEFORE_FINAL_DIAGNOSTIC_METRICS" and
            not (smoke and protocol["status"] == "PROSPECTIVE_FINAL_DIAGNOSTICS_BEFORE_FREEZE")):
        raise ValueError("a frozen final protocol is required before opening final inputs")
    if not protocol["inference_only"] or protocol["m2_gate_unchanged"] != "NO_GO":
        raise ValueError("UATD interventions must preserve inference-only scope and M2 NO_GO")
    if (output / "report.json").exists():
        raise FileExistsError("intervention report already exists")
    source_dir = Path(__file__).parent
    source_before = {name: file_digest(source_dir / name) for name in
                     ("uatd_interventions.py", "m3_diagnostics.py", "uatd_readout.py", "uatd_head.py", "model.py",
                      "frontends.py", "readout.py", "train.py", "data.py", "config_eval.py", "artifacts.py")}
    for name, actual in source_before.items():
        if actual != protocol["source_sha256"][name]:
            raise ValueError(f"execution source differs from frozen protocol: {name}")
    prepared = Path(protocol["prepared"])
    if file_digest(prepared / "manifest.json") != protocol["prepared_manifest_sha256"]:
        raise ValueError("prepared Training/validation manifest changed")
    if file_digest(prepared / "normalization.json") != protocol["normalization_sha256"]:
        raise ValueError("UATD train-only normalization changed")
    manifest = json.loads((prepared / "manifest.json").read_text())
    normalization = json.loads((prepared / "normalization.json").read_text())
    if normalization["source_manifest_sha256"] != protocol["prepared_manifest_sha256"]:
        raise ValueError("normalization and prepared manifest differ")
    if file_digest(Path(manifest["protocol_path"])) != manifest["protocol_sha256"]:
        raise ValueError("original UATD readout protocol changed")
    model = protocol["models"][variant]
    settings = model["steps"][f"step{step}"]
    parent_dir = Path(settings["parent_cache"])
    parent_sha = file_digest(parent_dir / "cache_manifest.json")
    if parent_sha != settings["parent_cache_manifest_sha256"]:
        raise ValueError("original train/validation cache manifest changed")
    parent = json.loads((parent_dir / "cache_manifest.json").read_text())
    provenance = parent["provenance"]
    if (provenance["task"] != "uatd" or provenance["variant"] != variant or provenance["checkpoint_step"] != step or
            Path(provenance["prepared"]).resolve() != prepared.resolve() or
            provenance["task_manifest_sha256"] != protocol["prepared_manifest_sha256"] or
            provenance["normalization_sha256"] != protocol["normalization_sha256"] or
            provenance["protocol_sha256"] != manifest["protocol_sha256"] or parent["class_names"] != manifest["class_names"]):
        raise ValueError("original image cache is not the declared variant/step/prepared task")
    head_dir = Path(settings["head_directory"])
    selection_sha = file_digest(head_dir / "report.json")
    if selection_sha != settings["head_selection_sha256"]:
        raise ValueError("clean image-only head selection changed")
    selection = json.loads((head_dir / "report.json").read_text())
    if selection["cache_manifest_sha256"] != parent_sha or selection["best_heads_sha256"] != settings["best_heads_sha256"]:
        raise ValueError("selected heads are not bound to the declared original cache/best hashes")
    heads = {"global": nn.Linear(384, 10).to(device), "dense": nn.Conv2d(384, 50, 1).to(device)}
    for name, head in heads.items():
        path = head_dir / f"best_{name}.pt"
        if file_digest(path) != settings["best_heads_sha256"][name]:
            raise ValueError(f"selected {name} best changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["provenance"] != provenance or saved["epoch"] != selection["selected"][name]["epoch"]:
            raise ValueError(f"selected {name} head/cache/epoch differ")
        head.load_state_dict(saved["head"])
        head.eval().requires_grad_(False)
    head_before = {name: state_digest(head.state_dict()) for name, head in heads.items()}
    if smoke:
        row_lookup = {row["row_id"]: row for row in manifest["validation"]}
        row_ids = protocol["smoke_row_ids"]
        if len(row_ids) != 32 or len(set(row_ids)) != 32 or not set(row_ids) <= row_lookup.keys():
            raise ValueError("smoke must use 32 distinct approved prepared.validation rows")
        rows = [row_lookup[row_id] for row_id in row_ids]
        cache_manifest_sha = parent_sha
        cache_path = parent_dir / "validation.pt"
        cache_sha = file_digest(cache_path)
        if cache_sha != parent["cache_files"]["validation.pt"]:
            raise ValueError("original validation cache changed")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
        if cache["row_ids"] != [row["row_id"] for row in manifest["validation"]]:
            raise ValueError("original validation cache rows differ from prepared.validation")
        indices = [cache["row_ids"].index(row_id) for row_id in row_ids]
        cache = {**cache, "features": cache["features"][indices], "row_ids": row_ids,
                 "targets": [cache["targets"][index] for index in indices]}
        actual_split = "validation"
    else:
        population = protocol["test_population"][test_split]
        source = Path(population["source_manifest"])
        if (source.resolve() != Path(manifest["final_manifests"][test_split]).resolve() or
                file_digest(source) != population["source_manifest_sha256"] or
                population["source_manifest_sha256"] != manifest["final_manifest_sha256"][test_split]):
            raise ValueError("final source manifest differs from the frozen/prepared test split")
        rows = prepared_rows([json.loads(line) for line in source.read_text().splitlines()],
                             Path(manifest["real_root"]), Path(manifest["source_prefix"]), manifest["class_names"])
        if population["rows"] != 796 or len(rows) != population["rows"]:
            raise ValueError("UATD final split must contain the frozen 796 rows")
        final_dir = Path(settings["final_caches"][test_split])
        cache_manifest_sha = file_digest(final_dir / "cache_manifest.json")
        final_manifest = json.loads((final_dir / "cache_manifest.json").read_text())
        if (final_manifest["parent_cache_manifest_sha256"] != parent_sha or final_manifest["provenance"] != provenance or
                final_manifest["test_split"] != test_split or final_manifest["class_names"] != parent["class_names"] or
                Path(final_manifest["source_manifest"]).resolve() != source.resolve() or
                final_manifest["source_manifest_sha256"] != population["source_manifest_sha256"]):
            raise ValueError("clean final cache differs from original train cache and frozen test source")
        cache_path = final_dir / "final.pt"
        cache_sha = file_digest(cache_path)
        if cache_sha != final_manifest["cache_files"]["final.pt"]:
            raise ValueError("clean final cache changed")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
        actual_split = test_split
    if (cache["provenance"] != provenance or cache["row_ids"] != [row["row_id"] for row in rows] or
            tuple(cache["features"].shape) != (len(rows), 384, 16, 16) or cache["features"].dtype != torch.float32):
        raise ValueError("clean features/rows/provenance differ from the fixed image-only contract")
    for row, target in zip(rows, cache["targets"]):
        if (not torch.equal(target["labels"], torch.tensor(row["labels"], dtype=torch.long)) or
                not torch.equal(target["boxes"], torch.tensor(row["boxes"], dtype=torch.float32))):
            raise ValueError(f"clean cache targets differ from recipient ground truth: {row['row_id']}")
    constants_reference = model["train_constants"]
    constants_path = Path(constants_reference["report"])
    if file_digest(constants_path) != constants_reference["report_sha256"]:
        raise ValueError("original M3 development constants report changed")
    constants_report = json.loads(constants_path.read_text())
    constant_step = constants_report["steps"]["step10000"]
    constants = constant_step["train_constants"]
    original_protocol_path = Path(constants_report["provenance"]["protocol"])
    if file_digest(original_protocol_path) != constants_report["provenance"]["protocol_sha256"]:
        raise ValueError("original 192-frame constants protocol changed")
    original_protocol = json.loads(original_protocol_path.read_text())
    if (constant_step["provenance"]["variant"] != variant or constant_step["provenance"]["checkpoint_step"] != 10000 or
            constant_step["provenance"]["checkpoint_sha256"] != model["steps"]["step10000"]["checkpoint_sha256"] or
            constants["frames"] != 192 or original_protocol["constant_training_population"]["usage"] != "ssl_train" or
            [row["sample_id"] for row in constants["selected_sources"]] != original_protocol["constant_training_population"]["sample_ids"] or
            {key: constants[key] for key in ("descriptor_x", "descriptor_y", "gate_mean", "frames")} != constants_reference["values"]):
        raise ValueError("constants differ from the frozen same-variant 192 SSL train selection")
    checkpoint = Path(settings["checkpoint"])
    checkpoint_sha = file_digest(checkpoint)
    if (checkpoint_sha != settings["checkpoint_sha256"] or checkpoint_sha != provenance["checkpoint_sha256"] or
            checkpoint.resolve() != Path(provenance["checkpoint"]).resolve()):
        raise ValueError("encoder checkpoint differs from frozen protocol/original clean cache")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["resolved_config"]
    if config["variant"] != variant or saved["step"] != step:
        raise ValueError("checkpoint variant/step differ from requested control")
    if file_digest(Path(config["weights_path"])) != saved["provenance"]["weights"]:
        raise ValueError("foundation weights differ from checkpoint provenance")
    encoder = build_encoder(Path(config["code_dir"]), Path(config["weights_path"]), variant)
    encoder.load_state_dict(saved["student"])
    encoder.eval().requires_grad_(False).to(device)
    del saved
    encoder_before = state_digest(encoder.state_dict())
    if encoder_before != provenance["encoder_state_sha256"]:
        raise ValueError("actual encoder differs from the original clean/head encoder")
    zero_state = zero_residual_state(encoder) if step == 0 else None
    signs, donors, assignment_audit = intervention_assignments(rows)
    prefix = encode_condition(encoder, rows[:4], normalization, "clean", constants, protocol, signs, donors)
    if not torch.equal(prefix, cache["features"][:4]):
        raise ValueError("current encoder first4 clean features are not exact original-cache features")
    create_artifact_directory(output)
    conditions = list(protocol["conditions"]["all"])
    if variant in ("R013", "R021"):
        conditions.extend(protocol["conditions"]["gated_only"])
    reports = {}
    for condition in conditions:
        features = cache["features"] if condition == "clean" else encode_condition(
            encoder, rows, normalization, condition, constants, protocol, signs, donors)
        sensitivity = same_input_metrics(features, cache["features"], cache["row_ids"])
        if step == 0 and not sensitivity["all_patch_tokens_bit_identical"]:
            raise RuntimeError(f"actual step0 intervention is not exact no-op: {condition}")
        current_cache = {**cache, "features": features}
        metrics, predictions = evaluate_fixed_heads(heads["global"], heads["dense"], current_cache, device)
        encoder_after = state_digest(encoder.state_dict())
        head_after = {name: state_digest(head.state_dict()) for name, head in heads.items()}
        if encoder_before != encoder_after or head_before != head_after:
            raise RuntimeError("encoder or fixed heads changed during interventions")
        condition_output = output / condition
        condition_output.mkdir()
        torch.save(predictions, condition_output / "predictions.pt")
        report = {"condition": condition, "variant": variant, "checkpoint_step": step,
                  "split": actual_split, "requested_test_split": test_split,
                  "final_test_evaluated": not smoke, "count": len(rows), "row_ids": cache["row_ids"], "metrics": metrics,
                  "same_input": sensitivity, "selected": selection["selected"], "step0_exact_noop": step == 0,
                  "encoder_hash_before": encoder_before, "encoder_hash_after": encoder_after,
                  "head_state_sha256_before": head_before, "head_state_sha256_after": head_after,
                  "predictions_sha256": file_digest(condition_output / "predictions.pt")}
        if condition.startswith("wrong_"):
            level = protocol["wrong_calibration"]["levels"][condition.removeprefix("wrong_")]
            report["per_row_stress"] = [{"row_id": row["row_id"], "factor": level ** signs[row["row_id"]],
                                         "stratum": "expansion" if signs[row["row_id"]] == 1 else "contraction"} for row in rows]
        if condition == "metadata_reassignment":
            report["metadata_reassignment"] = assignment_audit["metadata_reassignment"]
        reports[condition] = report
        del features, current_cache
        print(json.dumps({"condition": condition, "split": actual_split, "count": len(rows),
                          "global_map": metrics["global"]["map"], "dense_map50": metrics["dense"]["map50"]}), flush=True)
    checkpoint_after = file_digest(checkpoint)
    best_after = {name: file_digest(head_dir / f"best_{name}.pt") for name in heads}
    source_after = {name: file_digest(source_dir / name) for name in source_before}
    if (checkpoint_sha != checkpoint_after or best_after != settings["best_heads_sha256"] or source_before != source_after or
            selection_sha != file_digest(head_dir / "report.json")):
        raise RuntimeError("checkpoint, selected heads, or execution source changed during inference")
    result = {"task": "uatd_image_only_fixed_head_interventions", "variant": variant, "checkpoint_step": step,
              "split": actual_split, "requested_test_split": test_split, "count": len(rows), "final_test_evaluated": not smoke,
              "inference_only": True, "m2_gate_unchanged": "NO_GO", "encoder_parameters_in_optimizer": False,
              "head_finetuning_or_reselection": False, "conditions": reports, "assignments": assignment_audit,
              "step0_actual_zero_state": zero_state, "current_first4_clean_exact_original_cache": True,
              "train_constants_reference": constants_reference,
              "provenance": {"original_clean_cache_and_selected_heads": provenance,
                             "parent_cache_manifest_sha256": parent_sha, "clean_cache_manifest_sha256": cache_manifest_sha,
                             "clean_cache": str(cache_path.resolve()), "clean_cache_sha256": cache_sha, "clean_cache_loaded_with_mmap": True,
                             "head_selection_sha256": selection_sha, "original_selected_head_source_sha256": selection["head_source_sha256"],
                             "best_heads_sha256_before": settings["best_heads_sha256"], "best_heads_sha256_after": best_after,
                             "checkpoint_sha256_before": checkpoint_sha, "checkpoint_sha256_after": checkpoint_after,
                             "encoder_hash_before": encoder_before, "encoder_hash_after": encoder_after,
                             "head_state_sha256_before": head_before, "head_state_sha256_after": head_after,
                             "execution_source_sha256_before": source_before, "execution_source_sha256_after": source_after,
                             "protocol": str(protocol_path.resolve()), "protocol_sha256": protocol_sha,
                             "device": str(device), "torch": str(torch.__version__)},
              "limitations": ["Approximate UATD uniform polar calibration; no exact physical geometry or scene-independent efficacy claim.",
                              "Recipient intensity, H/W, validity and clean labels/boxes stay fixed; only model calibration/gate inputs change.",
                              "Metadata donor identity is a row identifier, not an acquisition group; equal Range/FOV is an actual no-op.",
                              "Same token-index cosine is descriptive semantic sensitivity; worst-stratum analysis is a separate frozen analysis."]}
    for condition, report in reports.items():
        report["provenance"] = result["provenance"]
        report["train_constants_reference"] = constants_reference
        write_json(output / condition / "report.json", report)
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", choices=("R013", "R020", "R021"), required=True)
    parser.add_argument("--step", type=int, choices=(0, 10000), required=True)
    parser.add_argument("--test-split", choices=("test1", "test2"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = run_interventions(args.protocol, args.variant, args.step, args.test_split, args.device, args.output, args.smoke)
    print(json.dumps({"output": str(args.output), "variant": args.variant, "step": args.step,
                      "split": result["split"], "conditions": list(result["conditions"])}))


if __name__ == "__main__":
    main()
