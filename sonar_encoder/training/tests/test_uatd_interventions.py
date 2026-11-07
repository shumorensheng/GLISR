import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from sonar_encoder.training.model import SingleChannelEncoder
from sonar_encoder.training.readout import file_digest, state_digest
from sonar_encoder.training.tests.test_native_model import SmallBackbone


def synthetic_rows(tmp_path):
    rows = []
    for index, (maximum, fov) in enumerate(((5., 60.), (5., 60.), (8., 90.), (8., 90.))):
        image = tmp_path / f"{index}.bmp"
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (32, 48), dtype=np.uint8)).save(image)
        rows.append({"row_id": f"row{index}", "image": str(image), "image_sha256": file_digest(image),
                     "shape": [32, 48], "range_m": maximum, "horizontal_fov_deg": fov, "frequency": 700.,
                     "labels": list(range(10)), "boxes": [[.2, .3, .4, .5]] * 10})
    return rows


PROTOCOL = {"conditions": {"all": ["clean", "wrong_L", "wrong_M", "wrong_H", "metadata_reassignment"],
                           "gated_only": ["constant_footprint", "flat_gate"]},
            "wrong_calibration": {"levels": {"L": 1.10, "M": 1.25, "H": 1.50}},
            "branch_masks": {"radial_only": [1, 0], "tangential_only": [0, 1], "identity": [0, 0]}}
CONSTANTS = {"descriptor_x": -2., "descriptor_y": -1., "gate_mean": [.3, .7]}
NORMALIZATION = {"mean": .2, "std": .1}


@pytest.mark.parametrize("variant", ["R013", "R020", "R021"])
def test_native_interventions_preserve_truth_and_step0_exact_noop(tmp_path, monkeypatch, variant):
    from sonar_encoder.training import uatd_interventions as interventions

    torch.set_num_threads(1)
    rows = synthetic_rows(tmp_path)
    saved_rows = copy.deepcopy(rows)
    signs, donors, audit = interventions.intervention_assignments(list(reversed(rows)))
    assert signs == {"row0": 1, "row1": -1, "row2": 1, "row3": -1}
    assert [entry["donor_row_id"] for entry in audit["metadata_reassignment"]] == ["row1", "row2", "row3", "row0"]
    assert [entry["header_noop"] for entry in audit["metadata_reassignment"]] == [True, False, True, False]
    samples = {row["row_id"]: interventions.native_sample(row) for row in rows}
    assert samples["row0"]["configuration"]["Azimuth"] == 60.
    saved_samples = copy.deepcopy(samples)
    monkeypatch.setattr(interventions, "native_sample", lambda row: samples[row["row_id"]])
    encoder = SingleChannelEncoder(SmallBackbone(), variant).eval().requires_grad_(False)
    conditions = list(PROTOCOL["conditions"]["all"])
    if variant != "R020":
        conditions.extend(PROTOCOL["conditions"]["gated_only"])
    with torch.inference_mode():
        zero = interventions.zero_residual_state(encoder)
        assert zero["residual_scale"] == [0., 0.]
        assert zero["polar_pe"] == "not_present_in_R013_R020_R021"
        clean = interventions.encode_condition(encoder, rows, NORMALIZATION, "clean", CONSTANTS, PROTOCOL, signs, donors)
        before = state_digest(encoder.state_dict())
        for condition in conditions:
            actual = interventions.encode_condition(encoder, rows, NORMALIZATION, condition, CONSTANTS, PROTOCOL, signs, donors)
            assert torch.equal(actual, clean), condition
            assert state_digest(encoder.state_dict()) == before
        encoder.stem.residual_scale.fill_(.3)
        if variant != "R020":
            for gate in encoder.stem.gates:
                gate[2].weight.fill_(.2)
        with pytest.raises(ValueError, match="not exactly zero"):
            interventions.zero_residual_state(encoder)
        active_clean = interventions.encode_condition(encoder, rows, NORMALIZATION, "clean", CONSTANTS, PROTOCOL, signs, donors)
        before = state_digest(encoder.state_dict())
        for condition in ("wrong_H", "metadata_reassignment"):
            actual = interventions.encode_condition(encoder, rows, NORMALIZATION, condition, CONSTANTS, PROTOCOL, signs, donors)
            assert not torch.equal(actual, active_clean), condition
            if condition == "metadata_reassignment":
                assert torch.equal(actual[[0, 2]], active_clean[[0, 2]])
            assert state_digest(encoder.state_dict()) == before
    assert rows == saved_rows
    for row_id, sample in samples.items():
        for key in ("intensity", "valid", "range_m", "bearing_rad"):
            assert torch.equal(sample[key], saved_samples[row_id][key])
        assert sample["configuration"] == saved_samples[row_id]["configuration"]
    if variant != "R020":
        assert all(not gate._forward_hooks and not gate._forward_pre_hooks for gate in encoder.stem.gates)


def test_fixed_heads_and_saved_predictions_reproduce_task_metrics(tmp_path):
    from sonar_encoder.training.uatd_interventions import evaluate_fixed_heads
    from sonar_encoder.training.uatd_head import detection_metrics, global_targets, multilabel_metrics

    torch.set_num_threads(1)
    torch.manual_seed(5)
    heads = (nn.Linear(384, 10), nn.Conv2d(384, 50, 1))
    with torch.no_grad():
        heads[1].bias[:10].fill_(-10.)
    cache = {"features": torch.randn(4, 384, 16, 16), "row_ids": [f"row{i}" for i in range(4)],
             "targets": [{"labels": torch.arange(10), "boxes": torch.tensor([[.2, .3, .4, .5]] * 10)}] * 4}
    before = [state_digest(head.state_dict()) for head in heads]
    metrics, predictions = evaluate_fixed_heads(*heads, cache, "cpu")
    torch.save(predictions, tmp_path / "predictions.pt")
    saved = torch.load(tmp_path / "predictions.pt", weights_only=True)
    assert metrics["global"] == multilabel_metrics(saved["global_scores"], global_targets(cache["targets"], "cpu"))
    assert metrics["dense"] == detection_metrics(saved["detections"], cache["targets"])
    assert saved["row_ids"] == cache["row_ids"]
    assert np.isfinite(metrics["loss"]["total"])
    assert [state_digest(head.state_dict()) for head in heads] == before
    assert all(parameter.grad is None for head in heads for parameter in head.parameters())


def test_prospective_protocol_does_not_open_final_inputs(tmp_path):
    from sonar_encoder.training.uatd_interventions import run_interventions

    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"status": "PROSPECTIVE_FINAL_DIAGNOSTICS_BEFORE_FREEZE"}))
    with pytest.raises(ValueError, match="frozen final"):
        run_interventions(protocol, "R021", 0, "test1", "cpu", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_validation_smoke_binds_original_cache_heads_and_external_predictions(tmp_path, monkeypatch, capsys):
    from sonar_encoder.training import artifacts, uatd_interventions as interventions
    from sonar_encoder.training.uatd_head import detection_metrics, global_targets, multilabel_metrics

    torch.set_num_threads(1)
    rows = []
    base = synthetic_rows(tmp_path)
    for index in range(32):
        rows.append({**base[index % 4], "row_id": f"validation/{index:02d}"})
    original_protocol = tmp_path / "readout_protocol.json"
    original_protocol.write_text("{}")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    prepared_manifest = {"protocol_path": str(original_protocol), "protocol_sha256": file_digest(original_protocol),
                         "validation": rows, "class_names": [str(i) for i in range(10)],
                         "final_manifests": {"test1": str(tmp_path / "sealed_test1.jsonl"),
                                             "test2": str(tmp_path / "sealed_test2.jsonl")}}
    (prepared / "manifest.json").write_text(json.dumps(prepared_manifest))
    for split in ("test1", "test2"):
        (tmp_path / f"sealed_{split}.jsonl").write_text("SEALED_NOT_JSON_DO_NOT_PARSE")
    normalization = {**NORMALIZATION, "source_manifest_sha256": file_digest(prepared / "manifest.json")}
    (prepared / "normalization.json").write_text(json.dumps(normalization))
    encoder = SingleChannelEncoder(SmallBackbone(), "R021").eval().requires_grad_(False)
    checkpoint = tmp_path / "encoder.pt"
    foundation = tmp_path / "foundation.bin"
    foundation.write_bytes(b"synthetic foundation, no DINO")
    torch.save({"student": encoder.state_dict(), "step": 0,
                "resolved_config": {"variant": "R021", "code_dir": "unused", "weights_path": str(foundation)},
                "provenance": {"weights": file_digest(foundation)}}, checkpoint)
    provenance = {"task": "uatd", "variant": "R021", "checkpoint_step": 0, "prepared": str(prepared),
                  "task_manifest_sha256": file_digest(prepared / "manifest.json"),
                  "normalization_sha256": file_digest(prepared / "normalization.json"),
                  "protocol_sha256": file_digest(original_protocol), "checkpoint": str(checkpoint),
                  "checkpoint_sha256": file_digest(checkpoint), "encoder_state_sha256": state_digest(encoder.state_dict()),
                  "source_sha256": {"uatd_readout.py": "original_archived_source_sha"}}
    parent = tmp_path / "parent_cache"
    parent.mkdir()
    signs, donors, _ = interventions.intervention_assignments(rows)
    features = interventions.encode_condition(encoder, rows, normalization, "clean", CONSTANTS, PROTOCOL, signs, donors)
    targets = [{"labels": torch.tensor(row["labels"]), "boxes": torch.tensor(row["boxes"])} for row in rows]
    torch.save({"features": features, "row_ids": [row["row_id"] for row in rows], "targets": targets,
                "provenance": provenance}, parent / "validation.pt")
    (parent / "cache_manifest.json").write_text(json.dumps({"provenance": provenance,
        "cache_files": {"validation.pt": file_digest(parent / "validation.pt")}, "class_names": prepared_manifest["class_names"]}))
    head_dir = tmp_path / "heads"
    head_dir.mkdir()
    selected = {"global": {"epoch": 5, "score": .1}, "dense": {"epoch": 10, "score": .1}}
    for name, head in (("global", nn.Linear(384, 10)), ("dense", nn.Conv2d(384, 50, 1))):
        if name == "dense":
            with torch.no_grad():
                head.bias[:10].fill_(-20.)
        torch.save({"head": head.state_dict(), "epoch": selected[name]["epoch"], "provenance": provenance},
                   head_dir / f"best_{name}.pt")
    head_hashes = {name: file_digest(head_dir / f"best_{name}.pt") for name in selected}
    (head_dir / "report.json").write_text(json.dumps({"cache_manifest_sha256": file_digest(parent / "cache_manifest.json"),
        "best_heads_sha256": head_hashes, "selected": selected, "head_source_sha256": "original_selected_head_source_sha"}))
    constants_protocol = tmp_path / "constants_protocol.json"
    train_ids = [f"ssl_train/{i}" for i in range(192)]
    constants_protocol.write_text(json.dumps({"constant_training_population": {"usage": "ssl_train", "sample_ids": train_ids}}))
    constants_report = tmp_path / "constants_report.json"
    values = {**CONSTANTS, "frames": 192}
    constants_report.write_text(json.dumps({"steps": {"step10000": {
        "train_constants": {**values, "selected_sources": [{"sample_id": row_id} for row_id in train_ids]},
        "provenance": {"variant": "R021", "checkpoint_step": 10000, "checkpoint_sha256": "fixed_step10000_sha"}}},
        "provenance": {"protocol": str(constants_protocol), "protocol_sha256": file_digest(constants_protocol)}}))
    settings = {"parent_cache": str(parent), "parent_cache_manifest_sha256": file_digest(parent / "cache_manifest.json"),
                "head_directory": str(head_dir), "head_selection_sha256": file_digest(head_dir / "report.json"),
                "best_heads_sha256": head_hashes, "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint)}
    protocol = {**PROTOCOL, "status": "PROSPECTIVE_FINAL_DIAGNOSTICS_BEFORE_FREEZE", "inference_only": True,
                "m2_gate_unchanged": "NO_GO", "source_sha256": {name: file_digest(Path(interventions.__file__).with_name(name))
                    for name in ("uatd_interventions.py", "m3_diagnostics.py", "uatd_readout.py", "uatd_head.py", "model.py",
                                 "frontends.py", "readout.py", "train.py", "data.py", "config_eval.py", "artifacts.py")},
                "prepared": str(prepared),
                "prepared_manifest_sha256": file_digest(prepared / "manifest.json"),
                "normalization_sha256": file_digest(prepared / "normalization.json"), "smoke_row_ids": [row["row_id"] for row in rows],
                "models": {"R021": {"steps": {"step0": settings, "step10000": {"checkpoint_sha256": "fixed_step10000_sha"}},
                    "train_constants": {"report": str(constants_report), "report_sha256": file_digest(constants_report), "values": values}}}}
    protocol_path = tmp_path / "prospective.json"
    protocol_path.write_text(json.dumps(protocol))
    monkeypatch.setattr(interventions, "build_encoder", lambda *args: SingleChannelEncoder(SmallBackbone(), "R021"))
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path / "external_artifacts")
    output = tmp_path / "interventions"
    result = interventions.run_interventions(protocol_path, "R021", 0, "test1", "cpu", output, True)
    assert output.is_symlink() and output.resolve() == tmp_path / "external_artifacts/interventions"
    assert result["split"] == "validation" and not result["final_test_evaluated"] and result["count"] == 32
    assert result["current_first4_clean_exact_original_cache"]
    assert result["provenance"]["original_clean_cache_and_selected_heads"] == provenance
    assert result["provenance"]["execution_source_sha256_before"] == result["provenance"]["execution_source_sha256_after"]
    assert result["provenance"]["original_selected_head_source_sha256"] == "original_selected_head_source_sha"
    assert all(report["same_input"]["all_patch_tokens_bit_identical"] for report in result["conditions"].values())
    for condition, report in result["conditions"].items():
        saved = torch.load(output / condition / "predictions.pt", weights_only=True)
        assert report["metrics"]["global"] == multilabel_metrics(saved["global_scores"], global_targets(targets, "cpu"))
        assert report["metrics"]["dense"] == detection_metrics(saved["detections"], targets)
        assert json.loads((output / condition / "report.json").read_text())["provenance"] == result["provenance"]
    assert file_digest(checkpoint) == settings["checkpoint_sha256"]
    assert all(file_digest(head_dir / f"best_{name}.pt") == digest for name, digest in head_hashes.items())
    assert capsys.readouterr().out
