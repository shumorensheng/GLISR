import copy
import json

import numpy as np
import pytest
import torch

from sonar_encoder.training.frontends import footprint_descriptor
from sonar_encoder.training.model import SingleChannelEncoder
from sonar_encoder.training.readout import state_digest
from sonar_encoder.training.tests.test_native_model import SmallBackbone


def native_samples():
    torch.manual_seed(13)
    samples = []
    for config, maximum in (("C00", 5.), ("C01", 3.)):
        valid = torch.ones(32, 32, dtype=torch.bool)
        valid[0, 0] = False
        samples.append({"sample_id": f"pose/{config}", "pose_id": "pose", "config_id": config,
                        "group_id": "group", "scene_id": "scene",
                        "intensity": torch.rand(1, 32, 32), "valid": valid,
                        "range_m": torch.linspace(.6, maximum - .1, 32),
                        "bearing_rad": torch.linspace(.5, -.5, 32),
                        "configuration": {"RangeMin": .5, "RangeMax": maximum, "Azimuth": 60.}})
    return samples


PROTOCOL = {"wrong_calibration": {"levels": {"L": 1.1, "M": 1.25, "H": 1.5},
                                  "strata": ["expansion", "contraction"]},
            "branch_masks": {"radial_only": [1, 0], "tangential_only": [0, 1], "identity": [0, 0]}}
CONSTANTS = {"descriptor_x": -2., "descriptor_y": -1., "gate_mean": [.3, .7]}
NORMALIZATION = {"mean": .2, "std": .1}


def test_calibration_replaces_only_model_geometry_and_preserves_truth():
    from sonar_encoder.training.m3_diagnostics import metadata_calibration, wrong_calibration

    truth = native_samples()[1]
    before = copy.deepcopy(truth)
    wrong = wrong_calibration(truth, 1 / 1.5)
    donor = {"RangeMin": 1., "RangeMax": 8., "Azimuth": 120.}
    reassigned = metadata_calibration(truth, donor)
    for changed in (wrong, reassigned):
        assert changed["intensity"] is truth["intensity"]
        assert changed["valid"] is truth["valid"]
        assert changed["range_m"].shape == truth["range_m"].shape
        for key in ("range_m", "bearing_rad", "intensity", "valid"):
            assert torch.equal(truth[key], before[key])
        assert truth["configuration"] == before["configuration"]
    torch.testing.assert_close(wrong["range_m"], .5 + (before["range_m"] - .5) / 1.5)
    assert wrong["range_m"].min() >= .5
    assert wrong["configuration"]["RangeMax"] == pytest.approx(.5 + 2.5 / 1.5)
    torch.testing.assert_close(reassigned["range_m"], 1. + (before["range_m"] - .5) * 7 / 2.5)
    torch.testing.assert_close(reassigned["bearing_rad"], 2 * before["bearing_rad"])


def test_constant_footprint_recomputes_area_and_anisotropy_channels():
    from sonar_encoder.training.m3_diagnostics import constant_descriptor

    sample = native_samples()[0]
    clean = footprint_descriptor(sample["range_m"], sample["bearing_rad"], sample["valid"], (.5, 5.))
    changed = constant_descriptor(clean, CONSTANTS)
    assert torch.equal(changed[..., [0, 1, 2, 7]], clean[..., [0, 1, 2, 7]])
    for channel, expected in ((3, -2.), (4, -1.), (5, -3.), (6, 1.)):
        assert torch.equal(changed[..., channel], torch.full((32, 32), expected))
    assert not torch.equal(clean[..., 3:7], changed[..., 3:7])


def test_architecture_condition_uses_its_own_gallery_and_query_only_uses_clean_gallery():
    from sonar_encoder.training.m3_diagnostics import condition_patches

    torch.set_num_threads(1)
    encoder = SingleChannelEncoder(SmallBackbone(), "R021").eval()
    with torch.no_grad():
        encoder.stem.residual_scale.fill_(.3)
    samples = native_samples()
    signs = {sample["sample_id"]: 1 for sample in samples}
    donors = {sample["sample_id"]: {"RangeMin": .5, "RangeMax": 8., "Azimuth": 100.} for sample in samples}
    with torch.inference_mode():
        clean = condition_patches(encoder, samples, NORMALIZATION, "clean", CONSTANTS, PROTOCOL, signs, donors, None)
        identity = condition_patches(encoder, samples, NORMALIZATION, "identity", CONSTANTS, PROTOCOL, signs, donors, clean)
        wrong = condition_patches(encoder, samples, NORMALIZATION, "wrong_H", CONSTANTS, PROTOCOL, signs, donors, clean)
        reassigned = condition_patches(encoder, samples, NORMALIZATION, "metadata_reassignment", CONSTANTS, PROTOCOL, signs, donors, clean)
    assert not torch.equal(identity[0], clean[0])
    assert torch.equal(wrong[0], clean[0])
    assert torch.equal(reassigned[0], clean[0])
    assert not torch.equal(wrong[1], clean[1])


@pytest.mark.parametrize("variant", ["R013", "R020", "R021"])
def test_zero_residual_all_conditions_are_exact_noops_and_restore_encoder(variant):
    from sonar_encoder.training.m3_diagnostics import condition_patches, stem_condition

    torch.set_num_threads(1)
    encoder = SingleChannelEncoder(SmallBackbone(), variant).eval()
    samples = native_samples()
    signs = {sample["sample_id"]: -1 for sample in samples}
    donors = {sample["sample_id"]: {"RangeMin": .5, "RangeMax": 8., "Azimuth": 100.} for sample in samples}
    before = state_digest(encoder.state_dict())
    conditions = ["wrong_L", "wrong_M", "wrong_H", "metadata_reassignment", "radial_only", "tangential_only", "identity"]
    if variant != "R020":
        conditions.extend(["constant_footprint", "flat_gate"])
    with torch.inference_mode():
        clean = condition_patches(encoder, samples, NORMALIZATION, "clean", CONSTANTS, PROTOCOL, signs, donors, None)
        for condition in conditions:
            changed = condition_patches(encoder, samples, NORMALIZATION, condition, CONSTANTS, PROTOCOL, signs, donors, clean)
            assert torch.equal(changed, clean), condition
            assert state_digest(encoder.state_dict()) == before
        with pytest.raises(RuntimeError, match="intentional interruption"):
            with stem_condition(encoder.stem, "identity", CONSTANTS, PROTOCOL["branch_masks"]):
                raise RuntimeError("intentional interruption")
    assert state_digest(encoder.state_dict()) == before
    if variant != "R020":
        assert all(not gate._forward_hooks and not gate._forward_pre_hooks for gate in encoder.stem.gates)


def test_metadata_audit_collapses_equal_group_headers_and_checks_each_query(tmp_path):
    from sonar_encoder.training.m3_diagnostics import audit_metadata

    frames = []
    for group, pose in (("a", "p0"), ("b", "p1")):
        headers = {}
        for config, maximum in (("C00", 5.), ("C01", 3.)):
            sample_id = f"{pose}/{config}"
            frames.append({"sample_id": sample_id, "pose_id": pose, "group_id": group,
                           "config_id": config, "metadata_relpath": f"{pose}.json"})
            headers[config] = {"configuration": {"RangeMin": .5, "RangeMax": maximum, "Azimuth": 60.}}
        (tmp_path / f"{pose}.json").write_text(json.dumps({"frames": headers}))
    donors, signs, audit = audit_metadata(frames, tmp_path, {"C01": "C00"})
    assert audit["query_pairs"] == 2
    assert audit["group_controls_collapsed"]
    assert audit["all_query_calibration_changed"]
    assert audit["pairs"][0]["across_donor_sample_id"] == "p1/C00"
    assert donors["p0/C01"]["RangeMax"] == 5.
    assert signs["p0/C01"] == 1
    assert signs["p1/C01"] == 1
    header = json.loads((tmp_path / "p0.json").read_text())
    header["frames"]["C01"]["configuration"]["RangeMax"] = 5.
    (tmp_path / "p0.json").write_text(json.dumps(header))
    with pytest.raises(ValueError, match="no-op"):
        audit_metadata(frames, tmp_path, {"C01": "C00"})


def test_streamed_report_uses_clean_metric_geometry_and_empty_smoke_stratum(tmp_path, capsys):
    from sonar_encoder.training.data import SonarDataset
    from sonar_encoder.training.m3_diagnostics import audit_metadata, evaluate_conditions
    from sonar_encoder.training.train import file_digest

    frames = []
    for pose, group in (("p0", "a"), ("p1", "b")):
        headers = {}
        for sample in native_samples():
            sample.update(pose_id=pose, group_id=group, sample_id=f"{pose}/{sample['config_id']}")
            relative = f"{pose}-{sample['config_id']}.npz"
            np.savez(tmp_path / relative, intensity=sample["intensity"][0].numpy(),
                     range_coordinates_m=sample["range_m"].numpy(),
                     bearing_coordinates_rad=sample["bearing_rad"].numpy(),
                     intensity_valid_region_mask=sample["valid"].numpy())
            frames.append({key: sample[key] for key in ("sample_id", "pose_id", "group_id", "scene_id", "config_id")} |
                          {"usage": "ssl_validation", "metadata_relpath": f"{pose}.json",
                           "npz_relpath": relative, "sha256": file_digest(tmp_path / relative)})
            headers[sample["config_id"]] = {"configuration": sample["configuration"]}
        (tmp_path / f"{pose}.json").write_text(json.dumps({"frames": headers}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"frames": frames}))
    dataset = SonarDataset(manifest_path, tmp_path, "ssl_validation")
    donors, signs, audit = audit_metadata(frames, tmp_path, {"C01": "C00"})
    encoder = SingleChannelEncoder(SmallBackbone(), "R021").eval()
    with torch.no_grad():
        encoder.stem.residual_scale.fill_(.3)
    before = state_digest(encoder.state_dict())
    protocol = {**PROTOCOL, "population": {"configs": ["C00", "C01"]}}
    with torch.inference_mode():
        report, sources = evaluate_conditions(encoder, dataset, NORMALIZATION, protocol,
                                              ["clean", "wrong_H", "identity", "metadata_reassignment"], CONSTANTS,
                                              donors, signs, audit, False, {"development_exposed_configs": []}, "development")
    assert len(sources) == 4
    assert report["gallery_size"] == 2
    audit = report["derived_input_audit"]
    assert (audit["range_changed_queries"], audit["bearing_changed_queries"], audit["both_changed_queries"]) == (2, 0, 0)
    assert all(row["range_changed"] and not row["bearing_changed"] for row in audit["per_query"])
    clean, wrong = report["conditions"]["clean"], report["conditions"]["wrong_H"]
    assert wrong["stress_strata"]["contraction"]["frames"] == 0
    assert wrong["stress_strata"]["contraction"]["mean_patch_cosine"] is None
    for clean_row, wrong_row in zip(clean["per_pose"], wrong["per_pose"]):
        for key in ("range_bounds_m", "bearing_bounds_rad", "coverage", "valid_points"):
            assert wrong_row["dense"][key] == clean_row["dense"][key]
    assert clean["per_pose"][0]["clean_branches"][0]["gate"]["pearson_log_r_dtheta_over_dr"] is None
    assert report["conditions"]["identity"]["gallery_condition"] == "identity"
    assert state_digest(encoder.state_dict()) == before
    assert capsys.readouterr().out


@pytest.mark.parametrize("exact_noop", [False, True])
def test_final_population_roles_clean_truth_and_analytical_step0(tmp_path, monkeypatch, exact_noop):
    from sonar_encoder.training import m3_diagnostics as diagnostics
    from sonar_encoder.training.config_eval import select_frames
    from sonar_encoder.training.data import SonarDataset
    from sonar_encoder.training.train import file_digest

    torch.set_num_threads(1)
    configs = [f"C{index:02d}" for index in range(14)]
    roles = {"development_exposed_configs": ["C02", "C04", "C06", "C10"],
             "development_unseen_configs": ["C08", "C09", "C13"]}
    frames = []
    for pose, group in (("p0", "a"), ("p1", "b")):
        headers = {}
        reference, query = native_samples()
        for config in configs:
            sample = copy.deepcopy(reference if config == "C00" else query)
            sample.update(config_id=config, pose_id=pose, group_id=group, sample_id=f"{pose}/{config}")
            relative = f"{pose}-{config}.npz"
            np.savez(tmp_path / relative, intensity=sample["intensity"][0].numpy(),
                     range_coordinates_m=sample["range_m"].numpy(),
                     bearing_coordinates_rad=sample["bearing_rad"].numpy(),
                     intensity_valid_region_mask=sample["valid"].numpy())
            usage = ("test_reference" if config == "C11" else "test_unseen_combination" if config in {"C09", "C10"} else
                     "test_unseen_single" if config in {"C02", "C04", "C06", "C08", "C13"} else "test_seen")
            frames.append({key: sample[key] for key in ("sample_id", "pose_id", "group_id", "scene_id", "config_id")} |
                          {"split": "test", "usage": usage, "metadata_relpath": f"{pose}.json",
                           "npz_relpath": relative, "sha256": file_digest(tmp_path / relative)})
            headers[config] = {"configuration": sample["configuration"]}
        (tmp_path / f"{pose}.json").write_text(json.dumps({"frames": headers}))
    manifest = {"frames": frames}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    dataset = SonarDataset(manifest_path, tmp_path, "test_seen")
    dataset.frames = select_frames(manifest, "test", configs, None, roles)
    protocol = {**PROTOCOL, "population": {"split": "test", "configs": configs,
                                          "poses": 2, "sample_ids": [frame["sample_id"] for frame in dataset.frames]}}
    donors, signs, audit = diagnostics.audit_metadata(dataset.frames, tmp_path, {config: "C00" for config in configs[1:]})
    encoder = SingleChannelEncoder(SmallBackbone(), "R021").eval()
    if not exact_noop:
        with torch.no_grad():
            encoder.stem.residual_scale.fill_(.3)
    before = state_digest(encoder.state_dict())
    observed_truth, forward_conditions = [], []
    original = diagnostics.condition_patches

    def observe_truth(encoder, samples, normalization, condition, *arguments):
        forward_conditions.append(condition)
        if condition == "clean":
            observed_truth.extend((sample, copy.deepcopy(sample)) for sample in samples)
        return original(encoder, samples, normalization, condition, *arguments)

    monkeypatch.setattr(diagnostics, "condition_patches", observe_truth)
    conditions = ["clean", "wrong_L", "wrong_M", "wrong_H", "metadata_reassignment",
                  "radial_only", "tangential_only", "identity", "constant_footprint", "flat_gate"]
    with torch.inference_mode():
        report, sources = diagnostics.evaluate_conditions(encoder, dataset, NORMALIZATION, protocol, conditions,
                                                          CONSTANTS, donors, signs, audit, exact_noop, roles, "test")
    assert report["split"] == "test" and report["final_test_evaluated"]
    assert report["configs"] == configs and report["gallery_size"] == 2
    assert [source["sample_id"] for source in sources] == protocol["population"]["sample_ids"]
    groups = report["conditions"]["clean"]["summary"]["configuration_groups"]
    assert {role: value["configs"] for role, value in groups.items()} == {
        "seen": ["C01", "C03", "C05", "C07", "C12"],
        "development_exposed_single": ["C02", "C04", "C06"],
        "development_exposed_combination": ["C10"],
        "development_unseen_single": ["C08", "C13"],
        "development_unseen_combination": ["C09"]}
    for condition, result in report["conditions"].items():
        gallery = "clean" if condition.startswith("wrong_") or condition == "metadata_reassignment" else condition
        assert result["split"] == "test" and result["final_test_evaluated"]
        assert result["gallery_condition"] == gallery
        for clean_row, row in zip(report["conditions"]["clean"]["per_pose"], result["per_pose"]):
            assert row["condition"] == condition and row["gallery_condition"] == gallery
            assert row["split"] == "test" and row["final_test_evaluated"]
            for key in ("range_bounds_m", "bearing_bounds_rad", "coverage", "valid_points"):
                assert row["dense"][key] == clean_row["dense"][key]
        if exact_noop:
            assert result["summary"] == report["conditions"]["clean"]["summary"]
            assert result["all_patch_tokens_bit_identical_to_clean"]
            assert result["execution"] == ("computed" if condition == "clean" else "analytical_zero_residual_noop")
    assert forward_conditions == (["clean", "clean"] if exact_noop else conditions * 2)
    for sample, saved in observed_truth:
        for key in ("intensity", "valid", "range_m", "bearing_rad"):
            assert torch.equal(sample[key], saved[key])
        assert sample["configuration"] == saved["configuration"]
    assert state_digest(encoder.state_dict()) == before
    if exact_noop:
        assert report["step0_noop_basis"]["residual_scale"] == [0., 0.]
        with torch.no_grad():
            encoder.stem.residual_scale.fill_(.3)
        with pytest.raises(ValueError, match="not exactly zero"), torch.inference_mode():
            diagnostics.evaluate_conditions(encoder, dataset, NORMALIZATION, protocol, conditions,
                                            CONSTANTS, donors, signs, audit, True, roles, "test")


def test_prospective_final_requires_freeze_before_any_data_access(tmp_path):
    from sonar_encoder.training.m3_diagnostics import run_final_diagnostics

    path = tmp_path / "prospective.json"
    path.write_text(json.dumps({"status": "PROSPECTIVE_FINAL_DIAGNOSTICS_BEFORE_FREEZE", "inference_only": True,
                                "trained_m3_controls_complete": False, "m2_gate_unchanged": "NO_GO"}))
    with pytest.raises(ValueError, match="frozen final"):
        run_final_diagnostics(path, "R021", "cpu", tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("final", [False, True])
def test_cli_routes_final_explicitly_and_keeps_development_default(tmp_path, monkeypatch, capsys, final):
    import sys
    from sonar_encoder.training import m3_diagnostics as diagnostics

    called = []
    monkeypatch.setattr(diagnostics, "run_diagnostics",
                        lambda *arguments: called.append(("development", arguments)) or {"mode": "dev", "steps": {}})
    monkeypatch.setattr(diagnostics, "run_final_diagnostics",
                        lambda *arguments: called.append(("final", arguments)) or {"mode": "final_smoke", "steps": {}})
    arguments = ["m3_diagnostics", "--protocol", str(tmp_path / "protocol.json"), "--variant", "R021",
                 "--output", str(tmp_path / "report.json"), "--smoke"]
    if final:
        arguments.append("--final")
    monkeypatch.setattr(sys, "argv", arguments)
    diagnostics.main()
    assert called == [("final" if final else "development",
                       (tmp_path / "protocol.json", "R021", "cpu", tmp_path / "report.json", True))]
    assert capsys.readouterr().out
