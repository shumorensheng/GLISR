import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sonar_encoder.training.config_eval import token_geometry
from sonar_encoder.training.train import file_digest, manifest_digest


def test_unique_matching_is_identity_in_all_three_maps():
    from sonar_encoder.training.composition_eval import composition_metrics

    ranges = torch.linspace(1, 5, 16, dtype=torch.double)
    bearings = torch.linspace(-0.5, 0.5, 16, dtype=torch.double)
    valid = torch.ones(16, 16, dtype=torch.bool)
    native = {"range_m": ranges, "bearing_rad": bearings, "valid": valid}
    geometry = (ranges, bearings, valid)
    field = torch.eye(256).reshape(16, 16, 256)

    result = composition_metrics([field] * 3, [geometry] * 3, [native] * 3)

    assert result["valid_points"] == 256
    for key in ("composition_error_m", "composition_max_error_m", "direct_error_m",
                "direct_max_error_m", "composed_error_m", "composed_max_error_m"):
        assert result[key] == 0
    for key in ("ab_unique_target_fraction", "bc_unique_target_fraction",
                "ac_unique_target_fraction", "composed_unique_target_fraction"):
        assert result[key] == 1


def test_actual_feature_argmax_chain_can_be_non_transitive():
    from sonar_encoder.training.composition_eval import composition_metrics

    ranges = torch.linspace(1, 5, 16, dtype=torch.double)
    bearings = torch.linspace(-0.5, 0.5, 16, dtype=torch.double)
    valid = torch.ones(16, 16, dtype=torch.bool)
    native = {"range_m": ranges, "bearing_rad": bearings, "valid": valid}
    geometry = (ranges, bearings, valid)
    fields = []
    # A0=0° -> B0=40° -> C1=70°, while A0 -> C0=-20°.
    for ordinary, first in ((-90, 0), (-60, 40), (70, -20)):
        angles = torch.full((256,), math.radians(ordinary))
        angles[0] = math.radians(first)
        fields.append(torch.stack([angles.cos(), angles.sin()], dim=1).reshape(16, 16, 2))

    result = composition_metrics(fields, [geometry] * 3, [native] * 3)

    chord = 2 * math.sin(1 / 30)
    assert result["composition_error_m"] == pytest.approx(chord / 256)
    assert result["composition_max_error_m"] == pytest.approx(chord)
    assert result["ab_unique_target_fraction"] == 2 / 256
    assert result["bc_unique_target_fraction"] == 2 / 256
    assert result["ac_unique_target_fraction"] == 1 / 256
    assert result["composed_unique_target_fraction"] == 2 / 256


def test_constant_matching_has_zero_composition_but_positive_localization_error():
    from sonar_encoder.training.composition_eval import composition_metrics

    ranges = torch.linspace(1, 5, 16, dtype=torch.double)
    bearings = torch.linspace(-0.5, 0.5, 16, dtype=torch.double)
    valid = torch.ones(16, 16, dtype=torch.bool)
    native = {"range_m": ranges, "bearing_rad": bearings, "valid": valid}
    geometry = (ranges, bearings, valid)
    field = torch.ones(16, 16, 3)

    result = composition_metrics([field] * 3, [geometry] * 3, [native] * 3)

    assert result["composition_error_m"] == result["composition_max_error_m"] == 0
    assert result["direct_error_m"] > 1
    assert result["direct_error_m"] == result["composed_error_m"]
    assert result["direct_max_error_m"] == result["composed_max_error_m"]
    assert result["ac_unique_target_fraction"] == result["composed_unique_target_fraction"] == 1 / 256


def test_common_support_uses_three_native_grids_and_preserves_reversed_bearing():
    from sonar_encoder.training.composition_eval import composition_metrics

    samples = []
    for height, width in ((16, 16), (32, 32), (48, 48)):
        samples.append({"range_m": torch.linspace(1, 5, height),
                        "bearing_rad": torch.linspace(0.5, -0.5, width),
                        "valid": torch.ones(height, width, dtype=torch.bool)})
    samples[2]["valid"][22, 23] = False
    geometries = [token_geometry(sample) for sample in samples]
    fields = []
    for ranges, bearings, _ in geometries:
        r, a = torch.meshgrid(ranges, bearings, indexing="ij")
        fields.append(torch.stack([r, a, torch.ones_like(r)], dim=-1).float())

    result = composition_metrics(fields, geometries, samples)
    reversed_samples = [samples[0],
                        {**samples[1], "bearing_rad": samples[1]["bearing_rad"].flip(0),
                         "valid": samples[1]["valid"].flip(1)}, samples[2]]
    reversed_geometry = [geometries[0], token_geometry(reversed_samples[1]), geometries[2]]
    reversed_result = composition_metrics([fields[0], fields[1].flip(1), fields[2]],
                                         reversed_geometry, reversed_samples)

    assert 0 < result["valid_points"] < 256
    assert result["coverage"] == result["valid_points"] / 256
    assert result["range_bounds_m"] == pytest.approx([1 + 4 / 47, 5 - 4 / 47], abs=1e-6)
    assert result["bearing_bounds_rad"] == pytest.approx([-0.5 + 1 / 47, 0.5 - 1 / 47], abs=1e-6)
    for key in ("composition_error_m", "direct_error_m", "composed_error_m"):
        assert result[key] == reversed_result[key] == 0
    assert result["valid_points"] == reversed_result["valid_points"]


def test_prospective_final_protocol_fails_before_loading_inputs_or_encoder(tmp_path):
    from sonar_encoder.training.composition_eval import run_evaluation

    protocol = tmp_path / "prospective.json"
    protocol.write_text(json.dumps({"status": "PROSPECTIVE", "population": {"split": "test"}}))

    with pytest.raises(ValueError, match="FROZEN"):
        run_evaluation(protocol, "R012", "cpu", tmp_path / "unwritten.json")
    assert not (tmp_path / "unwritten.json").exists()


@pytest.mark.parametrize("extents", [(20, 30, 50), (20, 50, 120)])
def test_long_range_triads_measure_only_the_shared_near_range(extents):
    from sonar_encoder.training.composition_eval import composition_metrics

    samples = [{"range_m": torch.linspace(1, extent, 32),
                "bearing_rad": torch.linspace(0.5, -0.5, 32),
                "valid": torch.ones(32, 32, dtype=torch.bool)} for extent in extents]
    geometries = [token_geometry(sample) for sample in samples]
    fields = []
    for ranges, bearings, _ in geometries:
        r, a = torch.meshgrid(ranges, bearings, indexing="ij")
        fields.append(torch.stack([r, a, torch.ones_like(r)], dim=-1).float())

    result = composition_metrics(fields, geometries, samples)

    assert result["valid_points"] == 256
    assert result["range_bounds_m"] == pytest.approx([1 + (max(extents) - 1) / 62,
                                                     20 - 19 / 62], abs=1e-6)
    assert result["range_bounds_m"][1] < 20


@pytest.mark.parametrize("split", ["development", "test"])
def test_full_protocol_encodes_each_needed_config_once_per_pose_and_stage(tmp_path, monkeypatch, split):
    from sonar_encoder.training import composition_eval

    torch.set_num_threads(1)
    configs = ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C10", "C12"]
    if split == "test":
        configs = [f"C{index:02d}" for index in range(14) if index != 11]
    triads = composition_eval.DEVELOPMENT_TRIADS if split == "development" else composition_eval.FINAL_TRIADS
    seen = {"C00", "C01", "C03", "C05", "C07", "C12"}
    frames = []
    for pose in ("p0", "p1", "outside"):
        metadata = {"frames": {}}
        for config in configs:
            frame_id = f"CPU_{config}_R00"
            path = tmp_path / f"{pose}_{config}.npz"
            np.savez(path, intensity=np.ones((16, 16), dtype=np.float32),
                     range_coordinates_m=np.linspace(1, 5, 16).astype("float32"),
                     bearing_coordinates_rad=np.linspace(0.5, -0.5, 16).astype("float32"),
                     intensity_valid_region_mask=np.ones((16, 16), dtype=bool))
            metadata["frames"][frame_id] = {"configuration": {"RangeMin": 0.5, "RangeMax": 5.5}}
            frames.append({"sample_id": f"{pose}/{frame_id}", "pose_id": pose, "group_id": f"g{pose[-1]}",
                           "scene_id": "synthetic", "config_id": config,
                           "split": "validation" if split == "development" else "test",
                           "usage": ("ssl_validation" if config in seen else "unused") if split == "development"
                                    else ("test_seen" if config in seen else "test_unseen_single"),
                           "npz_relpath": path.name, "metadata_relpath": f"{pose}.json",
                           "sha256": file_digest(path)})
        (tmp_path / f"{pose}.json").write_text(json.dumps(metadata))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frames": frames}))
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"mean": 0, "std": 1,
                                         "source_manifest_sha256": manifest_digest({"frames": frames})}))
    role = tmp_path / "role.json"
    role.write_text(json.dumps({"source_manifest_sha256": file_digest(manifest),
                               "development_exposed_configs": ["C02", "C04", "C06", "C10"],
                               "development_unseen_configs": ["C08", "C09", "C13"],
                               "development_sample_ids": [frame["sample_id"] for frame in frames]}))
    weights = tmp_path / "foundation.pth"
    weights.write_bytes(b"synthetic-foundation")
    calls = []

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward_native(self, samples, normalization, masks):
            assert masks is None
            calls.append((float(self.scale), [sample["sample_id"] for sample in samples]))
            fields = []
            for sample in samples:
                r, a, _ = token_geometry(sample)
                rr, aa = torch.meshgrid(r, a, indexing="ij")
                fields.append(torch.stack([rr, aa, torch.ones_like(rr)], dim=-1).float().reshape(256, 3))
            return {"patch_tokens": self.scale * torch.stack(fields)}

    monkeypatch.setattr(composition_eval, "build_encoder", lambda *_: Encoder())
    checkpoints = {}
    for variant in ("R011", "R012", "R020", "R021"):
        config = {"variant": variant, "seed": 0, "steps": 10000,
                  "weights_path": str(weights), "code_dir": str(tmp_path)}
        checkpoints[variant] = {}
        for step, scale in ((0, 1.0), (10000, 2.0)):
            checkpoint = tmp_path / f"{variant}_{step}.pt"
            torch.save({"step": step, "student": {"scale": torch.tensor(scale)}, "resolved_config": config,
                        "provenance": {"manifest": file_digest(manifest), "normalization": file_digest(normalization),
                                       "weights": file_digest(weights)}}, checkpoint)
            checkpoints[variant][str(step)] = {"path": str(checkpoint), "sha256": file_digest(checkpoint)}
    package = Path(composition_eval.__file__).parent
    selected = [frame for frame in frames if frame["pose_id"] != "outside"]
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema_version": 1, "status": "FROZEN", "data_root": str(tmp_path),
        "input_files": {key: {"path": str(path), "sha256": file_digest(path)}
                        for key, path in [("manifest", manifest), ("normalization", normalization), ("role_protocol", role)]},
        "source_sha256": {name: file_digest(package / name) for name in composition_eval.SOURCE_FILES},
        "population": {"split": split, "configs": configs, "pose_ids": ["p0", "p1"],
                       "sample_ids": [frame["sample_id"] for frame in selected], "pose_count": 2,
                       "frame_count": len(configs) * 2, "per_configuration_counts": {config: 2 for config in configs}},
        "triads": [list(triad) for triad in triads],
        "checkpoints": checkpoints,
    }))

    result = composition_eval.run_evaluation(protocol, "R012", "cpu", tmp_path / "result.json")

    assert len(calls) == 4
    assert [scale for scale, _ in calls] == [1, 1, 2, 2]
    assert all(len(ids) == len(set(ids)) == len(configs) for _, ids in calls)
    assert all(any("C12" in sample_id for sample_id in ids) for _, ids in calls)
    assert all("C11" not in sample_id for _, ids in calls for sample_id in ids)
    assert set(result["stages"]) == {"0", "10000"}
    for stage in result["stages"].values():
        assert len(stage["per_pose"]) == 2 * len(triads)
        assert len(stage["summary"]["per_triad"]) == len(triads)
        assert set(stage["summary"]["per_group"]) == {"g0", "g1"}
        assert stage["summary"]["macro"]["composition_error_m"] == 0
        assert stage["summary"]["macro"]["direct_error_m"] == 0
        assert stage["provenance"]["encoder_hash_before"] == stage["provenance"]["encoder_hash_after"]
    assert result["final_test_evaluated"] == (split == "test")
    assert (tmp_path / "result.json").is_file()
    assert not (tmp_path / "cache").exists()
