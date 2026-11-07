import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from sonar_encoder.training import artifacts, uatd_readout
from sonar_encoder.training.readout import state_digest


def input_files(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    rows = []
    for name, height, width, value in [("a", 17, 16, 0), ("b", 31, 32, 255), ("v", 19, 32, 128)]:
        path = real_root / f"{name}.bmp"
        Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(path)
        rows.append({"id": name, "split": "UATD_Training", "image_path": f"/original/{name}.bmp",
                     "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixel_sha256": name,
                     "shape": [height, width, 3], "range_m": 10., "horizontal_fov_deg": 120.,
                     "frequency": "720k", "rgb_channels_equal": True,
                     "objects": [{"name": "ball", "bbox_xyxy": [2, 3, 10, 11], "valid": True}]})
    for name in ("conflict1", "conflict2"):
        rows.append({**rows[0], "id": name, "pixel_sha256": "conflict",
                     "image_path": f"/original/{name}.bmp"})
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source.with_name("test1.jsonl").write_text(json.dumps({**rows[0], "id": "test1/a"}) + "\n")
    source.with_name("test2.jsonl").write_text("opaque test bytes: never parse in prepare or test1 extraction")
    protocol = {"source_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
                "input_contract": {"source_manifest": str(source), "source_prefix": "/original"},
                "labels": {"class_names": ["ball", "cube"]},
                "split": {"train_ids": ["a", "b"], "validation_ids": ["v"]},
                "duplicate_conflict_policy": {"excluded_groups": [{"row_ids": ["conflict1", "conflict2"]}]},
                "approximate_geometry": {"assumption_set_id": "uniform-test"}}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    return protocol_path, real_root, source


def test_prepare_uses_frozen_memberships_excludes_conflicts_and_fits_train_pixels_only(tmp_path):
    protocol, real_root, _ = input_files(tmp_path)
    output = tmp_path / "prepared"

    manifest = uatd_readout.prepare_task(protocol, real_root, output)
    normalization = json.loads((output / "normalization.json").read_text())

    assert [row["row_id"] for row in manifest["train"]] == ["a", "b"]
    assert [row["row_id"] for row in manifest["validation"]] == ["v"]
    n_black, n_white = 17 * 16, 31 * 32
    expected_mean = np.log(2) * n_white / (n_black + n_white)
    expected_std = np.sqrt(np.log(2) ** 2 * n_white / (n_black + n_white) - expected_mean ** 2)
    assert normalization["mean"] == pytest.approx(expected_mean)
    assert normalization["std"] == pytest.approx(expected_std)
    assert normalization["pixel_count"] == n_black + n_white
    assert manifest["train"][0]["boxes"] == [[2 / 16, 3 / 17, 10 / 16, 11 / 17]]
    assert manifest["train"][0]["labels"] == [0]
    assert not (real_root / "conflict1.bmp").exists()
    assert manifest["final_manifest_sha256"]["test2"] == hashlib.sha256((tmp_path / "test2.jsonl").read_bytes()).hexdigest()


def test_native_sample_preserves_nondivisible_raster_and_declared_coordinates(tmp_path):
    protocol, real_root, _ = input_files(tmp_path)
    manifest = uatd_readout.prepare_task(protocol, real_root, tmp_path / "prepared")

    sample = uatd_readout.native_sample(manifest["train"][0])

    assert sample["intensity"].shape == (1, 17, 16)
    assert sample["valid"].all()
    assert torch.count_nonzero(sample["intensity"]) == 0
    torch.testing.assert_close(sample["range_m"][[0, -1]], torch.tensor([5 / 17, 165 / 17]))
    torch.testing.assert_close(sample["bearing_rad"][[0, -1]], torch.deg2rad(torch.tensor([56.25, -56.25])))
    assert sample["configuration"] == {"RangeMin": 0., "RangeMax": 10.}


def test_prepare_rejects_changed_source_manifest(tmp_path):
    protocol, real_root, source = input_files(tmp_path)
    source.write_text(source.read_text() + "\n")

    with pytest.raises(ValueError, match="source manifest changed"):
        uatd_readout.prepare_task(protocol, real_root, tmp_path / "prepared")


class NativeOnlyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.))

    def forward_native(self, samples, normalization, masks):
        assert not self.training and not self.scale.requires_grad and not torch.is_grad_enabled()
        assert masks is None
        values = []
        for sample in samples:
            assert sample["intensity"].shape[-2] in (17, 19, 31)
            value = ((sample["intensity"].log1p() - normalization["mean"]) / normalization["std"]).mean()
            values.append(value * self.scale + sample["range_m"].mean())
        return {"patch_tokens": torch.stack(values)[:, None, None].expand(-1, 256, 384)}


def test_cache_calls_full_native_encoder_and_preserves_frozen_state_and_targets(tmp_path, monkeypatch):
    protocol, real_root, _ = input_files(tmp_path)
    prepared = tmp_path / "prepared"
    uatd_readout.prepare_task(protocol, real_root, prepared)
    model = NativeOnlyEncoder()
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"student": model.state_dict(), "step": 10000,
                "resolved_config": {"variant": "R013", "code_dir": "unused", "weights_path": "unused"}}, checkpoint)
    monkeypatch.setattr(uatd_readout, "build_encoder", lambda code, weights, variant: model)
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path / "training_artifacts")
    before = state_digest(model.state_dict())

    result = uatd_readout.extract_features(prepared, checkpoint, "R013", tmp_path / "cache", "cpu", 2)

    cached = torch.load(tmp_path / "cache/train.pt", weights_only=True)
    assert (tmp_path / "cache").is_symlink()
    assert (tmp_path / "cache").resolve() == tmp_path / "training_artifacts/cache"
    assert cached["features"].shape == (2, 384, 16, 16)
    assert cached["features"].dtype == torch.float32
    assert cached["row_ids"] == ["a", "b"]
    torch.testing.assert_close(cached["targets"][0]["boxes"], torch.tensor([[2 / 16, 3 / 17, 10 / 16, 11 / 17]]))
    assert cached["targets"][0]["labels"].dtype == torch.long
    assert result["encoder_hash_after_extraction"] == before == state_digest(model.state_dict())
    assert result["provenance"]["variant"] == "R013"
    assert "artifacts.py" in result["provenance"]["source_sha256"]
    assert set(result["cache_files"]) == {"train.pt", "validation.pt"}
    assert cached["features"][0, 0, 0, 0] < cached["features"][1, 0, 0, 0]


def test_final_opens_only_explicit_test_and_reuses_parent_features_contract(tmp_path, monkeypatch):
    protocol, real_root, source = input_files(tmp_path)
    prepared = tmp_path / "prepared"
    uatd_readout.prepare_task(protocol, real_root, prepared)
    model = NativeOnlyEncoder()
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"student": model.state_dict(), "step": 10000,
                "resolved_config": {"variant": "R013", "code_dir": "unused", "weights_path": "unused"}}, checkpoint)
    monkeypatch.setattr(uatd_readout, "build_encoder", lambda code, weights, variant: model)
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path / "training_artifacts")
    cache = tmp_path / "cache"
    parent = uatd_readout.extract_features(prepared, checkpoint, "R013", cache, "cpu", 2)
    norm_before = (prepared / "normalization.json").read_bytes()

    result = uatd_readout.extract_final(cache, "test1", tmp_path / "final", "cpu", 2)

    final = torch.load(tmp_path / "final/final.pt", weights_only=True)
    assert (tmp_path / "final").is_symlink()
    assert (tmp_path / "final").resolve() == tmp_path / "training_artifacts/final"
    assert final["row_ids"] == ["test1/a"]
    assert final["provenance"] == parent["provenance"] == result["provenance"]
    assert result["parent_cache_manifest_sha256"] == hashlib.sha256((cache / "cache_manifest.json").read_bytes()).hexdigest()
    assert result["test_split"] == "test1"
    assert result["cache_files"] == {"final.pt": hashlib.sha256((tmp_path / "final/final.pt").read_bytes()).hexdigest()}
    torch.testing.assert_close(final["features"][0], torch.load(cache / "train.pt", weights_only=True)["features"][0])
    assert (prepared / "normalization.json").read_bytes() == norm_before
    assert source.with_name("test2.jsonl").read_text().startswith("opaque")

    source.with_name("test1.jsonl").write_text("changed after preparation")
    with pytest.raises(ValueError, match="final manifest changed"):
        uatd_readout.extract_final(cache, "test1", tmp_path / "changed-final", "cpu", 2)


def test_checkpoint_requires_declared_variant_even_when_state_keys_would_match(tmp_path, monkeypatch):
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"student": NativeOnlyEncoder().state_dict(), "step": 10000,
                "resolved_config": {"variant": "R012", "code_dir": "unused", "weights_path": "unused"}}, checkpoint)
    monkeypatch.setattr(uatd_readout, "build_encoder", lambda code, weights, variant: NativeOnlyEncoder())
    with pytest.raises(ValueError, match="checkpoint variant R012 differs from requested R020"):
        uatd_readout.checkpoint_encoder(checkpoint, "R020", "cpu")
