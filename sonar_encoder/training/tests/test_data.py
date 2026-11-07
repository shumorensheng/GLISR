import hashlib
import json

import numpy as np
import pytest
import torch

from sonar_encoder.training.data import SonarDataset, collate_native
from sonar_encoder.training.prepare import fit_normalization


def test_native_samples_preserve_shapes_coordinates_and_unlabelled_intensity(tmp_path):
    frames = []
    configurations = {}
    for config_id, shape in (("C05", (256, 128)), ("C12", (1024, 512))):
        key = f"CPU_{config_id}_R00"
        intensity = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.prod(shape)
        ranges = np.linspace(0.5, 20.0, shape[0], dtype=np.float32)
        bearings = np.linspace(0.7, -0.7, shape[1], dtype=np.float32)
        np.savez(tmp_path / f"{key}.npz", intensity=intensity,
                 range_coordinates_m=ranges, bearing_coordinates_rad=bearings,
                 intensity_valid_region_mask=np.ones(shape, dtype=bool),
                 hit_mask=np.zeros(shape, dtype=bool),
                 semantic_valid_mask=np.zeros(shape, dtype=bool))
        configurations[key] = {"configuration": {"RangeBins": shape[0], "AzimuthBins": shape[1]}}
        frames.append({
            "sample_id": f"pose/{key}", "pose_id": "pose", "group_id": "group",
            "scene_id": "Shipwreck", "split": "train", "usage": "ssl_train",
            "config_id": config_id, "npz_relpath": f"{key}.npz",
            "metadata_relpath": "metadata.json", "sha256": "recorded-upstream",
        })
    frames.append({"usage": "test_seen"})
    (tmp_path / "metadata.json").write_text(json.dumps({"frames": configurations}))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"frames": frames}))
    dataset = SonarDataset(manifest_path, tmp_path, "ssl_train")
    samples = collate_native([dataset[0], dataset[1]])
    assert len(dataset) == len(samples) == 2
    assert [tuple(sample["intensity"].shape) for sample in samples] == [(1, 256, 128), (1, 1024, 512)]
    for sample in samples:
        with np.load(tmp_path / f"CPU_{sample['config_id']}_R00.npz") as original:
            torch.testing.assert_close(sample["intensity"][0], torch.from_numpy(original["intensity"]).float())
            torch.testing.assert_close(sample["range_m"], torch.from_numpy(original["range_coordinates_m"]))
            torch.testing.assert_close(sample["bearing_rad"], torch.from_numpy(original["bearing_coordinates_rad"]))
        assert sample["bearing_rad"][0] > sample["bearing_rad"][-1]
        assert sample["valid"].dtype == torch.bool
        assert sample["valid"].all()
        assert sample["configuration"]["RangeBins"] == sample["intensity"].shape[1]


def test_normalization_uses_only_train_seen_valid_pixels(tmp_path):
    frames = []
    for name, usage, value in (("train", "ssl_train", 1),
                               ("val", "ssl_validation", 100),
                               ("unseen", "unused", 10000)):
        intensity = np.full((2, 3), value, dtype=np.float32)
        intensity[0, 0] = 99999
        valid = np.ones((2, 3), dtype=bool)
        valid[0, 0] = False
        np.savez(tmp_path / f"{name}.npz", intensity=intensity, intensity_valid_region_mask=valid)
        frames.append({"npz_relpath": f"{name}.npz", "usage": usage})
    manifest = {"frames": frames}
    statistics = fit_normalization(manifest, tmp_path)
    assert statistics["mean"] == pytest.approx(np.log(2))
    assert statistics["std"] == pytest.approx(0, abs=1e-15)
    assert statistics["count"] == 5
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert statistics["source_manifest_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_normalization_weights_pixels_and_reports_population_std(tmp_path):
    frames = []
    for name, values in (("a", [[0, 3]]), ("b", [[1, 1, 1, 1]])):
        array = np.array(values, dtype=np.float32)
        np.savez(tmp_path / f"{name}.npz", intensity=array,
                 intensity_valid_region_mask=np.ones(array.shape, dtype=bool))
        frames.append({"npz_relpath": f"{name}.npz", "usage": "ssl_train"})
    statistics = fit_normalization({"frames": frames}, tmp_path)
    assert statistics["count"] == 6
    assert statistics["mean"] == pytest.approx(np.log(2))
    assert statistics["std"] == pytest.approx(np.log(2) / np.sqrt(3))
