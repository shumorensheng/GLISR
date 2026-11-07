import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sonar_encoder.training import artifacts, train
from sonar_encoder.training.model import prepare_images


class TinyEncoder(nn.Module):
    """Fast context-dependent encoder for exercising the real training loop."""

    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 4)

    def forward(self, images, masks=None):
        pixels = torch.nn.functional.adaptive_avg_pool2d(images, (16, 16))
        tokens = pixels.flatten(2).transpose(1, 2)
        tokens = self.projection(tokens + tokens.mean(1, keepdim=True))
        return {"patch_tokens": tokens, "global_features": tokens.mean(1)}

    def forward_native(self, samples, normalization, masks):
        return self(prepare_images(samples, normalization, masks), masks)


def test_accumulation_updates_teacher_once_after_optimizer():
    student = nn.Linear(1, 1, bias=False)
    student.weight.data.fill_(1)
    teacher = copy.deepcopy(student).requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    for _ in range(4):
        (student(torch.ones(1, 1)).square().mean() / 4).backward()
        assert teacher.weight.item() == 1
    updated, _ = train.optimizer_update(student, teacher, optimizer, scaler, 0.9, 10)
    assert updated
    assert student.weight.item() == pytest.approx(0.8)
    assert teacher.weight.item() == pytest.approx(0.98)
    assert teacher.weight.grad is None


def test_amp_overflow_does_not_update_student_or_teacher():
    student = nn.Linear(1, 1, bias=False)
    teacher = copy.deepcopy(student).requires_grad_(False)
    before = student.weight.detach().clone()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(student(torch.ones(1, 1)).sum() * float("inf")).backward()
    updated, _ = train.optimizer_update(student, teacher, optimizer, scaler, 0.9, 1)
    assert not updated
    assert scaler.get_scale() == 4
    torch.testing.assert_close(student.weight, before)
    torch.testing.assert_close(teacher.weight, before)


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    # Only the large, externally supplied DINO asset is substituted.
    monkeypatch.setattr(train, "build_encoder", lambda *_: TinyEncoder())
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path / "training_artifacts")
    rows = []
    for index in range(7):
        path = tmp_path / f"sample{index}.npz"
        intensity = np.random.default_rng(index).uniform(0, 1, (32, 32)).astype("float32")
        np.savez(path, intensity=intensity, range_coordinates_m=np.linspace(1, 4, 32),
                 bearing_coordinates_rad=np.linspace(.5, -.5, 32),
                 intensity_valid_region_mask=np.ones((32, 32), dtype=bool))
        metadata = {"frames": {"CPU_C00_R00": {"configuration": {"RangeMax": 4}}}}
        (tmp_path / f"metadata{index}.json").write_text(json.dumps(metadata))
        rows.append({"sample_id": f"pose{index}/CPU_C00_R00", "pose_id": f"pose{index}",
                     "group_id": f"g{index}", "scene_id": "test", "config_id": "C00",
                     "split": "train" if index < 5 else "validation",
                     "usage": "ssl_train" if index < 5 else "ssl_validation",
                     "npz_relpath": path.name, "metadata_relpath": f"metadata{index}.json",
                     "sha256": "fixture"})
    manifest = {"frames": rows}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    digest = train.manifest_digest(manifest)
    norm_path = tmp_path / "normalization.json"
    norm_path.write_text(json.dumps({"mean": 0.4, "std": 0.2, "count": 5120,
                                    "source_manifest_sha256": digest}))
    config = json.loads(Path("sonar_encoder/training/configs/r010_v1.json").read_text())
    (tmp_path / "weights").write_bytes(b"fixture")
    config.update(variant="R010", device="cpu", cpu_threads=1, weights_path=str(tmp_path / "weights"),
                  code_dir=str(tmp_path), micro_batch=2, accumulation=2, amp=False,
                  steps=4, warmup_steps=1, validation_every=2, log_every=1,
                  diagnostics_per_config=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, manifest_path, tmp_path, norm_path


def test_resume_matches_uninterrupted_parameters_masks_and_data(tiny_run):
    args = tiny_run
    whole = args[2] / "whole"
    resumed = args[2] / "resumed"
    train.run_training(*args, whole, 4, None)
    train.run_training(*args, resumed, 2, None)
    train.run_training(*args, resumed, 4, resumed / "checkpoints/last.pt")
    a = torch.load(whole / "checkpoints/last.pt", weights_only=False)
    b = torch.load(resumed / "checkpoints/last.pt", weights_only=False)
    assert a["step"] == b["step"] == 4
    for field in ("student", "teacher", "predictor"):
        for name in a[field]:
            torch.testing.assert_close(a[field][name], b[field][name], rtol=0, atol=0)
    for field in ("epoch_order", "mask_rng", "sample_rng"):
        torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)
    assert a["sample_cursor"] == b["sample_cursor"]
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert (whole / "checkpoints").is_symlink()
    assert (whole / "checkpoints").resolve() == args[2] / "training_artifacts/whole/checkpoints"
    assert {path.name for path in (whole / "checkpoints").iterdir()} == {"initial.pt", "last.pt"}
    assert "artifacts.py" in a["provenance"]["source"]
    assert (whole / "code_snapshot/artifacts.py").read_bytes() == Path(artifacts.__file__).read_bytes()
    lines = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in lines if row["kind"] == "train"] == [1, 2, 3, 4]


def test_changed_input_refuses_resume(tiny_run):
    args = tiny_run
    output = args[2] / "resume_check"
    train.run_training(*args, output, 1, None)
    norm = json.loads(args[3].read_text())
    norm["mean"] = 0.8
    args[3].write_text(json.dumps(norm))
    with pytest.raises(ValueError, match="provenance"):
        train.run_training(*args, output, 2, output / "checkpoints/last.pt")


def test_effective_rank_reports_constant_collapse():
    stats = train.feature_statistics(torch.ones(8, 4))
    assert stats["effective_rank"] == 0
    assert stats["mean_std"] == 0


def test_patch_diagnostics_detect_input_independent_spatial_template():
    template = torch.eye(4).unsqueeze(0).repeat(8, 1, 1)
    stats = train.feature_statistics(template)
    assert stats["mean_std"] == 0
    assert stats["effective_rank"] == 0
    assert stats["pairwise_cosine_quantiles"] == [1.] * 5


def test_patch_cosine_compares_vectors_at_matching_positions_across_images():
    features = torch.tensor([[[1., 0.], [1., 0.]], [[-1., 0.], [-1., 0.]]])
    stats = train.feature_statistics(features)
    assert stats["pairwise_cosine_quantiles"] == [-1.] * 5
