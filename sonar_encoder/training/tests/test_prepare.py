import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from sonar_encoder.training.prepare import build_manifest


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "data"
    records = []
    assignment = {"group-a": "train", "group-b": "validation", "group-c": "test"}
    for run, group in zip(("run-a", "run-b", "run-c"), assignment):
        anchor = root / run / "A000001"
        anchor.mkdir(parents=True)
        frames = {}
        for number in range(14):
            key = f"CPU_C{number:02d}_R00"
            payload = f"{run}/{key}".encode()
            (anchor / f"{key}.npz").write_bytes(payload)
            frames[key] = {
                "config_id": f"CPU_C{number:02d}",
                "file": f"{key}.npz",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        (anchor / "metadata.json").write_text(json.dumps({"frames": frames}))
        records.append({
            "source_pose_file": f"/capture/oceangym_dataset/{run}-poses.json",
            "source_pose_index_zero_based": 0,
            "scene_id": "Shipwreck",
            "group_id": group,
            "run": f"/capture/oceangym_dataset/{run}",
            "anchor": f"/capture/oceangym_dataset/{run}/A000001",
            "origin": "base",
        })
    shutil.copytree(root / "run-a", root / "replaced-run")
    index = root / "canonical-index.json"
    index.write_text(json.dumps(records))
    return root, index, assignment


def test_manifest_preserves_canonical_identity_and_all_configuration_usages(capture, tmp_path):
    root, index, assignment = capture
    manifest = build_manifest(index, assignment, root)
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert build_manifest(relocated / index.name, assignment, relocated) == manifest
    assert manifest["source_index_sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    assert len(manifest["pose_assignments"]) == 3
    assert len(manifest["frames"]) == 42
    assert len({frame["sample_id"] for frame in manifest["frames"]}) == 42
    for frame in manifest["frames"]:
        assert (root / frame["npz_relpath"]).is_file()
        assert (relocated / frame["npz_relpath"]).is_file()
        assert (relocated / frame["metadata_relpath"]).is_file()
        assert "replaced-run" not in frame["npz_relpath"]
        assert frame["split"] == assignment[frame["group_id"]]
    train = [f for f in manifest["frames"] if f["split"] == "train"]
    assert {f["config_id"] for f in train if f["usage"] == "ssl_train"} == {
        "C00", "C01", "C03", "C05", "C07", "C12"
    }
    assert next(f for f in train if f["config_id"] == "C02")["usage"] == "unused"
    assert manifest["counts"]["frames_by_usage"] == {
        "ssl_train": 6, "ssl_validation": 6, "test_seen": 6,
        "test_unseen_single": 5, "test_unseen_combination": 2,
        "test_reference": 1, "unused": 16,
    }
    source = "/capture/oceangym_dataset/run-a-poses.json\n0"
    pose_id = hashlib.sha256(source.encode()).hexdigest()
    assert train[0]["sample_id"] == f"{pose_id}/CPU_C00_R00"


def test_manifest_rejects_unassigned_groups(capture):
    root, index, assignment = capture
    del assignment["group-b"]
    with pytest.raises(ValueError, match="group"):
        build_manifest(index, assignment, root)


def test_cli_rejects_changed_index_snapshot(capture, tmp_path):
    root, index, assignment = capture
    config = tmp_path / "split.json"
    config.write_text(json.dumps({
        "source_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "assignment": assignment,
    }))
    index.write_text(json.dumps(json.loads(index.read_text())[:-1]))
    result = subprocess.run([
        sys.executable, "-m", "sonar_encoder.training.prepare",
        "--index", str(index), "--assignment", str(config),
        "--data-root", str(root), "--output", str(tmp_path / "manifest.json"),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "source index SHA-256" in result.stderr
    assert not (tmp_path / "manifest.json").exists()
