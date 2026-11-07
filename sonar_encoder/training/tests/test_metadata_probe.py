import json
import math
from pathlib import Path
import pytest
import torch
from torch import nn

from sonar_encoder.training import artifacts, uatd_head
from sonar_encoder.training.readout import file_digest, state_digest


def metadata_input_files(tmp_path, include_final=False):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = [{"row_id": f"row_{index}", "shape": [32, 64, 3], "range_m": 16.,
             "horizontal_fov_deg": 90., "image": str(tmp_path / "absent.bmp"),
             "frequency": "unused", "labels": list(range(10)),
             "boxes": [[.2, .3, .4, .5]] * 10} for index in range(2)]
    image_protocol = tmp_path / "image_protocol.json"
    image_protocol.write_text("{}")
    manifest = {"task": "uatd", "protocol_path": str(image_protocol),
                "protocol_sha256": file_digest(image_protocol), "class_names": [str(k) for k in range(10)],
                "geometry_assumption_set": "uatd_uniform_polar_top_near_positive_left_v1",
                "train": rows, "validation": [{**row, "row_id": f"val_{i}"} for i, row in enumerate(rows)],
                "final_manifests": {"test1": str(tmp_path / "sealed_test1"),
                                    "test2": str(tmp_path / "sealed_test2")}}
    if include_final:
        manifest.update(real_root=str(tmp_path / "missing_images"), source_prefix="/source",
                        final_manifest_sha256={})
        for split in ("test1", "test2"):
            final_rows = [{"id": f"{split}_{i}", "image_path": f"/source/{split}_{i}.bmp",
                           "image_sha256": "unused", "pixel_sha256": "unused", "shape": [32, 64, 3],
                           "range_m": 16., "horizontal_fov_deg": 90., "frequency": "unused",
                           "objects": [{"name": str(k), "bbox_xyxy": [12.8, 9.6, 25.6, 16.]} for k in range(10)]}
                          for i in range(2)]
            source = tmp_path / f"sealed_{split}"
            source.write_text("\n".join(json.dumps(row) for row in final_rows))
            manifest["final_manifest_sha256"][split] = file_digest(source)
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    protocol = {"prepared": str(prepared), "prepared_manifest_sha256": file_digest(prepared / "manifest.json"),
                "image_readout_protocol": str(image_protocol), "image_readout_protocol_sha256": file_digest(image_protocol),
                "counts": {"train": 2, "validation": 2}, "input_8": "calibration descriptors",
                "token_coordinates": "nominal token centers with native sampling scales",
                "network": "independent pointwise 8->16->384 phi per task",
                "scope": "limited physical-configuration shortcut comparator"}
    protocol_path = tmp_path / "metadata_protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    return protocol_path, manifest


def test_descriptor_uses_native_scales_and_nominal_positive_left_centers_without_pixels():
    from sonar_encoder.training.metadata_probe import metadata_features

    row = {"shape": [32, 64, 3], "range_m": 16., "horizontal_fov_deg": 90.}
    features = metadata_features([row, {**row, "shape": [64, 128, 3]}])
    assert features.shape == (2, 8, 16, 16)
    assert features.dtype == torch.float32
    torch.testing.assert_close(features[0, 0, [0, 15], 0], torch.tensor([.03125, .96875]))
    angle = math.radians(42.1875)
    assert float(features[0, 1, 0, 0]) == pytest.approx(math.sin(angle))
    assert float(features[0, 1, 0, 15]) == pytest.approx(-math.sin(angle))
    assert float(features[0, 2, 0, 0]) == pytest.approx(math.cos(angle))
    x, y = math.log(.5), math.log(.5 * math.pi / 2 / 64)
    torch.testing.assert_close(features[0, 3:, 0, 0], torch.tensor([x, y, x + y, y - x, 1.]))
    torch.testing.assert_close(features[1, :3], features[0, :3])
    torch.testing.assert_close(features[1, 3:5], features[0, 3:5] - math.log(2))


def test_prepare_caches_only_trainval_metadata_and_keeps_gt_in_targets(tmp_path, monkeypatch):
    from sonar_encoder.training.metadata_probe import metadata_features, prepare_cache

    protocol, manifest = metadata_input_files(tmp_path)
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path / "external")
    cache = tmp_path / "cache"
    result = prepare_cache(protocol, cache)
    assert cache.is_symlink() and cache.resolve() == tmp_path / "external/cache"
    assert set(result["cache_files"]) == {"train.pt", "validation.pt"}
    assert result["provenance"]["task"] == "uatd_metadata_only"
    assert "checkpoint" not in result["provenance"]
    assert "encoder_state_sha256" not in result["provenance"]
    for split in ("train", "validation"):
        saved = torch.load(cache / f"{split}.pt", weights_only=True)
        assert saved["row_ids"] == [row["row_id"] for row in manifest[split]]
        torch.testing.assert_close(saved["features"], metadata_features(manifest[split]))
        torch.testing.assert_close(saved["targets"][0]["labels"], torch.arange(10))
        torch.testing.assert_close(saved["targets"][0]["boxes"], torch.tensor([[.2, .3, .4, .5]] * 10))
    altered = [{**row, "labels": [9], "boxes": [[.5, .5, .9, .9]], "row_id": "other",
                "image": "other_absent.bmp", "frequency": "different"} for row in manifest["train"]]
    torch.testing.assert_close(metadata_features(altered), metadata_features(manifest["train"]))
    assert not (tmp_path / "absent.bmp").exists()


def test_global_metadata_head_applies_phi_before_pooling_and_receives_gradients():
    from sonar_encoder.training.uatd_head import MetadataGlobalHead

    head = MetadataGlobalHead(nn.Linear(384, 10))
    features = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        head.head.weight.fill_(1.)
        head.head.bias.zero_()
    logits = head(features)
    torch.testing.assert_close(logits, head.head(head.phi(features).mean((-2, -1))))
    assert not torch.allclose(logits, head.head(head.phi(features.mean((-2, -1), keepdim=True)).flatten(1)))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.eye(10)[:2]).backward()
    assert all(parameter.grad.abs().sum() > 0 for parameter in head.phi.parameters())


def test_metadata_training_preserves_common_initial_heads_and_saves_independent_best_phi(tmp_path, monkeypatch):
    from sonar_encoder.training.metadata_probe import prepare_cache

    torch.set_num_threads(1)
    protocol, _ = metadata_input_files(tmp_path)
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path / "external")
    cache, output = tmp_path / "cache", tmp_path / "head"
    prepare_cache(protocol, cache)
    torch.manual_seed(0)
    common_global, common_dense = nn.Linear(384, 10), nn.Conv2d(384, 50, 1)
    with torch.no_grad():
        common_dense.bias[:10].fill_(math.log(.01 / .99))
    initial = {"global": state_digest(common_global.state_dict()), "dense": state_digest(common_dense.state_dict())}
    trained_heads = []
    evaluate = uatd_head.evaluate_heads

    def evaluate_and_capture(global_head, dense_head, *args):
        if not trained_heads:
            trained_heads.extend((global_head, dense_head))
        return evaluate(global_head, dense_head, *args)

    monkeypatch.setattr(uatd_head, "evaluate_heads", evaluate_and_capture)

    report = uatd_head.train_head(cache, output, "cpu")

    assert report["task"] == "uatd_metadata_only"
    assert report["epochs"] == 100 and report["updates"] == 100
    assert report["initial_base_head_sha256"] == initial
    assert report["phi_parameters"] == {"global": 6672, "dense": 6672}
    assert {id(p) for p in trained_heads[0].phi.parameters()}.isdisjoint(
        {id(p) for p in trained_heads[1][0].parameters()})
    assert report["selected"]["global"]["epoch"] == 5
    assert not report["final_test_evaluated"] and "encoder_hash_before" not in report
    assert output.is_symlink() and output.resolve() == tmp_path / "external/head"
    global_head = uatd_head.MetadataGlobalHead(nn.Linear(384, 10))
    dense_head = nn.Sequential(nn.Sequential(nn.Conv2d(8, 16, 1), nn.GELU(), nn.Conv2d(16, 384, 1)),
                               nn.Conv2d(384, 50, 1))
    for name, head, phi in (("global", global_head, global_head.phi), ("dense", dense_head, dense_head[0])):
        saved = torch.load(output / f"best_{name}.pt", weights_only=True)
        head.load_state_dict(saved["head"])
        assert state_digest(head.state_dict()) != report["initial_head_sha256"][name]
        assert state_digest(phi.state_dict()) != report["initial_phi_sha256"][name]
        assert state_digest(phi.state_dict()) == report["best_phi_sha256"][name]
    assert not (output / "initial.pt").exists() and not (output / "last.pt").exists()
    validation = torch.load(cache / "validation.pt", weights_only=True)
    metrics, _ = evaluate(global_head, dense_head, validation, "cpu", metadata_only=True)
    assert metrics["global"]["map"] == 1.


def paired_input_files(tmp_path, monkeypatch, include_final=False):
    from sonar_encoder.training.metadata_probe import metadata_features

    metadata_protocol, prepared_manifest = metadata_input_files(tmp_path, include_final)
    monkeypatch.setattr(artifacts, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACT_ROOT", tmp_path / "external")
    original_source = tmp_path / "original_source"
    original_source.mkdir()
    for name in ("metadata_probe.py", "artifacts.py"):
        (original_source / name).write_text(f"original synthetic {name}\n")
    (original_source / "uatd_head.py").write_bytes(Path(uatd_head.__file__).read_bytes())
    source_hashes = {name: file_digest(original_source / name) for name in ("metadata_probe.py", "artifacts.py")}
    metadata_provenance = {"task": "uatd_metadata_only", "prepared": str(tmp_path / "prepared"),
                           "task_manifest_sha256": file_digest(tmp_path / "prepared/manifest.json"),
                           "protocol_path": str(metadata_protocol), "protocol_sha256": file_digest(metadata_protocol),
                           "image_readout_protocol": prepared_manifest["protocol_path"],
                           "image_readout_protocol_sha256": prepared_manifest["protocol_sha256"],
                           "scope": "synthetic calibration shortcut comparison",
                           "source_sha256": source_hashes}
    checkpoint = tmp_path / "encoder.pt"
    state = {"weight": torch.tensor([1., 2.])}
    torch.save({"student": state}, checkpoint)
    image_provenance = {"task": "uatd", "variant": "R021", "checkpoint_step": 10000,
                        "prepared": metadata_provenance["prepared"],
                        "task_manifest_sha256": metadata_provenance["task_manifest_sha256"],
                        "protocol_sha256": prepared_manifest["protocol_sha256"],
                        "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
                        "encoder_state_sha256": state_digest(state)}
    parents = {}
    for name, provenance, channels in (("image", image_provenance, 384), ("metadata", metadata_provenance, 8)):
        directory = tmp_path / f"{name}_cache"
        directory.mkdir()
        cache_files = {}
        for split in ("train", "validation"):
            rows = prepared_manifest[split]
            features = (torch.zeros(len(rows), channels, 16, 16) if name == "image" else metadata_features(rows))
            path = directory / f"{split}.pt"
            torch.save({"features": features,
                        "targets": [{"labels": torch.tensor(row["labels"]),
                                     "boxes": torch.tensor(row["boxes"])} for row in rows],
                        "row_ids": [row["row_id"] for row in rows], "provenance": provenance}, path)
            cache_files[path.name] = file_digest(path)
        (directory / "cache_manifest.json").write_text(json.dumps({
            "provenance": provenance, "cache_files": cache_files, "class_names": prepared_manifest["class_names"]}))
        parents[name] = directory
    receipt = tmp_path / "source_snapshot.json"
    receipt.write_text(json.dumps({"source_root": str(original_source),
                                   "source_sha256": {**source_hashes, "uatd_head.py": file_digest(original_source / "uatd_head.py")},
                                   "protocol_sha256": file_digest(metadata_protocol),
                                   "cache_output": str(parents["metadata"])}))
    protocol = tmp_path / "combined_protocol.json"
    protocol.write_text(json.dumps({"image_cache": str(parents["image"]), "metadata_cache": str(parents["metadata"]),
                                   "image_cache_manifest_sha256": file_digest(parents["image"] / "cache_manifest.json"),
                                   "metadata_cache_manifest_sha256": file_digest(parents["metadata"] / "cache_manifest.json"),
                                   "network": "independent residual phi", "scope": "synthetic calibration shortcut comparison"}))
    return protocol, receipt, parents


def test_final_metadata_uses_original_receipt_without_opening_pixels(tmp_path, monkeypatch):
    from sonar_encoder.training.metadata_probe import extract_final, metadata_features

    _, receipt, parents = paired_input_files(tmp_path, monkeypatch, include_final=True)
    output = tmp_path / "final_metadata"
    manifest = extract_final(parents["metadata"], receipt, "test1", output)
    parent = json.loads((parents["metadata"] / "cache_manifest.json").read_text())
    saved = torch.load(output / "final.pt", weights_only=True)
    assert output.is_symlink()
    assert saved["provenance"] == parent["provenance"] == manifest["provenance"]
    assert saved["row_ids"] == ["test1_0", "test1_1"]
    assert manifest["parent_cache_manifest_sha256"] == file_digest(parents["metadata"] / "cache_manifest.json")
    assert manifest["source_receipt_sha256"] == file_digest(receipt)
    assert manifest["source_root"] == str(tmp_path / "original_source")
    assert manifest["preparation_source_sha256"]["metadata_probe.py"] != parent["provenance"]["source_sha256"]["metadata_probe.py"]
    torch.testing.assert_close(saved["features"], metadata_features([
        {"shape": [32, 64, 3], "range_m": 16., "horizontal_fov_deg": 90.}] * 2))
    torch.testing.assert_close(saved["targets"][0]["boxes"], torch.tensor([[.2, .3, .4, .5]] * 10))
    assert not (tmp_path / "missing_images").exists()
    other = tmp_path / "other_cache"
    other.mkdir()
    (other / "cache_manifest.json").write_bytes((parents["metadata"] / "cache_manifest.json").read_bytes())
    with pytest.raises(ValueError, match="receipt belongs to a different cache"):
        extract_final(other, receipt, "test1", tmp_path / "wrong_final")


def paired_final_input_files(tmp_path, monkeypatch):
    from sonar_encoder.training.metadata_probe import extract_final, prepare_paired_cache

    protocol, metadata_receipt, parents = paired_input_files(tmp_path, monkeypatch, include_final=True)
    parent_cache = tmp_path / "combined_parent"
    parent = prepare_paired_cache(protocol, metadata_receipt, parent_cache)
    source_root = tmp_path / "combined_original"
    source_root.mkdir()
    for name in parent["provenance"]["source_sha256"]:
        (source_root / name).write_bytes(Path(uatd_head.__file__).read_bytes() if name == "uatd_head.py" else
                                         f"original combined synthetic {name}\n".encode())
    parent["provenance"]["source_sha256"] = {name: file_digest(source_root / name)
                                                for name in parent["provenance"]["source_sha256"]}
    (parent_cache / "cache_manifest.json").write_text(json.dumps(parent))
    receipt = tmp_path / "combined_receipt.json"
    receipt.write_text(json.dumps({"source_root": str(source_root),
                                   "source_sha256": parent["provenance"]["source_sha256"],
                                   "protocol_sha256": file_digest(protocol), "cache_output": str(parent_cache)}))
    metadata_final = tmp_path / "metadata_final"
    metadata_manifest = extract_final(parents["metadata"], metadata_receipt, "test1", metadata_final)
    saved = torch.load(metadata_final / "final.pt", weights_only=True)
    image_final = tmp_path / "image_final"
    image_final.mkdir()
    image_provenance = parent["provenance"]["image_provenance"]
    torch.save({"features": torch.zeros(2, 384, 16, 16), "row_ids": saved["row_ids"],
                "targets": saved["targets"], "provenance": image_provenance}, image_final / "final.pt")
    image_manifest = {key: metadata_manifest[key] for key in
                      ("class_names", "test_split", "source_manifest", "source_manifest_sha256")}
    image_manifest.update(provenance=image_provenance,
                          parent_cache_manifest_sha256=file_digest(parents["image"] / "cache_manifest.json"),
                          cache_files={"final.pt": file_digest(image_final / "final.pt")})
    (image_final / "cache_manifest.json").write_text(json.dumps(image_manifest))
    return parent_cache, receipt, image_final, metadata_final


def test_final_pair_references_sealed_parent_sources_and_separate_tensors(tmp_path, monkeypatch):
    from sonar_encoder.training.metadata_probe import load_paired_final_cache, prepare_paired_final

    parent_cache, receipt, image_final, metadata_final = paired_final_input_files(tmp_path, monkeypatch)
    output = tmp_path / "paired_final"
    manifest = prepare_paired_final(parent_cache, receipt, image_final, metadata_final, output)
    assert list(output.iterdir()) == [output / "cache_manifest.json"]
    assert manifest["provenance"] == json.loads((parent_cache / "cache_manifest.json").read_text())["provenance"]
    assert manifest["parent_cache_manifest_sha256"] == file_digest(parent_cache / "cache_manifest.json")
    assert manifest["source_receipt_sha256"] == file_digest(receipt)
    cache = load_paired_final_cache(manifest)
    assert cache["row_ids"] == ["test1_0", "test1_1"]
    assert cache["features"].shape == (2, 384, 16, 16)
    assert cache["metadata"].shape == (2, 8, 16, 16)
    assert not (output / "final.pt").exists()


@pytest.mark.parametrize("change", ["test_split", "parent", "row_ids", "targets"])
def test_final_pair_rejects_mixed_splits_parents_and_misaligned_targets(tmp_path, monkeypatch, change):
    from sonar_encoder.training.metadata_probe import prepare_paired_final

    parent_cache, receipt, image_final, metadata_final = paired_final_input_files(tmp_path, monkeypatch)
    path = metadata_final / "cache_manifest.json"
    manifest = json.loads(path.read_text())
    if change == "test_split":
        manifest["test_split"] = "test2"
    elif change == "parent":
        manifest["parent_cache_manifest_sha256"] = "wrong original cache"
    else:
        cache = torch.load(metadata_final / "final.pt", weights_only=True)
        if change == "row_ids":
            cache["row_ids"].reverse()
        else:
            cache["targets"][0]["boxes"][0, 0] = .1
        torch.save(cache, metadata_final / "final.pt")
        manifest["cache_files"]["final.pt"] = file_digest(metadata_final / "final.pt")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="final (parent|row IDs|targets)"):
        prepare_paired_final(parent_cache, receipt, image_final, metadata_final, tmp_path / "mixed_pair")


def test_pair_manifest_references_original_sources_and_separate_parent_features(tmp_path, monkeypatch):
    from sonar_encoder.training.metadata_probe import load_paired_cache, prepare_paired_cache

    protocol, receipt, parents = paired_input_files(tmp_path, monkeypatch)
    output = tmp_path / "paired"
    manifest = prepare_paired_cache(protocol, receipt, output)
    assert list(output.iterdir()) == [output / "cache_manifest.json"]
    assert manifest["provenance"]["task"] == "uatd_image_metadata"
    assert manifest["provenance"]["metadata_source_receipt"] == str(receipt)
    assert manifest["provenance"]["metadata_source_receipt_sha256"] == file_digest(receipt)
    assert manifest["provenance"]["metadata_source_root"] == str(tmp_path / "original_source")
    train, validation = load_paired_cache(manifest)
    for cache, split in ((train, "train"), (validation, "validation")):
        assert cache["features"].shape == (2, 384, 16, 16)
        assert cache["metadata"].shape == (2, 8, 16, 16)
        assert cache["row_ids"] == (["row_0", "row_1"] if split == "train" else ["val_0", "val_1"])
    (tmp_path / "original_source/metadata_probe.py").write_text("changed original source\n")
    with pytest.raises(ValueError, match="original metadata source changed"):
        load_paired_cache(manifest)


@pytest.mark.parametrize("changed_file, error", [
    ("image_cache/cache_manifest.json", "paired image cache manifest changed"),
    ("metadata_cache/train.pt", "paired cache tensor changed"),
    ("combined_protocol.json", "combined readout protocol changed"),
    ("source_snapshot.json", "original metadata source receipt changed"),
])
def test_pair_verifies_parent_protocol_and_original_source_receipt_hashes(tmp_path, monkeypatch, changed_file, error):
    from sonar_encoder.training.metadata_probe import load_paired_cache, prepare_paired_cache

    protocol, receipt, _ = paired_input_files(tmp_path, monkeypatch)
    manifest = prepare_paired_cache(protocol, receipt, tmp_path / "paired")
    path = tmp_path / changed_file
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match=error):
        load_paired_cache(manifest)


@pytest.mark.parametrize("change", ["row_ids", "labels", "boxes"])
def test_pair_rejects_misaligned_parent_rows_or_targets(tmp_path, monkeypatch, change):
    from sonar_encoder.training.metadata_probe import prepare_paired_cache

    protocol, receipt, parents = paired_input_files(tmp_path, monkeypatch)
    path = parents["metadata"] / "train.pt"
    cache = torch.load(path, weights_only=True)
    if change == "row_ids":
        cache["row_ids"].reverse()
    elif change == "labels":
        cache["targets"][0]["labels"][0] = 9
    else:
        cache["targets"][0]["boxes"][0, 0] = .1
    torch.save(cache, path)
    manifest_path = parents["metadata"] / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cache_files"]["train.pt"] = file_digest(path)
    manifest_path.write_text(json.dumps(manifest))
    design = json.loads(protocol.read_text())
    design["metadata_cache_manifest_sha256"] = file_digest(manifest_path)
    protocol.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="paired train (row IDs|targets) differ"):
        prepare_paired_cache(protocol, receipt, tmp_path / "paired")


def test_combined_training_preserves_common_heads_and_records_best_phi_without_encoder_baseline(tmp_path, monkeypatch):
    from sonar_encoder.training.metadata_probe import prepare_paired_cache

    torch.set_num_threads(1)
    protocol, receipt, _ = paired_input_files(tmp_path, monkeypatch)
    cache, output = tmp_path / "paired", tmp_path / "combined_head"
    prepare_paired_cache(protocol, receipt, cache)
    torch.manual_seed(0)
    common_global, common_dense = nn.Linear(384, 10), nn.Conv2d(384, 50, 1)
    with torch.no_grad():
        common_dense.bias[:10].fill_(math.log(.01 / .99))
    expected = {"global": state_digest(common_global.state_dict()), "dense": state_digest(common_dense.state_dict())}
    report = uatd_head.train_head(cache, output, "cpu")
    assert report["task"] == "uatd_image_metadata"
    assert report["epochs"] == 100 and report["updates"] == 100
    assert report["initial_base_head_sha256"] == expected
    assert report["phi_parameters"] == {"global": 6672, "dense": 6672}
    assert report["initial_phi_sha256"]["global"] != report["initial_phi_sha256"]["dense"]
    assert report["selected"]["global"]["epoch"] == 5
    assert report["provenance"]["metadata_source_receipt_sha256"] == file_digest(receipt)
    assert not report["encoder_parameters_in_optimizer"]
    assert not report["final_test_evaluated"] and "encoder_hash_before" not in report
    assert output.is_symlink()
    for name, base in (("global", nn.Linear(384, 10)), ("dense", nn.Conv2d(384, 50, 1))):
        head = uatd_head.ImageMetadataHead(base, global_readout=name == "global")
        saved = torch.load(output / f"best_{name}.pt", weights_only=True)
        head.load_state_dict(saved["head"])
        assert state_digest(head.phi.state_dict()) == report["best_phi_sha256"][name]
        assert report["best_phi_sha256"][name] != report["initial_phi_sha256"][name]
    assert not (output / "initial.pt").exists() and not (output / "last.pt").exists()
