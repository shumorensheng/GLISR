import unittest
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from unittest import mock

import torch
import pytest
from torch import nn


class UATDHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_different_classes_in_same_cell_keep_both_box_targets(self):
        from sonar_encoder.training.uatd_head import box_targets

        targets = [{"labels": torch.tensor([2, 7]),
                    "boxes": torch.tensor([[.20, .20, .30, .30], [.21, .21, .31, .31]])}]
        heatmap, regression, positive = box_targets(targets, "cpu")
        self.assertEqual(int(positive.sum()), 2)
        self.assertEqual(float(heatmap.sum()), 2)
        torch.testing.assert_close(regression[0, 2, :2, 4, 4], torch.zeros(2))
        torch.testing.assert_close(regression[0, 7, :2, 4, 4], torch.full((2,), .16))
        torch.testing.assert_close(regression[0, 2, 2:, 4, 4], torch.full((2,), 1.6).log())

    def test_dense_loss_updates_positive_boxes_only(self):
        from sonar_encoder.training.uatd_head import box_targets, dense_loss

        targets = [{"labels": torch.tensor([3]), "boxes": torch.tensor([[.2, .3, .4, .5]])}]
        output = torch.zeros(1, 50, 16, 16, requires_grad=True)
        loss = dense_loss(output, box_targets(targets, "cpu"))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(output.grad[:, :10].abs().sum()), 0)
        regression_grad = output.grad[:, 10:].reshape(1, 10, 4, 16, 16)
        active_cells = regression_grad.abs().sum(2) > 0
        self.assertEqual(int(active_cells.sum()), 1)
        self.assertTrue(active_cells[0, 3, 6, 4])

    def test_decode_recovers_class_specific_boxes_and_keeps_adjacent_objects(self):
        from sonar_encoder.training.uatd_head import decode_detections

        output = torch.zeros(1, 50, 16, 16)
        output[:, :10] = -20
        for label in (2, 7):
            output[0, label, 4, 4] = 10
            output[0, 10 + label * 4 + 2:10 + label * 4 + 4, 4, 4] = torch.tensor(1.6).log()
        output[0, 2, 4, 5] = 9
        output[0, 20:22, 4, 5] = torch.tensor(1.6).log()
        output[0, 4, 4, 4] = 12
        output[0, 28:30, 4, 4] = -20
        decoded = decode_detections(output)[0]
        self.assertEqual(set(decoded["labels"].tolist()), {2, 7})
        self.assertEqual(len(decoded["boxes"]), 3)
        torch.testing.assert_close(decoded["boxes"][:2], torch.tensor([[.23125, .23125, .33125, .33125]] * 2))

    def test_detection_ap_matches_by_image_and_penalizes_higher_scoring_false_positive(self):
        from sonar_encoder.training.uatd_head import detection_metrics

        targets = [{"labels": torch.arange(10), "boxes": torch.tensor([[0., 0., .5, .5]] * 10)}]
        predictions = [{"labels": torch.tensor([0] + list(range(10))),
                        "boxes": torch.tensor([[.6, .6, 1., 1.]] + [[0., 0., .5, .5]] * 10),
                        "scores": torch.tensor([.99] + [.9] * 10)}]
        result = detection_metrics(predictions, targets)
        self.assertAlmostEqual(result["per_class_ap50"][0], .5)
        self.assertAlmostEqual(result["map50"], .95)

    def test_multilabel_ap_uses_ranking_and_global_targets_include_all_objects(self):
        from sonar_encoder.training.uatd_head import global_targets, multilabel_metrics

        targets = [{"labels": torch.tensor([1, 1, 4]), "boxes": torch.zeros(3, 4)}]
        truth = global_targets(targets, "cpu")
        self.assertEqual(float(truth.sum()), 2)
        labels = torch.eye(10)
        result = multilabel_metrics(labels * .8 + .1, labels)
        self.assertAlmostEqual(result["map"], 1.)
        self.assertAlmostEqual(result["macro_f1_at_0_5"], 1.)

    def test_fixed_budget_trains_both_heads_and_leaves_source_encoder_unchanged(self):
        from sonar_encoder.training import artifacts
        from sonar_encoder.training.readout import file_digest, state_digest
        from sonar_encoder.training.uatd_head import final_test, train_head

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "encoder.pt"
            state = {"weight": torch.tensor([1., 2.])}
            torch.save({"student": state}, checkpoint)
            provenance = {"task": "uatd", "checkpoint": str(checkpoint), "checkpoint_sha256": file_digest(checkpoint),
                          "encoder_state_sha256": state_digest(state)}
            cache = root / "cache"
            cache.mkdir()
            files = {}
            targets = [{"labels": torch.arange(10), "boxes": torch.tensor([[.2, .3, .4, .5]] * 10)} for _ in range(10)]
            for split in ("train", "validation"):
                path = cache / f"{split}.pt"
                torch.save({"features": torch.zeros(10, 384, 16, 16), "targets": targets,
                            "row_ids": [str(k) for k in range(10)], "provenance": provenance}, path)
                files[path.name] = file_digest(path)
            (cache / "cache_manifest.json").write_text(json.dumps({"provenance": provenance, "cache_files": files,
                                                                   "class_names": [str(k) for k in range(10)]}))
            output = root / "head"
            with (mock.patch.object(artifacts, "RUN_ROOT", root),
                  mock.patch.object(artifacts, "ARTIFACT_ROOT", root / "training_artifacts"),
                  redirect_stdout(io.StringIO())):
                report = train_head(cache, output, "cpu")
            self.assertTrue(output.is_symlink())
            self.assertEqual(output.resolve(), root / "training_artifacts/head")
            self.assertEqual(report["epochs"], 100)
            self.assertEqual(report["updates"], 100)
            self.assertEqual(report["selected"]["global"]["epoch"], 5)
            self.assertEqual(report["selected"]["dense"]["epoch"], 5)
            self.assertEqual(report["encoder_hash_before"], report["encoder_hash_after"])
            self.assertFalse(report["final_test_evaluated"])
            self.assertEqual(set(report["initial_head_sha256"]), {"global", "dense"})
            self.assertTrue(all(len(value) == 64 for value in report["initial_head_sha256"].values()))
            self.assertTrue((output / "best_global.pt").exists())
            self.assertTrue((output / "best_dense.pt").exists())
            for name in ("global", "dense"):
                best = torch.load(output / f"best_{name}.pt", weights_only=True)["head"]
                self.assertNotEqual(state_digest(best), report["initial_head_sha256"][name])
            self.assertFalse((output / "initial.pt").exists())
            self.assertFalse((output / "last.pt").exists())
            final_cache = root / "final_cache"
            final_cache.mkdir()
            (final_cache / "final.pt").write_bytes((cache / "validation.pt").read_bytes())
            (final_cache / "cache_manifest.json").write_text(json.dumps({
                "provenance": provenance, "parent_cache_manifest_sha256": file_digest(cache / "cache_manifest.json"),
                "cache_files": {"final.pt": file_digest(final_cache / "final.pt")},
                "class_names": [str(k) for k in range(10)], "test_split": "synthetic_test"}))
            with (mock.patch.object(artifacts, "RUN_ROOT", root),
                  mock.patch.object(artifacts, "ARTIFACT_ROOT", root / "training_artifacts")):
                final_report = final_test(final_cache, output, root / "final_output", "cpu")
            self.assertTrue((root / "final_output").is_symlink())
            self.assertEqual(final_report["selected"], report["selected"])
            self.assertEqual(final_report["metrics"]["global"]["map"], 1.)
            self.assertEqual(file_digest(checkpoint), provenance["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("global_readout", [True, False])
def test_combined_initial_readout_matches_image_only_and_metadata_becomes_active(global_readout):
    from sonar_encoder.training.uatd_head import ImageMetadataHead, box_targets, dense_loss

    torch.set_num_threads(1)
    torch.manual_seed(0)
    base = nn.Linear(384, 10) if global_readout else nn.Conv2d(384, 50, 1)
    image = torch.randn(2, 384, 16, 16)
    metadata = torch.randn(2, 8, 16, 16)
    features = torch.cat([image, metadata], dim=1)
    expected = base(image.mean((-2, -1)) if global_readout else image).detach()
    combined = ImageMetadataHead(base, global_readout=global_readout)
    assert torch.equal(combined(features), expected)
    assert sum(parameter.numel() for parameter in combined.phi.parameters()) == 6672
    optimizer = torch.optim.AdamW(combined.parameters(), lr=1e-3, weight_decay=1e-4)
    targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[.2, .3, .4, .5]])}] * 2
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = combined(features)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(output, torch.eye(10)[:2]) if global_readout
                else dense_loss(output, box_targets(targets, "cpu")))
        loss.backward()
        assert combined.phi[-1].weight.grad.abs().sum() > 0
        assert combined.phi[-1].bias.grad.abs().sum() > 0
        if step == 0:
            assert combined.phi[0].weight.grad.count_nonzero() == 0
        else:
            assert combined.phi[0].weight.grad.abs().sum() > 0
            assert combined.phi[0].bias.grad.abs().sum() > 0
        optimizer.step()
    changed = features.clone()
    changed[:, 384:] += 1
    assert not torch.equal(combined(features), combined(changed))


@pytest.mark.parametrize("combined", [False, True])
def test_final_probe_loads_original_selected_heads_and_preserves_phi_predictions(tmp_path, monkeypatch, combined):
    from sonar_encoder.training import artifacts, uatd_head
    from sonar_encoder.training.metadata_probe import prepare_paired_final
    from sonar_encoder.training.readout import file_digest
    from sonar_encoder.training.tests.test_metadata_probe import paired_final_input_files

    torch.set_num_threads(1)
    parent_cache, receipt, image_final, metadata_final = paired_final_input_files(tmp_path, monkeypatch)
    if combined:
        final_cache = tmp_path / "combined_final"
        manifest = prepare_paired_final(parent_cache, receipt, image_final, metadata_final, final_cache)
    else:
        final_cache = metadata_final
        manifest = json.loads((final_cache / "cache_manifest.json").read_text())
        parent_cache = Path(manifest["parent_cache"])
        receipt = Path(manifest["source_receipt"])
    provenance = manifest["provenance"]
    parent = json.loads((parent_cache / "cache_manifest.json").read_text())
    if combined:
        global_head = uatd_head.ImageMetadataHead(nn.Linear(384, 10), global_readout=True)
        dense_head = uatd_head.ImageMetadataHead(nn.Conv2d(384, 50, 1), global_readout=False)
        global_base, dense_base = global_head.head, dense_head.head
        dense_phi = dense_head.phi
    else:
        global_head = uatd_head.MetadataGlobalHead(nn.Linear(384, 10))
        dense_head = nn.Sequential(nn.Sequential(nn.Conv2d(8, 16, 1), nn.GELU(), nn.Conv2d(16, 384, 1)),
                                   nn.Conv2d(384, 50, 1))
        global_base, dense_base = global_head.head, dense_head[1]
        dense_phi = dense_head[0]
    with torch.no_grad():
        global_base.weight.fill_(.002)
        global_base.bias.fill_(.1)
        dense_base.weight.fill_(.001)
        dense_base.bias.fill_(0.)
        dense_base.bias[:10].fill_(-20.)
        dense_base.bias[0] = 2.
        for phi in (global_head.phi, dense_phi):
            phi[0].weight.fill_(.02)
            phi[0].bias.fill_(.1)
            phi[2].weight.fill_(.01)
            phi[2].bias.fill_(.4)
    heads = tmp_path / "selected_heads"
    artifacts.create_artifact_directory(heads)
    selected = {"global": {"epoch": 5, "score": 1.}, "dense": {"epoch": 10, "score": .1}}
    for name, head in (("global", global_head), ("dense", dense_head)):
        torch.save({"head": head.state_dict(), "provenance": provenance,
                    "epoch": selected[name]["epoch"]}, heads / f"best_{name}.pt")
    hashes = {name: file_digest(heads / f"best_{name}.pt") for name in selected}
    receipt_data = json.loads(receipt.read_text())
    selection = {"cache_manifest_sha256": file_digest(parent_cache / "cache_manifest.json"),
                 "selected": selected, "best_heads_sha256": hashes,
                 "head_source_sha256": receipt_data["source_sha256"]["uatd_head.py"]}
    (heads / "report.json").write_text(json.dumps(selection))
    cache = torch.load(metadata_final / "final.pt", weights_only=True)
    features = cache["features"]
    if combined:
        features = torch.cat([torch.load(image_final / "final.pt", weights_only=True)["features"], features], dim=1)
    expected_scores = global_head(features).sigmoid().detach()
    expected_detections = uatd_head.decode_detections(dense_head(features))
    output = tmp_path / "final_predictions"
    report = uatd_head.final_test(final_cache, heads, output, "cpu")
    assert output.is_symlink()
    predictions = torch.load(output / "predictions.pt", weights_only=True)
    torch.testing.assert_close(predictions["global_scores"], expected_scores)
    for actual, expected in zip(predictions["detections"], expected_detections):
        for key in ("scores", "boxes", "labels"):
            torch.testing.assert_close(actual[key], expected[key])
    assert report["metrics"]["global"]["map"] == 1.
    assert report["metrics"]["dense"] == uatd_head.detection_metrics(expected_detections, cache["targets"])
    assert report["selected"] == selected
    assert report["task"] == ("uatd_image_metadata" if combined else "uatd_metadata_only")
    assert report["final_test_evaluated"]
    assert report["provenance"] == parent["provenance"]
    assert report["source_receipt_sha256"] == file_digest(receipt)
    assert all(file_digest(heads / f"best_{name}.pt") == digest for name, digest in hashes.items())
