import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
from torch import nn


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_prepare_uses_only_inner_training_pixels_and_preserves_test_pixels_unopened(self):
        from sonar_encoder.training.readout import prepare_task

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            rows = []
            for index, pixels in enumerate(([[0, 255]], [[128, 128]])):
                path = real / f"{index}.png"
                Image.fromarray(np.array(pixels, dtype=np.uint8)).save(path)
                rows.append({"index": index, "classID": index, "className": f"class{index}", "image": f"/media/hit/大磁盘/pmca_real_dataset/{index}.png", "pixelSHA256": str(index)})
            train = root / "train.jsonl"
            outer = root / "val.jsonl"
            train.write_text("".join(json.dumps(row) + "\n" for row in rows))
            outer.write_text(json.dumps({"index": 2, "classID": 0, "image": "/media/hit/大磁盘/pmca_real_dataset/missing-test.png"}) + "\n")
            protocol = {"source_sha256": {str(train): hashlib.sha256(train.read_bytes()).hexdigest(), str(outer): hashlib.sha256(outer.read_bytes()).hexdigest()}, "nksid": {"outer_partition": {"train_manifest": str(train), "evaluation_manifest": str(outer)}, "inner_split": {"train_indices": [0], "val_indices": [1]}}}
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            prepared = prepare_task("nksid", protocol_path, real, root / "prepared")
            self.assertEqual([row["row_id"] for row in prepared["train"]], [0])
            self.assertEqual([row["row_id"] for row in prepared["validation"]], [1])
            normalization = json.loads((root / "prepared/normalization.json").read_text())
            self.assertEqual(normalization["pixel_count"], 2)
            self.assertAlmostEqual(normalization["mean"], np.log(2) / 2)
            self.assertAlmostEqual(normalization["std"], np.log(2) / 2)

    def test_confusion_metrics_include_all_classes_background_and_wall(self):
        from sonar_encoder.training.readout import confusion_matrix, metrics_from_confusion

        target = torch.arange(12)
        predicted = target.clone()
        predicted[1] = 0
        confusion = confusion_matrix(target, predicted, 12)
        self.assertEqual(confusion[1, 0].item(), 1)
        metrics = metrics_from_confusion(confusion, "marine")
        self.assertAlmostEqual(metrics["miou12"], 10.5 / 12)
        self.assertAlmostEqual(metrics["debris_miou10"], 0.9)
        self.assertEqual(metrics["per_class_iou"][11], 1)
        classification = torch.tensor([[2, 1], [1, 0]])
        self.assertAlmostEqual(metrics_from_confusion(classification, "nksid")["macro_f1"], 1 / 3)

    def test_dense_logits_use_original_grid_and_preserve_mask_labels(self):
        from sonar_encoder.training.readout import head_logits

        head = nn.Conv2d(384, 12, 1)
        features = torch.zeros(2, 384, 16, 16)
        targets = torch.tensor([[[0, 11, 1], [3, 4, 5]]] * 2, dtype=torch.uint8)
        before = targets.clone()
        logits = head_logits(head, features, targets, "marine")
        self.assertEqual(tuple(logits.shape), (2, 12, 2, 3))
        torch.testing.assert_close(targets, before)
        nn.functional.cross_entropy(logits, targets.long()).backward()
        self.assertIsNotNone(head.weight.grad)

    def test_feature_extraction_is_fp32_unmasked_and_leaves_encoder_unchanged(self):
        from sonar_encoder.training import artifacts, readout
        from sonar_encoder.training.readout import encode_rows, state_digest

        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(384))

            def forward(self, images):
                average = images.mean((1, 2, 3))[:, None] * self.scale[None]
                return {"global_features": average, "patch_tokens": average[:, None].expand(-1, 256, -1)}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.fromarray(np.full((16, 8), 255, dtype=np.uint8)).save(path)
            rows = [{"row_id": 7, "image": str(path), "classID": 3}]
            encoder = TinyEncoder().eval().requires_grad_(False)
            before = state_digest(encoder.state_dict())
            features = encode_rows(encoder, rows, {"mean": 0.0, "std": 1.0}, "nksid", "cpu", 2)
            self.assertEqual(features.dtype, torch.float32)
            self.assertEqual(tuple(features.shape), (1, 384))
            self.assertAlmostEqual(features[0, 0].item(), np.log(2), places=6)
            self.assertFalse(features.requires_grad)
            self.assertEqual(state_digest(encoder.state_dict()), before)
            self.assertIsNone(encoder.scale.grad)

            prepared = Path(temporary) / "prepared"
            prepared.mkdir()
            manifest = {"task": "nksid", "protocol_sha256": "fixture", "train": rows,
                        "validation": rows, "class_names": [str(index) for index in range(8)]}
            (prepared / "manifest.json").write_text(json.dumps(manifest))
            normalization = {"source_manifest_sha256": hashlib.sha256(
                (prepared / "manifest.json").read_bytes()).hexdigest(), "mean": 0.0, "std": 1.0}
            (prepared / "normalization.json").write_text(json.dumps(normalization))
            checkpoint = Path(temporary) / "encoder.pt"
            checkpoint.write_bytes(b"fixture")
            output = Path(temporary) / "cache"
            with (mock.patch.object(artifacts, "RUN_ROOT", Path(temporary)),
                  mock.patch.object(artifacts, "ARTIFACT_ROOT", Path(temporary) / "training_artifacts"),
                  mock.patch.object(readout, "checkpoint_encoder", return_value=(encoder, "fixture", 0)),
                  redirect_stdout(io.StringIO())):
                result = readout.extract_features(prepared, checkpoint, output, "cpu", 2)
            self.assertTrue(output.is_symlink())
            self.assertEqual(output.resolve(), Path(temporary) / "training_artifacts/cache")
            self.assertEqual(len(result["provenance"]["artifacts_source_sha256"]), 64)

    def test_fixed_head_budget_selects_earliest_tie_without_changing_encoder(self):
        from sonar_encoder.training import artifacts
        from sonar_encoder.training.readout import state_digest, train_head

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "encoder.pt"
            state = {"weight": torch.tensor([1.0, 2.0])}
            torch.save({"student": state}, checkpoint)
            for task in ("nksid", "marine"):
                with self.subTest(task=task):
                    cache_dir = root / task / "cache"
                    cache_dir.mkdir(parents=True)
                    provenance = {"task": task, "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "encoder_state_sha256": state_digest(state)}
                    if task == "nksid":
                        features = torch.zeros(8, 384)
                        targets = torch.arange(8)
                    else:
                        features = torch.zeros(2, 384, 16, 16)
                        targets = torch.arange(12, dtype=torch.uint8).reshape(1, 4, 3).repeat(2, 1, 1)
                    files = {}
                    for split in ("train", "validation"):
                        path = cache_dir / f"{split}.pt"
                        torch.save({"features": features, "targets": targets, "row_ids": list(range(len(features))), "provenance": provenance}, path)
                        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    (cache_dir / "cache_manifest.json").write_text(json.dumps({"provenance": provenance, "cache_files": files}))
                    output = root / task / "head"
                    with (mock.patch.object(artifacts, "RUN_ROOT", root),
                          mock.patch.object(artifacts, "ARTIFACT_ROOT", root / "training_artifacts"),
                          redirect_stdout(io.StringIO())):
                        report = train_head(cache_dir, output, "cpu")
                    self.assertTrue(output.is_symlink())
                    self.assertEqual(output.resolve(), root / "training_artifacts" / task / "head")
                    self.assertEqual(report["optimizer_updates"], 100)
                    self.assertEqual(report["selected_epoch"], 1)
                    self.assertEqual(report["encoder_hash_before"], report["encoder_hash_after"])
                    self.assertFalse(report["final_test_evaluated"])
                    self.assertEqual(len((output / "metrics.jsonl").read_text().splitlines()), 100)
                    self.assertEqual(len(report["initial_head_sha256"]), 64)
                    self.assertTrue((output / "best.pt").exists())
                    best = torch.load(output / "best.pt", weights_only=True)["head"]
                    self.assertNotEqual(state_digest(best), report["initial_head_sha256"])
                    self.assertFalse((output / "initial.pt").exists())
                    self.assertFalse((output / "last.pt").exists())


if __name__ == "__main__":
    unittest.main()
