import math
import json
from pathlib import Path
import tempfile
import unittest

import torch


def role_fixture():
    seen = {"C00", "C01", "C03", "C05", "C07", "C12"}
    frames = []
    for pose, split in (("v0", "validation"), ("v1", "validation"), ("train", "train"), ("test", "test")):
        for config in (f"C{index:02d}" for index in range(14)):
            if split == "test":
                usage = "test_seen" if config in seen else "test_reference" if config == "C11" else "test_unseen_combination" if config in {"C09", "C10"} else "test_unseen_single"
            else:
                usage = ("ssl_validation" if split == "validation" else "ssl_train") if config in seen else "unused"
            frames.append({"sample_id": f"{pose}/{config}", "pose_id": pose, "config_id": config, "split": split, "usage": usage})
    exposed = ["C02", "C04", "C06", "C10"]
    role = {"source_manifest_sha256": "set-by-file-test",
            "development_sample_ids": [row["sample_id"] for row in frames
                                       if row["split"] == "validation" and row["config_id"] in seen | set(exposed)],
            "development_exposed_configs": exposed, "development_unseen_configs": ["C08", "C09", "C13"]}
    return {"frames": frames}, role


class ConfigEvalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_signed_physical_axes_reproduce_affine_field_and_complete_valid_stencil(self):
        from sonar_encoder.training.config_eval import bilinear_sample

        ranges = torch.tensor([1., 2., 4.])
        bearings = torch.tensor([.5, .1, -.5])
        field = (2 * ranges[:, None] + 6 * bearings[None] + 5)[..., None]
        points = torch.tensor([[1.5, .3], [3., -.2], [4., -.5], [.9, .3]])
        valid = torch.ones(3, 3, dtype=torch.bool)
        values, usable = bilinear_sample(field, ranges, bearings, valid, points)
        torch.testing.assert_close(values[:3, 0], torch.tensor([9.8, 9.8, 10.]))
        self.assertEqual(usable.tolist(), [True, True, True, False])
        valid[1, 1] = False
        _, usable = bilinear_sample(field, ranges, bearings, valid, points[2:3])
        self.assertFalse(usable.item())  # Zero-weight corner still belongs to the complete stencil.

    def test_token_coordinates_follow_resize_centers_and_invalid_native_footprint(self):
        from sonar_encoder.training.config_eval import token_geometry

        sample = {"range_m": torch.arange(32).float() + .5,
                  "bearing_rad": 1 - .01 * torch.arange(64).float(),
                  "valid": torch.ones(32, 64, dtype=torch.bool)}
        ranges, bearings, valid = token_geometry(sample)
        torch.testing.assert_close(ranges, torch.arange(1, 32, 2).double())
        torch.testing.assert_close(bearings, 1 - .01 * (torch.arange(16).double() * 4 + 1.5))
        self.assertTrue(valid.all())
        sample["valid"][0, 0] = False
        self.assertFalse(token_geometry(sample)[2][0, 0])

    def test_retrieval_ranks_same_pose_and_stably_breaks_ties(self):
        from sonar_encoder.training.config_eval import retrieval_metrics

        gallery = torch.eye(6)
        query = gallery.clone()
        query[0] = gallery[1]
        result = retrieval_metrics(query, gallery, list("abcdef"), list("abcdef"))
        self.assertEqual([row["rank"] for row in result], [2, 1, 1, 1, 1, 1])
        self.assertEqual(sum(row["top1"] for row in result), 5)
        self.assertEqual(sum(row["top5"] for row in result), 6)
        self.assertAlmostEqual(result[0]["reciprocal_rank"], .5)

    def test_dense_identity_respects_descending_bearing_and_physical_distance(self):
        from sonar_encoder.training.config_eval import dense_metrics

        ranges = torch.linspace(1, 5, 16, dtype=torch.double)
        bearings = torch.linspace(.5, -.5, 16, dtype=torch.double)
        tokens = torch.eye(256).reshape(16, 16, 256)
        valid = torch.ones(16, 16, dtype=torch.bool)
        geometry = (ranges, bearings, valid)
        native = {"range_m": ranges, "bearing_rad": bearings, "valid": valid}
        result = dense_metrics(tokens, geometry, native, tokens.flip(1),
                               (ranges, bearings.flip(0), valid),
                               {**native, "bearing_rad": bearings.flip(0)})
        self.assertEqual(result["valid_points"], 256)
        self.assertAlmostEqual(result["cosine"], 1.)
        self.assertEqual(result["bidirectional_error_m"], 0.)
        shifted = dense_metrics(tokens, geometry, native, tokens.roll(1, dims=1), geometry, native)
        expected_chord = 2 * 3 * (15 * math.sin(1 / 30) + math.sin(.5)) / 16
        self.assertAlmostEqual(shifted["forward_error_m"], expected_chord)
        self.assertAlmostEqual(shifted["reverse_error_m"], expected_chord)

    def test_validation_selection_cannot_include_unused_config_or_test_pose(self):
        from sonar_encoder.training.config_eval import select_frames

        frames = [{"pose_id": pose, "config_id": config, "split": split, "usage": usage}
                  for pose, split, usage in [("z", "validation", "ssl_validation"),
                                             ("a", "test", "test_seen")]
                  for config in ("C00", "C01")]
        manifest = {"frames": frames}
        selected = select_frames(manifest, "validation", ["C00", "C01"], 2)
        self.assertEqual({row["pose_id"] for row in selected}, {"z"})
        with self.assertRaises(ValueError):
            select_frames(manifest, "validation", ["C00", "C02"], None)

    def test_development_requires_exact_overlay_and_preserves_old_validation_and_test_members(self):
        from sonar_encoder.training.config_eval import select_frames

        manifest, role = role_fixture()
        selected = select_frames(manifest, "development", ["C00", "C02", "C10"], None, role)
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["pose_id"] for row in selected}, {"v0", "v1"})
        for forbidden in ("C08", "C09", "C13", "C11"):
            with self.subTest(config=forbidden), self.assertRaises(ValueError):
                select_frames(manifest, "development", ["C00", forbidden], None, role)
        for wrong_id in ("train/C02", "test/C02", "v0/C08"):
            bad_role = {**role, "development_sample_ids": role["development_sample_ids"][:-1] + [wrong_id]}
            with self.subTest(sample=wrong_id), self.assertRaisesRegex(ValueError, "development membership"):
                select_frames(manifest, "development", ["C00", "C02"], None, bad_role)
        with self.assertRaises(ValueError):
            select_frames(manifest, "validation", ["C00", "C02"], None, role)
        tested = select_frames(manifest, "test", ["C00", "C02", "C09", "C11"], None, role)
        self.assertEqual({row["pose_id"] for row in tested}, {"test"})
        self.assertEqual({row["config_id"] for row in tested}, {"C00", "C02", "C09", "C11"})

    def test_role_manifest_hash_is_checked_before_opening_encoder_or_native_files(self):
        from sonar_encoder.training.config_eval import run_evaluation
        from sonar_encoder.training.train import manifest_digest

        manifest, role = role_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "manifest.json"
            source.write_text(json.dumps(manifest))
            normalization = root / "normalization.json"
            normalization.write_text(json.dumps({"source_manifest_sha256": manifest_digest(manifest)}))
            role_path = root / "role.json"
            role_path.write_text(json.dumps(role))
            with self.assertRaisesRegex(ValueError, "role protocol.*manifest"):
                run_evaluation(root / "unopened-checkpoint.pt", source, root / "unopened-data", normalization,
                               root / "metrics-protocol.json", root / "output.json", "R010", "development",
                               ["C00", "C02"], "cpu", role_protocol_path=role_path)
            with self.assertRaisesRegex(ValueError, "role protocol"):
                run_evaluation(root / "unopened-checkpoint.pt", source, root / "unopened-data", normalization,
                               root / "metrics-protocol.json", root / "output.json", "R010", "test",
                               ["C00", "C02"], "cpu")

    def test_summaries_separate_development_exposure_and_exclude_reference_checks(self):
        from sonar_encoder.training.config_eval import summarize

        _, role = role_fixture()
        rows = [{"config_id": config,
                 "global": {"top1": value, "top5": value, "reciprocal_rank": value},
                 "dense": {"cosine": value, "forward_error_m": 1 - value, "reverse_error_m": 1 - value,
                           "bidirectional_error_m": 1 - value, "coverage": 1., "valid_points": 256}}
                for config, value in [("C00", 1.), ("C01", .4), ("C03", .6), ("C02", .3), ("C08", .8),
                                      ("C13", .6), ("C09", .2), ("C10", .7), ("C11", 0.)]]
        result = summarize(rows, [row["config_id"] for row in rows], "test", role)
        self.assertEqual(set(result["configuration_groups"]), {"seen", "development_exposed_single", "development_exposed_combination",
                                                               "development_unseen_single", "development_unseen_combination"})
        seen = result["configuration_groups"]["seen"]
        self.assertEqual(seen["configs"], ["C01", "C03"])
        self.assertEqual(seen["macro"]["top1"], .5)
        self.assertEqual(seen["worst"]["top1"], .4)
        self.assertEqual(seen["worst"]["bidirectional_error_m"], .6)
        self.assertEqual(result["configuration_groups"]["development_exposed_single"]["configs"], ["C02"])
        self.assertEqual(result["configuration_groups"]["development_exposed_combination"]["configs"], ["C10"])
        unseen = result["configuration_groups"]["development_unseen_single"]
        self.assertEqual(unseen["configs"], ["C08", "C13"])
        self.assertAlmostEqual(unseen["macro"]["mrr"], .7)
        self.assertEqual(result["configuration_groups"]["development_unseen_combination"]["configs"], ["C09"])
        self.assertIn("C11", result["per_config"])
        development = [row for row in rows if row["config_id"] not in {"C08", "C09", "C13", "C11"}]
        dev_summary = summarize(development, [row["config_id"] for row in development], "development", role)
        self.assertEqual(set(dev_summary["configuration_groups"]), {"seen", "development_single", "development_combination"})


if __name__ == "__main__":
    unittest.main()
