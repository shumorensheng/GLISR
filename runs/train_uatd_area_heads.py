"""One fixed pair of UATD heads per whole-image frozen encoding path."""

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonar_encoder.training.readout import file_digest, state_digest, write_json
from sonar_encoder.training.uatd_head import box_targets, dense_loss, evaluate_heads, global_targets


def main(run, arm, device):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    protocol = json.loads((run / "protocol.json").read_text())
    extracted = json.loads((run / "cache_manifest.json").read_text())
    assert extracted["protocol_sha256"] == file_digest(run / "protocol.json")
    assert {p: file_digest(Path(p)) for p in protocol["inputs_sha256"]} == protocol["inputs_sha256"]
    manifest = json.loads(Path(protocol["prepared_manifest"]).read_text())
    cache = {}
    for split in ("train", "validation"):
        path = run / f"{arm}_{split}.npy"
        assert file_digest(path) == extracted["cache_files_sha256"][path.name]
        cache[split] = {"features": torch.from_numpy(np.load(path, mmap_mode="c")).to(device),
                        "targets": [{"boxes": torch.tensor(row["boxes"], dtype=torch.float32), "labels": torch.tensor(row["labels"])} for row in manifest[split]],
                        "row_ids": [row["row_id"] for row in manifest[split]]}
    train, validation = cache["train"], cache["validation"]
    output = run / f"heads_{arm}"
    output.mkdir()
    settings = protocol["heads"]
    torch.manual_seed(settings["seed"])
    global_head = nn.Linear(384, 10).to(device)
    dense_head = nn.Conv2d(384, 50, 1).to(device)
    with torch.no_grad():
        dense_head.bias[:10].fill_(math.log(.01 / .99))
    initial = {"global": state_digest(global_head.state_dict()), "dense": state_digest(dense_head.state_dict())}
    train_boxes = tuple(value.to(device) for value in box_targets(train["targets"], "cpu"))
    train_presence = global_targets(train["targets"], "cpu").to(device)
    optimizer = torch.optim.AdamW(list(global_head.parameters()) + list(dense_head.parameters()), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(settings["seed"])
    best = {name: {"score": -1., "epoch": 0} for name in ("global", "dense")}
    provenance = {"arm": arm, "protocol_sha256": file_digest(run / "protocol.json"), "cache_manifest_sha256": file_digest(run / "cache_manifest.json"),
                  "encoder_state_sha256": extracted["model_state_sha256_after"], "encoder_parameters_in_optimizer": False}
    updates = 0
    with (output / "metrics.jsonl").open("w") as log:
        for epoch in range(1, settings["epochs"] + 1):
            global_head.train()
            dense_head.train()
            order = torch.randperm(len(train["features"]), generator=generator)
            loss_sum = 0.
            for indices in order.split(settings["batch_size"]):
                device_indices = indices.to(device)
                features = train["features"][device_indices]
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(global_head(features.mean((-2, -1))), train_presence[device_indices]) + dense_loss(dense_head(features), tuple(value[device_indices] for value in train_boxes))
                loss.backward()
                optimizer.step()
                updates += 1
                loss_sum += float(loss.detach()) * len(indices)
            row = {"epoch": epoch, "updates": updates, "train_loss": loss_sum / len(order)}
            if epoch % settings["evaluate_every_epochs"] == 0:
                metrics, predictions = evaluate_heads(global_head, dense_head, validation, device)
                row["validation"] = metrics
                for name, head, key in (("global", global_head, "map"), ("dense", dense_head, "map50")):
                    score = metrics[name][key]
                    if score > best[name]["score"]:
                        best[name] = {"score": score, "epoch": epoch}
                        torch.save({"head": head.state_dict(), "epoch": epoch, "metrics": metrics[name], "provenance": provenance}, output / f"best_{name}.pt")
                        torch.save({"row_ids": predictions["row_ids"], "predictions": predictions["global_scores" if name == "global" else "detections"]}, output / f"best_{name}_validation_predictions.pt")
                print(json.dumps({"arm": arm, "epoch": epoch, "global_map": metrics["global"]["map"], "dense_map50": metrics["dense"]["map50"]}), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            assert time.monotonic() - started < protocol["budgets"]["head_seconds_per_arm"]
    assert {p: file_digest(Path(p)) for p in protocol["inputs_sha256"]} == protocol["inputs_sha256"]
    resources = {"wall_seconds": time.monotonic() - started, "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 "max_gpu_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
                 "max_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device.startswith("cuda") else 0}
    assert resources["wall_seconds"] < protocol["budgets"]["head_seconds_per_arm"] and resources["max_rss_bytes"] < protocol["budgets"]["rss_bytes"] and resources["max_gpu_reserved_bytes"] < protocol["budgets"]["gpu_bytes"]
    report = {"status": "COMPLETE_FIXED_DEVELOPMENT_HEADS", "arm": arm, "epochs": settings["epochs"], "updates": updates,
              "selected": best, "epoch100_metrics": row["validation"], "initial_head_sha256": initial, "provenance": provenance, "resources": resources,
              "class_names": manifest["class_names"], "best_heads_sha256": {name: file_digest(output / f"best_{name}.pt") for name in best},
              "best_predictions_sha256": {name: file_digest(output / f"best_{name}_validation_predictions.pt") for name in best},
              "final_test_evaluated": False, "scope": "Validation-selected image-split development results; whole encoding path comparison, no single-factor or scene-independent claim."}
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "arm": arm, "selected": best, "resources": resources}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=("aa", "area"), required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    main(args.run, args.arm, args.device)
