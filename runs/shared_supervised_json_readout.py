"""JSON metadata repair and the one certified T0/FSOD epoch5 continuation."""

import hashlib
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from runs import shared_supervised_readout as original
from runs.shared_supervised_readout import evaluate, load_targets, make_head, state_digest, write_json


_original_canonical_evaluate = original.canonical_evaluate


def plain_image_ids(score):
    for category in map(str, range(1, 11)):
        ordering = score["per_class"][category]["ordering"]
        ordering["image_ids"] = [int(i) for i in ordering["image_ids"]]
    return score


def canonical_evaluate(coco_gt, predictions):
    return plain_image_ids(_original_canonical_evaluate(coco_gt, predictions))


def install_serialization_fix():
    original.canonical_evaluate = canonical_evaluate


make_fsod_head = original.make_fsod_head
targetextract = original.targetextract
score_readout = original.score_readout
score_and_summarize = original.score_and_summarize


def resume_t0_fsod():
    from runs import shared_supervised_json_resume as core

    state, task = "T0", "fsod"
    from runs.cfc_frozen_detector import scale_boxes

    core.configure_runtime(task)
    protocol = core.load_protocol()
    target = protocol["targets"][task]
    folder = core.RUN / "readouts" / state / task
    extracted = json.loads((folder / "extraction.json").read_text())
    protocol_sha = core.digest(core.RUN / "protocol.json")
    addendum = core.effective_addendum()
    assert core.digest(folder / "extraction.json") == addendum["reused_T0_FSOD_extraction_sha256"]
    assert (extracted["state"], extracted["task"]) == (state, task)
    assert extracted["protocol_sha256"] == protocol_sha
    assert extracted["encoder_before_sha256"] == extracted["encoder_after_sha256"]
    rows, arrays = target["rows"], {}
    for split in ("train", "validation"):
        cache = extracted["caches"][split]
        path = Path(cache["path"])
        reused_link = folder / f"{split}.npy"
        assert reused_link.is_symlink() and reused_link.resolve() == path.resolve()
        assert cache["row_ids"] == [r["image_id" if task == "fsod" else "row_id"] for r in rows[split]]
        arrays[split] = np.load(path, mmap_mode="r", allow_pickle=False)
        assert list(arrays[split].shape) == cache["shape"] and arrays[split].dtype == np.float32
    if task == "fsod":
        coco = json.loads(Path(target["validation_coco"]).read_text())
        assert core.digest(Path(target["validation_coco"])) == protocol["inputs_sha256"][target["validation_coco"]]
        targets = []
        for row in rows["train"]:
            annotations = row["annotations"]
            boxes = torch.tensor([a["bbox"] for a in annotations], dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            targets.append({"boxes": scale_boxes(boxes, (row["height"], row["width"]), (224, 224)),
                            "labels": torch.tensor([a["category_id"] for a in annotations], dtype=torch.int64)})
        epochs, batch_size, updates_required = 60, 8, 17700
    else:
        coco = None
        targets = load_targets(rows["train"], "nksid")
        epochs, batch_size, updates_required = 100, 64, 2700
    torch.manual_seed(0)
    head = make_head(task)
    initial = state_digest(head.state_dict())
    assert initial == target["initial_head_sha256"]
    head.to("cuda:0")
    optimizer = (torch.optim.SGD(head.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
                 if task == "fsod" else torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001,
                                                         betas=(.9, .999), eps=1e-8))
    generator = torch.Generator().manual_seed(0)
    order_hash = hashlib.sha256()
    metric = "AP50" if task == "fsod" else "macro_f1"
    extension = ".npy" if task == "fsod" else ".pt"
    best_score, best_epoch, updates = -1., 0, 0
    begun = time.monotonic()
    selector = {"best_score": -1., "best_epoch": 0}
    stream = {"epoch": 0, "cursor": 0, "order": torch.empty(0, dtype=torch.int64), "updates": 0}

    certificate = addendum["root_certificate"]
    assert addendum["resource_contract_sha256"] == "c31b7c522c015bc6fa7a38ed171a671540cda8e06230994c12ab091184f29a3e"
    assert addendum["scientific_protocol_sha256"] == protocol_sha == "3f15f8bd4212ddd985a8843b16e241acfb51a4f8756492026b942af08b54c590"
    assert certificate["last_head_sha256"] == "19ffa06ca3e464949a12088507c1c6d871d6af25a2235e4d183846c957d66dee"
    assert certificate["head_state_sha256"] == "2245443358cc177e004355e28853794cd582cd2b8e9ddf7b4e0d185ecf5e2f86"
    assert (certificate["epoch"], certificate["cursor"], certificate["completed_updates"], certificate["selected_epoch"]) == (5, 2357, 1475, 5)
    copies = addendum["copied_head_files"]
    for name in ("last", "best", "best_predictions", "metrics"):
        assert core.digest(Path(copies[name]["path"])) == copies[name]["sha256"]
    assert Path(copies["last"]["path"]) == folder / "last_head.pt"
    assert Path(copies["best"]["path"]) == folder / "best_head.pt"
    assert Path(copies["best_predictions"]["path"]) == folder / "best_validation.npy"
    assert Path(copies["metrics"]["path"]) == folder / "metrics.jsonl"
    saved = torch.load(folder / "last_head.pt", map_location="cpu", weights_only=False)
    assert saved["failure_not_safe_for_automatic_resume"] is True
    assert saved["protocol_sha256"] == protocol_sha
    assert saved["head_state_sha256"] == certificate["head_state_sha256"] == state_digest(saved["head"])
    assert (saved["stream"]["epoch"], saved["stream"]["cursor"], saved["stream"]["updates"]) == (5, 2357, 1475)
    assert saved["selector"]["best_epoch"] == 5
    head.load_state_dict(saved["head"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    assert state_digest(head.state_dict()) == certificate["head_state_sha256"]
    restored_optimizer = optimizer.state_dict()
    assert restored_optimizer["param_groups"] == saved["optimizer"]["param_groups"]
    assert set(restored_optimizer["state"]) == set(saved["optimizer"]["state"])
    assert len(restored_optimizer["state"]) == 14
    for parameter_id in saved["optimizer"]["state"]:
        assert torch.equal(restored_optimizer["state"][parameter_id]["momentum_buffer"].cpu(),
                           saved["optimizer"]["state"][parameter_id]["momentum_buffer"])
    selected = torch.load(folder / "best_head.pt", map_location="cpu", weights_only=True)
    assert selected["epoch"] == 5 and selected["protocol_sha256"] == protocol_sha
    assert state_digest(selected["head"]) == selected["head_state_sha256"] == certificate["head_state_sha256"]
    assert selected["validation"] == saved["selector"]["best_metrics"]
    for _ in range(5):
        fifth_order = torch.randperm(2357, generator=generator)
        order_hash.update(fifth_order.numpy().tobytes())
    assert order_hash.hexdigest() == certificate["private_order_prefix_sha256"] == "2602f7c0a4752af2eb14301ad4967bb83a8b3ecead7333c84f9e6f54bdef5364"
    assert torch.equal(fifth_order, saved["stream"]["order"])
    assert torch.equal(generator.get_state(), saved["private_rng"])
    stream, selector = saved["stream"], saved["selector"]
    updates, best_score, best_epoch = stream["updates"], selector["best_score"], selector["best_epoch"]
    best_metrics, best_head_state = selector["best_metrics"], selector["best_head_state_sha256"]
    assert best_score == best_metrics["AP50"] and best_head_state == certificate["head_state_sha256"]
    best_path = folder / "best_validation.npy"
    previous = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert [entry["epoch"] for entry in previous] == [1, 2, 3, 4, 5]
    assert previous[-1]["updates"] == 1475 and previous[-1]["LR"] == optimizer.param_groups[0]["lr"] == .005
    assert previous[-1]["train_loss"] == stream["loss_sum"] / 2357
    assert previous[-1]["validation"] == best_metrics
    generator.set_state(saved["private_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"], 0)
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    assert torch.equal(generator.get_state(), saved["private_rng"])
    assert torch.equal(torch.get_rng_state(), saved["torch_rng"])
    assert torch.equal(torch.cuda.get_rng_state(0), saved["cuda_rng"])
    assert random.getstate() == saved["python_rng"]
    numpy_rng = np.random.get_state()
    assert numpy_rng[0] == saved["numpy_rng"][0]
    assert np.array_equal(numpy_rng[1], saved["numpy_rng"][1]) and numpy_rng[2:] == saved["numpy_rng"][2:]
    del saved, selected, restored_optimizer
    imported_certificate = {"state": state, "task": task, "phase": "certified_resume_before_first_forward",
        "epoch": stream["epoch"], "cursor": stream["cursor"], "completed_updates": updates,
        "next_epoch": 6, "next_update": 1476, "head_state_sha256": best_head_state,
        "reconstructed_order_prefix_sha256": order_hash.hexdigest(),
        "origin_last_head_sha256": certificate["last_head_sha256"], "protocol_sha256": protocol_sha,
        "root_contract_sha256": addendum["resource_contract_sha256"], "all_rng_match": True,
        "extraction_sha256": addendum["reused_T0_FSOD_extraction_sha256"], "cache_bytes_rehashed": False,
        "cache_binding": "Existing extraction receipt, file-level readonly link identity, NPY header and row IDs"}
    write_json(folder / "imported_epoch5_certificate.json", imported_certificate)
    print(json.dumps(imported_certificate), flush=True)

    def save_boundary(failure):
        payload = {"head": head.state_dict(), "optimizer": optimizer.state_dict(),
            "stream": stream, "selector": selector, "private_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(0),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "head_state_sha256": state_digest(head.state_dict()), "protocol_sha256": protocol_sha,
            "failure_not_safe_for_automatic_resume": failure is not None}
        if failure is not None:
            payload["failure"] = failure
            payload["gradients"] = {name: parameter.grad for name, parameter in head.named_parameters()}
        core.atomic_torch(folder / "last_head.pt", payload)

    def interrupted(signum, frame):
        raise InterruptedError("Target head received SIGTERM; stop without automatic resume")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    event = {"stage": "certified_epoch5_resume", "epoch": 5}
    try:
        with (folder / "metrics.jsonl").open("a") as log:
            for epoch in range(6, epochs + 1):
                event = {"stage": "epoch_start", "epoch": epoch}
                head.train()
                order = torch.randperm(len(rows["train"]), generator=generator)
                order_hash.update(order.numpy().tobytes())
                stream.update(epoch=epoch, cursor=0, order=order,
                              epoch_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest())
                lr = (.005 if epoch <= 40 else (.0005 if epoch <= 55 else .00005)) if task == "fsod" else .001
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss_sum = 0.
                for indices in order.split(batch_size):
                    event = {"stage": "batch_forward", "epoch": epoch, "indices": indices.tolist(), "update_committed": False}
                    features = torch.from_numpy(np.array(arrays["train"][indices.numpy()], copy=True)).to("cuda:0")
                    optimizer.zero_grad(set_to_none=True)
                    if task == "fsod":
                        current_targets = [{key: value.to("cuda:0") for key, value in targets[i].items()} for i in indices.tolist()]
                        loss = sum(head(features, current_targets).values())
                    else:
                        loss = F.cross_entropy(head(features), targets[indices].to("cuda:0"))
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Nonfinite target head loss")
                    event["stage"] = "backward"
                    loss.backward()
                    if not all(torch.isfinite(p.grad).all().item() for p in head.parameters()):
                        raise FloatingPointError("Nonfinite target head gradient")
                    event["stage"] = "optimizer_step"
                    optimizer.step()
                    if not all(torch.isfinite(p).all().item() for p in head.parameters()):
                        raise FloatingPointError("Nonfinite target head parameter")
                    updates += 1
                    loss_sum += loss.item() * len(indices)
                    stream.update(cursor=stream["cursor"] + len(indices), updates=updates, loss_sum=loss_sum)
                    event.update(stage="accepted_update", update_committed=True)
                    if updates == 1476:
                        print(json.dumps({"state": state, "task": task, "phase": "first_continued_update",
                                          "epoch": epoch, "updates": updates, "parent_updates_not_replayed": 1475,
                                          "image_IDs": [rows["train"][i]["image_id"] for i in indices.tolist()]}), flush=True)
                    core.resources()
                    if updates % 100 == 0:
                        print(json.dumps({"state": state, "task": task, "phase": "head", "epoch": epoch,
                                          "updates": updates, "seconds": time.monotonic() - begun}), flush=True)
                entry = {"epoch": epoch, "updates": updates, "LR": lr, "train_loss": loss_sum / len(rows["train"])}
                if task == "nksid" or epoch % 5 == 0:
                    event = {"stage": "validation", "epoch": epoch}
                    metrics, predictions = evaluate(head, arrays["validation"], rows["validation"], task, coco)
                    entry["validation"] = metrics
                    if metrics[metric] > best_score:
                        best_score, best_epoch, best_metrics = metrics[metric], epoch, metrics
                        best_head_state = state_digest(head.state_dict())
                        selector.update(best_score=best_score, best_epoch=best_epoch, best_metrics=best_metrics,
                                        best_head_state_sha256=best_head_state)
                        event["stage"] = "best_artifact_write"
                        core.atomic_torch(folder / "best_head.pt", {"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                            "epoch": epoch, "validation": metrics, "head_state_sha256": best_head_state,
                            "initial_head_sha256": initial, "encoder_state_sha256": extracted["encoder_after_sha256"],
                            "protocol_sha256": protocol_sha})
                        best_path = folder / ("best_validation" + extension)
                        if task == "fsod":
                            core.atomic_predictions(best_path, predictions)
                        else:
                            core.atomic_torch(best_path, {**predictions, "epoch": epoch, "protocol_sha256": protocol_sha})
                    if epoch == epochs:
                        event["stage"] = "terminal_evaluation_output_write"
                        terminal_metrics, terminal_head_state = metrics, state_digest(head.state_dict())
                        terminal_path = best_path if best_epoch == epochs else folder / ("terminal_validation" + extension)
                        if best_epoch != epochs:
                            if task == "fsod":
                                core.atomic_predictions(terminal_path, predictions)
                            else:
                                core.atomic_torch(terminal_path, {**predictions, "epoch": epoch, "protocol_sha256": protocol_sha})
                    del predictions
                log.write(json.dumps(entry, allow_nan=False) + "\n")
                log.flush()
                event = {"stage": "epoch_boundary_save", "epoch": epoch}
                save_boundary(None)
        event = {"stage": "final_report", "epoch": epochs}
        assert updates == updates_required
        write_json(folder / "report.json", {"state": state, "task": task, "epochs": epochs, "updates": updates,
            "initial_head_sha256": initial, "private_order_sha256": order_hash.hexdigest(),
            "best_epoch": best_epoch, "best": best_metrics, "terminal": terminal_metrics,
            "best_head_state_sha256": best_head_state, "terminal_head_state_sha256": terminal_head_state,
            "best_head_file_sha256": core.digest(folder / "best_head.pt"),
            "best_prediction": str(best_path), "terminal_prediction": str(terminal_path),
            "best_prediction_sha256": core.digest(best_path), "terminal_prediction_sha256": core.digest(terminal_path),
            "logical_endpoints": 2, "unique_prediction_artifacts": 1 if terminal_path == best_path else 2,
            "terminal_weight_retained": best_epoch == epochs, "protocol_sha256": protocol_sha,
            "resume_origin_last_head_sha256": certificate["last_head_sha256"],
            "resume_head_state_sha256": certificate["head_state_sha256"], "new_physical_updates": updates - 1475,
            "reconstructed_order_prefix_sha256": certificate["private_order_prefix_sha256"],
            "prefix_order_provenance": "CPU seed0 five-permutation reconstruction, not full observed runtime batch trace",
            "resources": core.resources()})
    except Exception as error:
        failure = {"state": state, "task": task, "event": event,
                   "exception_type": type(error).__name__, "message": str(error),
                   "actual_epoch": stream["epoch"], "actual_cursor": stream["cursor"],
                   "completed_updates": stream["updates"],
                   "failure_not_safe_for_automatic_resume": True, "checkpoint_saved": False,
                   "protocol_sha256": protocol_sha}
        try:
            failure["checkpoint_saved"] = True
            save_boundary(failure)
            failure["checkpoint_sha256"] = core.digest(folder / "last_head.pt")
        except Exception as save_error:
            failure["checkpoint_saved"] = False
            failure["save_error"] = {"exception_type": type(save_error).__name__, "message": str(save_error)}
        try:
            write_json(folder / "failure.json", failure)
        except Exception as record_error:
            print(json.dumps({**failure, "failure_record_error": str(record_error)}), flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)



def fit_head(state, task):
    install_serialization_fix()
    if (state, task) == ("T0", "fsod"):
        return resume_t0_fsod()
    return original.fit_head(state, task)
