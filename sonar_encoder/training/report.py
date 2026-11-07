"""Render the completed R010 run's measured curves and health diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_report(run_dir: Path):
    config = json.loads((run_dir / "resolved_config.json").read_text())
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    training = [row for row in records if row["kind"] == "train"]
    validation = [row for row in records if row["kind"] == "validation"]
    step = training[-1]["step"]
    initial = json.loads((run_dir / "diagnostics_initial.json").read_text())
    final = json.loads((run_dir / f"diagnostics_step{step}.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot([r["step"] for r in training], [r["loss"] for r in training], label="Train")
    axes[0, 0].plot([r["step"] for r in validation], [r["macro_loss"] for r in validation], "o-", label="Seen validation macro")
    axes[0, 0].set(xlabel="Successful optimizer updates", ylabel="Masked SmoothL1")
    axes[0, 0].legend()
    for key in validation[-1]["per_config"]:
        axes[0, 1].plot([r["step"] for r in validation], [r["per_config"][key] for r in validation], "o-", label=key)
    axes[0, 1].set(xlabel="Successful optimizer updates", ylabel="Seen validation loss")
    axes[0, 1].legend(ncol=2)
    for axis, measure in zip(axes[1], ("mean_std", "effective_rank")):
        axis.bar([0, 1], [initial[k][measure] for k in ("global", "patch")], width=.35, label="Step 0")
        axis.bar([.35, 1.35], [final[k][measure] for k in ("global", "patch")], width=.35, label=f"Step {step}")
        axis.set(xticks=[.175, 1.175], xticklabels=["Global", "Patch across images"], ylabel=measure)
        axis.legend()
    fig.savefig(run_dir / "training_diagnostics.png", dpi=150)
    plt.close(fig)
    steady = [row for row in training if row["step"] > 20]
    seconds = sum(row["seconds"] for row in steady)
    updates = sum(row["updates"] for row in steady)
    summary = {
        "step": step,
        "measurement_updates": updates,
        "mean_seconds_per_update": seconds / updates,
        "data_preparation_fraction": sum(row["data_seconds"] for row in steady) / seconds,
        "peak_allocated_bytes": max(row["peak_memory_bytes"] for row in training),
        "last_validation_seconds": validation[-1]["seconds"],
        "estimated_remaining_training_hours": (config["steps"] - step) * seconds / updates / 3600,
        "estimate_excludes": "future validation, diagnostics, checkpoint IO; shared-GPU load may change",
        "initial_masked_loss": initial["masked_loss"],
        "final_masked_loss": final["masked_loss"],
        "final_constant_loss": final["constant_loss"],
        "final_shuffled_target_loss": final["shuffled_target_loss"],
        "scope": "Engineering and seen-validation diagnostics; no unseen test or task-performance claim",
    }
    (run_dir / "measurement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(render_report(parser.parse_args().run_dir), indent=2))
