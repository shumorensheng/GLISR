# GLISR: Global–Local Interaction Sparse Re-encoding for Sonar Perception

Code for the paper on whole-image sonar encoding with a frozen DINOv2 backbone.
GLISR first encodes the full image with a frozen backbone, selects a small
budget of informative regions by attention, re-encodes them through a shared
sparse local branch, interacts with a global memory, and fuses gated residual
corrections back onto the global feature map:

```
F = B(G) + Scatter(g ⊙ R)
```

where `G` is the global feature, `R` the sparse local re-encoding, `g` a
learnable gate, and `B` the global branch output. Unselected positions keep
the global base features; task heads (detection / segmentation / classification
/ point localization / tracking) read the fused map.

## Method summary

1. **Global encoding** — the full 224×224 sonar image passes through frozen
   DINOv2 (all 12 layers), producing global features and last-layer attention.
2. **Attention-based region selection** — CLS→patch attention selects 64–256
   patches (80% attention-mass budget, τ-filtered in the main `Rtau` variant)
   on the 448 view.
3. **Sparse local re-encoding** — only the selected patches run through a
   shared DINOv2 (first 6 layers) with their original 448-view positions.
4. **Global memory interaction** — the local branch reads global features at
   its layers 3 and 6.
5. **Gated residual fusion** — local corrections are scattered back through a
   learnable gate onto the re-grid global base.

Ablation variants (`R256`, `A1`–`A4`, `B1`) and the source-training / downstream
protocol cover six source interfaces (FSOD, Marine, Xing detection, Xing
segmentation, LSOD, CFC22) with equal weighting, plus downstream tasks (UATD,
NKSID, HighResolution, DeeperSense Diver, CFC26 tracking).

## Repository layout

```
runs/                     experiment & method code (package name kept from the
                          research codebase for provenance)
  entry_snapshot.py           frozen core: GlobalMemoryGSR, SparseFusion
  glisr_supplement_variants.py  Rtau/R256/A1-A4/B1 variants + source training
  glisr_supplement_targets.py   downstream fine-tuning worker
  glisr_supplement_queue.py     source/downstream queue driver
  gsr_multidata_prepare.py      six-interface dataset preparation
  cfc26_matched_native_training.py  unified matched CFC26 training (12 configs)
  cfc26_matched_prediction_completion.py  resumable prediction/eval pipeline
  cfc26_unified_tracking.py     unified tracking/counting evaluation
  cfc26_tracking_interface.py   association, motion filter, counting (scipy only)
  cfc26_observation_quality.py  track-quality screening (train-GT supervised)
  featsharp_marine_matched.py / hsmla_marine_matched.py /
  gsr_same_source_dino.py / gsr_siglip2_fullft.py   comparison methods
sonar_encoder/            shared sonar data/IO layer
third_party/
  yolov5/                 YOLOv5 (teacher & detection head), see its LICENSE
  featsharp_adapted/      official FeatSharp (ICML 2025) with the sonar batch fix
  fisheye/                CFC26 authors' Fisheye branch (author-weights eval)
  CFC26-main/             CFC26 authors' counting/eval scripts
  TrackEval/              official HOTA/IDF1/MOTA implementation
  ByteTrack/ OC_SORT/     mature tracker backends used by the evaluation
  deps/                   two small pure-python deps (structlog, filterpy)
tests/                    unit tests (formula-level checks)
```

## Installation

```bash
conda create -n glisr python=3.10 -y && conda activate glisr
pip install torch torchvision numpy scipy pandas scikit-learn opencv-python \
            Pillow pycocotools transformers pyyaml
# optional mature trackers used by the CFC26 evaluation:
pip install git+https://github.com/ifzhang/ByteTrack.git   # provides yolox
pip install trackers                                        # provides trackers.ocsort_tracker
# structlog and filterpy are vendored under third_party/deps/
```

`third_party/yolov5`, `third_party/TrackEval`, `third_party/featsharp` and
`third_party/fisheye` are vendored; no separate clone is needed.

## Data

See [DATASETS.md](DATASETS.md). Datasets are not redistributed; the document
lists each public source, the splits used, and the directory layout expected
by the code.

## Configuration

All machine-specific paths are read from environment variables (defaults keep
the original research layout):

| Variable | Meaning |
|---|---|
| `GLISR_ARTIFACTS_ROOT` | writable root for run artifacts / prepared data (default: `./artifacts`) |
| `GLISR_REPO_ROOT` | repository root (where `runs/` lives; default: auto-detected) |
| `GLISR_OCEANGYM_DATA` | legacy simulation data root (only used by helper modules) |

Checkpoints are not stored in git. Download links for the trained source /
downstream weights will be provided with the paper release.

## Reproducing the main results

```bash
# 1. prepare the six source interfaces + downstream tasks
python -m runs.gsr_multidata_prepare --task FSOD     # repeat per task

# 2. source training (six interfaces, equal weight) per variant/seed
python -m runs.glisr_supplement_variants source --variant Rtau --seed 0

# 3. downstream fine-tuning
python -m runs.glisr_supplement_targets target --variant Rtau --task FSOD --seed 0

# 4. CFC26 matched tracking round (12 configs x 3 seeds)
python -m runs.cfc26_matched_native_training

# 5. unified tracking / counting evaluation
python -m runs.cfc26_unified_tracking
```

## Comparison methods and attribution

- **FeatSharp** — we use the official ICML 2025 implementation
  (`third_party/featsharp`), adapted to the matched sonar training recipe.
  This is an architecture adaptation under matched supervision, not a claim of
  reproducing the original paper's numbers.
- **HSMLA** — the authors of HSMLA have **not released source code or
  pretrained weights**; the paper alone does not fully specify the network
  (missing configuration details, two inconsistent gate definitions). We
  therefore reimplemented the method from the paper's formulas and report it
  as an **"HSMLA formula-based adaptation"** (`runs/hsmla_formula.py` +
  `runs/hsmla_marine_matched.py`). This is **not** an official reproduction,
  and results should not be cited as the original method's performance.
- **Same-source DINO / SigLIP 2** — frozen-backbone baselines trained with the
  identical matched recipe.

## License

This repository's own code is released under the [MIT License](LICENSE).
Vendored third-party code keeps its original license (see each directory).
