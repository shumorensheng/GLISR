# Datasets

None of the datasets are redistributed by this repository. Each entry lists the
public source, the split used in the paper, and notes. All dataset roots are
configured through the environment (see README); the code expects the layout
produced by `runs/gsr_multidata_prepare.py`.

## Source-training interfaces (six, equal weight 1/6)

| Interface | Public source | Splits used | Notes |
|---|---|---|---|
| FSOD | Forward-Looking Sonar Object Detection (RSOD), Li et al., AAAI 2026 — [author repository](https://github.com/LICZ9/RSOD) | official 2,357 / 785 / 787 | COCO format; `segmentation` fields empty (box annotations only) |
| Marine (segmentation) | Marine Debris Watertank — segmentation release (images resized by authors to 320×480) | 1,000 / 251 / 617 (local image-level split) | 12-class mIoU including background/wall |
| Xing detection | local prepared copy (see paper) | per prepared manifest | |
| Xing segmentation | local prepared copy (see paper) | per prepared manifest | |
| LSOD | local prepared copy (see paper) | per prepared manifest | |
| CFC22 | Caltech Fish Counting Dataset — [official page](https://data.caltech.edu/records/1y23m-j8r69), [paper](https://arxiv.org/abs/2207.09295) | Kenai train 162,198 / val 30,454 | single-class fish; used for source training only |

## Downstream tasks

| Task | Public source | Splits used | Metric |
|---|---|---|---|
| UATD (detection) | UATD dataset (sonar object detection, [paper](https://arxiv.org/abs/2306.05794)) | official 7,600 / 800 / 800 (effective 7,589 / 796 / 796 after duplicate-frame audit) | AP50 |
| NKSID (classification) | [NK-Sonar-Image-Dataset](https://github.com/Jorwnpay/NK-Sonar-Image-Dataset), Jiao et al., ESWA 2024 ([DOI](https://doi.org/10.1016/j.eswa.2024.123495)) | p0/k0 training population 1,680 / 417 internal split; 8-class closed-set | macro-F1 |
| HighResolution | local prepared copy (see paper) | per prepared manifest | mIoU |
| Diver (point localization) | DeeperSense Diver — [Zenodo 10220989](https://zenodo.org/records/10220989) | set1+set2 train 394 / set3 val 425 / set4 test 374 | pixel error |
| CFC26 (tracking) | Caltech Fish Counting 2026 — [HF dataset](https://huggingface.co/datasets/perona-lab/cfc26) | local 10,000-frame subset: Eel + Nanaimo, 7,000 train (26 clips) / 1,000 val (9) / 2,000 test (13); author-split order preserved, continuous clips kept | HOTA / IDF1 / MOTA (TrackEval), counting nMAE (line-crossing) |

## CFC26 subset construction

The tracking experiments use a 10,000-frame continuous-clip subset (not the
full release) built by: Eel site complete annotated 2,900 frames + Nanaimo
7,100 frames, following the author train/val/test split order, keeping
continuous sequences and original trajectory IDs (46,669 boxes total). The
subset manifest (frame index, COCO and MOT annotations) is produced by the
data preparation scripts; the original full release is not required.

## Usage boundaries honored by the code

- Head training reads only the task train split; validation selects epochs;
  test is evaluated once per endpoint and never feeds back into selection.
- Test splits (UATD Test1/Test2, NKSID outer folds, CFC22 Channel) are kept
  test-only.
- Prepared manifests record per-image identity so that splits are reproducible.
