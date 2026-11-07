# CFC26: Building Evaluations for Deployment in Sonar-Based Fish Counting

CFC26 is a benchmark for deployable sonar-based fish counting, designed to better reflect the challenges of real-world deployment. 
It expands CFC22 from 1,567 to 4,363 annotated videos and from 515,933 to over 1M bounding boxes across eleven deployment locations spanning nine ecologically distinct river systems in Alaska, California, Washington, and British Columbia.

**Contributions:**
- **Revised evaluation protocol** with in-distribution (ID) and held-out-river OOD splits that directly simulate deployment conditions
- **Temporal data splits** — train/val/test windows are contiguous in time per location, mirroring how training and deployment data are collected in practice
- **Two novel counting metrics** — nMNE and nMANE — that detect directional bias in aggregate passage counts, complementing nMAE from [Caltech Fish Counting 2022](https://github.com/visipedia/caltech-fish-counting/)

## Dataset Access

The dataset is hosted on [HuggingFace](https://huggingface.co/datasets/perona-lab/cfc26).
A curated mini version (CFC26-Mini, 1 clip per split per location) is also [available](https://huggingface.co/datasets/perona-lab/cfc26-mini) for demonstration purposes.

## Train/Val/Test Splits

Splits are assigned **temporally per location** — clips are ordered chronologically and divided into contiguous train, validation, and test windows.
Pre-generated clip lists are also available in [HuggingFace](https://huggingface.co/datasets/perona-lab/cfc26/blob/main/splits.tar).

## Training Baselines

Baseline experiments use YOLOv5s/YOLOv5m and YOLOv26s/YOLOv26m. 

```bash
# YOLOv5m (train → inference → eval)
chmod +x scripts/run_experiment.sh
bash scripts/run_experiment.sh

# YOLO26 (train only)
chmod +x scripts/train_yolov26.sh
bash scripts/train_yolov26.sh
```

## Inference

Inference uses [Fisheye](https://github.com/fisheye-sonar/fisheye/tree/217-mvh-load-specified-frame-indices) on the `217-mvh-load-specified-frame-indices` branch and exports MOT files.

## Evaluation

```bash
python scripts/eval_count_infer.py \
  --infer_dir  /path/to/inference \  
  --gt_dir     /path/to/gt_mot \
  --frames_dir /path/to/3_channels \
  --output_dir /path/to/results
```

Outputs per location CSVs and a summary table with nMAE, nMNE, and nMANE.

## Useful Format Conversion

```bash
# COCO → MOT 
python converters/coco_to_mot.py --input_dir /path/to/coco --output_dir /path/to/gt_mot

# COCO → YOLO
python converters/coco_to_yolo.py \
  --coco_dir   /path/to/coco \
  --yolo_root  /path/to/yolo \
  --splits_dir /path/to/splits
```

## Citation

If you use CFC26 in your research, please cite our paper:

```bibtex
@inproceedings{cfc26neurips,
  title     = {CFC26: Building Evaluations for Deployment in Sonar-Based Fish Counting},
  author    = {Van Horn, Madison and Stathatos, Suzanne and Brodjian, Sevan and Kay, Justin and Van Brunt, Kai and Young, Erik and Beery, Sara and Perona, Pietro and Hobley, Michael},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2026},
  url       = {https://arxiv.org/abs/2610.22549}
}
```

## License

CFC26 is released under the Creative Commons Attribution 4.0 International [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/deed.en).