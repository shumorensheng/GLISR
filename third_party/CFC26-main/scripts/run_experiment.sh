#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=""

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_VENV=""  # Path to YOLOv5
INFER_VENV=""  # Path to FishEye (inference pipeline) virtual env
YOLO_REPO_PATH=""  # Path to cloned YOLOv5 repo
WEIGHTS=""
EXPERIMENT_NAME=""
FRAMES_ROOT=""  # Path to 3-channel frames
OUTPUT_DIR=""
GT_MOT_PATH=""
DATA_YAML="$OUTPUT_DIR/$EXPERIMENT_NAME/dataset.yaml"  # YOLO dataset yaml created from coco_to_yolo.py

EPOCHS=2
IMG_SIZE=896
BATCH_SIZE=16
WORKERS=6

SELECTION_METRIC="mAP50"  # For YOLOv26 only
MODEL_OVERRIDE="platform.model.weights"  # For FishEye
OUTPUT_OVERRIDE="output_dir"  # For FishEye
# ──────────────────────────────────────────────────────────────────────────────

# ── Training ──────────────────────────────────────────────────────────────────
echo "=== Activating training venv ==="
source "$TRAIN_VENV/bin/activate"

echo "=== Starting training: $EXPERIMENT_NAME ==="

python "$REPO_ROOT/train_yolov5_multiple_val_sets.py" \
    --yolov5_path "$YOLO_REPO_PATH" \
    --weights     "$WEIGHTS" \
    --data        "$DATA_YAML" \
    --project     "$OUTPUT_DIR" \
    --name        "$EXPERIMENT_NAME" \
    --epochs      "$EPOCHS" \
    --imgsz       "$IMG_SIZE" \
    --batch       "$BATCH_SIZE" \
    --workers     "$WORKERS" \
    --device      0 \
    --val_by_river


CHECKPOINT="$OUTPUT_DIR/$EXPERIMENT_NAME/weights/best.pt"
echo ""
echo "=== Training complete ==="
echo "Results CSV:     $OUTPUT_DIR/$EXPERIMENT_NAME/results.csv"
echo "Weights:         $OUTPUT_DIR/$EXPERIMENT_NAME/weights/"
echo "Val metrics CSV: $OUTPUT_DIR/$EXPERIMENT_NAME/val_metrics.csv"
echo "Best checkpoint: $CHECKPOINT"

deactivate

# ── Inference ─────────────────────────────────────────────────────────────────
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT" >&2
    exit 1
fi

INFER_OUT="$OUTPUT_DIR/$EXPERIMENT_NAME/inference"
INFER_SCRIPT="$REPO_ROOT/infer.py"

echo ""
echo "=== Activating inference venv ==="
source "$INFER_VENV/bin/activate"

echo "=== Running fisheye inference ==="
echo "    Checkpoint: $CHECKPOINT"
echo "    Output:     $INFER_OUT"

python "$INFER_SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --output     "$INFER_OUT"

echo ""
echo "=== Inference complete ==="
echo "Results: $INFER_OUT"

# ── Count Eval ─────────────────────────────────────────────────────────────────

echo ""
echo "=== Running Count Eval ==="
COUNT_EVAL_OUT="$OUTPUT_DIR/$EXPERIMENT_NAME/count_eval"
python "$REPO_ROOT/CFC26/scripts/eval_count_infer.py" \
    --infer_dir  "$INFER_OUT" \
    --gt_dir     "$GT_MOT_PATH" \
    --frames_dir "$FRAMES_ROOT" \
    --output_dir "$COUNT_EVAL_OUT" \

echo ""
echo "=== Count eval complete ==="
