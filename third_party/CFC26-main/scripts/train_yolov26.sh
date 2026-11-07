#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=""

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_VENV=""  # Path to YOLOv5 or Ultralytics virtual env

DATA_YAML=""
WEIGHTS=""
EPOCHS=2
IMG_SIZE=896
BATCH_SIZE=16
WORKERS=6
SELECTION_METRIC="mAP50"

# ──────────────────────────────────────────────────────────────────────────────

echo "=== Activating venv ==="
source "$TRAIN_VENV/bin/activate"

echo "=== Starting training: $EXPERIMENT_NAME ==="
python "$REPO_ROOT/train_yolov26.py" \
    --model    "$WEIGHTS" \
    --data     "$DATA_YAML" \
    --project  "$OUTPUT_DIR" \
    --name     "$EXPERIMENT_NAME" \
    --epochs   "$EPOCHS" \
    --imgsz    "$IMG_SIZE" \
    --batch    "$BATCH_SIZE" \
    --workers  "$WORKERS" \
    --selection_metric "$SELECTION_METRIC" \
    --val_by_river

echo ""
echo "=== Training complete ==="
echo "Results CSV:     $OUTPUT_DIR/$EXPERIMENT_NAME/results.csv"
echo "Val metrics CSV: $OUTPUT_DIR/$EXPERIMENT_NAME/val_metrics.csv"
echo "Weights:         $OUTPUT_DIR/$EXPERIMENT_NAME/weights/"