"""
Convert COCO JSON to MOT txt files.

Output layout: {output_dir}/{location}/{clip_name}/gt.txt

# whole tree
python converters/coco_to_mot.py \
    --input_dir /path/to/coco_root \
    --output_dir /path/to/gt_mot

# single JSON (location derived from the JSON's parent directory name)
python converters/coco_to_mot.py \
    --coco_json /path/to/chemainus/clip.json \
    --output_dir /path/to/gt_mot
"""
import json
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser


def get_clip_id(filename):
    stem = Path(filename).stem
    return "_".join(stem.split("_")[:-1])


def get_frame_id(filename):
    """Return 0-indexed frame number encoded in the filename."""
    stem = Path(filename).stem
    return int(stem.split("_")[-1])


def convert_coco_to_mot(coco_json_path, output_dir, location=None):
    """
    Convert a COCO JSON to per-clip MOT txt files.

    Input COCO track_ids and frame indices are 0-indexed.
    Output MOT files are 1-indexed for both frame and track_id.

    Output: {output_dir}/{location}/{clip_name}/gt.txt
    MOT columns: frame_id, track_id, left, top, width, height, conf, -1, -1, -1
    """
    if location is None:
        location = Path(coco_json_path).parent.name

    with open(coco_json_path) as f:
        coco = json.load(f)

    image_id_to_info = {img['id']: img for img in coco['images']}

    clip_rows = defaultdict(list)

    for ann in coco['annotations']:
        img = image_id_to_info[ann['image_id']]
        filename = img['file_name']

        clip_name = get_clip_id(filename)
        coco_frame_idx = get_frame_id(filename)

        track_id = ann.get('track_id')
        if track_id is None:
            track_id = ann.get('attributes', {}).get('track_id')

        if track_id is None:
            print(f"Warning: no track_id for annotation id={ann['id']}, skipping")
            continue

        x, y, w, h = ann['bbox']
        conf = ann.get('score', 1.0)

        # MOT is 1-indexed, track_ids are already 1-indexed
        clip_rows[clip_name].append((coco_frame_idx + 1, track_id, x, y, w, h, conf))

    for clip_name, rows in clip_rows.items():
        rows.sort(key=lambda r: (r[0], r[1]))
        output_path = Path(output_dir) / location / clip_name
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / 'gt.txt', 'w') as f:
            for frame_id, track_id, x, y, w, h, conf in rows:
                f.write(f'{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n')

    print(f'  [{Path(coco_json_path).name}] wrote {len(clip_rows)} clips to {Path(output_dir) / location}')


def convert_directory_to_mot(input_dir, output_dir):
    """
    Convert a directory tree of COCO JSONs to MOT txt files.

    Expected input layout:  {input_dir}/{location}/*.json
    Output layout:          {output_dir}/{location}/{clip_name}/gt.txt
    """
    input_dir = Path(input_dir)

    json_files = [
        p for p in input_dir.glob('*/*.json')
        if not any(part.startswith('.') for part in p.parts)
    ]

    if not json_files:
        print(f"No JSON files found under {input_dir}")
        return

    for json_path in sorted(json_files):
        convert_coco_to_mot(str(json_path), output_dir, location=json_path.parent.name)


if __name__ == '__main__':
    parser = ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--coco_json', type=str, help='Path to a single input COCO JSON')
    group.add_argument('--input_dir', type=str, help='Root directory containing {location}/*.json COCO files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write per-clip MOT txt files')
    args = parser.parse_args()

    if args.coco_json:
        convert_coco_to_mot(args.coco_json, args.output_dir)
    else:
        convert_directory_to_mot(args.input_dir, args.output_dir)
