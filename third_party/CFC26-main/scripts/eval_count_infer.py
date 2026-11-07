"""
Run count eval on inference outputs organized as:
  {infer_dir}/{location}/{clip_name}/{clip_name}.txt

GT MOT layout:   {gt_dir}/{location}/{clip_name}/gt.txt
Frames layout:   {frames_dir}/{location}/{clip_name}/*.png

Outputs per location:
  {output_dir}/{location}/clip_count_error_results.csv

Summary across all locations:
  {output_dir}/eval_summary.csv  (columns: location, nMAE, nMNE, nANE, nMANE, n_clips)

Single location:
  python scripts/eval_count_infer.py \
      --infer_dir /datadrive/experiments/my_exp/inference/kenai-val \
      --gt_dir /path/to/gt_mot \
      --frames_dir /data_drive/3_channels \
      --output_dir /path/to/eval_output

All locations:
  python scripts/eval_count_infer.py \
      --infer_dir /datadrive/experiments/my_exp/inference \
      --gt_dir /path/to/gt_mot \
      --frames_dir /data_drive/3_channels \
      --output_dir /path/to/eval_output
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from types import SimpleNamespace

from count import Count
from eval_count import AE, NE, nMAE, nMNE, nANE, nMANE

EPSILON = 1e-6


def eval_location(count, location, pred_location_dir, gt_location_dir, frames_location_dir):
    """
    Evaluate one location.

    pred_location_dir:   {infer_dir}/{location}  — contains {clip_name}/{clip_name}.txt
    gt_location_dir:     {gt_dir}/{location}      — contains {clip_name}/gt.txt
    frames_location_dir: {frames_dir}/{location}  — contains {clip_name}/

    Returns (df, metrics) or (None, None) if no data.
    """
    clip_dirs = [
        d for d in os.listdir(pred_location_dir)
        if os.path.isdir(os.path.join(pred_location_dir, d))
        and not d.startswith('.')
        and not d.startswith('FCe_')
    ]
    if not clip_dirs:
        print(f"  No clip directories in {pred_location_dir}, skipping")
        return None, None

    gt_videopath_to_count = {}
    clip_to_error = {}
    rows = []

    for clip_name in sorted(clip_dirs):
        pred_path = os.path.join(pred_location_dir, clip_name, f'{clip_name}.txt')
        gt_path = os.path.join(gt_location_dir, clip_name, 'gt.txt')
        video_path = os.path.join(frames_location_dir, clip_name)

        if not os.path.exists(pred_path):
            print(f"  Missing pred: {pred_path}, skipping")
            continue
        if not os.path.exists(gt_path):
            print(f"  Missing GT: {gt_path}, skipping")
            continue
        if not os.path.exists(video_path):
            print(f"  Missing frames dir: {video_path}, skipping")
            continue

        gt_left, gt_right, gt_n = count.count(gt_path, video_path)
        pred_left, pred_right, pred_n = count.count(pred_path, video_path)

        gt_counts = {'left': gt_left, 'right': gt_right}
        pred_counts = {'left': pred_left, 'right': pred_right}
        gt_videopath_to_count[video_path] = gt_counts
        clip_error = AE(gt_counts, pred_counts)
        clip_net_error = NE(gt_counts, pred_counts)
        clip_to_error[clip_name] = clip_error

        n_total = gt_left + gt_right or EPSILON
        rows.append({
            'clip': clip_name,
            'gt_left': gt_left,
            'gt_right': gt_right,
            'pred_left': pred_left,
            'pred_right': pred_right,
            'absolute_error': clip_error,
            'normalized_error': clip_error / n_total,
            'net_gt': gt_left - gt_right,
            'net_error': clip_net_error,
        })

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    net_errors = df['net_error'].tolist()
    net_gts = df['net_gt'].tolist()

    micro = nMAE(clip_to_error, gt_videopath_to_count, save_path=None)
    sum_ae = df['absolute_error'].sum()
    sum_gt = sum(v['left'] + v['right'] for v in gt_videopath_to_count.values())
    sum_net_error = float(np.sum(net_errors))
    sum_net_gt = float(np.sum(net_gts))
    sum_abs_net_error = float(np.sum(np.abs(net_errors)))
    sum_abs_net_gt = float(np.sum(np.abs(net_gts)))
    signs = np.where(np.array(net_gts) >= 0, 1, -1)
    sum_aligned_net_error = float(np.sum(signs * np.array(net_errors)))

    return df, {
        'nMAE': micro,
        'nMNE': nMNE(net_errors, net_gts),
        'nANE': nANE(net_errors, net_gts),
        'nMANE': nMANE(net_errors, net_gts),
        'sum_ae': sum_ae,
        'sum_gt': sum_gt,
        'sum_net_error': sum_net_error,
        'sum_net_gt': sum_net_gt,
        'sum_abs_net_error': sum_abs_net_error,
        'sum_abs_net_gt': sum_abs_net_gt,
        'sum_aligned_net_error': sum_aligned_net_error,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument('--infer_dir', type=str, required=True,
                        help='Inference root: {infer_dir}/{location}/{clip}/{clip}.txt')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='GT MOT root: {gt_dir}/{location}/{clip}/gt.txt')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Frames root: {frames_dir}/{location}/{clip}/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write eval results')
    parser.add_argument('--count_protocol', type=str, default='LOI')
    args = parser.parse_args()

    count = Count(SimpleNamespace(count_protocol=args.count_protocol))

    # Auto-detect depth: if infer_dir's immediate subdirs contain clip subdirs with .txt files,
    # it's a single-location dir ({clip}/{clip}.txt).
    # Otherwise it's the multi-location root ({location}/{clip}/{clip}.txt).
    def is_single_location(path):
        for entry in os.scandir(path):
            if entry.is_dir() and not entry.name.startswith('.'):
                inner = [
                    f for f in os.listdir(entry.path)
                    if f.endswith('.txt') and not f.startswith('FCe_') and not f.startswith('.')
                ]
                if inner:
                    return True
        return False

    infer_path = args.infer_dir
    locations = []  # (location, pred_location_dir, gt_location_dir, frames_location_dir)

    if is_single_location(infer_path):
        location = os.path.basename(infer_path.rstrip('/'))
        locations.append((location, infer_path, args.gt_dir, args.frames_dir))
    else:
        for location in sorted(os.listdir(infer_path)):
            if location.startswith('.'):
                continue
            pred_location_dir = os.path.join(infer_path, location)
            if not os.path.isdir(pred_location_dir):
                continue
            locations.append((
                location,
                pred_location_dir,
                os.path.join(args.gt_dir, location),
                os.path.join(args.frames_dir, location),
            ))

    print(f"Found {len(locations)} location(s)")

    summary_rows = []

    for location, pred_location_dir, gt_location_dir, frames_location_dir in locations:
        print(f"\n{location}")

        df, metrics = eval_location(count, location, pred_location_dir, gt_location_dir, frames_location_dir)

        if df is None:
            continue

        out_dir = os.path.join(args.output_dir, location)
        os.makedirs(out_dir, exist_ok=True)
        clip_csv = os.path.join(out_dir, 'clip_count_error_results.csv')
        df.to_csv(clip_csv, index=False)

        print(f"  nMAE={metrics['nMAE']:.4f}  nMNE={metrics['nMNE']:.4f}  "
              f"nANE={metrics['nANE']:.4f}  nMANE={metrics['nMANE']:.4f}  "
              f"n_clips={len(df)}  -> {clip_csv}")
        summary_rows.append({
            'location': location,
            'nMAE': metrics['nMAE'],
            'nMNE': metrics['nMNE'],
            'nANE': metrics['nANE'],
            'nMANE': metrics['nMANE'],
            'n_clips': len(df),
            'sum_ae': metrics['sum_ae'],
            'sum_gt': metrics['sum_gt'],
            'sum_net_error': metrics['sum_net_error'],
            'sum_net_gt': metrics['sum_net_gt'],
            'sum_abs_net_error': metrics['sum_abs_net_error'],
            'sum_abs_net_gt': metrics['sum_abs_net_gt'],
            'sum_aligned_net_error': metrics['sum_aligned_net_error'],
        })

    if not summary_rows:
        print("No results produced.")
        return

    summary_df = pd.DataFrame(summary_rows)

    if summary_df['location'].nunique() > 1:
        g = summary_df
        sum_abs_net_gt = g['sum_abs_net_gt'].sum()
        all_row = {
            'location': 'ALL',
            'nMAE': g['sum_ae'].sum() / g['sum_gt'].sum(),
            'nMNE': (g['sum_net_error'].sum() / g['sum_net_gt'].sum()) if g['sum_net_gt'].sum() != 0 else np.nan,
            'nANE': (g['sum_abs_net_error'].sum() / sum_abs_net_gt) if sum_abs_net_gt != 0 else np.nan,
            'nMANE': (g['sum_aligned_net_error'].sum() / sum_abs_net_gt) if sum_abs_net_gt != 0 else np.nan,
            'n_clips': g['n_clips'].sum(),
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([all_row])], ignore_index=True)

    accum_cols = ['sum_ae', 'sum_gt', 'sum_net_error', 'sum_net_gt',
                  'sum_abs_net_error', 'sum_abs_net_gt', 'sum_aligned_net_error']
    summary_df = summary_df.drop(columns=[c for c in accum_cols if c in summary_df.columns])
    summary_df = summary_df.sort_values('location').reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(args.output_dir, 'eval_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nEval summary saved to {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()