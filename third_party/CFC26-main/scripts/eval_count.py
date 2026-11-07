'''
Evaluation metrics for counting tasks

Iterate over pred_mot and annotation_mot files to generate absolute left and right counts for each video,
calculating the absolute left and right count errors for each video, and then calculate the
normalized mean absolute counting error (nMAE) for each domain.

To run standalone:
  python scripts/eval_count.py \
    --dataset kenai-val \
    --frames_dir /data_drive/3_channels \
    --annotations_dir /some/dir/gt_mot \
    --track_dir /datadrive/experiments/{experiment_name}/inference \
    --output_dir /path/to/output
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from count import Count

EPSILON = 1e-6


def AE(gt_counts, pred_counts):
    return np.abs(gt_counts['left'] - pred_counts['left']) + np.abs(gt_counts['right'] - pred_counts['right'])


def NE(gt_counts, pred_counts):
    """Signed net error per clip: (pred_left - pred_right) - (gt_left - gt_right)"""
    return (pred_counts['left'] - pred_counts['right']) - (gt_counts['left'] - gt_counts['right'])


def nMAE(clip_to_error, gt_videopath_to_count, save_path, debug=False):
    errors = list(clip_to_error.values())
    total_counted_fish = [count['left'] + count['right'] for count in gt_videopath_to_count.values()]
    sum_of_errors = sum(errors)
    sum_of_gt_counts = sum(total_counted_fish)
    if debug and save_path and os.path.dirname(save_path):
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, edgecolor='black')
        plt.title('Histogram of Counting Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(save_path, 'error_histogram.png'))
        plt.close()

        print(f"Error statistics:")
        print(f"Mean: {np.mean(errors):.2f}")
        print(f"Median: {np.median(errors):.2f}")
        print(f"Standard Deviation: {np.std(errors):.2f}")
        print(f"Min: {min(errors):.2f}")
        print(f"Max: {max(errors):.2f}")

    assert sum_of_gt_counts > 0, 'Sum of GT counts is 0'
    return sum_of_errors / sum_of_gt_counts


def nMNE(net_errors, net_gts):
    """Normalized Mean Net Error: sum(net_error) / sum(net_gt). Measures directional bias."""
    sum_net_gt = float(np.sum(net_gts))
    if sum_net_gt == 0:
        return np.nan
    return float(np.sum(net_errors)) / sum_net_gt


def nANE(net_errors, net_gts):
    """Normalized Absolute Net Error: sum(|net_error|) / sum(|net_gt|)."""
    sum_abs_net_gt = float(np.sum(np.abs(net_gts)))
    if sum_abs_net_gt == 0:
        return np.nan
    return float(np.sum(np.abs(net_errors))) / sum_abs_net_gt


def nMANE(net_errors, net_gts):
    """Normalized Mean Aligned Net Error: sum(sign(net_gt) * net_error) / sum(|net_gt|)."""
    net_errors = np.array(net_errors)
    net_gts = np.array(net_gts)
    signs = np.where(net_gts >= 0, 1, -1)
    sum_abs_net_gt = float(np.sum(np.abs(net_gts)))
    if sum_abs_net_gt == 0:
        return np.nan
    return float(np.sum(signs * net_errors)) / sum_abs_net_gt


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kenai-val', help='Dataset name')
    parser.add_argument('--frames_dir', type=str, required=True, help='Directory with frames')
    parser.add_argument('--annotations_dir', type=str, required=True, help='Directory with ground truth MOT annotations')
    parser.add_argument('--track_dir', type=str, required=True, help='Directory with predicted MOT tracks')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save eval output')
    parser.add_argument('--count_protocol', type=str, default='LOI', help='Count protocol')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def main():
    args = parse_args()
    count = Count(args)

    domain = args.dataset
    frames_path = os.path.join(args.frames_dir, domain)
    print(f"Frames path: {frames_path} exists? {os.path.exists(frames_path)}")

    assert os.path.exists(frames_path), f'Frames directory {frames_path} does not exist'
    all_clips = os.listdir(frames_path)

    gt_videopath_to_count = defaultdict(lambda: {'left': 0, 'right': 0})
    pred_videopath_to_count = defaultdict(lambda: {'left': 0, 'right': 0})
    clip_to_error = dict()

    df = pd.DataFrame(
        columns=['clip', 'gt_left', 'pred_left', 'gt_right', 'pred_right', 'absolute_error', 'normalized_error',
                 'net_gt', 'net_error'])

    for clip in all_clips:
        if clip.startswith('.') or clip.endswith('.txt') or clip.endswith('.csv'):
            continue
        if clip.startswith('FCe_'):  
            continue
        video_path = os.path.join(args.frames_dir, domain, clip)
        mot_path = os.path.join(args.annotations_dir, domain, clip, 'gt.txt')
        track_pred_path = os.path.join(args.track_dir, domain, f'{clip}.txt')

        assert os.path.exists(mot_path), f'MOT path {mot_path} does not exist'
        gt_absolute_left_counts, gt_absolute_right_counts, gt_n_unique_fish = count.count(mot_path, video_path)

        if not os.path.exists(track_pred_path):
            pred_absolute_left_counts, pred_absolute_right_counts, pred_n_unique_fish = 0, 0, 0
        else:
            pred_absolute_left_counts, pred_absolute_right_counts, pred_n_unique_fish = count.count(track_pred_path, video_path)

        gt_videopath_to_count[video_path] = {'left': gt_absolute_left_counts, 'right': gt_absolute_right_counts,
                                             'n_unique_fish': gt_n_unique_fish}
        pred_videopath_to_count[video_path] = {'left': pred_absolute_left_counts, 'right': pred_absolute_right_counts,
                                               'n_unique_fish': pred_n_unique_fish}

        clip_error = AE(gt_videopath_to_count[video_path], pred_videopath_to_count[video_path])
        clip_net_error = NE(gt_videopath_to_count[video_path], pred_videopath_to_count[video_path])
        clip_to_error[clip] = clip_error

        n_total_counted_fish = gt_absolute_left_counts + gt_absolute_right_counts
        if n_total_counted_fish == 0:
            n_total_counted_fish = EPSILON
        new_row = [{'clip': clip,
                    'gt_left': gt_absolute_left_counts,
                    'pred_left': pred_absolute_left_counts,
                    'gt_right': gt_absolute_right_counts,
                    'pred_right': pred_absolute_right_counts,
                    'absolute_error': clip_error,
                    'normalized_error': clip_error / n_total_counted_fish,
                    'net_gt': gt_absolute_left_counts - gt_absolute_right_counts,
                    'net_error': clip_net_error}]
        df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=True)

    exp_name = os.path.basename(os.path.dirname(args.track_dir))  # .../experiments/{exp_name}/inference
    counts_save_path = os.path.join(args.output_dir, exp_name)
    metric = nMAE(clip_to_error, gt_videopath_to_count, os.path.join(counts_save_path, domain), debug=args.debug)
    print(f'{domain} nMAE: {metric}')

    net_errors = df['net_error'].tolist()
    net_gts = df['net_gt'].tolist()
    print(f'{domain} nMNE:  {nMNE(net_errors, net_gts):.4f}')
    print(f'{domain} nANE:  {nANE(net_errors, net_gts):.4f}')
    print(f'{domain} nMANE: {nMANE(net_errors, net_gts):.4f}')

    results_csv_path = os.path.join(counts_save_path, domain, 'clip_count_error_results.csv')
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)

    high_error_clips = df.sort_values(by='normalized_error', ascending=False).head(10)
    print(f'Highest error clips:\n{high_error_clips}')
    low_error_clips = df.sort_values(by='normalized_error', ascending=True).head(10)
    print(f'Lowest error clips:\n{low_error_clips}')
    df.to_csv(results_csv_path, index=False)
    print(f'Count results saved to {results_csv_path}')


if __name__ == '__main__':
    main()