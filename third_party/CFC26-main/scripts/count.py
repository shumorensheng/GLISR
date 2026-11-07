"""LOI (Line of Interest) fish counting from MOT track files."""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

__all__ = ['Count']

def load_mot_file(file_path):
    df = pd.read_csv(file_path, delimiter=',', header=None)
    df.columns = ['frame', 'id', 'x1', 'y1', 'width', 'height', 'conf', 'x', 'y', 'z']
    df['x'] = df['x1'].astype(int)
    df['y'] = df['y1'].astype(int)
    df['frame'] = df['frame'].astype(int)
    df['id'] = df['id'].astype(int)
    df['width'] = df['width'].astype(int)
    df['height'] = df['height'].astype(int)
    df['conf'] = df['conf'].astype(float)
    df['kp_x'] = df['x'] + df['width'] / 2
    df['kp_y'] = df['y'] + df['height'] / 2
    return df


class Count:
    def __init__(self, args):
        self.args = args
        self.protocol = args.count_protocol

    def count(self, mot_tracks_file, video_path):
        if not os.path.exists(mot_tracks_file):
            print(f"Skipping {mot_tracks_file} because it does not exist")
            return 0, 0, 0
        if os.path.getsize(mot_tracks_file) == 0:
            print(f"Skipping {mot_tracks_file} because it is empty")
            return 0, 0, 0

        if self.protocol == 'LOI':
            frames = [f for f in os.listdir(video_path) if not f.startswith('.')]
            image = Image.open(os.path.join(video_path, frames[0]))
            w, h = image.size
            line_x = w / 2

            mot_tracks_df = load_mot_file(mot_tracks_file)
            left, right = self.LOI_count(mot_tracks_df, line_x)
            n_unique_fish = len(mot_tracks_df['id'].unique())
            return left, right, n_unique_fish
        else:
            raise ValueError(f'Protocol {self.protocol} not supported')

    def LOI_count(self, mot_tracks_df, line):
        unique_tracks = mot_tracks_df['id'].unique()
        track_counts = defaultdict(lambda: {'left': 0, 'right': 0})

        for track in unique_tracks:
            track_df = mot_tracks_df[mot_tracks_df['id'] == track].sort_values('frame', ascending=True)
            x_coords = track_df['kp_x'].values
            first_x, last_x = x_coords[0], x_coords[-1]

            if last_x <= line and first_x > line:
                track_counts[track]['left'] += 1
            elif last_x >= line and first_x < line:
                track_counts[track]['right'] += 1

        # Calculate absolute counts
        absolute_left_counts = 0
        absolute_right_counts = 0

        for track, count in track_counts.items():
            net_counts = count['left'] - count['right']
            assert net_counts <= 1 and net_counts >= -1, f'Track {track} has net counts of {net_counts}'
            if net_counts > 0:
                absolute_left_counts += 1
            else:
                absolute_right_counts += 1

        print(f'Absolute left counts: {absolute_left_counts}, Absolute right counts: {absolute_right_counts}')
        return absolute_left_counts, absolute_right_counts