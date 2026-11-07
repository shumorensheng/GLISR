"""Read native sonar arrays without geometric or intensity preprocessing."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SonarDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path, usage: str):
        manifest = json.loads(manifest_path.read_text())
        self.frames = [frame for frame in manifest["frames"] if frame["usage"] == usage]
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        metadata = json.loads((self.data_root / frame["metadata_relpath"]).read_text())
        frame_key = frame["sample_id"].rsplit("/", 1)[1]
        with np.load(self.data_root / frame["npz_relpath"], allow_pickle=False) as arrays:
            return {
                **frame,
                "intensity": torch.from_numpy(arrays["intensity"]).float().unsqueeze(0),
                "range_m": torch.from_numpy(arrays["range_coordinates_m"]).float(),
                "bearing_rad": torch.from_numpy(arrays["bearing_coordinates_rad"]).float(),
                "valid": torch.from_numpy(arrays["intensity_valid_region_mask"]).bool(),
                "configuration": metadata["frames"][frame_key]["configuration"],
            }


def collate_native(samples: list[dict]) -> list[dict]:
    return samples
