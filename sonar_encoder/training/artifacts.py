"""Place large training artifacts on the dataset disk behind local symlinks."""

import os
from pathlib import Path


RUN_ROOT = Path(os.environ.get('GLISR_REPO_ROOT', str(Path(__file__).resolve().parents[1])))/"runs"
ARTIFACT_ROOT = Path(os.environ.get('GLISR_ARTIFACTS_ROOT', str(Path(__file__).resolve().parents[1] / 'artifacts')))


def create_artifact_directory(path: Path) -> None:
    path = path if path.is_absolute() else Path.cwd() / path
    target = ARTIFACT_ROOT / path.relative_to(RUN_ROOT)
    target.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_symlink():
        path.symlink_to(target, target_is_directory=True)
