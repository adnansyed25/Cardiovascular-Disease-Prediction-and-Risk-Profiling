from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m cardiac.mri.preprocess` first.")
    d = np.load(path, allow_pickle=False)
    return d["X"], d["y"], d["ids"]


class ACDCVolumes(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, seed: int = 0):
        keep = y >= 0
        self.X = X[keep].astype(np.float32)
        self.y = y[keep].astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def _augment(self, v: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            v = v[..., ::-1]
        if self.rng.random() < 0.5:
            v = v[..., ::-1, :]
        k = int(self.rng.integers(0, 4))
        v = np.rot90(v, k, axes=(-2, -1))
        v = v * self.rng.uniform(0.9, 1.1) + self.rng.normal(0, 0.05)
        return np.ascontiguousarray(v, dtype=np.float32)

    def __getitem__(self, i: int):
        v = self.X[i]
        if self.augment:
            v = self._augment(v)
        return torch.from_numpy(np.ascontiguousarray(v)), torch.tensor(self.y[i])
