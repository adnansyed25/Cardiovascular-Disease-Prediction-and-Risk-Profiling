from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9c6644", "#6d597a"]


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_grid(cms: dict[str, np.ndarray], labels: Sequence[str], path: Path) -> Path:
    n = len(cms)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax, (name, cm) in zip(axes.flat, cms.items()):
        _confusion(ax, cm, labels, name)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return save(fig, path)


def confusion_single(cm: np.ndarray, labels: Sequence[str], title: str, path: Path) -> Path:
    size = max(4.5, 0.9 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    _confusion(ax, cm, labels, title)
    return save(fig, path)


def _confusion(ax, cm, labels, title):
    ax.imshow(cm, cmap="Blues")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)


def roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for (name, (fpr, tpr, auc)), color in zip(curves.items(), PALETTE * 3):
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return save(fig, path)


def metric_bars(table: dict[str, dict[str, float]], metrics: Sequence[str], title: str, path: Path) -> Path:
    names = list(table)
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names)), 4.8))
    for i, (m, color) in enumerate(zip(metrics, PALETTE)):
        vals = [table[n][m] for n in names]
        bars = ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width, label=m, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(ncol=len(metrics), fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    return save(fig, path)


def horizontal_importance(names: Sequence[str], values: Sequence[float], errors, title: str, path: Path) -> Path:
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(names) + 1.5))
    ax.barh(np.array(names)[order], np.array(values)[order],
            xerr=None if errors is None else np.array(errors)[order], color=PALETTE[0])
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    return save(fig, path)


def training_curves(history: dict[str, list[float]], path: Path) -> Path:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    epochs = range(1, len(history["train_loss"]) + 1)
    a1.plot(epochs, history["train_loss"], label="train")
    a1.plot(epochs, history["val_loss"], label="validation")
    a1.set_title("Loss")
    a2.plot(epochs, history["train_acc"], label="train")
    a2.plot(epochs, history["val_acc"], label="validation")
    a2.set_ylim(0, 1)
    a2.set_title("Accuracy")
    for a in (a1, a2):
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(alpha=0.3)
    return save(fig, path)
