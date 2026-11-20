from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from cardiac import config, plots
from cardiac.mri.dataset import ACDCVolumes, load_npz
from cardiac.mri.model import CardiacCNN3D

CLASS_NAMES = [config.ACDC_CLASS_NAMES[c] for c in config.ACDC_CLASSES]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    probs, labels = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
            probs.append(torch.softmax(logits, 1).detach().cpu().numpy())
            labels.append(y.cpu().numpy())
    return total_loss / n, correct / n, np.concatenate(probs), np.concatenate(labels)


def train(epochs: int = 100, batch_size: int = 8, lr: float = 3e-4, weight_decay: float = 1e-4,
          val_size: float = 0.2, patience: int = 25, device: str | None = None,
          data_dir: Path = config.PROCESSED_DIR, out_dir: Path = config.REPORTS_DIR,
          seed: int = config.RANDOM_STATE) -> dict:
    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir, fig_dir = Path(out_dir), Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_npz(Path(data_dir) / "acdc_training.npz")
    labelled = y >= 0
    X, y = X[labelled], y[labelled]
    idx_tr, idx_val = train_test_split(np.arange(len(y)), test_size=val_size, stratify=y, random_state=seed)
    train_ds = ACDCVolumes(X[idx_tr], y[idx_tr], augment=True, seed=seed)
    val_ds = ACDCVolumes(X[idx_val], y[idx_val])

    test_path = Path(data_dir) / "acdc_testing.npz"
    test_ds = None
    if test_path.exists():
        Xt, yt, _ = load_npz(test_path)
        if (yt >= 0).any():
            test_ds = ACDCVolumes(Xt, yt)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    print(f"Device {device} | train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds) if test_ds else 0}")

    model = CardiacCNN3D(num_classes=len(config.ACDC_CLASSES), in_channels=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt = config.MODELS_DIR / "cnn3d_best.pt"
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    best_acc, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta, _, _ = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va, _, _ = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        for k, v in zip(history, (tl, vl, ta, va)):
            history[k].append(v)
        if va > best_acc:
            best_acc, best_epoch, stale = va, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
        print(f"Epoch {epoch:3d}/{epochs} | train {tl:.3f}/{ta:.3f} | val {vl:.3f}/{va:.3f} "
              f"| {time.time() - t0:.1f}s")
        if stale >= patience:
            print(f"Early stopping: no validation improvement for {patience} epochs.")
            break

    pd.DataFrame(history).rename_axis("epoch").to_csv(out_dir / "mri_training_history.csv")
    plots.training_curves(history, fig_dir / "mri_training_curves.png")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    eval_ds, eval_name = (test_ds, "test") if test_ds else (val_ds, "validation")
    _, acc, probs, labels = run_epoch(model, DataLoader(eval_ds, batch_size=batch_size), criterion, device)
    preds = probs.argmax(1)
    all_labels = list(range(len(CLASS_NAMES)))

    report = classification_report(labels, preds, labels=all_labels, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=all_labels)
    plots.confusion_single(cm, config.ACDC_CLASSES, f"3D-CNN: {eval_name} set", fig_dir / "mri_confusion.png")

    curves = {}
    for i, name in enumerate(config.ACDC_CLASSES):
        if 0 < (labels == i).sum() < len(labels):
            fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
            curves[name] = (fpr, tpr, roc_auc_score(labels == i, probs[:, i]))
    if curves:
        plots.roc_curves(curves, f"3D-CNN one-vs-rest ROC: {eval_name} set", fig_dir / "mri_roc.png")

    summary = {"best_epoch": best_epoch, "best_val_accuracy": best_acc,
               f"{eval_name}_accuracy": acc, "per_class": report}
    (out_dir / "mri_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest val acc {best_acc:.3f} (epoch {best_epoch}) | {eval_name} acc {acc:.3f}")
    print(classification_report(labels, preds, labels=all_labels, target_names=CLASS_NAMES, zero_division=0))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25, help="early-stopping patience (epochs)")
    p.add_argument("--device", help="cuda / cpu (default: auto)")
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--out", type=Path, default=config.REPORTS_DIR)
    a = p.parse_args(argv)
    train(a.epochs, a.batch_size, a.lr, a.weight_decay, patience=a.patience, device=a.device,
          data_dir=a.data_dir, out_dir=a.out)


if __name__ == "__main__":
    main()
