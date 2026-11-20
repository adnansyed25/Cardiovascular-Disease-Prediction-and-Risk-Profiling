from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from cardiac import config

SEG_LABELS = {"RV": 1, "MYO": 2, "LV": 3}
MYO_DENSITY = 1.05


def parse_info_cfg(path: Path) -> dict[str, str]:
    info = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            info[key.strip()] = val.strip()
    return info


def resize_volume(vol: np.ndarray, target_shape=config.MRI_TARGET_SHAPE) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    out = zoom(vol.astype(np.float32), factors, order=1)
    out = out[: target_shape[0], : target_shape[1], : target_shape[2]]
    pad = [(0, t - s) for t, s in zip(target_shape, out.shape)]
    return np.pad(out, pad, mode="edge") if any(p[1] for p in pad) else out


def zscore(vol: np.ndarray) -> np.ndarray:
    return ((vol - vol.mean()) / (vol.std() + 1e-8)).astype(np.float32)


def structure_volumes(seg: np.ndarray, voxel_size) -> dict[str, float]:
    voxel_ml = float(np.prod(voxel_size[:3])) / 1000.0
    return {name: float((seg == lab).sum()) * voxel_ml for name, lab in SEG_LABELS.items()}


def ejection_fraction(edv: float, esv: float) -> float:
    return float("nan") if edv <= 0 else 100.0 * (edv - esv) / edv


def heart_metrics(ed_seg, es_seg, voxel_size, info: dict[str, str]) -> dict[str, float]:
    ed, es = structure_volumes(ed_seg, voxel_size), structure_volumes(es_seg, voxel_size)
    m = {
        "EDV_LV": ed["LV"], "ESV_LV": es["LV"], "EF_LV": ejection_fraction(ed["LV"], es["LV"]),
        "EDV_RV": ed["RV"], "ESV_RV": es["RV"], "EF_RV": ejection_fraction(ed["RV"], es["RV"]),
        "MYO_MASS_ED": ed["MYO"] * MYO_DENSITY, "MYO_MASS_ES": es["MYO"] * MYO_DENSITY,
    }
    try:
        h, w = float(info["Height"]), float(info["Weight"])
        bsa = np.sqrt(h * w / 3600.0)
        m.update({"height": h, "weight": w, "EDV_LV_index": m["EDV_LV"] / bsa,
                  "EDV_RV_index": m["EDV_RV"] / bsa, "MYO_MASS_index": m["MYO_MASS_ED"] / bsa,
                  "RV_LV_ratio": m["EDV_RV"] / max(m["EDV_LV"], 1e-6)})
    except (KeyError, ValueError):
        pass
    return m


def _load_nii(path: Path):
    import nibabel as nib

    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.header.get_zooms()


def patient_dirs(split_dir: Path) -> list[Path]:
    dirs = sorted(p for p in Path(split_dir).iterdir() if p.is_dir() and re.fullmatch(r"patient\d{3}", p.name))
    if not dirs:
        raise FileNotFoundError(f"No patientXXX folders found in {split_dir}")
    return dirs


def load_patient(pdir: Path, target_shape=config.MRI_TARGET_SHAPE):
    info = parse_info_cfg(pdir / "Info.cfg")
    pid = pdir.name
    ed_f, es_f = int(info["ED"]), int(info["ES"])
    ed, vox = _load_nii(pdir / f"{pid}_frame{ed_f:02d}.nii.gz")
    es, _ = _load_nii(pdir / f"{pid}_frame{es_f:02d}.nii.gz")

    vol = np.stack([zscore(resize_volume(ed, target_shape)), zscore(resize_volume(es, target_shape))])
    vol = np.transpose(vol, (0, 3, 1, 2))

    group = info.get("Group", "")
    label = config.ACDC_CLASSES.index(group) if group in config.ACDC_CLASSES else -1

    metrics = {}
    ed_gt, es_gt = pdir / f"{pid}_frame{ed_f:02d}_gt.nii.gz", pdir / f"{pid}_frame{es_f:02d}_gt.nii.gz"
    if ed_gt.exists() and es_gt.exists():
        metrics = heart_metrics(_load_nii(ed_gt)[0], _load_nii(es_gt)[0], vox, info)
    return pid, vol.astype(np.float32), label, group, metrics


def process_split(split_dir: Path, out_dir: Path, split: str, target_shape=config.MRI_TARGET_SHAPE):
    X, y, ids, rows = [], [], [], []
    dirs = patient_dirs(split_dir)
    for i, pdir in enumerate(dirs, 1):
        try:
            pid, vol, label, group, metrics = load_patient(pdir, target_shape)
        except (FileNotFoundError, KeyError) as e:
            print(f"  ! skipping {pdir.name}: {e}")
            continue
        X.append(vol)
        y.append(label)
        ids.append(pid)
        rows.append({"patient": pid, "split": split, "group": group, **metrics})
        print(f"\r  {split}: {i}/{len(dirs)}", end="", flush=True)
    print()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"acdc_{split}.npz", X=np.stack(X), y=np.array(y), ids=np.array(ids))
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acdc-dir", type=Path, default=config.ACDC_DIR,
                   help="folder containing 'training' and 'testing' sub-folders")
    p.add_argument("--out", type=Path, default=config.PROCESSED_DIR)
    a = p.parse_args(argv)

    rows = []
    for split in ("training", "testing"):
        split_dir = a.acdc_dir / split
        if split_dir.exists():
            rows += process_split(split_dir, a.out, split)
        else:
            print(f"(no '{split}' folder in {a.acdc_dir}, skipped)")
    if not rows:
        raise SystemExit(f"Nothing processed. Check --acdc-dir (currently {a.acdc_dir}).")
    pd.DataFrame(rows).to_csv(a.out / "acdc_heart_metrics.csv", index=False)
    print(f"Saved arrays and heart metrics to {a.out}")


if __name__ == "__main__":
    main()
