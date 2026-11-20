from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CARDIAC_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "cardiovascular.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ACDC_DIR = DATA_DIR / "acdc"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42

TARGET = "cardio"

PLAUSIBLE_RANGES = {
    "height": (120, 220),
    "weight": (30, 200),
    "ap_hi": (70, 250),
    "ap_lo": (40, 150),
    "bmi": (12, 60),
}

NUMERIC_FEATURES = [
    "age_years", "height", "weight", "bmi",
    "ap_hi", "ap_lo", "pulse_pressure", "mean_arterial_pressure",
]
ORDINAL_FEATURES = ["cholesterol", "gluc"]
BINARY_FEATURES = ["is_male", "smoke", "alco", "active"]
FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES

ACDC_CLASSES = ["NOR", "MINF", "DCM", "HCM", "RV"]
ACDC_CLASS_NAMES = {
    "NOR": "Normal",
    "MINF": "Myocardial infarction",
    "DCM": "Dilated cardiomyopathy",
    "HCM": "Hypertrophic cardiomyopathy",
    "RV": "Abnormal right ventricle",
}
MRI_TARGET_SHAPE = (128, 128, 8)


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
