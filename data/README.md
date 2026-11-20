# Data

## Clinical dataset (included)

`raw/cardiovascular.csv`: 70,000 patient records from the Kaggle
[Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).

| Column | Meaning |
|---|---|
| `age` | age in **days** |
| `gender` | 1 = female, 2 = male |
| `height`, `weight` | cm, kg |
| `ap_hi`, `ap_lo` | systolic / diastolic blood pressure (mmHg) |
| `cholesterol`, `gluc` | 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke`, `alco`, `active` | 0 / 1 |
| `cardio` | **target**: 1 = cardiovascular disease present |

Cleaning (see `src/cardiac/tabular/data.py`) removes duplicates and physiologically
impossible values (e.g. `ap_hi = 16020`, negative pressures, diastolic ≥ systolic),
leaving 68,561 patients.

## Cardiac MRI dataset (not included: download it yourself)

The ACDC dataset (150 patients, ~3 GB) is too large for GitHub and has its own licence.

1. Download it from the ACDC challenge page (<https://www.creatis.insa-lyon.fr/Challenge/acdc/>)
   or from the Hugging Face mirror `MedOtter/ACDC` (notebook 03 does this automatically).
2. Put it here so the layout is:

```
data/acdc/training/patient001/Info.cfg
data/acdc/training/patient001/patient001_frame01.nii.gz
...
data/acdc/testing/patient101/...
```

3. Run `python -m cardiac.mri.preprocess`. Outputs go to `data/processed/`.

Both `data/acdc/` and `data/processed/` are in `.gitignore`, so they are never pushed.
