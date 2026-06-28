# BreastDCEDL Run Guide

## 1) Install

```bash
python -m pip install numpy pandas nibabel matplotlib scikit-learn pillow jupyter
```

If you run the ViT notebook:

```bash
python -m pip install torch torchvision transformers
```

## 2) Configure local paths

Use the sample data already in the repo.

```python
import os, sys
sys.path.append(os.path.abspath("utils"))
import data_utils as ds

base_path = "."
nifti_path = {
    "spy2": os.path.join(base_path, "ISPY2", "data_samples", "dce"),
    "spy1": os.path.join(base_path, "ISPY1", "data_samples", "dce"),
    "duke": os.path.join(base_path, "DUKE", "data_samples", "dce"),
}
mask_path = {
    "spy2": os.path.join(base_path, "ISPY2", "data_samples", "mask"),
    "spy1": os.path.join(base_path, "ISPY1", "data_samples", "mask"),
    "duke": os.path.join(base_path, "DUKE", "data_samples", "mask"),
}
ds.setup_paths(base_path, nifti_path, mask_path)
```

## 3) Smoke test

```python
acqs = ds.get_all_nifti_acquisitions("ISPY1_1072")
mask = ds.get_nifti_mask("ISPY1_1072")

acqs_duke = ds.get_all_nifti_acquisitions("Breast_MRI_001")
mask_duke = ds.get_nifti_mask("Breast_MRI_001")
```

Notes:
- I-SPY1 / I-SPY2 use real masks.
- Duke has no `mask/` folder in sample data; `data_utils.py` falls back to bbox metadata.

## 4) Full preprocessing

Run this once after cloning, or let `run_everything.py` do it for you:

```bash
python preprocess_full_dataset.py --output-dir preprocessed_output
```

Quick test:

```bash
python preprocess_full_dataset.py --output-dir preprocessed_output --limit 5
```

Outputs:
- `preprocessed_output/preprocess_manifest.csv`
- `preprocessed_output/<dataset>/<pid>/preview.npy`
- `preprocessed_output/<dataset>/<pid>/cropped_acquisitions.npz`

## 5) Reproduce the Duke pCR table + plots

Open:

```text
DUKE/duke_modeling_with_niftii_files.ipynb
```

Run the notebook cells that:
- load `DUKE/TCIA_metadata/Imaging_Features.xlsx`
- merge with `BreastDCEDL_duke_metadata.csv`
- train the model
- call `ds.report_full(...)`
- plot feature importance

The relevant code is in the pCR section around the Random Forest / Gradient Boosting cells.

## 6) One-click run everything

This runs preprocess + notebook execution and writes images into HTML:

```bash
python run_everything.py --output-dir run_outputs
```

Outputs:
- `run_outputs/preprocessed_output/`
- `run_outputs/duke_modeling_executed.ipynb`
- `run_outputs/duke_modeling_executed.html`
