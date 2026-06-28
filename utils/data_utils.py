b=0#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_utils.py – cleaned-up utilities for Breast-MRI work

Main changes
============
* removed duplicated imports / warnings
* fixed stray “n img” token & bad indent in `get_ser_acquisitions`
* added missing helpers (`cont_br`, `plot_cm`)
* grouped code into clear sections
* added type hints, small doc-strings and safe defaults
"""

# --------------------------------------------------------------------- #
#                               Imports                                 #
# --------------------------------------------------------------------- #


import os, re, warnings, time
from glob   import glob
from pathlib import Path

import numpy  as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from PIL import Image

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------- #
#                         tiny helper-functions                         #
# --------------------------------------------------------------------- #
def cont_br(img: np.ndarray, clip: float = .01) -> np.ndarray:
    """Simple contrast/brightness stretching to [0,1]."""
    lo, hi = np.quantile(img, [clip, 1 - clip])
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo + 1e-8)


def plot_cm(cm: np.ndarray, labels, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) if arr.sum() else arr


# --------------------------------------------------------------------- #
#                          Display helpers                              #
# --------------------------------------------------------------------- #
def show_n_images(
    imgs,
    cmap="gray",
    titles=None,
    enlarge=4,
    mtitle=None,
    cut=False,
    axis_off=True,
    fontsize=15,
    cb=False,
):
    """Visualise *n* images side-by-side."""
    _ = plt.figure(figsize=(enlarge * len(imgs), enlarge * 2))
    for i, im in enumerate(imgs, 1):
        ax = plt.subplot(1, len(imgs), i)
        if cb and len(np.unique(im)) > 5:
            im = cont_br(im)
        ax.imshow(im[50:290, 75:450] if cut else im, cmap=cmap, origin="lower")
        if titles is not None:
            ax.set_title(titles[i - 1], fontsize=fontsize)
        if axis_off:
            ax.axis("off")
    if mtitle:
        plt.suptitle(mtitle, fontsize=fontsize + 2)
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------- #
#           Dataset / file-system constants & path registration         #
# --------------------------------------------------------------------- #
NIFTI_SUFFIX = {
    "spy2": "_spy2_vis1_dce_aqc_",
    "spy1": "_spy1_vis1_acq",
    "duke": "_duke_aqc_",
}
MASK_SUFFIX = {
    "spy2": "_spy2_vis1_mask",
    "spy1": "_spy1_vis1_mask",
    "duke": "_duke_mask",
}

base_path: str | None = None
nifti_path: dict[str, str] = {}
mask_path:  dict[str, str] = {}
_duke_metadata_cache: pd.DataFrame | None = None


def setup_paths(base: str, nifti_dirs: dict[str, str], mask_dirs: dict[str, str]):
    """Register root-folders for NIfTI volumes and masks."""
    global base_path, nifti_path, mask_path
    base_path = base
    nifti_path = nifti_dirs
    mask_path  = mask_dirs
    print("NIfTI roots :", nifti_path)
    print("Mask  roots :", mask_path)


# --------------------------------------------------------------------- #
#                       NIfTI I/O utilities                             #
# --------------------------------------------------------------------- #
def _ds_from_pid(pid: str) -> str:
    if "ISPY1" in pid:
        return "spy1"
    if ("ISPY2" in pid) or ("ACRIN-6698" in pid):
        return "spy2"
    if "Breast_MRI" in pid:
        return "duke"
    raise ValueError(f"Cannot infer dataset for pid={pid!r}")


def _build_file(pid: str, idx: str | int, kind="nifti") -> str:
    ds   = _ds_from_pid(pid)
    root = nifti_path[ds] if kind == "nifti" else mask_path[ds]
    suf  = NIFTI_SUFFIX[ds] if kind == "nifti" else MASK_SUFFIX[ds]
    return os.path.join(root, f"{pid}{suf}{idx}.nii.gz")


def read_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()


def get_nifti_acquisition(pid: str, idx: int = 0):
    f = _build_file(pid, idx)
    return read_nifti(f) if os.path.isfile(f) else None


def _last_int(path):
    m = re.search(r"(\d+)\.nii\.gz$", path)
    return int(m.group(1)) if m else -1


def get_all_nifti_acquisitions(pid,  deb=0):
        if deb: print('get_all_nifti_acquisitions',pid)
        if 'SPY1' in pid: fpath=nifti_path['spy1']
        elif 'SPY2' in pid: fpath=nifti_path['spy2']
        elif 'ACRIN-6698' in pid: fpath=nifti_path['spy2']
        elif 'MRI' in pid: fpath=nifti_path['duke']
        else: return None
        x=os.listdir(fpath)
        if deb: print(pid,fpath)
        x=[c for c in x if pid in c]
        if deb: print(x)
        files = sorted(x, key=_last_int)
        return [read_nifti(os.path.join(fpath,f)) for f in files]


def get_nifti_mask(pid: str, deb=0):
        if 'SPY1' in pid: fpath=mask_path['spy1']
        elif 'SPY2' in pid: fpath=mask_path['spy2']
        elif 'ACRIN-6698' in pid: fpath=mask_path['spy2']
        elif 'MRI' in pid: fpath=mask_path['duke']
        else: return None
        if not os.path.isdir(fpath):
            if deb: print('mask directory missing', pid, fpath)
            return _get_duke_bbox_mask(pid, deb=deb)
        x=os.listdir(fpath)
        if deb: print(pid,fpath)
        x=[c for c in x if pid in c]
        if deb: print(x)
        if len(x) == 0:
            if deb: print('mask not found', pid)
            return _get_duke_bbox_mask(pid, deb=deb)
        if len(x)>1:
            print('error in mask',pid)
        m=read_nifti(os.path.join(fpath,x[0]))
        return m


def _load_duke_metadata() -> pd.DataFrame | None:
        global _duke_metadata_cache
        if _duke_metadata_cache is not None:
            return _duke_metadata_cache
        if base_path is None:
            return None
        candidates = [
            Path(base_path) / "DUKE" / "BreastDCEDL_duke_metadata.csv",
            Path(base_path) / "DUKE" / "BreastDCEDL_duke_cropped.csv",
        ]
        for path in candidates:
            if path.is_file():
                _duke_metadata_cache = pd.read_csv(path)
                return _duke_metadata_cache
        return None


def _get_duke_bbox_mask(pid: str, deb=0):
        if "MRI" not in pid:
            return None
        meta = _load_duke_metadata()
        if meta is None:
            if deb: print("duke metadata missing", pid)
            return None
        row = meta[meta["pid"].astype(str) == str(pid)]
        if row.empty:
            if deb: print("duke bbox row missing", pid)
            return None
        row = row.iloc[0]
        required = ("mask_start", "mask_end", "sraw", "eraw", "scol", "ecol")
        if any(pd.isna(row[c]) for c in required):
            if deb: print("duke bbox columns incomplete", pid)
            return None
        a0 = get_nifti_acquisition(pid, idx=0)
        if a0 is None:
            if deb: print("duke acquisition missing for bbox mask", pid)
            return None
        startm = int(row["mask_start"])
        endm = int(row["mask_end"])
        sraw = int(row["sraw"])
        eraw = int(row["eraw"])
        scol = int(row["scol"])
        ecol = int(row["ecol"])
        mask = np.zeros(a0.shape, dtype=np.uint8)
        mask[startm:min(endm + 1, mask.shape[0]), sraw:min(eraw + 1, mask.shape[1]), scol:min(ecol + 1, mask.shape[2])] = 1
        if deb:
            print("using duke bbox fallback", pid, (startm, endm, sraw, eraw, scol, ecol))
        return mask

# --------------------------------------------------------------------- #
#                 Simple statistics / bounding boxes                    #
# --------------------------------------------------------------------- #
def find_first_last_planes(mask):
    active = np.any(mask, axis=(1, 2))
    idx    = np.where(active)[0]
    return (int(idx[0]), int(idx[-1])) if idx.size else (None, None)


def get_nonzero_bbox(arr):
    pos = np.nonzero(arr)
    if len(pos[0]) == 0:
        return None
    mins = [int(np.min(p)) for p in pos]
    maxs = [int(np.max(p)) + 1 for p in pos]
    return tuple(zip(mins, maxs))


# --------------------------------------------------------------------- #
#                       ML metrics / reporting                          #
# --------------------------------------------------------------------- #
def plot_roc(y_true, y_score, title="ROC"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(3, 3))
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend()
    plt.gca().set_aspect("equal", "box"); plt.tight_layout(); plt.show()


def report_full(
    y_true,
    y_score,
    title="",
    class_names=("Negative", "Positive"),
    thresh=0.5,
    show_conf=True,
    show_roc=True,
):
    y_pred  = (y_score > thresh).astype(int)
    acc     = accuracy_score(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_score)

    print(f"{title}  --  Acc: {acc:.3f}  AUC: {auc_val:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print(pd.DataFrame(cm, index=class_names, columns=class_names), "\n")

    if show_conf:
        plot_cm(cm, class_names)
    if show_roc:
        plot_roc(y_true, y_score, title)


# --------------------------------------------------------------------- #
#                               THE END                                 #
# --------------------------------------------------------------------- #
