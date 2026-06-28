#!/usr/bin/env python3
"""Preprocess the full BreastDCEDL dataset into per-patient artifacts.

This script keeps the current preprocessing logic intact:
- I-SPY1 / I-SPY2: use real masks when available
- Duke: use bbox fallback from metadata if mask files are absent

Output per patient:
- cropped acquisitions saved as .npz
- a compact RGB-like preview saved as .npy
- one manifest CSV for the whole run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import data_utils as ds


def dataset_roots(repo_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    dce = {
        "spy2": str(repo_root / "ISPY2" / "data_samples" / "dce"),
        "spy1": str(repo_root / "ISPY1" / "data_samples" / "dce"),
        "duke": str(repo_root / "DUKE" / "data_samples" / "dce"),
    }
    mask = {
        "spy2": str(repo_root / "ISPY2" / "data_samples" / "mask"),
        "spy1": str(repo_root / "ISPY1" / "data_samples" / "mask"),
        "duke": str(repo_root / "DUKE" / "data_samples" / "mask"),
    }
    return dce, mask


def infer_dataset(pid: str) -> str:
    if "ISPY1" in pid:
        return "spy1"
    if "ISPY2" in pid or "ACRIN-6698" in pid:
        return "spy2"
    if "Breast_MRI" in pid:
        return "duke"
    raise ValueError(f"Cannot infer dataset from pid={pid}")


def safe_int(value, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def crop_center(shape: tuple[int, int, int], crop_size: int) -> tuple[int, int, int, int, int, int]:
    z, y, x = shape
    half = crop_size // 2
    cz = z // 2
    cy = y // 2
    cx = x // 2
    z0 = max(0, cz - 1)
    z1 = min(z - 1, cz + 1)
    y0 = max(0, cy - half)
    y1 = min(y - 1, cy + half)
    x0 = max(0, cx - half)
    x1 = min(x - 1, cx + half)
    return z0, z1, y0, y1, x0, x1


def crop_bbox_from_row(row, shape: tuple[int, int, int], crop_size: int) -> tuple[int, int, int, int, int, int]:
    required = ("mask_start", "mask_end", "sraw", "eraw", "scol", "ecol")
    if all(col in row.index and not pd.isna(row[col]) for col in required):
        z0 = int(row["mask_start"])
        z1 = int(row["mask_end"])
        y0 = int(row["sraw"])
        y1 = int(row["eraw"])
        x0 = int(row["scol"])
        x1 = int(row["ecol"])
        return z0, z1, y0, y1, x0, x1
    return crop_center(shape, crop_size)


def pick_three_indices(row, total: int) -> list[int]:
    pre = safe_int(row.get("pre", 0), 0)
    early = safe_int(row.get("post_early", 1), min(1, total - 1))
    late = safe_int(row.get("post_late", total - 1), total - 1)
    indices = [pre, early, late]
    out = []
    for idx in indices:
        out.append(max(0, min(total - 1, idx)))
    return out


def stack_preview(volumes: list[np.ndarray]) -> np.ndarray:
    mid = volumes[0].shape[0] // 2
    slices = [ds.minmax(v[mid]) for v in volumes]
    while len(slices) < 3:
        slices.append(slices[-1].copy())
    return np.stack(slices[:3], axis=-1)


def process_patient(pid: str, row: pd.Series, out_dir: Path, crop_size: int):
    acquisitions = ds.get_all_nifti_acquisitions(pid)
    if not acquisitions:
        return None

    ds_name = infer_dataset(pid)
    mask = ds.get_nifti_mask(pid)

    ref = acquisitions[0]
    if mask is not None and np.any(mask):
        nonzero = np.argwhere(mask > 0)
        z0, y0, x0 = nonzero.min(axis=0)
        z1, y1, x1 = nonzero.max(axis=0)
        crop_coords = (
            int(z0),
            int(z1),
            int(y0),
            int(y1),
            int(x0),
            int(x1),
        )
    else:
        crop_coords = crop_bbox_from_row(row, ref.shape, crop_size)

    z0, z1, y0, y1, x0, x1 = crop_coords
    cropped = [acq[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] for acq in acquisitions]
    preview = stack_preview(cropped[:3])

    patient_dir = out_dir / ds_name / pid
    patient_dir.mkdir(parents=True, exist_ok=True)
    np.save(patient_dir / "preview.npy", preview)
    np.savez_compressed(
        patient_dir / "cropped_acquisitions.npz",
        **{f"acq_{i}": arr for i, arr in enumerate(cropped)},
        crop_coords=np.asarray(crop_coords, dtype=np.int32),
    )

    return {
        "pid": pid,
        "dataset": ds_name,
        "timepoints": len(cropped),
        "shape_z": int(cropped[0].shape[0]),
        "shape_y": int(cropped[0].shape[1]),
        "shape_x": int(cropped[0].shape[2]),
        "has_mask": bool(mask is not None),
        "preview_path": str(patient_dir / "preview.npy"),
        "npz_path": str(patient_dir / "cropped_acquisitions.npz"),
        "crop_z0": int(z0),
        "crop_z1": int(z1),
        "crop_y0": int(y0),
        "crop_y1": int(y1),
        "crop_x0": int(x0),
        "crop_x1": int(x1),
        "selected_indices": ",".join(map(str, pick_three_indices(row, len(acquisitions)))),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess full BreastDCEDL dataset")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--metadata", type=Path, default=None, help="Metadata CSV to use")
    parser.add_argument("--output-dir", type=Path, default=Path("preprocessed_output"))
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of patients for smoke testing")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    metadata_path = args.metadata or (repo_root / "BreastDCEDL_metadata.csv")
    df = pd.read_csv(metadata_path)

    dce_roots, mask_roots = dataset_roots(repo_root)
    ds.setup_paths(str(repo_root), dce_roots, mask_roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        if args.limit and idx >= args.limit:
            break
        pid = str(row["pid"])
        try:
            result = process_patient(pid, row, args.output_dir, args.crop_size)
            if result is not None:
                rows.append(result)
                print("OK", pid, result["dataset"], result["timepoints"], result["preview_path"])
        except Exception as exc:
            rows.append({"pid": pid, "error": str(exc)})
            print("ERROR", pid, exc)

    manifest = pd.DataFrame(rows)
    manifest_path = args.output_dir / "preprocess_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("\nManifest:", manifest_path)
    print("Patients processed:", len(manifest))


if __name__ == "__main__":
    main()
