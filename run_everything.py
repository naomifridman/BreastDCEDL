#!/usr/bin/env python3
"""One-click end-to-end runner for BreastDCEDL.

Runs:
1) full preprocessing
2) Duke modeling notebook execution

The notebook execution generates the tables/plots shown in the report and
saves an executed notebook plus HTML with embedded figures.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs"))
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-notebook", action="store_true")
    parser.add_argument("--preprocess-limit", type=int, default=0, help="Optional smoke-test limit for preprocess")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocess:
        preprocess_cmd = [
            sys.executable,
            str(repo_root / "preprocess_full_dataset.py"),
            "--output-dir",
            str(args.output_dir / "preprocessed_output"),
        ]
        if args.preprocess_limit > 0:
            preprocess_cmd += ["--limit", str(args.preprocess_limit)]
        run(preprocess_cmd, cwd=repo_root)

    if not args.skip_notebook:
        duke_dir = repo_root / "DUKE"
        executed_name = "duke_modeling_executed"
        executed_ipynb = args.output_dir / f"{executed_name}.ipynb"
        html_output = args.output_dir / f"{executed_name}.html"

        run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=-1",
                "--output-dir",
                str(args.output_dir),
                "--output",
                executed_name,
                "duke_modeling_with_niftii_files.ipynb",
            ],
            cwd=duke_dir,
        )

        run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                str(executed_ipynb),
                "--output-dir",
                str(args.output_dir),
            ]
        )

        print("\nNotebook outputs:")
        print(f"- {executed_ipynb}")
        print(f"- {html_output}")

    print("\nDone.")


if __name__ == "__main__":
    main()

