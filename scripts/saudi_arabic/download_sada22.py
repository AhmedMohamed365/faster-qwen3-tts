#!/usr/bin/env python3
"""Download SADA22 from Hugging Face (preferred) or Kaggle mirror."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _load_hf_subset(revision: str | None, max_samples: int | None):
    from datasets import Audio, Dataset, load_dataset

    if max_samples:
        stream = load_dataset("MohamedRashad/SADA22", split="train", revision=revision, streaming=True)
        stream = stream.cast_column("audio", Audio(decode=False))
        rows = []
        for idx, row in enumerate(stream):
            if idx >= max_samples:
                break
            rows.append(row)
        return Dataset.from_list(rows), f"train[:{max_samples}]"

    ds = load_dataset("MohamedRashad/SADA22", split="train", revision=revision)
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds, "train"


def download_hf(output_dir: Path, revision: str | None, max_samples: int | None) -> None:
    try:
        from datasets import DatasetDict
    except ImportError as exc:
        raise SystemExit(
            "`datasets` is required for Hugging Face downloads. Install with: pip install datasets"
        ) from exc

    train_ds, split = _load_hf_subset(revision, max_samples)
    dataset = DatasetDict({"train": train_ds})
    dataset.save_to_disk(str(output_dir))

    manifest = {
        "source": "huggingface",
        "dataset": "MohamedRashad/SADA22",
        "revision": revision,
        "split": split,
        "rows": len(train_ds),
        "saved_to": str(output_dir.resolve()),
    }
    (output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_kaggle(output_dir: Path, kaggle_dataset: str) -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Kaggle CLI not found. Install with `pip install kaggle` and configure ~/.kaggle/kaggle.json"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(output_dir), "-o"])
    zip_path = next(output_dir.glob("*.zip"), None)
    if zip_path is None:
        raise SystemExit(f"No zip archive downloaded in {output_dir}")
    run(["unzip", "-o", str(zip_path), "-d", str(output_dir / "raw")])

    manifest = {
        "source": "kaggle",
        "dataset": kaggle_dataset,
        "archive": str(zip_path.resolve()),
        "extract_dir": str((output_dir / "raw").resolve()),
    }
    (output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sada22"))
    parser.add_argument("--source", choices=["hf", "kaggle"], default="hf")
    parser.add_argument("--revision", type=str, default=None, help="HF revision/commit for reproducibility")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit rows downloaded from Hugging Face for fast smoke tests (ignored for Kaggle).",
    )
    parser.add_argument(
        "--kaggle-dataset",
        default="sdaiancai/sada2022",
        help="Kaggle dataset slug (owner/name) used when --source kaggle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        download_hf(args.output_dir, args.revision, args.max_samples)
    else:
        download_kaggle(args.output_dir, args.kaggle_dataset)

    print(f"Done. Metadata saved to {args.output_dir / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
