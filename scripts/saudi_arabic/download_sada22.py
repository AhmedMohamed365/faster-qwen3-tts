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


def _load_hf_subset(
    revision: str | None,
    max_samples: int | None,
    speaker: str | None = None,
    dialect: str | None = None,
):
    from datasets import Audio, Dataset, load_dataset
    from tqdm import tqdm

    print(f"  Streaming SADA22 — scanning for speaker={speaker!r}, dialect={dialect!r} ...")
    print("  (Each of the 28 shards may be 100–500 MB; this can take several minutes)")

    # Stream WITHOUT audio so each row is just metadata — much faster to scan.
    stream = load_dataset("MohamedRashad/SADA22", split="train", revision=revision, streaming=True)
    audio_cols = [c for c in stream.features if "audio" in c.lower()]
    if audio_cols:
        stream = stream.remove_columns(audio_cols)

    matched_rows: list = []
    scanned = 0
    with tqdm(desc="  Scanning rows", unit=" rows", dynamic_ncols=True) as pbar:
        for row in stream:
            scanned += 1
            pbar.update(1)
            pbar.set_postfix(matched=len(matched_rows), refresh=False)
            # Speaker / dialect filtering (uses SADA22's actual column names).
            if speaker and row.get("Speaker") != speaker:
                continue
            if dialect and row.get("SpeakerDialect") != dialect:
                continue
            matched_rows.append(row)
            if max_samples and len(matched_rows) >= max_samples:
                break

    if not matched_rows:
        raise SystemExit(
            f"No rows matched speaker={speaker!r} dialect={dialect!r} after scanning {scanned} rows."
        )

    print(f"  Found {len(matched_rows)} matching rows out of {scanned} scanned.")
    print("  Building Dataset and fetching audio for matched rows ...")

    # Build metadata-only Dataset, then fetch audio for just those rows.
    meta_ds = Dataset.from_list(matched_rows)

    # Re-stream to collect audio bytes only for matched rows (by text key).
    text_col = "ProcessedText" if "ProcessedText" in meta_ds.column_names else "text"
    matched_texts = set(meta_ds[text_col]) if text_col in meta_ds.column_names else set()

    audio_stream = load_dataset(
        "MohamedRashad/SADA22", split="train", revision=revision, streaming=True
    )
    audio_stream = audio_stream.cast_column("audio", Audio(decode=False))

    audio_map: dict = {}
    needed = len(matched_rows)
    with tqdm(desc="  Fetching audio", unit=" rows", total=needed, dynamic_ncols=True) as pbar2:
        for row in audio_stream:
            key = row.get(text_col, "")
            if key in matched_texts and key not in audio_map:
                audio_map[key] = row.get("audio")
                pbar2.update(1)
                if len(audio_map) >= needed:
                    break

    # Attach audio bytes back to metadata rows.
    rows_with_audio = []
    for r in matched_rows:
        key = r.get(text_col, "")
        rows_with_audio.append({**r, "audio": audio_map.get(key)})

    label = f"train[speaker={speaker or 'all'},dialect={dialect or 'all'},n={len(rows_with_audio)}]"
    return Dataset.from_list(rows_with_audio), label


def download_hf(
    output_dir: Path,
    revision: str | None,
    max_samples: int | None,
    speaker: str | None = None,
    dialect: str | None = None,
) -> None:
    try:
        from datasets import DatasetDict
    except ImportError as exc:
        raise SystemExit(
            "`datasets` is required for Hugging Face downloads. Install with: pip install datasets"
        ) from exc

    train_ds, split = _load_hf_subset(revision, max_samples, speaker=speaker, dialect=dialect)
    if len(train_ds) == 0:
        raise SystemExit(
            f"No rows matched speaker={speaker!r} dialect={dialect!r}. "
            "Check the column values in the dataset."
        )
    dataset = DatasetDict({"train": train_ds})
    dataset.save_to_disk(str(output_dir))

    manifest = {
        "source": "huggingface",
        "dataset": "MohamedRashad/SADA22",
        "revision": revision,
        "split": split,
        "speaker_filter": speaker,
        "dialect_filter": dialect,
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
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Keep only rows where Speaker == VALUE (SADA22 column). E.g. 'Speaker1متحدث'",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default=None,
        help="Keep only rows where SpeakerDialect == VALUE (SADA22 column). E.g. 'Najdi'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        download_hf(args.output_dir, args.revision, args.max_samples, speaker=args.speaker, dialect=args.dialect)
    else:
        download_kaggle(args.output_dir, args.kaggle_dataset)

    print(f"Done. Metadata saved to {args.output_dir / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
