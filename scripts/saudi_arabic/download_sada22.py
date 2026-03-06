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

    DATASET_ID = "MohamedRashad/SADA22"

    print(f"  Streaming SADA22 — scanning for speaker={speaker!r}, dialect={dialect!r} ...")
    print("  (Each of the 28 shards may be 100–500 MB; this can take several minutes)")

    # ── PASS 1: scan metadata — NO audio decoding ─────────────────────────────
    # Detect Audio columns by feature TYPE (not name) so we never miss them.
    stream = load_dataset(DATASET_ID, split="train", revision=revision, streaming=True)
    audio_cols = [col for col, feat in stream.features.items() if isinstance(feat, Audio)]
    if audio_cols:
        # Cast to decode=False BEFORE iterating — this prevents torchcodec from
        # being called. remove_columns() alone doesn't help because datasets
        # decodes first, then removes.
        for col in audio_cols:
            stream = stream.cast_column(col, Audio(decode=False))
        stream = stream.remove_columns(audio_cols)

    matched_indices: list[int] = []
    scanned = 0
    with tqdm(desc="  Scanning rows", unit=" rows", dynamic_ncols=True) as pbar:
        for idx, row in enumerate(stream):
            scanned += 1
            pbar.update(1)
            pbar.set_postfix(matched=len(matched_indices), refresh=False)
            spk = (row.get("Speaker") or "").strip()
            dia = (row.get("SpeakerDialect") or "").strip()
            if speaker and spk != speaker:
                continue
            if dialect and dia != dialect:
                continue
            matched_indices.append(idx)
            if max_samples and len(matched_indices) >= max_samples:
                break

    if not matched_indices:
        raise SystemExit(
            f"No rows matched speaker={speaker!r} dialect={dialect!r} after scanning {scanned} rows."
        )

    print(f"  Found {len(matched_indices)} matching rows out of {scanned} scanned.")
    print(f"  Fetching audio for {len(matched_indices)} matched rows ...")

    # ── PASS 2: fetch only the matched rows with raw audio bytes ──────────────
    # select() jumps directly to those indices — no full-dataset rescan needed.
    full_stream = load_dataset(DATASET_ID, split="train", revision=revision, streaming=True)
    for col in [c for c, f in full_stream.features.items() if isinstance(f, Audio)]:
        full_stream = full_stream.cast_column(col, Audio(decode=False))  # raw bytes, no torchcodec
    full_stream = full_stream.select(matched_indices)

    rows: list = []
    for row in tqdm(full_stream, total=len(matched_indices), desc="  Fetching audio", dynamic_ncols=True):
        rows.append(row)

    label = f"train[speaker={speaker or 'all'},dialect={dialect or 'all'},n={len(rows)}]"
    return Dataset.from_list(rows), label


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


def load_kaggle_input(
    kaggle_input_dir: Path,
    output_dir: Path,
    max_samples: int | None,
    speaker: str | None,
    dialect: str | None,
) -> None:
    """Build a HuggingFace Dataset from the pre-mounted Kaggle input dataset.

    Expects the layout::

        /kaggle/input/Sada22/
            train.csv          ← 11 columns (no header) or named header
            batch_1/*.wav
            batch_2/*.wav
            ...

    Column order when headerless:
        audio_path, ProcessedText, TotalDuration, SegmentName,
        SegmentDuration, StartTime, EndTime, AgeGroup, Gender,
        SpeakerDialect, Speaker
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from tqdm import tqdm

    COLS_NO_HEADER = [
        "audio_path", "ProcessedText", "TotalDuration", "SegmentName",
        "SegmentDuration", "StartTime", "EndTime", "AgeGroup", "Gender",
        "SpeakerDialect", "Speaker",
    ]

    csv_path = kaggle_input_dir / "train.csv"
    if not csv_path.exists():
        raise SystemExit(f"train.csv not found at {csv_path}")

    # Auto-detect header: peek at the first cell
    with open(csv_path, encoding="utf-8") as fh:
        first_cell = fh.readline().split(",")[0].strip().lower()
    has_header = first_cell in ("audio_path", "audio", "filepath", "path", "processedtext")

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)
        ncols = df.shape[1]
        if ncols == len(COLS_NO_HEADER):
            df.columns = COLS_NO_HEADER
        elif ncols == len(COLS_NO_HEADER) + 1:
            # Leading row-index column emitted by some exporters
            df.columns = ["_idx"] + COLS_NO_HEADER
            df = df.drop(columns=["_idx"])
        else:
            raise SystemExit(
                f"train.csv has {ncols} columns — expected {len(COLS_NO_HEADER)} or "
                f"{len(COLS_NO_HEADER) + 1} (with leading index).\n"
                f"First row: {list(df.iloc[0])}"
            )

    # Flexible column lookup (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}

    def find_col(*candidates: str) -> str | None:
        for name in candidates:
            if name.lower() in col_map:
                return col_map[name.lower()]
        return None

    audio_col = find_col("audio_path", "audio", "filepath", "path")
    text_col  = find_col("processedtext", "text", "transcript")
    spk_col   = find_col("speaker", "speaker_id")
    dia_col   = find_col("speakerdialect", "dialect")

    if not audio_col:
        raise SystemExit(f"Cannot find audio column. Columns present: {list(df.columns)}")

    # ── Filter ────────────────────────────────────────────────────────────────
    if speaker and spk_col:
        df = df[df[spk_col].astype(str).str.strip() == speaker.strip()]
    if dialect and dia_col:
        df = df[df[dia_col].astype(str).str.strip() == dialect.strip()]
    if len(df) == 0:
        raise SystemExit(
            f"No rows matched speaker={speaker!r} dialect={dialect!r}.\n"
            f"Available speakers: {list(df[spk_col].unique()) if spk_col else 'unknown'}"
        )
    if max_samples:
        df = df.head(max_samples)

    print(f"  Found {len(df)} matching rows (speaker={speaker!r}, dialect={dialect!r}).")
    print("  Building dataset — referencing audio by path (no copy needed) ...")

    # ── Build rows — audio stored as path dict ────────────────────────────────
    # preprocess_sada22.py already handles {"path": ..., "bytes": None} format.
    extra_cols = [c for c in df.columns if c not in [audio_col, text_col, spk_col, dia_col]]
    rows: list[dict] = []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Indexing rows"):
        audio_path = kaggle_input_dir / str(row[audio_col]).lstrip("/")
        if not audio_path.exists():
            missing += 1
            continue
        entry: dict = {
            "audio": {"path": str(audio_path), "bytes": None},
            "ProcessedText": str(row[text_col]) if text_col else "",
            "Speaker":        str(row[spk_col])  if spk_col  else "",
            "SpeakerDialect": str(row[dia_col])  if dia_col  else "",
        }
        for c in extra_cols:
            entry[c] = str(row.get(c, ""))
        rows.append(entry)

    if missing:
        print(f"  Warning: {missing} audio file(s) not found on disk — skipped.")
    if not rows:
        raise SystemExit("No valid audio entries after path check.")

    # ── Save to disk as HuggingFace Dataset ───────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    DatasetDict({"train": ds}).save_to_disk(str(output_dir))

    manifest = {
        "source": "kaggle-input",
        "input_dir": str(kaggle_input_dir.resolve()),
        "csv": str(csv_path),
        "speaker_filter": speaker,
        "dialect_filter": dialect,
        "rows": len(rows),
        "saved_to": str(output_dir.resolve()),
    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  ✓ Saved {len(rows)} rows → {output_dir}")


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
    parser.add_argument("--source", choices=["hf", "kaggle", "kaggle-input"], default="hf")
    parser.add_argument(
        "--kaggle-input-dir",
        type=Path,
        default=Path("/kaggle/input/datasets/sdaiancai/sada2022"),
        help="Path to the pre-mounted Kaggle input dataset (used with --source kaggle-input)",
    )
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
    elif args.source == "kaggle-input":
        load_kaggle_input(args.kaggle_input_dir, args.output_dir, args.max_samples, speaker=args.speaker, dialect=args.dialect)
    else:
        download_kaggle(args.output_dir, args.kaggle_dataset)

    print(f"Done. Metadata saved to {args.output_dir / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
