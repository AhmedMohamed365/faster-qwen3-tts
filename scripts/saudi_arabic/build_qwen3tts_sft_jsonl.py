#!/usr/bin/env python3
"""Convert normalized manifest into an SFT JSONL layout for Qwen3-TTS training scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/sada22_qwen3tts/train_manifest.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/sada22_qwen3tts/train_qwen3tts_sft.jsonl"))
    p.add_argument("--system-prompt", type=str, default="Speak naturally in Saudi Arabic dialect.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.manifest.parent
    rows = []

    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            audio_path = (base / rec["audio"]).as_posix()

            sample = {
                "id": rec["id"],
                "audio": audio_path,
                "text": rec["text"],
                "metadata": {
                    "language": rec.get("language", "Arabic"),
                    "locale": rec.get("locale", "ar-SA"),
                    "speaker": rec.get("speaker", "unknown"),
                },
                # Many community Qwen3-TTS SFT scripts expect instruction-style fields.
                "messages": [
                    {"role": "system", "content": args.system_prompt},
                    {
                        "role": "user",
                        "content": f"Generate speech in Saudi Arabic for: {rec['text']}",
                    },
                    {"role": "assistant", "content": "<speech>"},
                ],
            }
            rows.append(sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
