#!/usr/bin/env python3
"""Launch Qwen3-TTS fine-tuning (full or LoRA) with Accelerate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, default=Path("data/sada22_qwen3tts/train_qwen3tts_sft.jsonl"))
    p.add_argument("--val-jsonl", type=Path, default=Path("data/sada22_qwen3tts/val_qwen3tts_sft.jsonl"))
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-TTS-0.6B-Base")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3tts-sada22"))
    p.add_argument("--mode", choices=["full", "lora"], default="lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=16.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--trainer-entrypoint",
        type=str,
        default="qwen_tts.finetune.train_sft",
        help="Python module path shipped by your Qwen3-TTS training environment.",
    )
    return p.parse_args()


def build_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    return [
        "accelerate",
        "launch",
        "-m",
        args.trainer_entrypoint,
        "--config",
        str(config_path),
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "base_model": args.base_model,
        "train_file": str(args.train_jsonl),
        "validation_file": str(args.val_jsonl),
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_audio_seconds": args.max_seconds,
        "bf16": args.bf16,
        "save_strategy": "steps",
        "save_steps": 500,
        "logging_steps": 20,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "dialect": "ar-SA",
    }

    if args.mode == "lora":
        cfg["peft"] = {
            "type": "lora",
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
    else:
        cfg["peft"] = None

    cfg_path = args.output_dir / f"train_config_{args.mode}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    cmd = build_command(args, cfg_path)
    print(" ".join(cmd))

    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
