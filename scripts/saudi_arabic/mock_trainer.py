#!/usr/bin/env python3
"""Smoke trainer to validate generated config/dataset plumbing.

This is NOT real Qwen3-TTS training; it just proves that the generated JSONL can be
loaded and one optimizer step can run end-to-end (CPU or GPU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for the smoke train step.",
    )
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is not available (for GPU CI validation).",
    )
    return p.parse_args()


def load_first_sample(train_file: Path) -> dict:
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError(f"No training rows in {train_file}")


def resolve_device(requested: str, require_cuda: bool) -> torch.device:
    cuda_ok = torch.cuda.is_available()
    if require_cuda and not cuda_ok:
        raise RuntimeError("CUDA was required but is not available in this environment.")

    if requested == "cuda":
        if not cuda_ok:
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if cuda_ok else "cpu")


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    train_file = Path(cfg["train_file"])
    if not train_file.exists():
        raise FileNotFoundError(f"train_file not found: {train_file}")

    device = resolve_device(args.device, args.require_cuda)
    sample = load_first_sample(train_file)

    text = sample.get("text", "")
    length_feature = torch.tensor([[float(len(text))]], dtype=torch.float32, device=device)
    target = torch.tensor([[1.0]], dtype=torch.float32, device=device)

    model = torch.nn.Linear(1, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    pred = model(length_feature)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()
    optimizer.step()

    print(
        f"FIRST_TRAIN_STEP_OK device={device.type} loss={loss.item():.6f} "
        f"sample_id={sample.get('id', 'unknown')}"
    )


if __name__ == "__main__":
    main()
