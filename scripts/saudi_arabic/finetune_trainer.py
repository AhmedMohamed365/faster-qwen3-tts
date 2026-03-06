#!/usr/bin/env python3
"""
Standalone HuggingFace-Trainer fallback fine-tuner for Qwen3-TTS.

Used when `qwen_tts.finetune.train_sft` is not found in the installed
version of qwen-tts.  Applies LoRA (or full fine-tuning) to the talker LM
using the transformers Trainer + PEFT.

Usage (mirrors train_qwen3tts.py CLI):
    python scripts/saudi_arabic/finetune_trainer.py \
        --config outputs/qwen3tts-sada22-lora/train_config_lora.json

Or directly:
    python scripts/saudi_arabic/finetune_trainer.py \
        --train-jsonl data/sada22_small_qwen3tts/train_qwen3tts_sft.jsonl \
        --val-jsonl   data/sada22_small_qwen3tts/val_qwen3tts_sft.jsonl \
        --base-model  Qwen/Qwen3-TTS-0.6B-Base \
        --output-dir  outputs/qwen3tts-sada22-lora \
        --max-steps   50 \
        --mode        lora \
        --bf16
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_SR = 24_000
MAX_AUDIO_SECONDS = 16.0
MAX_TEXT_TOKENS = 256
CODEBOOK_SIZE = 4096  # Qwen3-TTS speech tokenizer vocab per codebook


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Sada22SFTDataset(Dataset):
    """
    Reads SFT JSONL produced by build_qwen3tts_sft_jsonl.py.

    Each record has:  id, audio (path), text, messages, metadata

    We return (text, audio_path) pairs; collation handles tokenization.
    """

    def __init__(self, jsonl_path: Path, base_dir: Optional[Path] = None) -> None:
        self.records: list[dict[str, Any]] = []
        self.base_dir = base_dir or jsonl_path.parent

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.records.append(rec)

        log.info("Loaded %d samples from %s", len(self.records), jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        audio_path = Path(rec["audio"])
        if not audio_path.is_absolute():
            audio_path = self.base_dir / audio_path
        return {"text": rec["text"], "audio_path": str(audio_path), "id": rec["id"]}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_audio_24k(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        src_t = np.linspace(0, 1, len(wav), endpoint=False)
        dst_n = int(math.ceil(len(wav) * TARGET_SR / sr))
        dst_t = np.linspace(0, 1, dst_n, endpoint=False)
        wav = np.interp(dst_t, src_t, wav).astype(np.float32)
    # Clip to max length
    max_samples = int(MAX_AUDIO_SECONDS * TARGET_SR)
    wav = wav[:max_samples]
    return wav


# ---------------------------------------------------------------------------
# Collator: builds model inputs from (text, audio_path) pairs
# ---------------------------------------------------------------------------

class TalkerCollator:
    """
    Tokenises text with the Qwen3-TTS thinker tokenizer,
    encodes audio to codec tokens via the speech tokenizer,
    and builds input_ids + labels for causal-LM training.

    The sequence layout (flattened over codebooks 1..16 interleaved):
        [text_token_ids ...] [task_token] [codec_flat ...] [eos]

    Only codec tokens are used in the loss (text tokens → -100).
    """

    def __init__(self, thinker_tokenizer, speech_tokenizer, device: str = "cpu") -> None:
        self.tok = thinker_tokenizer
        self.speech_tok = speech_tokenizer
        self.device = device

        # Special tokens used by qwen-tts (best-effort; fall back to generic)
        self._tts_start = getattr(self.tok, "tts_start_token_id", None)
        self._tts_end = getattr(self.tok, "tts_end_token_id", None)
        if self._tts_start is None:
            tts_start_ids = self.tok.convert_tokens_to_ids(["<|tts_bos|>"])
            self._tts_start = tts_start_ids[0] if tts_start_ids[0] != self.tok.unk_token_id else None
        if self._tts_end is None:
            tts_end_ids = self.tok.convert_tokens_to_ids(["<|tts_eos|>"])
            self._tts_end = tts_end_ids[0] if tts_end_ids[0] != self.tok.unk_token_id else None

    def _encode_audio(self, wav: np.ndarray) -> torch.Tensor:
        """
        Returns codec IDs of shape [n_codebooks, T].
        Handles both raw ndarray and tensor inputs from different speech_tokenizer
        implementations found in qwen-tts releases.
        """
        wav_t = torch.from_numpy(wav).float()
        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)  # [1, T]

        with torch.no_grad():
            # Try the standard qwen-tts API
            try:
                # speech_tokenizer.encode expects [B, T] or [T]
                wav_on_device = wav_t.to(next(iter(self.speech_tok.parameters()), wav_t).device) if hasattr(self.speech_tok, 'parameters') else wav_t
                codes = self.speech_tok.encode(wav_on_device)   # [n_codebooks, T]
            except Exception:
                try:
                    codes = self.speech_tok.encode(wav_t.squeeze(0))
                except Exception as exc:
                    raise RuntimeError(f"speech_tokenizer.encode failed: {exc}") from exc

        if isinstance(codes, (list, tuple)):
            codes = codes[0]  # some impls return (codes, lengths)
        if not isinstance(codes, torch.Tensor):
            codes = torch.tensor(codes)
        return codes.long()  # [n_codebooks, T]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for item in batch:
            try:
                wav = load_audio_24k(item["audio_path"])
                codes = self._encode_audio(wav)        # [n_codebooks, T]
            except Exception as exc:
                log.warning("Skipping %s: %s", item["id"], exc)
                continue

            # Tokenise text
            text_ids = self.tok(
                item["text"],
                add_special_tokens=True,
                max_length=MAX_TEXT_TOKENS,
                truncation=True,
            ).input_ids

            # Flatten codec tokens: interleave codebooks column-by-column → [n_codebooks * T]
            # (simple approach: concatenate codebook-0 then codebook-1 ... )
            codec_flat = codes.flatten().tolist()   # [n_codebooks * T]

            # Shift codec IDs into a separate vocab range if the model uses them inline.
            # For Qwen3-TTS the speech vocab is offset from text vocab; we use
            # the raw codec IDs here since the talker head maps 0..4095 codec space.
            # (finetune_trainer trains the talker directly; text tokens are frozen.)

            seq_ids: list[int] = []
            seq_ids.extend(text_ids)
            if self._tts_start is not None:
                seq_ids.append(self._tts_start)
            seq_ids.extend(codec_flat)
            if self._tts_end is not None:
                seq_ids.append(self._tts_end)

            # Labels: mask text portion with -100; only compute loss on codec tokens
            lab_ids: list[int] = [-100] * len(text_ids)
            if self._tts_start is not None:
                lab_ids.append(-100)            # don't predict the task start token
            lab_ids.extend(codec_flat)
            if self._tts_end is not None:
                lab_ids.append(self._tts_end)

            input_ids_list.append(torch.tensor(seq_ids, dtype=torch.long))
            labels_list.append(torch.tensor(lab_ids, dtype=torch.long))

        if not input_ids_list:
            # Return a dummy batch so Trainer doesn't crash on a bad batch
            dummy = torch.zeros(1, 4, dtype=torch.long)
            return {"input_ids": dummy, "labels": dummy.clone(), "attention_mask": torch.ones(1, 4)}

        # Pad to longest in batch
        max_len = max(t.size(0) for t in input_ids_list)
        pad_id = self.tok.pad_token_id or 0

        input_ids = torch.stack(
            [torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=pad_id) for t in input_ids_list]
        )
        labels = torch.stack(
            [torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=-100) for t in labels_list]
        )
        attention_mask = (input_ids != pad_id).long()

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model_and_components(base_model: str, bf16: bool):
    """
    Load Qwen3TTSModel via qwen_tts.
    Returns (talker, text_tokenizer, speech_tokenizer, full_wrapper).

    Model path (as confirmed from faster_qwen3_tts/model.py):
      wrapper                       = Qwen3TTSModel (inference wrapper)
      wrapper.model                 = Qwen3TTSForConditionalGeneration
      wrapper.model.talker          = Qwen3TTSTalkerForConditionalGeneration  ← fine-tune this
      wrapper.model.speech_tokenizer= Qwen3TTSTokenizer (VQ codec)
      wrapper.processor             = Qwen3TTSProcessor (has .tokenizer for text)
    """
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise SystemExit("qwen-tts is required. Install with: uv pip install qwen-tts") from exc

    log.info("Loading Qwen3TTSModel from %s …", base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if bf16 else torch.float32

    # Pass token explicitly so the model loads even when HF_HOME is
    # overridden (token may be in HF_TOKEN env var rather than the cache).
    import os as _os
    _hf_token = _os.environ.get("HF_TOKEN") or None

    wrapper = Qwen3TTSModel.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
        token=_hf_token,
    )

    full_model = wrapper.model  # Qwen3TTSForConditionalGeneration
    talker = full_model.talker  # Qwen3TTSTalkerForConditionalGeneration

    speech_tokenizer = full_model.speech_tokenizer
    if speech_tokenizer is None:
        raise RuntimeError("speech_tokenizer was not loaded by from_pretrained. "
                           "Make sure you are loading from an official Qwen3-TTS checkpoint.")

    # Processor has an inner text tokenizer
    processor = wrapper.processor
    text_tokenizer = getattr(processor, "tokenizer", None) or getattr(processor, "text_tokenizer", None)
    if text_tokenizer is None:
        # Fallback: use AutoTokenizer with the same path
        from transformers import AutoTokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(base_model)

    log.info("Talker type: %s", type(talker).__name__)
    log.info("Speech tokenizer type: %s", type(speech_tokenizer).__name__)
    log.info("Text tokenizer type: %s", type(text_tokenizer).__name__)

    return talker, text_tokenizer, speech_tokenizer, wrapper


def apply_lora(talker, lora_cfg: dict):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit("peft is required for LoRA. Install with: uv pip install peft") from exc

    config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    talker = get_peft_model(talker, config)
    talker.print_trainable_parameters()
    return talker


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="Path to JSON config written by train_qwen3tts.py")
    p.add_argument("--train-jsonl", type=Path)
    p.add_argument("--val-jsonl", type=Path)
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-TTS-0.6B-Base")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/qwen3tts-sada22-lora"))
    p.add_argument("--mode", choices=["full", "lora"], default="lora")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--logging-steps", type=int, default=5)
    return p.parse_args()


def merge_config_into_args(args: argparse.Namespace) -> argparse.Namespace:
    """If --config is given, override defaults with values from that JSON."""
    if args.config is None:
        return args
    cfg = json.loads(args.config.read_text())
    if args.train_jsonl is None:
        args.train_jsonl = Path(cfg["train_file"])
    if args.val_jsonl is None:
        args.val_jsonl = Path(cfg.get("validation_file", cfg.get("val_file", "")))
    args.base_model = cfg.get("base_model", args.base_model)
    args.output_dir = Path(cfg.get("output_dir", str(args.output_dir)))
    args.lr = float(cfg.get("learning_rate", args.lr))
    args.batch_size = int(cfg.get("per_device_train_batch_size", args.batch_size))
    args.grad_accum = int(cfg.get("gradient_accumulation_steps", args.grad_accum))
    args.bf16 = cfg.get("bf16", args.bf16)
    if "max_steps" in cfg and int(cfg["max_steps"]) > 0 and args.max_steps == 50:
        args.max_steps = int(cfg["max_steps"])
    if "num_train_epochs" in cfg:
        args.epochs = int(cfg["num_train_epochs"])
    if "peft" in cfg and cfg["peft"] is not None:
        args.mode = "lora"
    else:
        args.mode = "full"
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args = merge_config_into_args(args)

    if args.train_jsonl is None:
        raise SystemExit("--train-jsonl is required (or pass --config pointing to a JSON config file)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device / dtype selection (works on CPU and CUDA)
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 only on Ampere+ CUDA; FP32 on CPU or older GPUs
    bf16 = args.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = (device == "cuda") and not bf16   # fp16 as fallback on older CUDA
    log.info("Device: %s  | bf16: %s  | fp16: %s", device, bf16, use_fp16)

    # ------------------------------------------------------------------
    # Load model (device/dtype handled inside)
    # ------------------------------------------------------------------
    talker, text_tokenizer, speech_tokenizer, wrapper = load_model_and_components(args.base_model, bf16)

    if args.mode == "lora":
        lora_cfg = {
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        talker = apply_lora(talker, lora_cfg)

    # Move speech_tokenizer to device; talker is already on device via device_map
    try:
        speech_tokenizer.to(device)
    except Exception:
        pass  # some speech tokenizer impls are not nn.Module
    if next(talker.parameters(), None) is not None:
        talker.to(device)
    log.info("Training on device: %s  |  bf16: %s  |  fp16: %s", device, bf16, use_fp16)

    # ------------------------------------------------------------------
    # Datasets & collator
    # ------------------------------------------------------------------
    train_ds = Sada22SFTDataset(args.train_jsonl)
    val_ds = Sada22SFTDataset(args.val_jsonl) if args.val_jsonl and args.val_jsonl.exists() else None

    collator = TalkerCollator(text_tokenizer, speech_tokenizer, device=device)

    # ------------------------------------------------------------------
    # Trainer setup
    # ------------------------------------------------------------------
    from transformers import TrainingArguments, Trainer

    # Compute steps
    effective_max_steps = args.max_steps if args.max_steps > 0 else -1
    save_steps = max(10, (args.max_steps // 2) if args.max_steps > 0 else 500)
    eval_steps = save_steps
    logging_steps = max(1, min(args.logging_steps, save_steps // 2))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs if effective_max_steps < 0 else 100,  # large; capped by max_steps
        max_steps=effective_max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=bf16,
        fp16=use_fp16,  # fp16 fallback on CUDA without bf16 support; always False on CPU
        no_cuda=(device == "cpu"),  # ensure HF Trainer doesn't try to use GPU when on CPU
        logging_dir=str(args.output_dir / "logs"),
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        evaluation_strategy="steps" if val_ds else "no",
        eval_steps=eval_steps if val_ds else None,
        report_to="none",       # no W&B / wandb needed
        load_best_model_at_end=False,
        dataloader_num_workers=0,            # avoid mp issues in first run
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=talker,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=text_tokenizer,
    )

    log.info("Starting training: max_steps=%s, batch=%s, grad_accum=%s",
             effective_max_steps, args.batch_size, args.grad_accum)

    train_result = trainer.train()

    # ------------------------------------------------------------------
    # Save adapter (LoRA) or full weights
    # ------------------------------------------------------------------
    log.info("Saving model to %s …", args.output_dir)
    trainer.save_model(str(args.output_dir))
    try:
        text_tokenizer.save_pretrained(str(args.output_dir))
    except Exception as e:
        log.warning("Could not save text_tokenizer: %s", e)

    # Save training metrics for validation script
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Dump loss history for the validate script
    loss_history = [
        {"step": entry["step"], "loss": entry.get("loss", entry.get("train_loss"))}
        for entry in trainer.state.log_history
        if "loss" in entry or "train_loss" in entry
    ]
    (args.output_dir / "loss_history.json").write_text(
        json.dumps(loss_history, indent=2), encoding="utf-8"
    )
    log.info("Loss history: %s", loss_history)

    print(f"\nFINE_TUNE_OK  output_dir={args.output_dir}  steps={trainer.state.global_step}")
    print(f"Final loss: {metrics.get('train_loss', 'n/a'):.4f}" if isinstance(metrics.get('train_loss'), float) else "")


if __name__ == "__main__":
    main()
