#!/usr/bin/env python3
"""
Standalone HuggingFace-Trainer fine-tuner for Qwen3-TTS (12Hz).

Uses the official dual-channel sequence format from the Qwen3-TTS
finetuning/dataset.py recipe:
  - audio_codes are pre-encoded by build_qwen3tts_sft_jsonl.py
  - input_ids shape (B, T, 2): channel 0 = text IDs, channel 1 = codec IDs
  - Main loss = CE on codebook-0 tokens; sub-talker loss = CE on codebooks 1-15

Usage:
    python scripts/saudi_arabic/finetune_trainer.py \\
        --train-jsonl data/sada22_small_qwen3tts/train_qwen3tts_sft.jsonl \\
        --val-jsonl   data/sada22_small_qwen3tts/val_qwen3tts_sft.jsonl \\
        --base-model  Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --output-dir  outputs/qwen3tts-sada22-lora \\
        --max-steps   50 --mode lora --bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_CODEBOOKS = 16   # Qwen3-TTS uses 16 RVQ codebooks
SUB_TALKER_LOSS_WEIGHT = 0.3   # from official finetuning/train.py


# ---------------------------------------------------------------------------
# Dataset — reads pre-encoded audio_codes from JSONL
# ---------------------------------------------------------------------------

class Sada22SFTDataset(Dataset):
    """
    Reads SFT JSONL produced by build_qwen3tts_sft_jsonl.py.
    Each record must have: id, text, audio_codes (List[T × 16 ints]).
    Audio I/O happens at step [4/6] (build_qwen3tts_sft_jsonl.py), not here.
    """

    def __init__(self, jsonl_path: Path) -> None:
        self.records: list[dict[str, Any]] = []
        missing_codes = 0

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("audio_codes"):
                    missing_codes += 1
                    continue
                self.records.append(rec)

        if missing_codes:
            log.warning(
                "%d rows in %s have no audio_codes — skipped. "
                "Re-run step [4/6] (build_qwen3tts_sft_jsonl.py) to encode them.",
                missing_codes, jsonl_path,
            )
        if not self.records:
            raise RuntimeError(
                f"No valid records in {jsonl_path}. "
                "Run step [4/6] first to pre-encode audio codes."
            )
        log.info("Loaded %d samples from %s", len(self.records), jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        return {
            "id":          rec["id"],
            "text":        rec["text"],
            "audio_codes": rec["audio_codes"],  # List[T × 16 ints]
        }


# ---------------------------------------------------------------------------
# Collator — builds official dual-channel sequence
# ---------------------------------------------------------------------------

class TalkerCollator:
    """
    Mirrors finetuning/dataset.py from the official Qwen3-TTS repo.

    Sequence layout per sample (T_seq = 8 + L + T tokens):
      Pos  0-2    :  text_ids[0:3]       | 0, 0, 0             (prefix: im_start, assistant, \\n)
      Pos  3-5    :  tts_pad × 3         | nothink, think_bos, think_eos  (codec think tokens)
      Pos  6      :  tts_pad             | 0                              (speaker slot — zeroed)
      Pos  7      :  tts_bos             | codec_pad                      (TTS start)
      Pos  8..5+L :  text_ids[3:]        | codec_pad × (L-3)              (rest of text)
      Pos  6+L    :  tts_eos             | codec_pad                      (TTS end)
      Pos  7+L    :  tts_pad             | codec_bos                      (codec start marker)
      Pos  8+L..  :  tts_pad × T        | codes_col0[0..T-1]             (codebook-0)
      Pos  8+L+T  :  tts_pad             | codec_eos                      (codec end)

    batch keys returned:
      input_ids        (B, T_seq, 2)  — channel 0 = text, channel 1 = codec
      attention_mask   (B, T_seq)
      codec_ids        (B, T_seq, 16) — all 16 codebooks (zeros outside codec region)
      codec_0_labels   (B, T_seq)     — -100 everywhere except codec positions
    """

    def __init__(self, text_tokenizer, speech_tokenizer) -> None:
        self.tok = text_tokenizer
        self.stok = speech_tokenizer

        # ── Text-space special token IDs ──────────────────────────────────
        def _tid(token: str, fallback: int) -> int:
            ids = self.tok.convert_tokens_to_ids([token])
            if ids and ids[0] != self.tok.unk_token_id:
                return ids[0]
            return fallback

        self.tts_pad_id = _tid("<|tts_pad|>",  self.tok.pad_token_id or 0)
        self.tts_bos_id = _tid("<|tts_bos|>",  self.tok.bos_token_id or 1)
        self.tts_eos_id = _tid("<|tts_eos|>",  self.tok.eos_token_id or 2)

        # ── Codec-space special token IDs ─────────────────────────────────
        def _cid(attr: str, fallback: int) -> int:
            return int(getattr(self.stok, attr, fallback) or fallback)

        self.codec_pad_id       = _cid("pad_token_id",        0)
        self.codec_bos_id       = _cid("bos_token_id",        8193)
        self.codec_eos_id       = _cid("eos_token_id",        8194)
        self.codec_nothink_id   = _cid("nothink_token_id",    self.codec_pad_id)
        self.codec_think_bos_id = _cid("think_bos_token_id",  self.codec_pad_id)
        self.codec_think_eos_id = _cid("think_eos_token_id",  self.codec_pad_id)

        log.info(
            "TalkerCollator special tokens — "
            "tts_pad=%d tts_bos=%d tts_eos=%d | "
            "codec_pad=%d codec_bos=%d codec_eos=%d",
            self.tts_pad_id, self.tts_bos_id, self.tts_eos_id,
            self.codec_pad_id, self.codec_bos_id, self.codec_eos_id,
        )

    def _build_text_ids(self, text: str) -> list[int]:
        """
        Tokenise the instruction text and return token IDs with last-5 dropped.
        Prompt: <|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n
        """
        prompt = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        ids = self.tok(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].tolist()
        return ids[:-5] if len(ids) > 5 else ids   # drop trailing 5 tokens (official recipe)

    def _build_sample(
        self,
        text_ids: list[int],
        audio_codes: list[list[int]],   # List[T × 16 ints]
    ) -> tuple[list, list, list, list]:
        """
        Returns (text_channel, codec_channel, codec_ids_flat, labels) each length T_seq.
          codec_ids_flat: List[16 ints] per position (zeros outside codec region)
          labels:  -100 outside codec region, codec col-0 value at codec positions
        """
        L = len(text_ids)
        T = len(audio_codes)

        p   = self.tts_pad_id
        bos = self.tts_bos_id
        eos = self.tts_eos_id
        cp  = self.codec_pad_id
        cb  = self.codec_bos_id
        ce  = self.codec_eos_id
        cn  = self.codec_nothink_id
        ctb = self.codec_think_bos_id
        cte = self.codec_think_eos_id

        _ZEROS_16 = [0] * NUM_CODEBOOKS
        _PAD_16   = [cp] + [0] * (NUM_CODEBOOKS - 1)  # pad only codebook-0

        # Build channels position by position (matches official dataset.py layout)
        text_ch:  list[int]       = []
        codec_ch: list[int]       = []
        cids:     list[list[int]] = []
        labels:   list[int]       = []

        # Pos 0-2: first 3 text tokens, codec channel = 0
        for tok_id in text_ids[:3]:
            text_ch.append(tok_id);  codec_ch.append(0);  cids.append(_ZEROS_16);  labels.append(-100)

        # Pos 3-5: think tokens in codec channel
        for c in (cn, ctb, cte):
            text_ch.append(p);  codec_ch.append(c);  cids.append(_ZEROS_16);  labels.append(-100)

        # Pos 6: speaker slot (zeroed — no speaker encoder)
        text_ch.append(p);  codec_ch.append(0);  cids.append(_ZEROS_16);  labels.append(-100)

        # Pos 7: tts_bos | codec_pad
        text_ch.append(bos);  codec_ch.append(cp);  cids.append(_ZEROS_16);  labels.append(-100)

        # Pos 8 .. 8+(L-3)-1 = 8+L-4: text_ids[3:] | codec_pad
        for tok_id in text_ids[3:]:
            text_ch.append(tok_id);  codec_ch.append(cp);  cids.append(_ZEROS_16);  labels.append(-100)

        # Pos 8+L-3: tts_eos | codec_pad
        text_ch.append(eos);  codec_ch.append(cp);  cids.append(_ZEROS_16);  labels.append(-100)

        # Pos 8+L-2: tts_pad | codec_bos (codec sequence starts)
        text_ch.append(p);  codec_ch.append(cb);  cids.append(_ZEROS_16);  labels.append(-100)

        # Pos 8+L-1 .. 8+L-1+T-1: tts_pad | codec col-0; codec_ids = full 16-book row
        for frame in audio_codes:
            text_ch.append(p)
            codec_ch.append(frame[0])             # codebook-0 in codec channel
            cids.append(list(frame))              # all 16 codebooks
            labels.append(frame[0])              # CE loss on codebook-0

        # Pos 8+L-1+T: tts_pad | codec_eos  (predict EOS as final label)
        text_ch.append(p);  codec_ch.append(ce);  cids.append(_ZEROS_16);  labels.append(ce)

        return text_ch, codec_ch, cids, labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        all_text_ch:  list[torch.Tensor] = []
        all_codec_ch: list[torch.Tensor] = []
        all_cids:     list[torch.Tensor] = []    # (T_seq, 16) per sample
        all_labels:   list[torch.Tensor] = []

        for item in batch:
            text_ids    = self._build_text_ids(item["text"])
            audio_codes = item["audio_codes"]

            tc, cc, cids, labs = self._build_sample(text_ids, audio_codes)

            all_text_ch.append(torch.tensor(tc,   dtype=torch.long))
            all_codec_ch.append(torch.tensor(cc,  dtype=torch.long))
            all_cids.append(torch.tensor(cids,    dtype=torch.long))  # (T_seq, 16)
            all_labels.append(torch.tensor(labs,  dtype=torch.long))

        # Pad everything to the longest sequence in the batch
        max_len = max(t.size(0) for t in all_text_ch)
        pad_tok  = self.tts_pad_id
        pad_cid  = self.codec_pad_id

        def _pad(t: torch.Tensor, pad_val: int, target_len: int) -> torch.Tensor:
            diff = target_len - t.size(0)
            if diff == 0:
                return t
            padding = torch.full((diff,), pad_val, dtype=torch.long)
            return torch.cat([t, padding], dim=0)

        def _pad_cids(t: torch.Tensor, target_len: int) -> torch.Tensor:
            diff = target_len - t.size(0)
            if diff == 0:
                return t
            padding = torch.zeros(diff, NUM_CODEBOOKS, dtype=torch.long)
            return torch.cat([t, padding], dim=0)

        text_ch_padded  = torch.stack([_pad(t, pad_tok,  max_len) for t in all_text_ch])   # (B, T)
        codec_ch_padded = torch.stack([_pad(t, pad_cid,  max_len) for t in all_codec_ch])  # (B, T)
        cids_padded     = torch.stack([_pad_cids(t,      max_len) for t in all_cids])       # (B, T, 16)
        labels_padded   = torch.stack([_pad(t, -100,     max_len) for t in all_labels])     # (B, T)

        # Dual-channel input: (B, T, 2)
        input_ids = torch.stack([text_ch_padded, codec_ch_padded], dim=-1)  # (B, T, 2)

        attention_mask = (text_ch_padded != pad_tok).long()

        return {
            "input_ids":      input_ids,       # (B, T, 2)
            "attention_mask": attention_mask,   # (B, T)
            "codec_ids":      cids_padded,      # (B, T, 16)
            "codec_0_labels": labels_padded,    # (B, T)
        }


# ---------------------------------------------------------------------------
# Custom Trainer — passes dual-channel inputs, computes official loss
# ---------------------------------------------------------------------------

class Qwen3TTSTrainer:
    """
    Thin wrapper around the HF Trainer that overrides compute_loss to use
    the Qwen3-TTS dual-channel training forward.

    Forward call (mirrors official finetuning/train.py):
      outputs = talker(
          input_ids      = input_ids[:, :-1, :],      # (B, T-1, 2), shift for CLM
          attention_mask = attention_mask[:, :-1],
          labels         = codec_0_labels[:, 1:],      # (B, T-1), shift left
          output_hidden_states = True,
      )
      loss = outputs.loss + SUB_TALKER_LOSS_WEIGHT * sub_talker_loss

    Sub-talker loss: codebooks 1-15 predicted from main talker hidden states.
    """

    @staticmethod
    def compute_loss_fn(model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids      = inputs["input_ids"]       # (B, T, 2)
        attention_mask = inputs["attention_mask"]  # (B, T)
        codec_ids      = inputs["codec_ids"]       # (B, T, 16)
        codec_0_labels = inputs["codec_0_labels"]  # (B, T)

        # Standard CLM shift: predict token t from tokens 0..t-1
        outputs = model(
            input_ids      = input_ids[:, :-1, :],
            attention_mask = attention_mask[:, :-1],
            labels         = codec_0_labels[:, 1:],
            output_hidden_states=True,
        )
        loss = outputs.loss

        # Sub-talker loss (codebooks 1–15)
        try:
            hidden = outputs.hidden_states[-1]        # (B, T-1, H) last layer
            codec_mask = (codec_0_labels[:, 1:] != -100)  # (B, T-1) — codec positions

            talker_hidden = hidden[codec_mask]         # (N, H)
            talker_cids   = codec_ids[:, 1:][codec_mask]  # (N, 16)

            if talker_hidden.numel() > 0:
                _, sub_loss = model.forward_sub_talker_finetune(talker_cids, talker_hidden)
                loss = loss + SUB_TALKER_LOSS_WEIGHT * sub_loss
        except Exception as e:
            # sub-talker loss is optional; don't crash if unavailable
            log.debug("Sub-talker loss skipped: %s", e)

        if return_outputs:
            return loss, outputs
        return loss


def _make_trainer_class():
    """Dynamically create a Trainer subclass that uses Qwen3TTSTrainer.compute_loss_fn."""
    from transformers import Trainer

    class _Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return Qwen3TTSTrainer.compute_loss_fn(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    return _Trainer


# ---------------------------------------------------------------------------
# Model loading + cache fixes
# ---------------------------------------------------------------------------

def _ensure_speech_tokenizer_preprocessor(snapshot_dir: str) -> None:
    """Copy preprocessor_config.json into speech_tokenizer/ if missing (HF cache bug)."""
    import shutil
    snapshot = Path(snapshot_dir)
    root_cfg = snapshot / "preprocessor_config.json"
    sub_dir  = snapshot / "speech_tokenizer"
    sub_cfg  = sub_dir / "preprocessor_config.json"
    if not sub_dir.is_dir() or sub_cfg.exists():
        return
    if root_cfg.exists():
        log.info("Copying preprocessor_config.json → speech_tokenizer/ (cache fix)")
        shutil.copy2(str(root_cfg), str(sub_cfg))
    else:
        log.warning("preprocessor_config.json not found at snapshot root (%s)", snapshot_dir)


def load_model_and_components(base_model: str, bf16: bool):
    """
    Load Qwen3TTSModel and return (talker, text_tokenizer, speech_tokenizer, wrapper).
    """
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise SystemExit("qwen-tts is required. Install: uv pip install qwen-tts") from exc

    log.info("Loading Qwen3TTSModel from %s …", base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if bf16 else torch.float32
    hf_token = os.environ.get("HF_TOKEN") or None

    # Resolve to local snapshot (applies cache fix, re-uses cached weights)
    load_path = base_model
    if not Path(base_model).exists():
        try:
            from huggingface_hub import snapshot_download
            load_path = snapshot_download(
                base_model, token=hf_token,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
            _ensure_speech_tokenizer_preprocessor(load_path)
        except Exception as e:
            log.warning("snapshot_download failed (%s); loading directly", e)
            load_path = base_model
    else:
        _ensure_speech_tokenizer_preprocessor(base_model)

    wrapper = Qwen3TTSModel.from_pretrained(
        load_path, dtype=dtype, device_map=device, token=hf_token,
    )

    full_model    = wrapper.model
    talker        = full_model.talker
    speech_tok    = full_model.speech_tokenizer

    processor     = wrapper.processor
    text_tok = (
        getattr(processor, "tokenizer", None)
        or getattr(processor, "text_tokenizer", None)
    )
    if text_tok is None:
        from transformers import AutoTokenizer
        text_tok = AutoTokenizer.from_pretrained(load_path)

    log.info("Talker:            %s", type(talker).__name__)
    log.info("Speech tokenizer:  %s", type(speech_tok).__name__)
    log.info("Text tokenizer:    %s", type(text_tok).__name__)

    return talker, text_tok, speech_tok, wrapper


def apply_lora(talker, lora_cfg: dict):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit("peft is required: uv pip install peft") from exc

    config = LoraConfig(
        r               = lora_cfg.get("r", 64),
        lora_alpha      = lora_cfg.get("lora_alpha", 128),
        lora_dropout    = lora_cfg.get("lora_dropout", 0.05),
        target_modules  = lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        task_type       = TaskType.CAUSAL_LM,
        bias            = "none",
    )
    talker = get_peft_model(talker, config)
    talker.print_trainable_parameters()
    return talker


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",       type=Path, default=None)
    p.add_argument("--train-jsonl",  type=Path)
    p.add_argument("--val-jsonl",    type=Path)
    p.add_argument("--base-model",   type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    p.add_argument("--output-dir",   type=Path, default=Path("outputs/qwen3tts-sada22-lora"))
    p.add_argument("--mode",         choices=["full", "lora"], default="lora")
    p.add_argument("--max-steps",    type=int,   default=50)
    p.add_argument("--epochs",       type=int,   default=1)
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--grad-accum",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--bf16",         action="store_true", default=False)
    p.add_argument("--logging-steps",type=int,   default=5)
    return p.parse_args()


def merge_config_into_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    cfg = json.loads(args.config.read_text())
    if args.train_jsonl is None and "train_file" in cfg:
        args.train_jsonl = Path(cfg["train_file"])
    if args.val_jsonl is None:
        args.val_jsonl = Path(cfg.get("validation_file", cfg.get("val_file", "")))
    args.base_model  = cfg.get("base_model",                   args.base_model)
    args.output_dir  = Path(cfg.get("output_dir",              str(args.output_dir)))
    args.lr          = float(cfg.get("learning_rate",          args.lr))
    args.batch_size  = int(cfg.get("per_device_train_batch_size", args.batch_size))
    args.grad_accum  = int(cfg.get("gradient_accumulation_steps", args.grad_accum))
    args.bf16        = cfg.get("bf16",                         args.bf16)
    if "max_steps" in cfg and int(cfg["max_steps"]) > 0:
        args.max_steps = int(cfg["max_steps"])
    if "num_train_epochs" in cfg:
        args.epochs = int(cfg["num_train_epochs"])
    args.mode = "lora" if cfg.get("peft") else "full"
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args = merge_config_into_args(args)

    if args.train_jsonl is None:
        raise SystemExit("--train-jsonl is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16   = args.bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    fp16   = (device == "cuda") and not bf16
    log.info("Device: %s  | bf16: %s  | fp16: %s", device, bf16, fp16)

    # Load model components
    talker, text_tok, speech_tok, wrapper = load_model_and_components(args.base_model, bf16)

    if args.mode == "lora":
        talker = apply_lora(talker, {
            "r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
        })

    if next(talker.parameters(), None) is not None:
        talker.to(device)
    log.info("Training on device: %s  |  bf16: %s  |  fp16: %s", device, bf16, fp16)

    # Datasets
    train_ds = Sada22SFTDataset(args.train_jsonl)
    val_ds   = Sada22SFTDataset(args.val_jsonl) if (
        args.val_jsonl and args.val_jsonl.exists()
    ) else None

    collator = TalkerCollator(text_tok, speech_tok)

    # Training args
    from transformers import TrainingArguments

    eff_steps  = args.max_steps if args.max_steps > 0 else -1
    save_steps = max(10, (args.max_steps // 2) if args.max_steps > 0 else 500)
    log_steps  = max(1, min(args.logging_steps, save_steps // 2))

    training_args = TrainingArguments(
        output_dir                  = str(args.output_dir),
        num_train_epochs            = args.epochs if eff_steps < 0 else 9999,
        max_steps                   = eff_steps,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        bf16                        = bf16,
        fp16                        = fp16,
        no_cuda                     = (device == "cpu"),
        logging_dir                 = str(args.output_dir / "logs"),
        logging_steps               = log_steps,
        save_strategy               = "steps",
        save_steps                  = save_steps,
        eval_strategy               = "steps" if val_ds else "no",
        eval_steps                  = save_steps if val_ds else None,
        report_to                   = "none",
        load_best_model_at_end      = False,
        dataloader_num_workers      = 0,
        remove_unused_columns       = False,
        label_names                 = ["codec_0_labels"],
    )

    TrainerClass = _make_trainer_class()
    trainer = TrainerClass(
        model          = talker,
        args           = training_args,
        train_dataset  = train_ds,
        eval_dataset   = val_ds,
        data_collator  = collator,
        processing_class = text_tok,
    )

    log.info("Starting training: max_steps=%s, batch=%s, grad_accum=%s",
             eff_steps, args.batch_size, args.grad_accum)
    train_result = trainer.train()

    # Save
    log.info("Saving checkpoint to %s …", args.output_dir)
    trainer.save_model(str(args.output_dir))
    try:
        text_tok.save_pretrained(str(args.output_dir))
    except Exception as e:
        log.warning("Could not save text_tokenizer: %s", e)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    loss_history = [
        {"step": e["step"], "loss": e.get("loss", e.get("train_loss"))}
        for e in trainer.state.log_history
        if "loss" in e or "train_loss" in e
    ]
    (args.output_dir / "loss_history.json").write_text(
        json.dumps(loss_history, indent=2), encoding="utf-8"
    )
    log.info("Loss history: %s", loss_history)
    print(f"\nFINE_TUNE_OK  output_dir={args.output_dir}  steps={trainer.state.global_step}")


if __name__ == "__main__":
    main()

