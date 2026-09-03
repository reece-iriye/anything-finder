"""LoRA / QLoRA supervised fine-tuning for the restaurant ReAct agent.

    uv run scripts/train_lora.py --config configs/lora/qwen3_8-27b-qlora.yaml
    uv run scripts/train_lora.py --config configs/lora/smoke.yaml --dry-run
    accelerate launch scripts/train_lora.py --config configs/lora/qwen3_8-27b-qlora.yaml

``--dry-run`` builds the dataset, prints a token-length histogram, dumps one fully
rendered example (masked spans as ``·``, trained spans verbatim) and exits — no
GPU, no weights. That is the "is my data right?" gate before burning GPU hours.

Torch-free logic lives in ``src/training/``; every heavy import is inside
``main()`` so ``--help`` and the unit tests stay light.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(path: Path) -> list[str]:
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def _histogram(lengths: list[int], bins: int = 10) -> str:
    if not lengths:
        return "(no examples)"
    lo, hi = min(lengths), max(lengths)
    width = max((hi - lo) // bins + 1, 1)
    counts: Counter[int] = Counter((n - lo) // width for n in lengths)
    out = []
    for b in range(bins):
        start = lo + b * width
        c = counts.get(b, 0)
        out.append(f"  {start:>5}-{start + width - 1:<5} {'#' * c} {c}")
    p50 = sorted(lengths)[len(lengths) // 2]
    out.append(f"  n={len(lengths)} min={lo} p50={p50} max={hi}")
    return "\n".join(out)


def _load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _last_metric(trainer, key: str):
    """Most recent value of ``key`` in the trainer's log history, or None."""
    history = getattr(getattr(trainer, "state", None), "log_history", None) or []
    for entry in reversed(history):
        if key in entry:
            return round(float(entry[key]), 4)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="dotted-path override, e.g. --set train.learning_rate=5e-5",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _load_dotenv(REPO_ROOT / ".env")

    from src.training.config import config_to_dict, load_config
    from src.training.hub import upload_run_artifacts, write_run_artifacts

    cfg = load_config(args.config, args.overrides)
    print(f"run_name={cfg.run_name} base_model={cfg.model.base_model}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from src.training.masking import build_masked_example

    train_file = REPO_ROOT / cfg.data.train_file
    eval_file = REPO_ROOT / cfg.data.eval_file
    train_raw = _load_records(train_file)
    eval_raw = _load_records(eval_file) if eval_file.exists() else []

    def encode(records: list[dict]) -> list[dict]:
        out, dropped = [], 0
        for rec in records:
            ex = build_masked_example(
                tok,
                rec["messages"],
                tools=rec.get("tools"),
                max_seq_len=cfg.data.max_seq_len,
                on_overlong=cfg.data.on_overlong,
                template_kwargs=cfg.data.chat_template_kwargs,
            )
            if ex is None:
                dropped += 1
                continue
            out.append(ex)
        if dropped:
            print(f"  dropped {dropped} overlong example(s)")
        return out

    print(f"encoding {len(train_raw)} train / {len(eval_raw)} eval records")
    train_enc = encode(train_raw)
    eval_enc = encode(eval_raw)

    lengths = [len(e["input_ids"]) for e in train_enc]
    print("train token-length histogram:")
    print(_histogram(lengths))

    if args.dry_run:
        from src.training.masking import render_debug

        print("\n===== rendered example 0 (· = masked, verbatim = trained) =====\n")
        print(render_debug(tok, train_enc[0]))
        print("\n===== end example =====")
        n_trained = sum(1 for l in train_enc[0]["labels"] if l != -100)
        print(f"trained tokens: {n_trained}/{len(train_enc[0]['labels'])}")
        return

    # ---- real training path ------------------------------------------------
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    hf_cfg = AutoConfig.from_pretrained(
        cfg.model.base_model, trust_remote_code=cfg.model.trust_remote_code
    )
    arch = " ".join(getattr(hf_cfg, "architectures", []) or [])
    model_cls = (
        AutoModelForImageTextToText
        if ("ImageTextToText" in arch or "VL" in arch or "Omni" in arch)
        else AutoModelForCausalLM
    )
    print(f"architectures={arch or '?'} -> {model_cls.__name__}")

    quant_cfg = None
    if cfg.model.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.model.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.model.bnb_4bit_use_double_quant,
        )

    try:
        model = model_cls.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=getattr(torch, cfg.model.dtype),
            quantization_config=quant_cfg,
            attn_implementation=cfg.model.attn_implementation,
        )
    except (ImportError, ValueError) as exc:
        print(f"attn_implementation={cfg.model.attn_implementation} unavailable ({exc}); falling back to sdpa")
        model = model_cls.from_pretrained(
            cfg.model.base_model,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=getattr(torch, cfg.model.dtype),
            quantization_config=quant_cfg,
            attn_implementation="sdpa",
        )

    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.model.gradient_checkpointing
        )

    peft_cfg = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=cfg.lora.target_modules,
        exclude_modules=cfg.lora.exclude_modules or None,
        modules_to_save=cfg.lora.modules_to_save or None,
    )
    model = get_peft_model(model, peft_cfg)
    adapted = sorted({
        n.split(".lora_")[0].split(".")[-1]
        for n, _ in model.named_parameters()
        if "lora_" in n
    })
    print(f"adapted module types: {adapted}")
    print("(vision tower excluded)" if not any(
        "visual" in n or "vision" in n for n, _ in model.named_parameters() if "lora_" in n
    ) else "WARNING: vision modules were adapted — check lora.exclude_modules")
    model.print_trainable_parameters()

    def to_ds(enc: list[dict]) -> "Dataset":
        return Dataset.from_list(enc)

    def collate(batch: list[dict]) -> dict:
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id
        keys = ("input_ids", "attention_mask", "labels")
        pads = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}
        out = {k: [] for k in keys}
        for b in batch:
            n = maxlen - len(b["input_ids"])
            for k in keys:
                out[k].append(b[k] + [pads[k]] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}

    output_dir = str(REPO_ROOT / cfg.output_dir)
    sft_cfg = SFTConfig(
        output_dir=output_dir,
        run_name=cfg.run_name,
        seed=cfg.seed,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        bf16=cfg.train.bf16,
        logging_steps=cfg.train.logging_steps,
        eval_strategy=cfg.train.eval_strategy if eval_enc else "no",
        save_strategy=cfg.train.save_strategy,
        save_total_limit=cfg.train.save_total_limit,
        report_to=cfg.train.report_to,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=to_ds(train_enc),
        eval_dataset=to_ds(eval_enc) if eval_enc else None,
        data_collator=collate,
    )
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    (Path(output_dir) / "resolved_config.json").write_text(
        json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8"
    )

    stats = {
        "train_examples": len(train_enc),
        "eval_examples": len(eval_enc),
        "final_loss": _last_metric(trainer, "loss"),
        "eval_loss": _last_metric(trainer, "eval_loss"),
    }
    write_run_artifacts(Path(output_dir), cfg, stats)
    print(f"saved adapter + tokenizer + README.md + resolved_config.json -> {output_dir}")

    if cfg.hub.push_to_hub:
        model.push_to_hub(cfg.hub.hub_model_id, private=cfg.hub.hub_private_repo)
        tok.push_to_hub(cfg.hub.hub_model_id, private=cfg.hub.hub_private_repo)
        upload_run_artifacts(Path(output_dir), cfg.hub.hub_model_id, cfg.hub.hub_private_repo)
        print(f"pushed -> {cfg.hub.hub_model_id}")


if __name__ == "__main__":
    main()
