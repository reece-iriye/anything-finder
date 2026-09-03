"""Render the Hugging Face model card for a trained LoRA adapter.

The adapter repo is useless to anyone (including future-you) without the
provenance: which base model, which trajectories, which masking scheme, and how
to serve it. This builds that README from the resolved training config, so the
card can never drift from the run that produced it.

No torch, no transformers, no network — a pure string builder, unit-tested under
``tests/unit/training/test_model_card.py``.
"""

from __future__ import annotations

from typing import Any

from src.training.config import LoraTrainingConfig

# Frontmatter tags. `peft` + `base_model` are what make the Hub render the
# "adapter for <base>" banner and wire the inference widget.
_TAGS = ["peft", "lora", "qlora", "tool-calling", "agent", "langgraph", "openstreetmap"]


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(f"`{v}`" for v in value) if value else "(none)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"`{value}`"


def render_model_card(
    cfg: LoraTrainingConfig,
    stats: dict[str, Any] | None = None,
) -> str:
    """Return the full README.md text for ``cfg``'s adapter.

    ``stats`` is optional run metadata — ``train_examples``, ``eval_examples``,
    ``dropped``, ``final_loss``, ``eval_loss`` — folded into the card when present.
    """
    s = stats or {}
    base = cfg.model.base_model
    serve_model = f"{base}-FP8" if "27B" in base else base

    counts = []
    if s.get("train_examples") is not None:
        counts.append(f"- Training examples: **{s['train_examples']}**")
    if s.get("eval_examples") is not None:
        counts.append(f"- Validation examples: **{s['eval_examples']}**")
    if s.get("dropped"):
        counts.append(f"- Dropped during preparation: {s['dropped']}")
    counts_block = "\n".join(counts) or "- Counts unavailable for this run."

    losses = []
    if s.get("final_loss") is not None:
        losses.append(f"- Final training loss: `{s['final_loss']}`")
    if s.get("eval_loss") is not None:
        losses.append(f"- Final validation loss: `{s['eval_loss']}`")
    losses_block = ("\n" + "\n".join(losses)) if losses else ""

    return f"""---
base_model: {base}
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
{chr(10).join(f"  - {t}" for t in _TAGS)}
---

# {cfg.run_name}

A LoRA adapter for `{base}` that teaches the base model to drive the
**anything-finder** restaurant-search agent: a four-tool ReAct loop over
OpenStreetMap data (Nominatim geocoding + Overpass POI search).

The adapter is distilled from Claude Sonnet trajectories — the teacher ran the
*real* agent against a local Dallas OSM extract, and every tool call, tool
result, and final recommendation was captured verbatim.

## Tools the adapter is trained to call

| Tool | Purpose |
| --- | --- |
| `read_food_preferences` | Read the user's stored preference profile. |
| `get_current_location` | Fall back to the configured home city/state. |
| `geocode_location` | Place phrase -> coordinates (Nominatim). |
| `search_restaurants` | Nearby eateries within `radius_m` (Overpass); widen and retry when sparse. |

Tool JSON schemas ship with this repo as `tool_schemas.json`, generated from the
live `@tool` objects so they cannot drift from the app.

## Training data

{counts_block}

Records are OpenAI-style tool-calling chat transcripts built from the agent's
full message log — not flattened prompt/completion pairs. See
`src/training/dataset.py` in the source repo for the validation rules (a row is
dropped when a `tool_call_id` is unresolved or unanswered, when the final turn
is not a tool-call-free assistant message, or when that turn is empty).

## Training procedure

- Base model: {_fmt(base)} ({_fmt(cfg.model.dtype)}{", 4-bit NF4 QLoRA" if cfg.model.load_in_4bit else ""})
- LoRA rank / alpha / dropout: {_fmt(cfg.lora.r)} / {_fmt(cfg.lora.alpha)} / {_fmt(cfg.lora.dropout)}
- Target modules: {_fmt(cfg.lora.target_modules)}
- Excluded modules: {_fmt(cfg.lora.exclude_modules)}
- Sequence length: {_fmt(cfg.data.max_seq_len)} (overlong rows: {_fmt(cfg.data.on_overlong)})
- Epochs: {_fmt(cfg.train.num_train_epochs)}, LR {_fmt(cfg.train.learning_rate)} ({_fmt(cfg.train.lr_scheduler_type)}), optimizer {_fmt(cfg.train.optim)}
- Effective batch: {_fmt(cfg.train.per_device_train_batch_size)} x {_fmt(cfg.train.gradient_accumulation_steps)} grad-accum
- Seed: {_fmt(cfg.seed)}{losses_block}

### Two caveats worth knowing

1. **Assistant-only loss via incremental rendering.** Qwen chat templates lack
   `{{% generation %}}`, so TRL's `assistant_only_loss` cannot be used. Labels are
   masked by rendering `messages[:i]` and `messages[:i+1]` and unmasking the
   delta. System, user, and tool spans contribute no loss; every assistant turn
   does.
2. **Trained non-thinking.** The dataset renders with
   `enable_thinking: {str(cfg.data.chat_template_kwargs.get("enable_thinking", False)).lower()}` — the captured trajectories carry no
   reasoning content, so prompt it the same way at inference.

## Serving with vLLM

```bash
vllm serve {serve_model} \\
  --enable-lora --lora-modules af-lora=/adapters/af-lora \\
  --max-lora-rank {cfg.lora.r} --api-key EMPTY
```

Then point the app at it:

```bash
LLM_BACKEND=lora LLM_MODEL_LORA=af-lora LLM_BASE_URL=http://localhost:8000/v1
```
{
    _FP8_NOTE if "27B" in base else ""
}
## Evaluation

The source repo renders a three-way comparison (Claude Sonnet vs. the raw base
model vs. this adapter) over an identical slice of queries — final answers plus
telemetry (tool paths, per-call tokens, latency, cost) — with `make compare`.
"""


_FP8_NOTE = """
> **Train/serve precision mismatch.** The adapter is trained against BF16 base
> weights and served above against the FP8 repo. vLLM keeps adapters in BF16 and
> supports this, but there is a small distribution mismatch; serve the non-FP8
> base to remove it at the cost of VRAM.
"""
