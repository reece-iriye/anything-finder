"""YAML -> nested dataclass config for LoRA training.

One YAML holds every knob (the ``unbias.json`` spirit from
``reece-iriye/Mitigating-Bias-in-Stable-Diffusion-Models-Using-LoRA``). Unknown
keys are rejected so a typo fails loudly instead of being silently ignored.
``--set train.learning_rate=5e-5`` style overrides are applied before validation.

No torch, no transformers — importable anywhere.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised for malformed config: unknown keys, bad types, failed validation."""


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen3.8-27B"
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    train_file: str = "data/lora/train.jsonl"
    eval_file: str = "data/lora/val.jsonl"
    tool_schemas: str = "data/lora/tool_schemas.json"
    max_seq_len: int = 4096
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False}
    )
    train_on_assistant_only: bool = True
    on_overlong: str = "drop"  # drop | truncate


@dataclass
class LoraConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    exclude_modules: list[str] = field(
        default_factory=lambda: ["visual", "vision_tower"]
    )
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    logging_steps: int = 5
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    max_steps: int = -1


@dataclass
class HubConfig:
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_private_repo: bool = True


@dataclass
class LoraTrainingConfig:
    run_name: str = "qwen38-27b-restaurant-react-lora"
    seed: int = 20260902
    output_dir: str = "data/lora/runs/qwen38-27b-restaurant-react-lora"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    # ---- validation -------------------------------------------------------
    def validate(self) -> "LoraTrainingConfig":
        if self.data.on_overlong not in ("drop", "truncate"):
            raise ConfigError(
                f"data.on_overlong must be 'drop' or 'truncate', got {self.data.on_overlong!r}"
            )
        if self.data.max_seq_len <= 0:
            raise ConfigError("data.max_seq_len must be positive")
        if self.lora.r <= 0 or self.lora.alpha <= 0:
            raise ConfigError("lora.r and lora.alpha must be positive")
        if self.hub.push_to_hub and not self.hub.hub_model_id:
            raise ConfigError("hub.push_to_hub is true but hub.hub_model_id is unset")
        return self


# --------------------------------------------------------------------------
# construction helpers
# --------------------------------------------------------------------------
# Nested sub-config classes, keyed by their field name on LoraTrainingConfig.
_NESTED: dict[str, type] = {
    "model": ModelConfig,
    "data": DataConfig,
    "lora": LoraConfig,
    "train": TrainConfig,
    "hub": HubConfig,
}


def _build_leaf(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}"
        )
    return cls(**data)


def _build(cls: type, data: Any, path: str = "") -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{path or 'config'}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        child = f"{path}.{name}" if path else name
        if name in _NESTED:
            kwargs[name] = _build_leaf(_NESTED[name], value, child)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _coerce_scalar(raw: str) -> Any:
    low = raw.strip()
    if low.lower() in ("none", "null", "~"):
        return None
    if low.lower() == "true":
        return True
    if low.lower() == "false":
        return False
    for cast in (int, float):
        try:
            return cast(low)
        except ValueError:
            pass
    return raw


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.path=value`` overrides onto a plain config dict (in place)."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        keys = dotted.strip().split(".")
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ConfigError(f"--set {dotted}: {k} is not a mapping")
        node[keys[-1]] = _coerce_scalar(raw)
    return data


def load_config(
    path: str | Path, overrides: list[str] | None = None
) -> LoraTrainingConfig:
    import yaml  # pyyaml is in the `training` group

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    if overrides:
        apply_overrides(raw, list(overrides))
    return _build(LoraTrainingConfig, raw).validate()


def config_from_dict(data: dict[str, Any]) -> LoraTrainingConfig:
    """Rebuild a config from its ``config_to_dict`` output.

    Round-trips a run's ``resolved_config.json`` so a finished adapter can be
    re-published (``scripts/push_lora.py``) without the original YAML.
    """
    if not isinstance(data, dict):
        raise ConfigError("config_from_dict expects a mapping")
    return _build(LoraTrainingConfig, data).validate()


def config_to_dict(cfg: LoraTrainingConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
