from __future__ import annotations

import textwrap

import pytest

from src.training.config import (
    ConfigError,
    apply_overrides,
    load_config,
)

MINIMAL = "run_name: t\nmodel: {base_model: Qwen/Qwen3-0.6B}\n"


def _write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_defaults_applied(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.run_name == "t"
    assert cfg.model.base_model == "Qwen/Qwen3-0.6B"
    assert cfg.data.max_seq_len == 4096
    assert cfg.lora.r == 32
    assert cfg.data.chat_template_kwargs == {"enable_thinking": False}


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, MINIMAL + "bogus: 1\n"))


def test_unknown_nested_key_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "lora: {rank: 8}\n"))


def test_set_override_coercion(tmp_path):
    cfg = load_config(
        _write(tmp_path, MINIMAL),
        overrides=[
            "train.learning_rate=5e-5",
            "lora.r=8",
            "model.load_in_4bit=false",
            "hub.hub_model_id=none",
        ],
    )
    assert cfg.train.learning_rate == 5e-5 and isinstance(cfg.train.learning_rate, float)
    assert cfg.lora.r == 8 and isinstance(cfg.lora.r, int)
    assert cfg.model.load_in_4bit is False
    assert cfg.hub.hub_model_id is None


def test_apply_overrides_requires_kv():
    with pytest.raises(ConfigError):
        apply_overrides({}, ["train.lr"])


def test_validate_push_to_hub_needs_id(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, MINIMAL + "hub: {push_to_hub: true}\n"))


def test_validate_on_overlong(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "data: {on_overlong: nonsense}\n"))


def test_real_config_files_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for name in ("qwen3_8-27b-qlora.yaml", "smoke.yaml"):
        load_config(root / "configs" / "lora" / name)
