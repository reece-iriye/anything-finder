"""The model card is the adapter's only documentation on the Hub, so the facts
it states must come from the resolved config rather than from prose."""

import json

import pytest

from src.training.config import LoraTrainingConfig, load_config
from src.training.hub import EXTRA_FILES, write_run_artifacts
from src.training.model_card import render_model_card


@pytest.fixture
def cfg() -> LoraTrainingConfig:
    return LoraTrainingConfig()


def _frontmatter(card: str) -> dict[str, str]:
    assert card.startswith("---\n")
    block = card.split("---\n", 2)[1]
    out = {}
    for line in block.splitlines():
        if line.startswith("  - ") or not line.strip():
            continue
        key, _, val = line.partition(":")
        out[key.strip()] = val.strip()
    return out


def test_frontmatter_declares_base_model_and_peft(cfg):
    fm = _frontmatter(render_model_card(cfg))
    assert fm["base_model"] == cfg.model.base_model
    assert fm["library_name"] == "peft"
    assert fm["license"] and fm["pipeline_tag"] == "text-generation"


def test_hyperparameters_come_from_the_config(cfg):
    cfg.lora.r = 8
    cfg.lora.alpha = 16
    cfg.train.learning_rate = 5e-5
    cfg.train.num_train_epochs = 2
    card = render_model_card(cfg)
    assert "`8`" in card and "`16`" in card
    assert "5e-05" in card or "5e-5" in card
    assert f"`{cfg.data.max_seq_len}`" in card


def test_serve_command_uses_the_lora_rank(cfg):
    cfg.lora.r = 64
    assert "--max-lora-rank 64" in render_model_card(cfg)


def test_27b_card_carries_the_fp8_caveat_and_7b_does_not(cfg):
    cfg.model.base_model = "Qwen/Qwen3.8-27B"
    big = render_model_card(cfg)
    assert "Qwen/Qwen3.8-27B-FP8" in big
    assert "precision mismatch" in big.lower()

    cfg.model.base_model = "Qwen/Qwen2.5-7B-Instruct"
    small = render_model_card(cfg)
    assert "FP8" not in small
    assert "vllm serve Qwen/Qwen2.5-7B-Instruct" in small


def test_stats_are_folded_in_when_given(cfg):
    card = render_model_card(cfg, {"train_examples": 333, "eval_examples": 37, "eval_loss": 0.42})
    assert "**333**" in card and "**37**" in card and "0.42" in card


def test_card_survives_missing_stats(cfg):
    card = render_model_card(cfg, None)
    assert "Counts unavailable" in card
    assert "Final training loss" not in card


def test_all_configs_in_the_repo_render(tmp_path):
    """Every shipped config must produce a card — the push path depends on it."""
    from pathlib import Path

    for path in sorted(Path("configs/lora").glob("*.yaml")):
        card = render_model_card(load_config(path, []))
        assert card.startswith("---\n") and "## Training procedure" in card


def test_write_run_artifacts_populates_the_run_dir(tmp_path, cfg, monkeypatch):
    schemas = tmp_path / "schemas.json"
    schemas.write_text(json.dumps([{"type": "function"}]), encoding="utf-8")
    monkeypatch.setattr("src.training.hub.REPO_ROOT", tmp_path)
    cfg.data.tool_schemas = "schemas.json"

    run_dir = tmp_path / "run"
    written = write_run_artifacts(run_dir, cfg, {"train_examples": 10})

    assert {p.name for p in written} == set(EXTRA_FILES)
    assert "**10**" in (run_dir / "README.md").read_text(encoding="utf-8")
    assert json.loads((run_dir / "tool_schemas.json").read_text())[0]["type"] == "function"
    resolved = json.loads((run_dir / "resolved_config.json").read_text())
    assert resolved["model"]["base_model"] == cfg.model.base_model


def test_write_run_artifacts_tolerates_missing_schemas(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr("src.training.hub.REPO_ROOT", tmp_path)
    cfg.data.tool_schemas = "nope.json"
    written = write_run_artifacts(tmp_path / "run", cfg)
    assert {p.name for p in written} == {"README.md", "resolved_config.json"}
