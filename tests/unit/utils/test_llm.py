import importlib

import pytest


def _reload_llm(monkeypatch, **env):
    # The module reads env at import time, so set env then reload.
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import src.utils.llm as llm_mod

    return importlib.reload(llm_mod)


# ── vllm backend (default) ──────────────────────────────────────────────────


def test_default_model_is_qwen(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    llm_mod = _reload_llm(monkeypatch)
    assert "Qwen" in llm_mod.make_llm().model_name


def test_per_role_model_override(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    llm_mod = _reload_llm(
        monkeypatch,
        LLM_MODEL="base-model",
        LLM_MODEL_AGENT="agent-model",
    )
    assert llm_mod.make_llm("agent").model_name == "agent-model"
    # A role without its own override falls back to the global default.
    assert llm_mod.make_llm("other").model_name == "base-model"


def test_role_temperature_override(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    llm_mod = _reload_llm(
        monkeypatch,
        LLM_TEMPERATURE="0.1",
        LLM_TEMPERATURE_AGENT="0.7",
    )
    assert llm_mod.make_llm("agent").temperature == 0.7
    assert llm_mod.make_llm("other").temperature == 0.1


# ── claude backend ──────────────────────────────────────────────────────────


def test_claude_backend_returns_anthropic_client(monkeypatch):
    from langchain_anthropic import ChatAnthropic

    monkeypatch.setenv("LLM_BACKEND", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm_mod = _reload_llm(monkeypatch)
    llm = llm_mod.make_llm()
    assert isinstance(llm, ChatAnthropic)
    assert "claude" in llm.model.lower()


def test_claude_backend_respects_model_override(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_CLAUDE", "claude-haiku-4-5-20251001")
    llm_mod = _reload_llm(monkeypatch)
    llm = llm_mod.make_llm()
    assert llm.model == "claude-haiku-4-5-20251001"


# ── lora backend ────────────────────────────────────────────────────────────


def test_lora_backend_raises_without_model(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "lora")
    monkeypatch.delenv("LLM_MODEL_LORA", raising=False)
    llm_mod = _reload_llm(monkeypatch)
    with pytest.raises(RuntimeError, match="LLM_MODEL_LORA"):
        llm_mod.make_llm()


def test_lora_backend_uses_lora_model_path(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "lora")
    monkeypatch.setenv("LLM_MODEL_LORA", "myuser/qwen3-27b-lora-v1")
    llm_mod = _reload_llm(monkeypatch)
    llm = llm_mod.make_llm()
    assert llm.model_name == "myuser/qwen3-27b-lora-v1"
