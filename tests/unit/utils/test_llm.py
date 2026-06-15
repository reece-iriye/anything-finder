import importlib


def _reload_llm(monkeypatch, **env):
    # The module reads env at import time, so set env then reload.
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import src.utils.llm as llm_mod

    return importlib.reload(llm_mod)


def test_default_model_is_qwen(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    llm_mod = _reload_llm(monkeypatch)
    assert "Qwen" in llm_mod.make_llm().model_name


def test_per_role_model_override(monkeypatch):
    llm_mod = _reload_llm(
        monkeypatch,
        LLM_MODEL="base-model",
        LLM_MODEL_INTENT="intent-model",
    )
    assert llm_mod.make_llm("intent").model_name == "intent-model"
    # A role without its own override falls back to the global default.
    assert llm_mod.make_llm("synthesize").model_name == "base-model"


def test_role_temperature_override(monkeypatch):
    llm_mod = _reload_llm(
        monkeypatch,
        LLM_TEMPERATURE="0.1",
        LLM_TEMPERATURE_SYNTHESIZE="0.7",
    )
    assert llm_mod.make_llm("synthesize").temperature == 0.7
    assert llm_mod.make_llm("intent").temperature == 0.1
