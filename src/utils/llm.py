from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

import os

# ── Shared defaults ─────────────────────────────────────────────────────────
# Read at import time so tests can monkeypatch env and reload the module.
_DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
_BASE_URL: str = os.getenv(
    "LLM_BASE_URL",
    "http://vllm-service.llm.svc.cluster.local:8000/v1",
)
_API_KEY: str = os.getenv("LLM_API_KEY", "EMPTY")
_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

# ── Claude default ──────────────────────────────────────────────────────────
_DEFAULT_MODEL_CLAUDE: str = os.getenv("LLM_MODEL_CLAUDE", "claude-sonnet-4-6")


def _role_env(role: str | None, suffix: str, default: str) -> str:
    """Read a role-scoped env var (e.g. LLM_MODEL_AGENT) then fall back to global."""
    if role:
        scoped = os.getenv(f"LLM_{suffix}_{role.upper()}")
        if scoped is not None:
            return scoped
    return os.getenv(f"LLM_{suffix}", default)


def make_llm(role: str | None = None, **overrides) -> BaseChatModel:
    """Build a chat model for the configured backend.

    ``LLM_BACKEND`` selects the provider:
      - ``vllm``  (default) — self-hosted vLLM serving any HuggingFace model via
                              OpenAI-compatible API.
      - ``claude``          — Anthropic Claude via the Anthropic API.
                              Reads ``ANTHROPIC_API_KEY`` from env.
      - ``lora``            — Same vLLM endpoint as ``vllm`` but requires
                              ``LLM_MODEL_LORA`` set to the HuggingFace path of
                              the trained LoRA adapter (not yet deployed).

    All backends respect per-role env overrides: e.g. ``LLM_MODEL_AGENT``,
    ``LLM_TEMPERATURE_AGENT``.
    """
    backend = os.getenv("LLM_BACKEND", "vllm").lower()
    temperature = overrides.pop(
        "temperature", float(_role_env(role, "TEMPERATURE", "0.2"))
    )

    if backend == "claude":
        from langchain_anthropic import ChatAnthropic  # lazy: only needed for this path

        model = overrides.pop("model", None) or _role_env(
            role, "MODEL", _DEFAULT_MODEL_CLAUDE
        )
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            timeout=_TIMEOUT,
            max_retries=_MAX_RETRIES,
            **overrides,
        )

    if backend == "lora":
        lora_model = os.getenv("LLM_MODEL_LORA", "")
        if not lora_model:
            raise RuntimeError(
                "LLM_BACKEND=lora requires LLM_MODEL_LORA to be set to the "
                "HuggingFace path of the trained LoRA adapter."
            )
        default_model = lora_model
    else:
        default_model = _DEFAULT_MODEL

    model = overrides.pop("model", None) or _role_env(role, "MODEL", default_model)
    return ChatOpenAI(
        model=model,
        base_url=_BASE_URL,
        api_key=_API_KEY,
        temperature=temperature,
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
        **overrides,
    )
