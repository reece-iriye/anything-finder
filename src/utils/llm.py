from langchain_openai import ChatOpenAI

import os

# vLLM serves an OpenAI-compatible API. Default to a ~7B quantized Qwen sized to fit a
# 16 GB GPU with KV-cache headroom; override per-role via env so a small/fast model can
# handle intent parsing while a stronger one handles synthesis -- all from one server.
_DEFAULT_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
_BASE_URL: str = os.getenv(
    "LLM_BASE_URL",
    "http://vllm-service.llm.svc.cluster.local:8000/v1",
)
# vLLM ignores the key, but the OpenAI client requires a non-empty value.
_API_KEY: str = os.getenv("LLM_API_KEY", "EMPTY")
_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))


def _role_env(role: str | None, suffix: str, default: str) -> str:
    """Read ROLE-specific env (e.g. LLM_MODEL_INTENT) then fall back to the global."""
    if role:
        scoped = os.getenv(f"LLM_{suffix}_{role.upper()}")
        if scoped is not None:
            return scoped
    return os.getenv(f"LLM_{suffix}", default)


def make_llm(role: str | None = None, **overrides) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at the self-hosted vLLM endpoint.

    ``role`` selects per-role overrides: e.g. ``make_llm("intent")`` reads
    ``LLM_MODEL_INTENT`` / ``LLM_TEMPERATURE_INTENT`` before falling back to
    ``LLM_MODEL`` / ``LLM_TEMPERATURE``. All roles share one base_url/api_key.
    """
    model = overrides.pop("model", None) or _role_env(role, "MODEL", _DEFAULT_MODEL)
    temperature = overrides.pop(
        "temperature", float(_role_env(role, "TEMPERATURE", "0.2"))
    )
    return ChatOpenAI(
        model=model,
        base_url=_BASE_URL,
        api_key=_API_KEY,
        temperature=temperature,
        timeout=_TIMEOUT,
        max_retries=_MAX_RETRIES,
        **overrides,
    )
