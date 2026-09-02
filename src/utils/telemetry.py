"""Local JSON telemetry for agent LLM and tool calls.

Enabled by setting ``AF_TRACE_DIR``. Every agent invocation then writes one file::

    <AF_TRACE_DIR>/<mode>/<YYYYMMDD-HHMMSS>-<trace_id8>.json

``mode`` groups runs by which model stack produced them
(``claude`` | ``raw-open-source`` | ``finetuned-open-source``), taken from
``AF_TRACE_MODE`` or derived from ``LLM_BACKEND``. Each file holds the ordered
spans -- one per LLM call, one per tool call -- with inputs, outputs, token
usage and timing, plus rolled-up totals and (for ``claude``) an approximate
cost. The file is rewritten after every span, so a killed process still leaves a
readable partial trace.

Attach the tracer per invocation via ``RunnableConfig`` callbacks
(see ``src/services/geo_search/restaurants.py``); it is a no-op unless
``AF_TRACE_DIR`` is set. Browse the output with ``make trace``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# Approximate Anthropic list prices, USD per 1M tokens (input, output). Prefix
# match, so dated snapshots resolve too. Used only for the cost estimate.
_CLAUDE_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT = 1.25

_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def trace_mode() -> str:
    """Which model-stack bucket the current process is exercising."""
    explicit = os.getenv("AF_TRACE_MODE")
    if explicit:
        return explicit
    backend = os.getenv("LLM_BACKEND", "vllm").lower()
    return {
        "claude": "claude",
        "lora": "finetuned-open-source",
        "vllm": "raw-open-source",
    }.get(backend, backend or "unknown")


_tracer: JsonFileTracer | None = None
_tracer_lock = threading.Lock()


def get_tracer() -> JsonFileTracer | None:
    """Return the process tracer, or ``None`` when ``AF_TRACE_DIR`` is unset."""
    global _tracer
    root = os.getenv("AF_TRACE_DIR")
    if not root:
        return None
    with _tracer_lock:
        if _tracer is None or _tracer.root != Path(root):
            _tracer = JsonFileTracer(Path(root))
    return _tracer


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _dump_message(m: BaseMessage) -> dict[str, Any]:
    d: dict[str, Any] = {"role": _ROLES.get(m.type, m.type), "content": _jsonable(m.content)}
    if getattr(m, "tool_calls", None):
        d["tool_calls"] = _jsonable(m.tool_calls)
    if getattr(m, "tool_call_id", None):
        d["tool_call_id"] = m.tool_call_id
    if getattr(m, "name", None):
        d["name"] = m.name
    return d


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for t in tools or []:
        if isinstance(t, dict):
            names.append(
                t.get("name")
                or (t.get("function") or {}).get("name")
                or t.get("type")
                or "?"
            )
        else:
            names.append(getattr(t, "name", str(t)))
    return names


def _tokens(response: LLMResult, msg: Any) -> dict[str, int]:
    um = getattr(msg, "usage_metadata", None) if msg is not None else None
    if um:
        itd = um.get("input_token_details") or {}
        return {
            "input": um.get("input_tokens", 0) or 0,
            "output": um.get("output_tokens", 0) or 0,
            "total": um.get("total_tokens", 0) or 0,
            "cache_read": itd.get("cache_read", 0) or 0,
            "cache_creation": itd.get("cache_creation", 0) or 0,
        }
    lo = response.llm_output or {}
    usage = lo.get("usage") or lo.get("token_usage") or {}
    inp = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    out = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    return {
        "input": inp,
        "output": out,
        "total": usage.get("total_tokens", inp + out) or (inp + out),
        "cache_read": usage.get("cache_read_input_tokens", 0) or 0,
        "cache_creation": usage.get("cache_creation_input_tokens", 0) or 0,
    }


def _model_of(response: LLMResult, fallback: str | None) -> str | None:
    lo = response.llm_output or {}
    return lo.get("model_name") or lo.get("model") or fallback


def _finish_reason(gen: Any, msg: Any) -> str | None:
    info = getattr(gen, "generation_info", None) or {}
    if info.get("finish_reason"):
        return info["finish_reason"]
    meta = getattr(msg, "response_metadata", None) or {}
    return meta.get("finish_reason") or meta.get("stop_reason")


def _claude_cost(model: str | None, tokens: dict[str, int]) -> float | None:
    if not model:
        return None
    key = next((k for k in _CLAUDE_PRICES if model.startswith(k)), None)
    if key is None:
        return None
    price_in, price_out = _CLAUDE_PRICES[key]
    cost = (
        tokens["input"] * price_in
        + tokens["cache_read"] * price_in * _CACHE_READ_MULT
        + tokens["cache_creation"] * price_in * _CACHE_WRITE_MULT
        + tokens["output"] * price_out
    ) / 1_000_000
    return round(cost, 6)


class JsonFileTracer(BaseCallbackHandler):
    """LangChain callback handler that writes one JSON file per agent run."""

    run_inline = True  # deterministic span ordering; the writes are tiny

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._parent: dict[UUID, UUID | None] = {}
        self._traces: dict[UUID, dict[str, Any]] = {}  # trace_id -> doc
        self._open: dict[UUID, dict[str, Any]] = {}  # run_id -> in-flight span
        self._perf: dict[UUID, float] = {}  # run_id -> perf_counter start

    # -- run-tree bookkeeping ------------------------------------------------

    def _root_of(self, run_id: UUID, parent_run_id: UUID | None) -> UUID:
        self._parent.setdefault(run_id, parent_run_id)
        cur = run_id
        seen: set[UUID] = set()
        for _ in range(200):
            if cur in seen:
                return cur
            seen.add(cur)
            nxt = self._parent.get(cur)
            if nxt is None:
                return cur
            cur = nxt
        return cur

    def _ensure_trace(self, trace_id: UUID, metadata: dict[str, Any] | None) -> dict[str, Any]:
        md = metadata or {}
        doc = self._traces.get(trace_id)
        if doc is None:
            mode = trace_mode()
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            doc = {
                "trace_id": str(trace_id),
                "mode": mode,
                "backend": os.getenv("LLM_BACKEND", "vllm"),
                "started_at": _now(),
                "ended_at": None,
                "session_id": md.get("thread_id") or md.get("session_id"),
                "user_id": md.get("user_id"),
                "query": md.get("af_query"),
                "request": md.get("af_request"),
                "models": [],
                "totals": {},
                "spans": [],
                "_path": self.root / mode / f"{ts}-{trace_id}.json",
            }
            self._traces[trace_id] = doc
        else:
            for key, src in (
                ("session_id", "thread_id"),
                ("user_id", "user_id"),
                ("query", "af_query"),
                ("request", "af_request"),
            ):
                if not doc.get(key) and md.get(src):
                    doc[key] = md[src]
        return doc

    def _finish_span(self, trace_id: UUID, span: dict[str, Any]) -> None:
        doc = self._traces.get(trace_id)
        if doc is None:
            return
        doc["spans"].append(span)
        model = span.get("model")
        if span["type"] == "llm" and model and model not in doc["models"]:
            doc["models"].append(model)
        self._recompute(doc)
        self._write(doc)

    def _recompute(self, doc: dict[str, Any]) -> None:
        llm = [s for s in doc["spans"] if s["type"] == "llm"]
        tools = [s for s in doc["spans"] if s["type"] == "tool"]
        t_in = sum((s.get("tokens") or {}).get("input", 0) for s in llm)
        t_out = sum((s.get("tokens") or {}).get("output", 0) for s in llm)
        t_cache = sum((s.get("tokens") or {}).get("cache_read", 0) for s in llm)
        cost = sum(s.get("est_cost_usd") or 0.0 for s in llm)
        end = doc["ended_at"] or _now()
        doc["totals"] = {
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "input_tokens": t_in,
            "output_tokens": t_out,
            "total_tokens": t_in + t_out,
            "cache_read_tokens": t_cache,
            "est_cost_usd": round(cost, 6) if cost else None,
            "duration_s": round(
                (datetime.fromisoformat(end) - datetime.fromisoformat(doc["started_at"])).total_seconds(),
                3,
            ),
        }

    def _write(self, doc: dict[str, Any]) -> None:
        path: Path = doc["_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in doc.items() if not k.startswith("_")}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    def _safe(self, fn: Any, *args: Any) -> None:
        try:
            with self._lock:
                fn(*args)
        except Exception:  # telemetry must never break the agent
            logger.exception("telemetry handler error")

    # -- LLM --------------------------------------------------------------

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kwargs
    ):
        self._safe(self._llm_start, serialized, messages, run_id, parent_run_id, metadata, kwargs)

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, metadata=None, **kwargs
    ):
        wrapped = [[type("P", (), {"type": "human", "content": p})()] for p in prompts]
        self._safe(self._llm_start, serialized, wrapped, run_id, parent_run_id, metadata, kwargs)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self._safe(self._llm_end, response, run_id, None)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._safe(self._llm_end, None, run_id, str(error))

    def _llm_start(self, serialized, messages, run_id, parent_run_id, metadata, kwargs):
        trace_id = self._root_of(run_id, parent_run_id)
        self._ensure_trace(trace_id, metadata)
        self._perf[run_id] = time.perf_counter()
        inv = kwargs.get("invocation_params") or {}
        flat = messages[0] if messages and isinstance(messages[0], list) else messages
        self._open[run_id] = {
            "type": "llm",
            "run_id": str(run_id),
            "trace_id": str(trace_id),
            "name": (serialized or {}).get("name") or "llm",
            "model": inv.get("model") or inv.get("model_name") or inv.get("model_id"),
            "started_at": _now(),
            "params": {
                "temperature": inv.get("temperature"),
                "max_tokens": inv.get("max_tokens"),
                "tools": _tool_names(inv.get("tools")),
            },
            "input_messages": [_dump_message(m) for m in flat if isinstance(m, BaseMessage)]
            or [{"role": getattr(m, "type", "user"), "content": getattr(m, "content", str(m))} for m in flat],
        }

    def _llm_end(self, response: LLMResult | None, run_id: UUID, error: str | None):
        span = self._open.pop(run_id, None)
        if span is None:
            return
        started = self._perf.pop(run_id, None)
        span["ended_at"] = _now()
        span["duration_s"] = round(time.perf_counter() - started, 3) if started else None
        trace_id = UUID(span["trace_id"])

        if error is not None:
            span["error"] = error
            span["output"] = None
            span["tokens"] = {"input": 0, "output": 0, "total": 0, "cache_read": 0, "cache_creation": 0}
            span["est_cost_usd"] = None
            self._finish_span(trace_id, span)
            return

        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = getattr(gen, "message", None)
        text = ""
        blocks: list[Any] | None = None
        tool_calls: list[Any] = []
        if msg is not None:
            raw = msg.content
            if isinstance(raw, str):
                text = raw
            else:
                blocks = raw
                text = "".join(
                    b.get("text", "")
                    for b in raw
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            tool_calls = getattr(msg, "tool_calls", None) or []
            span["response_metadata"] = _jsonable(getattr(msg, "response_metadata", {}) or {})
        elif gen is not None:
            text = getattr(gen, "text", "")

        tokens = _tokens(response, msg)
        model = span.get("model") or _model_of(response, None)
        span["model"] = model
        span["output"] = {
            "text": text,
            "content_blocks": _jsonable(blocks) if blocks is not None else None,
            "tool_calls": [_jsonable(tc) for tc in tool_calls],
            "finish_reason": _finish_reason(gen, msg),
        }
        span["tokens"] = tokens
        span["est_cost_usd"] = _claude_cost(model, tokens)
        self._finish_span(trace_id, span)

    # -- tools ----------------------------------------------------------

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, metadata=None, inputs=None, **kwargs
    ):
        self._safe(
            self._tool_start, serialized, input_str, inputs, run_id, parent_run_id, metadata, kwargs
        )

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._safe(self._tool_end, output, run_id, None)

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._safe(self._tool_end, None, run_id, str(error))

    def _tool_start(self, serialized, input_str, inputs, run_id, parent_run_id, metadata, kwargs):
        trace_id = self._root_of(run_id, parent_run_id)
        self._ensure_trace(trace_id, metadata)
        self._perf[run_id] = time.perf_counter()
        # LangGraph passes the originating tool_call id through here so the viewer
        # can wire each tool run back to the LLM tool_call that requested it.
        tc_id = (metadata or {}).get("tool_call_id") or (kwargs or {}).get("tool_call_id")
        self._open[run_id] = {
            "type": "tool",
            "run_id": str(run_id),
            "trace_id": str(trace_id),
            "name": (serialized or {}).get("name") or "tool",
            "tool_call_id": tc_id,
            "started_at": _now(),
            "input": _jsonable(inputs if inputs is not None else input_str),
        }

    def _tool_end(self, output: Any, run_id: UUID, error: str | None):
        span = self._open.pop(run_id, None)
        if span is None:
            return
        started = self._perf.pop(run_id, None)
        span["ended_at"] = _now()
        span["duration_s"] = round(time.perf_counter() - started, 3) if started else None
        span["output"] = None if error is not None else _jsonable(getattr(output, "content", output))
        span["error"] = error
        if not span.get("tool_call_id"):
            span["tool_call_id"] = getattr(output, "tool_call_id", None)
        self._finish_span(UUID(span["trace_id"]), span)

    # -- chain tree + trace finalize ----------------------------------

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        self._safe(self._chain_start, run_id, parent_run_id, metadata)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        self._safe(self._finalize, run_id, None)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._safe(self._finalize, run_id, str(error))

    def _chain_start(self, run_id: UUID, parent_run_id: UUID | None, metadata):
        # Record every chain link so a later llm/tool span can walk up to the
        # run whose parent is None -- that run is the trace.
        self._parent[run_id] = parent_run_id
        if parent_run_id is None:
            self._ensure_trace(run_id, metadata)

    def _finalize(self, run_id: UUID, error: str | None):
        doc = self._traces.get(run_id)  # only a root run's id keys a trace
        if doc is None:
            return
        doc["ended_at"] = _now()
        if error:
            doc["error"] = error
        self._recompute(doc)
        self._write(doc)
        # kept in memory (not popped) so a trailing span can't recreate the file
