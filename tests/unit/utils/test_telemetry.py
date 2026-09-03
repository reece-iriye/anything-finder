import json
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult
import pytest

import src.utils.telemetry as tele


@pytest.fixture(autouse=True)
def _reset_tracer(monkeypatch):
    monkeypatch.setattr(tele, "_tracer", None)
    for var in ("AF_TRACE_DIR", "AF_TRACE_MODE", "LLM_BACKEND"):
        monkeypatch.delenv(var, raising=False)


# ── mode + factory ─────────────────────────────────────────────────────────


def test_trace_mode_derived_from_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "claude")
    assert tele.trace_mode() == "claude"
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    assert tele.trace_mode() == "raw-open-source"
    monkeypatch.setenv("LLM_BACKEND", "lora")
    assert tele.trace_mode() == "finetuned-open-source"


def test_trace_mode_explicit_override(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("AF_TRACE_MODE", "finetuned-open-source")
    assert tele.trace_mode() == "finetuned-open-source"


def test_get_tracer_is_none_without_env():
    assert tele.get_tracer() is None


def test_get_tracer_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    assert tele.get_tracer() is tele.get_tracer()


# ── run_config ─────────────────────────────────────────────────────────────


def test_run_config_without_trace_dir_is_plain():
    cfg = tele.run_config(thread_id="t-1", user_id="u-1", query="sushi")
    assert cfg == {"configurable": {"thread_id": "t-1", "user_id": "u-1"}}
    assert "callbacks" not in cfg and "metadata" not in cfg


def test_run_config_attaches_tracer_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    cfg = tele.run_config(
        thread_id="t-1",
        user_id="u-1",
        query="sushi",
        request={"query": "sushi", "radius_m": 800},
        eval_id="row-7",
    )
    assert cfg["configurable"] == {"thread_id": "t-1", "user_id": "u-1"}
    assert cfg["callbacks"] == [tele.get_tracer()]
    assert cfg["metadata"] == {
        "thread_id": "t-1",
        "user_id": "u-1",
        "af_query": "sushi",
        "af_request": {"query": "sushi", "radius_m": 800},
        "af_eval_id": "row-7",
    }


def test_run_config_omits_absent_optionals(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    md = tele.run_config(thread_id="t", user_id="u", query="q")["metadata"]
    assert "af_request" not in md and "af_eval_id" not in md


def test_eval_id_lands_in_trace(tmp_path, monkeypatch):
    """The eval row id is the join key between a trace and its query."""
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    tracer = tele.get_tracer()
    root = uuid4()
    cfg = tele.run_config(thread_id="sess-9", user_id="u-1", query="tacos", eval_id="row-7")
    tracer.on_chain_start({}, {}, run_id=root, parent_run_id=None, metadata=cfg["metadata"])
    tracer.on_chain_end({}, run_id=root)

    doc = json.loads(next((tmp_path / "raw-open-source").glob("*.json")).read_text())
    assert doc["eval_id"] == "row-7"
    assert doc["session_id"] == "sess-9"
    assert doc["query"] == "tacos"


def test_eval_id_backfilled_when_first_span_lacked_it(tmp_path, monkeypatch):
    """A span can open the trace before the metadata-carrying chain start."""
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    tracer = tele.JsonFileTracer(tmp_path)
    root, tool = uuid4(), uuid4()
    tracer.on_tool_start({"name": "t"}, "x", run_id=tool, parent_run_id=root, metadata={})
    tracer.on_tool_end("y", run_id=tool)
    tracer._ensure_trace(root, {"af_eval_id": "row-3", "af_query": "pho"})
    assert tracer._traces[root]["eval_id"] == "row-3"


# ── end-to-end span capture ────────────────────────────────────────────────


def _chat_result(text, *, tool_calls=None, model="claude-sonnet-4-6", tokens=(10, 5)):
    msg = AIMessage(
        content=text,
        tool_calls=tool_calls or [],
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
        },
        response_metadata={"stop_reason": "tool_use" if tool_calls else "end_turn"},
    )
    return LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output={"model_name": model},
    )


def test_full_trace_written_with_llm_and_tool_spans(tmp_path):
    tracer = tele.JsonFileTracer(tmp_path)
    root, llm1, tool1, llm2 = uuid4(), uuid4(), uuid4(), uuid4()
    md = {"thread_id": "sess-1", "user_id": "u-1", "af_query": "quiet sushi in Deep Ellum"}

    md["af_request"] = {"query": "quiet sushi in Deep Ellum", "include_casual": False, "radius_m": None}
    tracer.on_chain_start({}, {}, run_id=root, parent_run_id=None, metadata=md)

    tracer.on_chat_model_start(
        {"name": "ChatAnthropic"},
        [[SystemMessage("be helpful"), HumanMessage("quiet sushi in Deep Ellum")]],
        run_id=llm1,
        parent_run_id=root,
        metadata=md,
        invocation_params={"model": "claude-sonnet-4-6", "temperature": 0.3, "tools": [{"name": "geocode_location"}]},
    )
    tracer.on_llm_end(
        _chat_result("", tool_calls=[{"name": "geocode_location", "args": {"q": "Deep Ellum"}, "id": "t1"}]),
        run_id=llm1,
    )

    tracer.on_tool_start({"name": "geocode_location"}, "Deep Ellum", run_id=tool1, parent_run_id=root, inputs={"q": "Deep Ellum"})
    tracer.on_tool_end({"lat": 32.78, "lon": -96.78}, run_id=tool1)

    tracer.on_chat_model_start(
        {"name": "ChatAnthropic"},
        [[HumanMessage("(tool result)")]],
        run_id=llm2,
        parent_run_id=root,
        invocation_params={"model": "claude-sonnet-4-6", "temperature": 0.3},
    )
    tracer.on_llm_end(_chat_result("Try Near Sushi.", tokens=(20, 8)), run_id=llm2)

    tracer.on_chain_end({}, run_id=root)

    files = list((tmp_path / "raw-open-source").glob("*.json"))  # default mode (no LLM_BACKEND)
    assert len(files) == 1
    doc = json.loads(files[0].read_text())

    assert doc["query"] == "quiet sushi in Deep Ellum"
    assert doc["session_id"] == "sess-1"
    assert doc["request"]["query"] == "quiet sushi in Deep Ellum"
    assert doc["backend"] == "vllm"  # no LLM_BACKEND set
    assert [s["type"] for s in doc["spans"]] == ["llm", "tool", "llm"]

    first_llm = doc["spans"][0]
    assert first_llm["model"] == "claude-sonnet-4-6"
    assert first_llm["params"]["tools"] == ["geocode_location"]
    assert first_llm["output"]["tool_calls"][0]["name"] == "geocode_location"
    assert [m["role"] for m in first_llm["input_messages"]] == ["system", "user"]

    tool_span = doc["spans"][1]
    assert tool_span["name"] == "geocode_location"
    assert tool_span["input"] == {"q": "Deep Ellum"}
    assert tool_span["output"] == {"lat": 32.78, "lon": -96.78}

    assert doc["totals"]["llm_calls"] == 2
    assert doc["totals"]["tool_calls"] == 1
    assert doc["totals"]["input_tokens"] == 30
    assert doc["totals"]["output_tokens"] == 13
    assert doc["totals"]["est_cost_usd"] is not None  # claude-priced model


def test_claude_mode_and_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("AF_TRACE_MODE", "claude")
    tracer = tele.JsonFileTracer(tmp_path)
    root, llm = uuid4(), uuid4()

    tracer.on_chain_start({}, {}, run_id=root, parent_run_id=None, metadata={})
    tracer.on_chat_model_start(
        {"name": "ChatAnthropic"}, [[HumanMessage("hi")]],
        run_id=llm, parent_run_id=root,
        invocation_params={"model": "claude-sonnet-4-6"},
    )
    tracer.on_llm_end(_chat_result("hello", tokens=(1_000_000, 1_000_000)), run_id=llm)
    tracer.on_chain_end({}, run_id=root)

    doc = json.loads(next((tmp_path / "claude").glob("*.json")).read_text())
    # 1M input @ $3 + 1M output @ $15 = $18.00
    assert doc["spans"][0]["est_cost_usd"] == pytest.approx(18.0)


def test_llm_error_is_recorded(tmp_path):
    tracer = tele.JsonFileTracer(tmp_path)
    root, llm = uuid4(), uuid4()
    tracer.on_chain_start({}, {}, run_id=root, parent_run_id=None, metadata={})
    tracer.on_chat_model_start({"name": "m"}, [[HumanMessage("hi")]], run_id=llm, parent_run_id=root, invocation_params={})
    tracer.on_llm_error(ValueError("boom"), run_id=llm)
    tracer.on_chain_end({}, run_id=root)

    doc = json.loads(next((tmp_path / "raw-open-source").glob("*.json")).read_text())
    assert doc["spans"][0]["error"] == "boom"


def test_handler_never_raises_on_bad_payload(tmp_path):
    tracer = tele.JsonFileTracer(tmp_path)
    # end without a matching start must be a silent no-op
    tracer.on_llm_end(_chat_result("x"), run_id=uuid4())
    tracer.on_tool_end("x", run_id=uuid4())
