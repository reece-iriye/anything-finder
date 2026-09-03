"""The capture runner is what makes the comparison possible: it must honour the
selected backend, write to the requested file, and — crucially — leave a
telemetry trace joinable back to its eval row.

Hermetic: the agent, LLM, and HTTP clients are all stubbed, so nothing here
touches a model endpoint, Nominatim, or Overpass.
"""

import asyncio
import csv
import json
from argparse import Namespace
from uuid import uuid4

import pytest

from scripts import run_eval_queries as rq
import src.utils.telemetry as tele


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setattr(tele, "_tracer", None)
    # The runner loads the repo's real .env; neutralize it so these tests do not
    # depend on whether the developer happens to have credentials configured.
    monkeypatch.setattr(rq, "_load_dotenv", lambda path: [])
    for var in ("AF_TRACE_DIR", "AF_TRACE_MODE", "LLM_BACKEND", "LLM_MODEL_AGENT",
                "LLM_MODEL_CLAUDE", "LLM_MODEL_LORA", "LLM_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class _StubAgent:
    """Stands in for the compiled deep agent.

    Drives the tracer the way a real run would — one LLM span that requests a
    tool, the tool span, then the final answer — using the callbacks the runner
    put in the config. That is exactly the wiring under test.
    """

    def __init__(self, answer="Try Deep Sushi in Deep Ellum."):
        self.answer = answer
        self.configs = []

    async def ainvoke(self, payload, config=None):
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        self.configs.append(config)
        cbs = (config or {}).get("callbacks") or []
        md = (config or {}).get("metadata")
        root, llm1, tool1, llm2 = uuid4(), uuid4(), uuid4(), uuid4()

        def result(text, tool_calls=None):
            msg = AIMessage(
                content=text,
                tool_calls=tool_calls or [],
                usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16,
                                "input_token_details": {"cache_read": 0, "cache_creation": 0}},
            )
            return LLMResult(generations=[[ChatGeneration(message=msg)]],
                             llm_output={"model_name": "stub-model"})

        for cb in cbs:
            cb.on_chain_start({}, {}, run_id=root, parent_run_id=None, metadata=md)
            cb.on_chat_model_start({"name": "stub"}, [[HumanMessage("q")]], run_id=llm1,
                                   parent_run_id=root, metadata=md,
                                   invocation_params={"model": "stub-model"})
            cb.on_llm_end(result("", [{"name": "geocode_location", "args": {"q": "Deep Ellum"}, "id": "t1"}]),
                          run_id=llm1)
            cb.on_tool_start({"name": "geocode_location"}, "Deep Ellum", run_id=tool1,
                             parent_run_id=root, inputs={"q": "Deep Ellum"})
            cb.on_tool_end({"lat": 32.78, "lon": -96.78}, run_id=tool1)
            cb.on_chat_model_start({"name": "stub"}, [[HumanMessage("t")]], run_id=llm2,
                                   parent_run_id=root, invocation_params={"model": "stub-model"})
            cb.on_llm_end(result(self.answer), run_id=llm2)
            cb.on_chain_end({}, run_id=root)

        return {"messages": [HumanMessage("q"), AIMessage(content=self.answer)]}


class _StubClient:
    base_url = "http://stub"

    async def get(self, *a, **kw):
        return None

    async def aclose(self):
        return None


@pytest.fixture
def eval_csv(tmp_path, monkeypatch):
    path = tmp_path / "queries.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "user_id", "session_id", "query",
                                           "context_data", "city", "state",
                                           "target_neighborhood", "target_cuisine", "target_vibe"])
        w.writeheader()
        for i in (1, 2):
            w.writerow({"id": f"q{i}", "user_id": f"u{i}", "session_id": f"sess-q{i}",
                        "query": f"query {i}", "context_data": "# prefs", "city": "Dallas",
                        "state": "TX", "target_neighborhood": "Deep Ellum",
                        "target_cuisine": "sushi", "target_vibe": "quiet"})
    monkeypatch.setattr(rq, "IN_PATH", path)
    return path


@pytest.fixture
def stub_agent(monkeypatch):
    agent = _StubAgent()
    import src.agents.geo_search.agent as agent_mod
    import src.utils.llm as llm_mod
    import src.utils.nominatim as nom_mod
    import src.utils.overpass as ovp_mod

    monkeypatch.setattr(agent_mod, "build_restaurant_agent", lambda *a, **kw: agent)
    monkeypatch.setattr(llm_mod, "make_llm", lambda *a, **kw: object())
    monkeypatch.setattr(nom_mod, "make_nominatim_client", lambda *a, **kw: _StubClient())
    monkeypatch.setattr(ovp_mod, "make_overpass_client", lambda *a, **kw: _StubClient())
    return agent


def _run(tmp_path, **overrides):
    args = Namespace(limit=None, out=None, concurrency=1, resume=False, skip_preflight=True)
    for k, v in overrides.items():
        setattr(args, k, v)
    return asyncio.run(rq._main_async(args))


def _records(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── output routing ─────────────────────────────────────────────────────────


def test_writes_to_the_requested_out_path(tmp_path, eval_csv, stub_agent, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    out = tmp_path / "runs" / "raw-open-source.jsonl"
    assert _run(tmp_path, out=str(out)) == 0

    recs = _records(out)
    assert [r["id"] for r in recs] == ["q1", "q2"]
    assert recs[0]["completion"] == "Try Deep Sushi in Deep Ellum."
    assert recs[0]["mode"] == "raw-open-source"
    assert recs[0]["backend"] == "vllm"


def test_backend_from_env_is_not_overridden(tmp_path, eval_csv, stub_agent, monkeypatch):
    """The Makefile targets pick the backend; the runner must not force claude."""
    monkeypatch.setenv("LLM_BACKEND", "lora")
    monkeypatch.setenv("LLM_MODEL_LORA", "af-lora")
    out = tmp_path / "lora.jsonl"
    _run(tmp_path, out=str(out))
    rec = _records(out)[0]
    assert rec["backend"] == "lora"
    assert rec["mode"] == "finetuned-open-source"
    assert rec["model"] == "af-lora"


def test_claude_without_a_key_fails_before_spending_anything(tmp_path, eval_csv, stub_agent, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "claude")
    assert _run(tmp_path, out=str(tmp_path / "x.jsonl")) == 2
    assert not (tmp_path / "x.jsonl").exists()


def test_lora_without_an_adapter_name_fails_fast(tmp_path, eval_csv, stub_agent, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "lora")
    assert _run(tmp_path, out=str(tmp_path / "x.jsonl")) == 2


def test_limit_takes_the_first_rows_in_file_order(tmp_path, eval_csv, stub_agent, monkeypatch):
    """--limit N must be the SAME N queries for every backend."""
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    out = tmp_path / "one.jsonl"
    _run(tmp_path, out=str(out), limit=1)
    assert [r["id"] for r in _records(out)] == ["q1"]


def test_resume_skips_ids_already_written(tmp_path, eval_csv, stub_agent, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    out = tmp_path / "runs.jsonl"
    _run(tmp_path, out=str(out), limit=1)
    _run(tmp_path, out=str(out), resume=True)
    assert [r["id"] for r in _records(out)] == ["q1", "q2"]


# ── telemetry wiring ───────────────────────────────────────────────────────


def test_each_row_leaves_a_joinable_trace(tmp_path, eval_csv, stub_agent, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    trace_dir = tmp_path / "telemetry"
    monkeypatch.setenv("AF_TRACE_DIR", str(trace_dir))

    _run(tmp_path, out=str(tmp_path / "runs.jsonl"))

    files = sorted((trace_dir / "raw-open-source").glob("*.json"))
    assert len(files) == 2, "one trace per eval row"
    docs = {json.loads(f.read_text())["eval_id"]: json.loads(f.read_text()) for f in files}
    assert set(docs) == {"q1", "q2"}

    doc = docs["q1"]
    assert doc["session_id"] == "sess-q1"
    assert doc["query"] == "query 1"
    assert [s["type"] for s in doc["spans"]] == ["llm", "tool", "llm"]
    assert doc["totals"]["llm_calls"] == 2 and doc["totals"]["tool_calls"] == 1


def test_no_trace_dir_means_no_callbacks_and_no_files(tmp_path, eval_csv, stub_agent, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    _run(tmp_path, out=str(tmp_path / "runs.jsonl"))
    assert "callbacks" not in stub_agent.configs[0]
    assert not list(tmp_path.glob("telemetry/**/*.json"))


def test_traces_feed_the_comparison_report(tmp_path, eval_csv, stub_agent, monkeypatch):
    """End to end: capture -> trace -> joined report row."""
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    trace_dir = tmp_path / "telemetry"
    monkeypatch.setenv("AF_TRACE_DIR", str(trace_dir))
    _run(tmp_path, out=str(tmp_path / "runs.jsonl"))

    from scripts import build_compare_report as bcr

    data = bcr.build_dataset(
        bcr.load_queries(eval_csv),
        bcr.load_traces(trace_dir, tmp_path / "absent"),
        {},
    )
    assert data["modes"] == ["raw-open-source"]
    q1 = next(q for q in data["queries"] if q["id"] == "q1")
    result = q1["results"]["raw-open-source"]
    assert result["tool_path"] == ["geocode_location"]
    assert result["answer"] == "Try Deep Sushi in Deep Ellum."
    assert result["hit_neighborhood"] is True
    assert data["aggregates"][0]["runs"] == 2
