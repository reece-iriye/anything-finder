"""Stability guard for the comparison dataset inlined into the report.

``scripts/compare_report.html`` reads these keys by name from the JSON island,
and the frozen ``data/traces/*.jsonl.gz`` bundles are built against the trace
schema in ``src/utils/telemetry.py``. Pin both key sets so a rename fails here
rather than as a silently blank column in the report. Update intentionally when
the contract changes — and update the template in the same commit.
"""

from __future__ import annotations

from scripts import build_compare_report as bcr

_TRACE = {
    "trace_id": "t-1",
    "mode": "claude",
    "backend": "claude",
    "started_at": "2026-01-01T00:00:00+00:00",
    "ended_at": "2026-01-01T00:00:09+00:00",
    "session_id": "sess-1",
    "eval_id": "q1",
    "query": "sushi in Deep Ellum",
    "models": ["claude-sonnet-5"],
    "spans": [
        {
            "type": "llm",
            "model": "claude-sonnet-5",
            "input_messages": [{"role": "user", "content": "sushi"}],
            "output": {"text": "Try Deep Sushi in Deep Ellum.", "tool_calls": [], "content_blocks": None},
            "tokens": {"input": 10, "output": 5, "cache_read": 0},
        }
    ],
    "totals": {
        "llm_calls": 1,
        "tool_calls": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 0,
        "est_cost_usd": 0.0001,
        "duration_s": 9.0,
    },
}

_QUERIES = {
    "q1": {
        "id": "q1",
        "session_id": "sess-1",
        "query": "sushi in Deep Ellum",
        "city": "Dallas",
        "state": "TX",
        "target_neighborhood": "Deep Ellum",
        "target_cuisine": "sushi",
        "target_vibe": "quiet",
    }
}


def _dataset():
    return bcr.build_dataset(_QUERIES, {"claude": [_TRACE]}, {})


def test_top_level_keys():
    assert set(_dataset()) == {"generated_at", "modes", "aggregates", "queries"}


def test_query_row_keys():
    row = _dataset()["queries"][0]
    assert set(row) == {
        "id",
        "query",
        "city",
        "state",
        "target_neighborhood",
        "target_cuisine",
        "target_vibe",
        "results",
    }


def test_result_keys():
    result = _dataset()["queries"][0]["results"]["claude"]
    assert set(result) == {
        "model",
        "answer",
        "error",
        "latency_s",
        "tool_path",
        "tool_path_sig",
        "totals",
        "hit_neighborhood",
        "hit_cuisine",
        "grounded_search",
        "trace",
    }


def test_aggregate_keys():
    agg = _dataset()["aggregates"][0]
    assert set(agg) == {
        "mode",
        "models",
        "runs",
        "errors",
        "error_rate",
        "llm_calls_mean",
        "tool_calls_mean",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "tokens_per_run",
        "est_cost_usd",
        "latency_mean",
        "latency_p50",
        "hit_neighborhood",
        "hit_cuisine",
        "grounded_search",
        "tool_paths",
    }


def test_mode_names_match_the_telemetry_buckets():
    """The report's columns are the trace_mode() buckets — keep them in step."""
    import src.utils.telemetry as tele

    buckets = set()
    for backend in ("claude", "vllm", "lora"):
        import os

        old = os.environ.get("LLM_BACKEND")
        os.environ["LLM_BACKEND"] = backend
        os.environ.pop("AF_TRACE_MODE", None)
        buckets.add(tele.trace_mode())
        if old is None:
            os.environ.pop("LLM_BACKEND", None)
        else:
            os.environ["LLM_BACKEND"] = old
    assert buckets == set(bcr.MODE_ORDER)


def test_run_file_map_covers_every_mode():
    assert set(bcr.RUN_FILES) == set(bcr.MODE_ORDER)
