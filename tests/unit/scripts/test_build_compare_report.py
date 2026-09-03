"""The comparison report is only as good as its join: the same eval row has to
line up across three modes, and the file has to open from disk with no network.

Hermetic — builds a tiny telemetry tree + run files in tmp_path.
"""

import csv
import gzip
import json
import re

import pytest

from scripts import build_compare_report as bcr


# ── fixtures ───────────────────────────────────────────────────────────────


def _trace(eval_id, mode, *, tools=(), answer="Try Sushi Bar in Deep Ellum.", tokens=(100, 40), error=None):
    spans = [
        {
            "type": "llm",
            "model": f"{mode}-model",
            "input_messages": [{"role": "system", "content": "x" * 500}],
            "output": {"text": "", "tool_calls": [{"name": t, "args": {}} for t in tools], "content_blocks": None},
            "tokens": {"input": tokens[0], "output": tokens[1], "cache_read": 0},
        }
    ]
    spans += [{"type": "tool", "name": t, "input": {}, "output": {"ok": True}} for t in tools]
    spans.append(
        {
            "type": "llm",
            "model": f"{mode}-model",
            "input_messages": [{"role": "system", "content": "x" * 500}],
            "output": {"text": answer, "tool_calls": [], "content_blocks": None},
            "tokens": {"input": tokens[0], "output": tokens[1], "cache_read": 0},
        }
    )
    return {
        "trace_id": f"{mode}-{eval_id}",
        "mode": mode,
        "backend": mode,
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:12+00:00",
        "session_id": f"sess-{eval_id}",
        "eval_id": eval_id,
        "query": "quiet sushi in Deep Ellum",
        "models": [f"{mode}-model"],
        "error": error,
        "spans": spans,
        "totals": {
            "llm_calls": 2,
            "tool_calls": len(tools),
            "input_tokens": tokens[0] * 2,
            "output_tokens": tokens[1] * 2,
            "cache_read_tokens": 0,
            "est_cost_usd": 0.001 if mode == "claude" else None,
            "duration_s": 12.0,
        },
    }


@pytest.fixture
def workspace(tmp_path):
    """A trace dir with all three modes over two shared eval rows."""
    queries = tmp_path / "queries.csv"
    with queries.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["id", "session_id", "query", "city", "state",
                        "target_neighborhood", "target_cuisine", "target_vibe"],
        )
        w.writeheader()
        w.writerow({"id": "q1", "session_id": "sess-q1", "query": "quiet sushi in Deep Ellum",
                    "city": "Dallas", "state": "TX", "target_neighborhood": "Deep Ellum",
                    "target_cuisine": "sushi", "target_vibe": "quiet"})
        w.writerow({"id": "q2", "session_id": "sess-q2", "query": "tacos in Bishop Arts",
                    "city": "Dallas", "state": "TX", "target_neighborhood": "Bishop Arts",
                    "target_cuisine": "tacos", "target_vibe": "lively"})

    traces = tmp_path / "telemetry"
    plan = {
        "claude": {
            "q1": dict(tools=("read_food_preferences", "geocode_location", "search_restaurants")),
            "q2": dict(tools=("geocode_location", "search_restaurants"), answer="Try Taco Spot in Bishop Arts."),
        },
        "raw-open-source": {
            # never geocodes: searches blind, and does not name the area
            "q1": dict(tools=("search_restaurants",), answer="Some place downtown.", tokens=(300, 90)),
        },
        "finetuned-open-source": {
            "q1": dict(tools=("read_food_preferences", "geocode_location", "search_restaurants"),
                       tokens=(150, 50)),
            "q2": dict(tools=("geocode_location", "search_restaurants"),
                       answer="Try Taco Spot in Bishop Arts.", tokens=(150, 50)),
        },
    }
    for mode, rows in plan.items():
        d = traces / mode
        d.mkdir(parents=True)
        for eval_id, kwargs in rows.items():
            doc = _trace(eval_id, mode, **kwargs)
            (d / f"20260101-0000{eval_id}-{mode}.json").write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path, queries, traces


def _build(workspace, **kw):
    tmp_path, queries, traces = workspace
    return bcr.build_dataset(
        bcr.load_queries(queries),
        bcr.load_traces(traces, tmp_path / "no-frozen"),
        {},
        **kw,
    )


# ── joining ────────────────────────────────────────────────────────────────


def test_modes_are_ordered_teacher_then_student(workspace):
    assert _build(workspace)["modes"] == ["claude", "raw-open-source", "finetuned-open-source"]


def test_rows_join_on_eval_id_across_modes(workspace):
    data = _build(workspace)
    q1 = next(q for q in data["queries"] if q["id"] == "q1")
    assert set(q1["results"]) == {"claude", "raw-open-source", "finetuned-open-source"}
    assert q1["target_neighborhood"] == "Deep Ellum"
    assert q1["results"]["claude"]["model"] == "claude-model"


def test_a_mode_missing_a_query_degrades_gracefully(workspace):
    """raw-open-source never ran q2 — the row still renders, minus that column."""
    data = _build(workspace)
    q2 = next(q for q in data["queries"] if q["id"] == "q2")
    assert "raw-open-source" not in q2["results"]
    assert set(q2["results"]) == {"claude", "finetuned-open-source"}


# ── trimming for a smaller, focused report ─────────────────────────────────


def test_common_run_only_drops_rows_a_mode_never_attempted(workspace):
    data = _build(workspace)
    trimmed = bcr.filter_common_success(data, common_run_only=True)
    assert [q["id"] for q in trimmed["queries"]] == ["q1"]  # q2 never ran raw-open-source


def test_common_run_only_keeps_an_errored_attempt(workspace):
    """An error still counts as 'ran' — common_run_only shows it, doesn't drop it."""
    data = _build(workspace)
    data["queries"][0]["results"]["claude"]["error"] = "boom"
    trimmed = bcr.filter_common_success(data, common_run_only=True)
    assert [q["id"] for q in trimmed["queries"]] == ["q1"]
    q1 = trimmed["queries"][0]
    assert q1["results"]["claude"]["error"] == "boom"


def test_common_success_only_drops_rows_any_mode_missed_or_errored(workspace):
    data = _build(workspace)
    trimmed = bcr.filter_common_success(data, common_success_only=True)
    ids = [q["id"] for q in trimmed["queries"]]
    assert ids == ["q1"]  # q2 is missing from raw-open-source


def test_common_success_only_drops_rows_with_any_error(workspace):
    data = _build(workspace)
    data["queries"][0]["results"]["claude"]["error"] = "boom"
    trimmed = bcr.filter_common_success(data, common_success_only=True)
    assert trimmed["queries"] == []


def test_max_queries_caps_row_count_in_eval_set_order(workspace):
    data = _build(workspace)
    trimmed = bcr.filter_common_success(data, max_queries=1)
    assert [q["id"] for q in trimmed["queries"]] == ["q1"]


def test_trimming_recomputes_aggregates_over_the_kept_rows_only(workspace):
    data = _build(workspace)
    trimmed = bcr.filter_common_success(data, common_success_only=True)
    claude_agg = next(a for a in trimmed["aggregates"] if a["mode"] == "claude")
    assert claude_agg["runs"] == 1  # not the untrimmed 2


def test_no_flags_leaves_the_dataset_untouched(workspace):
    data = _build(workspace)
    same = bcr.filter_common_success(data)
    assert [q["id"] for q in same["queries"]] == [q["id"] for q in data["queries"]]


def test_trace_falls_back_to_session_id(tmp_path):
    """Traces captured before eval_id existed still join, via session_id."""
    doc = _trace("q1", "claude")
    del doc["eval_id"]
    d = tmp_path / "telemetry" / "claude"
    d.mkdir(parents=True)
    (d / "t.json").write_text(json.dumps(doc), encoding="utf-8")
    queries = {"q1": {"id": "q1", "session_id": "sess-q1", "query": "quiet sushi in Deep Ellum"}}
    data = bcr.build_dataset(queries, bcr.load_traces(tmp_path / "telemetry", tmp_path / "x"), {})
    assert data["queries"][0]["id"] == "q1"
    assert "claude" in data["queries"][0]["results"]


def test_run_records_fill_in_when_a_trace_is_missing(tmp_path):
    runs = {"claude": {"q9": {"id": "q9", "query": "pho", "completion": "Try Pho Bar.",
                              "latency_s": 4.2, "model": "claude-sonnet-5",
                              "tool_calls": [{"name": "search_restaurants", "args": {}}]}}}
    data = bcr.build_dataset({}, {}, runs)
    r = data["queries"][0]["results"]["claude"]
    assert r["answer"] == "Try Pho Bar." and r["latency_s"] == 4.2
    assert r["tool_path"] == ["search_restaurants"] and r["trace"] is None


def test_anthropic_content_blocks_are_flattened():
    blocks = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Try Pho Bar."}]
    runs = {"claude": {"q1": {"id": "q1", "completion": blocks}}}
    data = bcr.build_dataset({}, {}, runs)
    assert data["queries"][0]["results"]["claude"]["answer"] == "Try Pho Bar."


def test_frozen_bundles_are_read_when_no_live_traces(tmp_path):
    frozen = tmp_path / "traces"
    frozen.mkdir()
    with gzip.open(frozen / "claude.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_trace("q1", "claude")) + "\n")
    traces = bcr.load_traces(tmp_path / "absent", frozen)
    assert [d["eval_id"] for d in traces["claude"]] == ["q1"]


# ── derived metrics ────────────────────────────────────────────────────────


def test_tool_path_and_signature_collapse_repeats():
    assert bcr.path_signature(["a", "b", "b", "b", "c"]) == "a → b ×3 → c"
    assert bcr.path_signature([]) == "(no tools)"


def test_grounded_search_detects_searching_without_geocoding():
    assert bcr.geocode_before_search(["geocode_location", "search_restaurants"]) is True
    assert bcr.geocode_before_search(["search_restaurants"]) is False
    assert bcr.geocode_before_search(["read_food_preferences"]) is None  # never searched


def test_label_hits_are_scored_against_the_answer(workspace):
    data = _build(workspace)
    q1 = next(q for q in data["queries"] if q["id"] == "q1")
    assert q1["results"]["claude"]["hit_neighborhood"] is True
    assert q1["results"]["raw-open-source"]["hit_neighborhood"] is False
    assert q1["results"]["raw-open-source"]["grounded_search"] is False
    assert q1["results"]["finetuned-open-source"]["grounded_search"] is True


def test_aggregates_summarize_each_mode(workspace):
    aggs = {a["mode"]: a for a in _build(workspace)["aggregates"]}
    assert aggs["claude"]["runs"] == 2 and aggs["raw-open-source"]["runs"] == 1
    assert aggs["claude"]["llm_calls_mean"] == 2
    assert aggs["raw-open-source"]["grounded_search"] == 0.0
    assert aggs["finetuned-open-source"]["grounded_search"] == 1.0
    assert aggs["claude"]["est_cost_usd"] == pytest.approx(0.002)
    assert aggs["raw-open-source"]["est_cost_usd"] is None  # unpriced backend
    # the base model burns more tokens per run than the tuned one
    assert aggs["raw-open-source"]["tokens_per_run"] > aggs["finetuned-open-source"]["tokens_per_run"]


def test_aggregate_tool_paths_are_counted(workspace):
    aggs = {a["mode"]: a for a in _build(workspace)["aggregates"]}
    assert aggs["raw-open-source"]["tool_paths"] == {"search_restaurants": 1}


def test_error_rate_counts_failed_runs(tmp_path):
    d = tmp_path / "telemetry" / "claude"
    d.mkdir(parents=True)
    (d / "ok.json").write_text(json.dumps(_trace("q1", "claude")), encoding="utf-8")
    (d / "bad.json").write_text(
        json.dumps(_trace("q2", "claude", error="ValueError: No coordinates found")), encoding="utf-8"
    )
    data = bcr.build_dataset({}, bcr.load_traces(tmp_path / "telemetry", tmp_path / "x"), {})
    agg = data["aggregates"][0]
    assert agg["runs"] == 2 and agg["errors"] == 1 and agg["error_rate"] == 0.5


# ── slimming + rendering ───────────────────────────────────────────────────


def test_slim_drops_input_messages_but_keeps_the_count(workspace):
    data = _build(workspace)
    span = data["queries"][0]["results"]["claude"]["trace"]["spans"][0]
    assert span["input_messages"] == []
    assert span["input_message_count"] == 1


def test_full_keeps_input_messages(workspace):
    data = _build(workspace, full=True)
    span = data["queries"][0]["results"]["claude"]["trace"]["spans"][0]
    assert span["input_messages"] and "input_message_count" not in span


def test_report_is_self_contained(workspace):
    html = bcr.render(_build(workspace))
    assert not re.search(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', html), "external resource"
    assert "/*@include" not in html, "unresolved asset marker"
    assert "/*@dataset*/null" not in html, "dataset not inlined"
    # the shared renderer and palette came along
    assert "renderConversationInto" in html and "--user-bubble" in html


def test_inlined_dataset_parses_back(workspace):
    data = _build(workspace)
    html = bcr.render(data)
    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    parsed = json.loads(payload.replace("<\\/", "</"))
    assert parsed["modes"] == data["modes"]
    assert len(parsed["queries"]) == len(data["queries"])


def test_script_close_in_model_output_cannot_break_out(tmp_path):
    """An answer containing </script> must not terminate the data island."""
    runs = {"claude": {"q1": {"id": "q1", "completion": "hi </script><script>alert(1)</script>"}}}
    html = bcr.render(bcr.build_dataset({}, {}, runs))
    body = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert "</script>" not in body
    assert "<\\/script>" in body


def test_render_with_no_data_still_produces_a_page():
    html = bcr.render(bcr.build_dataset({}, {}, {}))
    assert "<title>" in html and '"queries": []' in html.replace(" ", " ")
