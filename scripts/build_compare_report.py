"""Render a self-contained HTML report comparing the model stacks side by side.

    make compare
    uv run scripts/build_compare_report.py --out data/compare/report.html

Joins three sources on the eval row id:

  * ``data/eval/dallas_food_queries.csv`` — the query and its ground-truth
    ``target_neighborhood`` / ``target_cuisine`` / ``target_vibe`` labels.
  * telemetry traces — ``<trace-dir>/<mode>/*.json`` as written by
    ``src/utils/telemetry.py``, or the frozen ``data/traces/<mode>.jsonl.gz``
    bundles from ``make traces-freeze``. These carry the spans: every LLM call,
    every tool call, tokens, cost, timings.
  * ``data/eval/runs/<mode>.jsonl`` and ``data/eval/*_runs.jsonl`` (optional) —
    the captured run records, used for the final answer and latency when a run
    has no trace.

Output is ONE html file with the joined data inlined — no server, no CDN, no
fetch. It opens from disk and can be attached to a PR.

By default the per-span ``input_messages`` are dropped (they re-send the whole
conversation on every call and dominate the file size); pass ``--full`` to keep
them.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEMPLATE = Path(__file__).with_name("compare_report.html")
ASSET_DIR = Path(__file__).parent

DEFAULT_TRACE_DIR = REPO_ROOT / "telemetry"
DEFAULT_FROZEN_DIR = REPO_ROOT / "data" / "traces"
DEFAULT_QUERIES = REPO_ROOT / "data" / "eval" / "dallas_food_queries.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "compare" / "report.html"

# Display order: teacher first, then the student before and after fine-tuning.
MODE_ORDER = ["claude", "raw-open-source", "finetuned-open-source"]

# Run files keyed by the mode they were captured under.
RUN_FILES = {
    "claude": ["data/eval/dallas_food_runs.jsonl", "data/eval/runs/claude.jsonl"],
    "raw-open-source": ["data/eval/runs/raw-open-source.jsonl"],
    "finetuned-open-source": ["data/eval/runs/finetuned-open-source.jsonl"],
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_queries(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["id"]: row for row in csv.DictReader(fh) if row.get("id")}


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_traces(trace_dir: Path, frozen_dir: Path) -> dict[str, list[dict]]:
    """All traces, grouped by mode. Live trace dir wins over a frozen bundle."""
    by_mode: dict[str, list[dict]] = {}
    if trace_dir.is_dir():
        for mode_dir in sorted(p for p in trace_dir.iterdir() if p.is_dir()):
            docs = []
            for f in sorted(mode_dir.glob("*.json")):
                try:
                    docs.append(json.loads(f.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    continue
            if docs:
                by_mode[mode_dir.name] = docs
    if frozen_dir.is_dir():
        for bundle in sorted(frozen_dir.glob("*.jsonl.gz")):
            mode = bundle.name[: -len(".jsonl.gz")]
            by_mode.setdefault(mode, _read_jsonl(bundle))
    return by_mode


def load_runs(root: Path) -> dict[str, dict[str, dict]]:
    """Captured run records per mode, keyed by eval id."""
    by_mode: dict[str, dict[str, dict]] = {}
    for mode, rels in RUN_FILES.items():
        merged: dict[str, dict] = {}
        for rel in rels:
            path = root / rel
            if not path.is_file():
                continue
            for rec in _read_jsonl(path):
                if rec.get("id"):
                    merged[rec["id"]] = rec
        if merged:
            by_mode[mode] = merged
    return by_mode


# ---------------------------------------------------------------------------
# per-run derived facts
# ---------------------------------------------------------------------------


def tool_path(doc: dict) -> list[str]:
    """Tool names in call order — the run's 'shape'."""
    return [s.get("name") or "?" for s in doc.get("spans") or [] if s.get("type") == "tool"]


def path_signature(names: Iterable[str]) -> str:
    """Collapse consecutive repeats: [a,b,b,b] -> 'a → b ×3'."""
    out: list[str] = []
    for name in names:
        if out and out[-1].split(" ×")[0] == name:
            head = out[-1].split(" ×")[0]
            n = int(out[-1].split(" ×")[1]) if " ×" in out[-1] else 1
            out[-1] = f"{head} ×{n + 1}"
        else:
            out.append(name)
    return " → ".join(out) or "(no tools)"


def as_text(content: Any) -> str:
    """Flatten a message content to plain text.

    Anthropic returns a list of content blocks; the OpenAI-compatible path
    returns a plain string. Captured run records carry whichever the backend
    produced, so normalize before any string work.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return "" if content is None else str(content)


def final_answer(doc: dict) -> str:
    """Last LLM span that produced text and requested no further tools."""
    for span in reversed(doc.get("spans") or []):
        if span.get("type") != "llm":
            continue
        out = span.get("output") or {}
        if out.get("tool_calls"):
            continue
        text = as_text(out.get("text"))
        if text.strip():
            return text
    return ""


def _mentions(answer: str, target: str) -> bool | None:
    """Did the answer name the target? None when there is no target to check."""
    if not target:
        return None
    return target.lower() in (answer or "").lower()


def geocode_before_search(names: list[str]) -> bool | None:
    """Did the run resolve coordinates before searching? None if it never searched."""
    if "search_restaurants" not in names:
        return None
    first_search = names.index("search_restaurants")
    return "geocode_location" in names[:first_search] or "get_current_location" in names[:first_search]


def slim_trace(doc: dict) -> dict:
    """Drop the bulky re-sent conversation, keep a count so the UI can say so."""
    out = dict(doc)
    spans = []
    for span in doc.get("spans") or []:
        s = dict(span)
        if s.get("type") == "llm":
            s["input_message_count"] = len(s.get("input_messages") or [])
            s["input_messages"] = []
            if (s.get("output") or {}).get("content_blocks"):
                s["output"] = {**s["output"], "content_blocks": None}
        spans.append(s)
    out["spans"] = spans
    return out


# ---------------------------------------------------------------------------
# joining
# ---------------------------------------------------------------------------


def trace_key(doc: dict) -> str | None:
    """How a trace ties back to an eval row: eval_id, else session_id."""
    return doc.get("eval_id") or doc.get("session_id") or None


def build_dataset(
    queries: dict[str, dict[str, str]],
    traces: dict[str, list[dict]],
    runs: dict[str, dict[str, dict]],
    *,
    full: bool = False,
) -> dict[str, Any]:
    known = set(traces) | set(runs)
    modes = [m for m in MODE_ORDER if m in known] + sorted(known - set(MODE_ORDER))

    # mode -> eval id -> trace (newest wins when a row was re-run)
    traces_by_key: dict[str, dict[str, dict]] = {}
    for mode, docs in traces.items():
        keyed: dict[str, dict] = {}
        for doc in sorted(docs, key=lambda d: d.get("started_at") or ""):
            key = trace_key(doc)
            if key:
                keyed[key] = doc
        traces_by_key[mode] = keyed

    # Every eval row that any mode touched, ordered by the eval set's own order.
    session_to_id = {
        row["session_id"]: rid for rid, row in queries.items() if row.get("session_id")
    }
    touched: set[str] = set()
    for mode in modes:
        touched |= set(runs.get(mode, {}))
        for key in traces_by_key.get(mode, {}):
            touched.add(session_to_id.get(key, key))
    order = [rid for rid in queries if rid in touched]
    order += sorted(touched - set(order))

    rows: list[dict[str, Any]] = []
    for rid in order:
        q = queries.get(rid, {})
        sess = q.get("session_id")
        entry: dict[str, Any] = {
            "id": rid,
            "query": q.get("query", ""),
            "city": q.get("city", ""),
            "state": q.get("state", ""),
            "target_neighborhood": q.get("target_neighborhood", ""),
            "target_cuisine": q.get("target_cuisine", ""),
            "target_vibe": q.get("target_vibe", ""),
            "results": {},
        }
        for mode in modes:
            doc = traces_by_key.get(mode, {}).get(rid) or (
                traces_by_key.get(mode, {}).get(sess) if sess else None
            )
            rec = runs.get(mode, {}).get(rid)
            if doc is None and rec is None:
                continue
            names = tool_path(doc) if doc else [
                tc.get("name", "?") for tc in (rec or {}).get("tool_calls") or []
            ]
            answer = (doc and final_answer(doc)) or as_text((rec or {}).get("completion"))
            totals = (doc or {}).get("totals") or {}
            if not entry["query"]:
                entry["query"] = (doc or {}).get("query") or (rec or {}).get("query") or ""
            entry["results"][mode] = {
                "model": ((doc or {}).get("models") or [None])[0]
                or (rec or {}).get("model")
                or "",
                "answer": answer,
                "error": (doc or {}).get("error") or (rec or {}).get("error") or None,
                "latency_s": totals.get("duration_s")
                if totals.get("duration_s") is not None
                else (rec or {}).get("latency_s"),
                "tool_path": names,
                "tool_path_sig": path_signature(names),
                "totals": totals,
                "hit_neighborhood": _mentions(answer, entry["target_neighborhood"]),
                "hit_cuisine": _mentions(answer, entry["target_cuisine"]),
                "grounded_search": geocode_before_search(names),
                "trace": (doc if full else slim_trace(doc)) if doc else None,
            }
        rows.append(entry)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "modes": modes,
        "aggregates": aggregate(modes, rows),
        "queries": rows,
    }


def filter_common_success(
    dataset: dict[str, Any],
    *,
    common_run_only: bool = False,
    common_success_only: bool = False,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Trim the query rows for a focused report.

    ``common_run_only`` keeps a row only when every mode in ``dataset["modes"]``
    was actually attempted (has a result entry) — a query only claude ran, say,
    is dropped everywhere, so the comparison never shows an empty column for a
    mode that was simply never captured for that row. An attempt that errored
    still counts as "ran" and is kept (and still shown as an error).

    ``common_success_only`` is the stricter version: also requires that result
    to have no error. ``max_queries`` caps the row count afterward, in the eval
    set's own order.

    Aggregates are recomputed over exactly the rows kept, so the leaderboard
    numbers always match what the report actually shows underneath it.
    """
    modes = dataset["modes"]
    rows = dataset["queries"]
    if common_success_only:
        rows = [
            row
            for row in rows
            if all(
                mode in row["results"] and row["results"][mode].get("error") is None
                for mode in modes
            )
        ]
    elif common_run_only:
        rows = [
            row for row in rows if all(mode in row["results"] for mode in modes)
        ]
    if max_queries is not None:
        rows = rows[:max_queries]
    return {
        **dataset,
        "aggregates": aggregate(modes, rows),
        "queries": rows,
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 2) if values else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _rate(flags: list[bool | None]) -> float | None:
    checked = [f for f in flags if f is not None]
    return round(sum(checked) / len(checked), 3) if checked else None


def aggregate(modes: list[str], rows: list[dict]) -> list[dict[str, Any]]:
    out = []
    for mode in modes:
        results = [r["results"][mode] for r in rows if mode in r["results"]]
        if not results:
            continue
        totals = [r["totals"] or {} for r in results]
        errors = sum(1 for r in results if r["error"])
        paths: dict[str, int] = {}
        for r in results:
            paths[r["tool_path_sig"]] = paths.get(r["tool_path_sig"], 0) + 1
        models = sorted({r["model"] for r in results if r["model"]})
        cost = sum(t.get("est_cost_usd") or 0.0 for t in totals)
        out.append(
            {
                "mode": mode,
                "models": models,
                "runs": len(results),
                "errors": errors,
                "error_rate": round(errors / len(results), 3),
                "llm_calls_mean": _mean([t.get("llm_calls", 0) for t in totals]),
                "tool_calls_mean": _mean([t.get("tool_calls", 0) for t in totals]),
                "input_tokens": sum(t.get("input_tokens", 0) for t in totals),
                "output_tokens": sum(t.get("output_tokens", 0) for t in totals),
                "cache_read_tokens": sum(t.get("cache_read_tokens", 0) for t in totals),
                "tokens_per_run": _mean(
                    [t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in totals]
                ),
                "est_cost_usd": round(cost, 6) if cost else None,
                "latency_mean": _mean(
                    [r["latency_s"] for r in results if r["latency_s"] is not None]
                ),
                "latency_p50": _median(
                    [r["latency_s"] for r in results if r["latency_s"] is not None]
                ),
                "hit_neighborhood": _rate([r["hit_neighborhood"] for r in results]),
                "hit_cuisine": _rate([r["hit_cuisine"] for r in results]),
                "grounded_search": _rate([r["grounded_search"] for r in results]),
                "tool_paths": dict(sorted(paths.items(), key=lambda kv: -kv[1])),
            }
        )
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render(dataset: dict[str, Any], template: Path = TEMPLATE, asset_dir: Path = ASSET_DIR) -> str:
    """Inline the shared assets and the dataset into the template."""
    from scripts.trace_ui import inline_assets

    html = inline_assets(template.read_text(encoding="utf-8"), asset_dir)
    payload = json.dumps(dataset, ensure_ascii=False, default=str)
    # </script> anywhere in captured model output would close the data island early.
    payload = payload.replace("</", "<\\/")
    return html.replace("/*@dataset*/null", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--full",
        action="store_true",
        help="keep per-span input_messages (much larger file)",
    )
    parser.add_argument(
        "--common-run-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only queries every mode actually attempted (default: on) — "
        "an error still counts as attempted and is shown; a mode that never "
        "ran a query drops that query everywhere, so no column is ever blank "
        "for 'not captured'. Pass --no-common-run-only to see every row a "
        "single mode touched.",
    )
    parser.add_argument(
        "--common-success-only",
        action="store_true",
        help="stricter than --common-run-only: also drop rows where any mode "
        "errored, instead of showing the error",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="cap the number of query rows in the report, in eval-set order",
    )
    args = parser.parse_args()

    queries = load_queries(Path(args.queries))
    traces = load_traces(Path(args.trace_dir), Path(args.frozen_dir))
    runs = load_runs(REPO_ROOT)

    if not traces and not runs:
        print(
            "nothing to compare: no traces under "
            f"{args.trace_dir}/<mode>/ or {args.frozen_dir}/<mode>.jsonl.gz, and no "
            "run files under data/eval/.\n"
            "  Capture some first:  make compare-run",
            file=sys.stderr,
        )
        raise SystemExit(2)

    dataset = build_dataset(queries, traces, runs, full=args.full)
    if args.common_run_only or args.common_success_only or args.max_queries is not None:
        dataset = filter_common_success(
            dataset,
            common_run_only=args.common_run_only,
            common_success_only=args.common_success_only,
            max_queries=args.max_queries,
        )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(dataset), encoding="utf-8")

    print(f"queries: {len(dataset['queries'])}  modes: {', '.join(dataset['modes'])}")
    for agg in dataset["aggregates"]:
        print(
            f"  {agg['mode']:<24} {agg['runs']:>4} runs  "
            f"{agg['errors']} err  "
            f"{agg['llm_calls_mean']} LLM/run  {agg['tool_calls_mean']} tool/run  "
            f"{agg['tokens_per_run']} tok/run"
            + (f"  ${agg['est_cost_usd']}" if agg["est_cost_usd"] else "")
        )
    size_kb = out_path.stat().st_size / 1024
    print(f"\nwrote {out_path} ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
