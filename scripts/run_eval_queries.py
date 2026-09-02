"""Execute every synthetic eval row through the restaurant-search workflow with
Claude as the inference engine.

Reads ``data/eval/dallas_food_queries.csv`` (see ``scripts/gen_eval_queries.py``),
runs each ``query`` + ``context_data`` pair through the *real* compiled deep agent
from ``src.agents.geo_search`` — same system prompt, same tools (Nominatim +
Overpass), same input formatting as the FastAPI route — and writes one JSONL
record per row to ``data/eval/dallas_food_runs.jsonl``.

Each record carries:
  * ``prompt`` / ``completion``      — flat pair for LoRA supervised fine-tuning.
  * ``messages``                     — OpenAI-style chat turns (system/user/assistant)
                                       for chat-template fine-tuning.
  * ``transcript``                   — full agent message log incl. tool calls and
                                       tool results, for judging tool use / grounding.
  * ``target_*``                     — ground-truth intent labels for LLM-as-a-judge.
  * ``error``                        — set instead of a completion when the run failed.

Backend: forces ``LLM_BACKEND=claude`` unless already set. Needs ``ANTHROPIC_API_KEY``
and reachable Nominatim / Overpass (honours the same env vars as ``make dev``:
``NOMINATIM_BASE_URL`` / ``OVERPASS_BASE_URL`` or the ``*_USE_EXTERNAL_API`` flags).

Usage:
    uv run scripts/run_eval_queries.py
    uv run scripts/run_eval_queries.py --limit 5 --concurrency 2
    uv run scripts/run_eval_queries.py --resume            # skip ids already written
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IN_PATH = REPO_ROOT / "data" / "eval" / "dallas_food_queries.csv"
OUT_PATH = REPO_ROOT / "data" / "eval" / "dallas_food_runs.jsonl"

SYSTEM_PROMPT_NAME = "restaurant_agent"


def _load_dotenv(path: Path) -> list[str]:
    """Minimal .env loader (no dependency). Existing env vars win, so
    `ANTHROPIC_API_KEY=... make eval-run` still overrides the file."""
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def _mask(secret: str) -> str:
    if not secret:
        return "MISSING"
    return f"set ({secret[:8]}…{secret[-4:]}, len {len(secret)})"


def _load_rows(limit: int | None) -> list[dict[str, str]]:
    with IN_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[:limit] if limit else rows


def _serialize_message(msg) -> dict:
    """Flatten a LangChain message to plain JSON."""
    out: dict = {"role": getattr(msg, "type", msg.__class__.__name__), "content": msg.content}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
        ]
    if getattr(msg, "name", None):
        out["name"] = msg.name
    if getattr(msg, "tool_call_id", None):
        out["tool_call_id"] = msg.tool_call_id
    return out


def _build_human_message(query: str) -> str:
    """Mirror src.services.geo_search.RestaurantSearch: the CSV carries no
    coordinate / radius overrides, so the query text is the whole human turn."""
    return query


async def _run_row(
    row: dict[str, str],
    *,
    agent,
    prefs_dir: Path,
    system_prompt: str,
    sem: asyncio.Semaphore,
) -> dict:
    user_id = row["user_id"]
    (prefs_dir / f"{user_id}.md").write_text(row["context_data"], encoding="utf-8")

    human_msg = _build_human_message(row["query"])
    config = {
        "configurable": {
            "thread_id": row["session_id"],
            "user_id": user_id,
        }
    }

    record: dict = {
        "id": row["id"],
        "query": row["query"],
        "context_data": row["context_data"],
        "city": row.get("city", ""),
        "state": row.get("state", ""),
        "target_neighborhood": row.get("target_neighborhood", ""),
        "target_cuisine": row.get("target_cuisine", ""),
        "target_vibe": row.get("target_vibe", ""),
        "backend": os.environ.get("LLM_BACKEND", "vllm"),
        "model": os.environ.get("LLM_MODEL_AGENT")
        or os.environ.get("LLM_MODEL_CLAUDE")
        or os.environ.get("LLM_MODEL", ""),
    }

    async with sem:
        started = time.monotonic()
        try:
            result = await agent.ainvoke({"messages": [("human", human_msg)]}, config=config)
        except Exception as exc:  # noqa: BLE001 — recorded, not raised
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_s"] = round(time.monotonic() - started, 2)
            return record
        record["latency_s"] = round(time.monotonic() - started, 2)

    messages = result.get("messages", [])
    transcript = [_serialize_message(m) for m in messages]
    response = next(
        (
            m.content
            for m in reversed(messages)
            if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", None)
        ),
        None,
    )

    record["transcript"] = transcript
    record["tool_calls"] = [
        {"name": t["tool_calls"][0]["name"], "args": t["tool_calls"][0]["args"]}
        for t in transcript
        if t.get("tool_calls")
    ]
    if response is None:
        record["error"] = "agent returned no final assistant message"
        return record

    record["completion"] = response
    record["prompt"] = (
        f"{system_prompt}\n\n"
        f"User food preferences:\n{row['context_data']}\n\n"
        f"User request:\n{row['query']}"
    )
    record["messages"] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"User food preferences:\n{row['context_data']}\n\n"
                f"User request:\n{row['query']}"
            ),
        },
        {"role": "assistant", "content": response},
    ]
    return record


async def _main_async(args: argparse.Namespace) -> int:
    loaded = _load_dotenv(REPO_ROOT / ".env")
    if loaded:
        print(f".env: loaded {', '.join(sorted(loaded))}")
    os.environ.setdefault("LLM_BACKEND", "claude")
    print(f"ANTHROPIC_API_KEY: {_mask(os.environ.get('ANTHROPIC_API_KEY', ''))}")
    print(
        f"backend={os.environ['LLM_BACKEND']}  "
        f"model={os.environ.get('LLM_MODEL_AGENT') or os.environ.get('LLM_MODEL_CLAUDE') or '(default)'}"
    )

    from src.agents.geo_search.agent import build_restaurant_agent
    from src.utils.llm import make_llm
    from src.utils.nominatim import make_nominatim_client
    from src.utils.overpass import make_overpass_client
    from src.utils.prompts import load_prompt
    from langgraph.checkpoint.memory import InMemorySaver

    prompts_dir = REPO_ROOT / "prompts"
    system_prompt = load_prompt(prompts_dir, SYSTEM_PROMPT_NAME)

    rows = _load_rows(args.limit)
    done: set[str] = set()
    if args.resume and OUT_PATH.exists():
        with OUT_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
        rows = [r for r in rows if r["id"] not in done]
        print(f"resume: {len(done)} already done, {len(rows)} remaining")

    if not rows:
        print("nothing to run")
        return 0

    if os.environ["LLM_BACKEND"] == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: LLM_BACKEND=claude but ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    nominatim = make_nominatim_client()
    overpass = make_overpass_client()

    # Preflight: the default base URLs are docker-compose service names
    # (http://nominatim-service:8080) that don't resolve from a bare host run.
    # Fail before spending Anthropic tokens on 200 doomed agent loops.
    if not args.skip_preflight:
        for name, client in (("Nominatim", nominatim), ("Overpass", overpass)):
            try:
                await client.get("/", timeout=5.0)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"ERROR: cannot reach {name} at {client.base_url} ({type(exc).__name__}).\n"
                    "  Set NOMINATIM_BASE_URL / OVERPASS_BASE_URL to your local containers,\n"
                    "  or export NOMINATIM_USE_EXTERNAL_API=true OVERPASS_USE_EXTERNAL_API=true\n"
                    "  to hit the public servers (then use --concurrency 1 for their rate limits).\n"
                    "  Pass --skip-preflight to bypass this check.",
                    file=sys.stderr,
                )
                await nominatim.aclose()
                await overpass.aclose()
                return 2

    llm = make_llm("agent")

    tmp_prefs = Path(tempfile.mkdtemp(prefix="af-eval-prefs-"))
    agent = build_restaurant_agent(
        llm,
        nominatim,
        overpass,
        prefs_dir=tmp_prefs,
        home_city=os.environ.get("HOME_CITY", "Dallas"),
        home_state=os.environ.get("HOME_STATE", "TX"),
        checkpointer=InMemorySaver(),
    )

    sem = asyncio.Semaphore(args.concurrency)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_lock = asyncio.Lock()
    counter = {"ok": 0, "err": 0}
    total = len(rows)

    mode = "a" if (args.resume and OUT_PATH.exists()) else "w"
    with OUT_PATH.open(mode, encoding="utf-8") as out_fh:

        async def _worker(row: dict[str, str]) -> None:
            rec = await _run_row(
                row,
                agent=agent,
                prefs_dir=tmp_prefs,
                system_prompt=system_prompt,
                sem=sem,
            )
            async with write_lock:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                bucket = "err" if rec.get("error") else "ok"
                counter[bucket] += 1
                n = counter["ok"] + counter["err"]
                tag = "ERR " + rec["error"][:60] if rec.get("error") else "ok"
                print(f"[{n:>3}/{total}] {rec['id']}  {rec.get('latency_s', '?')}s  {tag}")

        try:
            await asyncio.gather(*(_worker(r) for r in rows))
        finally:
            await nominatim.aclose()
            await overpass.aclose()

    print(f"\ndone: {counter['ok']} ok, {counter['err']} errors -> {OUT_PATH}")
    return 1 if counter["err"] and not counter["ok"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N rows")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="max in-flight agent runs (default 4)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append, skipping ids already present in the output file",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the Nominatim/Overpass reachability check",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
