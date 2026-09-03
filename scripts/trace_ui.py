"""Telemetry console for the restaurant agent -- query it, then drill into the trace.

    make trace
    # or: AF_TRACE_DIR=telemetry AF_API_BASE=http://localhost:9022 uv run scripts/trace_ui.py

One UI: submit a natural-language restaurant query, watch the run stream into a
trace (LLM + tool calls, tokens, cost), and browse past runs grouped by model
stack (``claude`` | ``raw-open-source`` | ``finetuned-open-source``). The query
is proxied to the running API; traces are written by ``src/utils/telemetry.py``.

Env:
    AF_TRACE_DIR   telemetry root (default ./telemetry)
    AF_API_BASE    agent API base URL (default http://localhost:9022)
    TRACE_PORT     port (default 7861)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

TRACE_DIR = Path(os.environ.get("AF_TRACE_DIR", "telemetry")).resolve()
API_BASE = os.environ.get("AF_API_BASE", "http://localhost:9022").rstrip("/")
DEFAULT_USER_ID = os.environ.get("AF_USER_ID", "00000000-0000-0000-0000-000000000001")
_INDEX = Path(__file__).with_name("trace_ui.html")
_ASSET_DIR = Path(__file__).parent

app = FastAPI(title="Anything Finder — Telemetry", docs_url=None, redoc_url=None)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _modes() -> list[str]:
    if not TRACE_DIR.is_dir():
        return []
    return sorted(
        p.name for p in TRACE_DIR.iterdir() if p.is_dir() and any(p.glob("*.json"))
    )


def _summaries(mode: str) -> list[dict[str, Any]]:
    out = []
    for p in (TRACE_DIR / mode).glob("*.json"):
        doc = _load(p)
        if not doc:
            continue
        out.append(
            {
                "file": f"{mode}/{p.name}",
                "trace_id": doc.get("trace_id"),
                "mode": doc.get("mode"),
                "backend": doc.get("backend"),
                "started_at": doc.get("started_at"),
                "ended_at": doc.get("ended_at"),
                "query": doc.get("query"),
                "models": doc.get("models") or [],
                "session_id": doc.get("session_id"),
                "totals": doc.get("totals") or {},
                "span_count": len(doc.get("spans") or []),
                "error": bool(doc.get("error")),
            }
        )
    return sorted(out, key=lambda d: d.get("started_at") or "", reverse=True)


def inline_assets(html: str, asset_dir: Path = _ASSET_DIR) -> str:
    """Replace ``/*@include <file>*/`` markers with that file's contents.

    The console and the generated comparison report share ``ui_common.css`` /
    ``ui_common.js``; both inline them rather than linking, so the report stays
    a single file that opens from disk. An unknown filename is left in place —
    a visible marker beats a silently half-rendered page.
    """
    for name in ("ui_common.css", "ui_common.js"):
        marker = f"/*@include {name}*/"
        if marker in html:
            html = html.replace(marker, (asset_dir / name).read_text(encoding="utf-8"))
    return html


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return inline_assets(_INDEX.read_text(encoding="utf-8"))


@app.get("/api/config")
async def config() -> JSONResponse:
    api: dict[str, Any] = {"reachable": False}
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{API_BASE}/meta")
            api = {"reachable": True, **r.json()}
        except (httpx.RequestError, ValueError):
            try:
                await client.get(f"{API_BASE}/health")
                api["reachable"] = True
            except httpx.RequestError:
                pass
    return JSONResponse(
        {
            "api_base": API_BASE,
            "trace_dir": str(TRACE_DIR),
            "default_user_id": DEFAULT_USER_ID,
            "api": api,
        }
    )


@app.get("/api/overview")
def overview() -> JSONResponse:
    modes = []
    for mode in _modes():
        agg = {
            "name": mode,
            "backend": None,
            "models": set(),
            "runs": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "est_cost_usd": 0.0,
            "first": None,
            "last": None,
        }
        for s in _summaries(mode):
            t = s["totals"]
            agg["runs"] += 1
            agg["backend"] = agg["backend"] or s.get("backend")
            agg["models"].update(s.get("models") or [])
            agg["llm_calls"] += t.get("llm_calls", 0)
            agg["tool_calls"] += t.get("tool_calls", 0)
            agg["input_tokens"] += t.get("input_tokens", 0)
            agg["output_tokens"] += t.get("output_tokens", 0)
            agg["cache_read_tokens"] += t.get("cache_read_tokens", 0)
            agg["est_cost_usd"] += t.get("est_cost_usd") or 0.0
            started = s.get("started_at")
            if started:
                agg["first"] = min(agg["first"] or started, started)
                agg["last"] = max(agg["last"] or started, started)
        agg["models"] = sorted(agg["models"])
        agg["est_cost_usd"] = round(agg["est_cost_usd"], 6)
        modes.append(agg)
    return JSONResponse({"modes": modes, "trace_dir": str(TRACE_DIR)})


@app.get("/api/traces")
def traces(mode: str = Query(...)) -> JSONResponse:
    if mode not in _modes():
        return JSONResponse([])
    return JSONResponse(_summaries(mode))


def _trace_path(file: str) -> Path:
    target = (TRACE_DIR / file).resolve()
    if TRACE_DIR not in target.parents or target.suffix != ".json":
        raise HTTPException(404)
    return target


@app.get("/api/trace")
def trace(file: str = Query(...)) -> JSONResponse:
    doc = _load(_trace_path(file))
    if doc is None:
        raise HTTPException(404)
    return JSONResponse(doc)


@app.delete("/api/trace")
def delete_trace(file: str = Query(...)) -> JSONResponse:
    target = _trace_path(file)
    try:
        target.unlink()
    except FileNotFoundError:
        raise HTTPException(404)
    # drop the mode dir too if it's now empty
    try:
        next(target.parent.iterdir())
    except StopIteration:
        target.parent.rmdir()
    return JSONResponse({"deleted": file})


class QueryBody(BaseModel):
    query: str
    user_id: str = DEFAULT_USER_ID
    session_id: str | None = None
    city: str | None = None
    state: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    radius_m: int | None = None
    include_casual: bool = False


@app.post("/api/query")
async def run_query(body: QueryBody) -> JSONResponse:
    """Proxy a query to the agent API, then point the UI at the trace it produced."""
    before = {p for p in TRACE_DIR.rglob("*.json")} if TRACE_DIR.is_dir() else set()

    payload = body.model_dump(exclude_none=True)
    user_id = payload.pop("user_id")
    payload.setdefault("include_casual", body.include_casual)
    url = f"{API_BASE}/api/geo-search/restaurants/{user_id}"

    async with httpx.AsyncClient(timeout=600) as client:
        try:
            r = await client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise HTTPException(502, f"cannot reach agent API at {url}: {exc}")

    try:
        body_json = r.json()
    except ValueError:
        body_json = {"raw": r.text}

    new = sorted(
        (p for p in TRACE_DIR.rglob("*.json") if p not in before),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if TRACE_DIR.is_dir() else []
    trace_file = str(new[0].relative_to(TRACE_DIR)) if new else None

    return JSONResponse(
        {"status": r.status_code, "response": body_json, "trace_file": trace_file}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("TRACE_PORT", "7861")),
        log_level="warning",
    )
