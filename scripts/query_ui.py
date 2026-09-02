"""Gradio front-end for the restaurant-search endpoint.

Talks to a running Anything Finder API over HTTP (default http://localhost:9022) so
you can iterate on queries without Postman/curl. Not part of the app image.

Run it (gradio is not a project dependency):

    uv run --with gradio scripts/query_ui.py

Env:
    AF_API_BASE   API base URL (default http://localhost:9022)
    AF_USER_ID    user UUID for knowledge-based filtering
                  (default 00000000-0000-0000-0000-000000000001)
"""

from __future__ import annotations

import os
import uuid

import gradio as gr
import httpx

API_BASE = os.environ.get("AF_API_BASE", "http://localhost:9022").rstrip("/")
DEFAULT_USER_ID = os.environ.get(
    "AF_USER_ID", "00000000-0000-0000-0000-000000000001"
)
# Agent turns run an LLM plus geocoding / Overpass calls - keep this generous.
REQUEST_TIMEOUT = float(os.environ.get("AF_TIMEOUT_SECONDS", "180"))


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def search(
    message: str,
    history: list[dict],
    user_id: str,
    session_id: str,
    city: str,
    state: str,
    latitude: float | None,
    longitude: float | None,
    radius_m: float | None,
    include_casual: bool,
) -> str:
    """One turn: POST the message to the restaurant endpoint and return the reply."""
    user_id = _clean(user_id) or DEFAULT_USER_ID
    payload: dict = {
        "query": message,
        "session_id": _clean(session_id),
        "include_casual": bool(include_casual),
    }

    city, state = _clean(city), _clean(state)
    has_place = bool(city or state)
    has_coord = latitude is not None or longitude is not None
    if has_place and has_coord:
        return "⚠️ Provide city/state OR latitude/longitude, not both."
    if has_place:
        payload["city"], payload["state"] = city, state
    if has_coord:
        if latitude is None or longitude is None:
            return "⚠️ latitude and longitude must be provided together."
        payload["latitude"], payload["longitude"] = float(latitude), float(longitude)
    if radius_m:
        payload["radius_m"] = int(radius_m)

    url = f"{API_BASE}/api/geo-search/restaurants/{user_id}"
    try:
        resp = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    except httpx.RequestError as exc:
        return f"❌ Could not reach {url}\n\n`{exc!r}`"

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        return f"❌ HTTP {resp.status_code}\n\n```\n{detail}\n```"

    return resp.json().get("response") or "_(empty response)_"


def _health() -> str:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        return f"🟢 {API_BASE} — {r.json().get('status', r.text)}"
    except httpx.RequestError:
        return f"🔴 {API_BASE} — unreachable"


with gr.Blocks(title="Anything Finder — restaurant search") as demo:
    gr.Markdown(f"### Anything Finder — restaurant search\nAPI: `{API_BASE}`")
    health = gr.Markdown(_health())
    demo.load(_health, outputs=health)

    with gr.Accordion("Request options", open=True):
        with gr.Row():
            user_id = gr.Textbox(DEFAULT_USER_ID, label="user_id (UUID)")
            session_id = gr.Textbox(
                str(uuid.uuid4()),
                label="session_id (checkpointer thread — keep stable for multi-turn)",
            )
        with gr.Row():
            city = gr.Textbox(label="city (optional override)")
            state = gr.Textbox(label="state (optional override)")
        with gr.Row():
            latitude = gr.Number(label="latitude (optional)", value=None)
            longitude = gr.Number(label="longitude (optional)", value=None)
        with gr.Row():
            radius_m = gr.Number(label="radius_m (optional)", value=None)
            include_casual = gr.Checkbox(
                label="include_casual (fast food, cafes, bars)", value=False
            )
        gr.Markdown(
            "_Location comes from the query text unless you set city/state **or** "
            "lat/lon (not both)._"
        )

    gr.ChatInterface(
        fn=search,
        additional_inputs=[
            user_id,
            session_id,
            city,
            state,
            latitude,
            longitude,
            radius_m,
            include_casual,
        ],
        examples=[
            ["quiet sushi near Grandscape in The Colony, TX"],
            ["cheap tacos open late in Deep Ellum"],
            ["date-night Italian in Uptown Dallas"],
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("AF_GRADIO_PORT", "7860")))
