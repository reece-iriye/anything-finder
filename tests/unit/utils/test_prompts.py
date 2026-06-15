from pathlib import Path

from src.utils.prompts import load_prompt

_AGENT_PROMPTS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "agents"
    / "geo_search"
    / "prompts"
)


def test_loads_existing_markdown_prompt():
    text = load_prompt(_AGENT_PROMPTS, "intent")
    assert "dining intent" in text


def test_formats_placeholders():
    text = load_prompt(
        _AGENT_PROMPTS, "search", default_radius=2000, min_results=5, max_radius=16000
    )
    assert "2000 meters" in text
    assert "{default_radius}" not in text  # placeholder interpolated
