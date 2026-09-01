from typing import Any

import httpx
from langchain_core.messages import AIMessage
import pytest

# Stable base URLs so pytest-httpx can match requests deterministically.
NOMINATIM_BASE = "http://test-nominatim"
OVERPASS_BASE = "http://test-overpass"


class FakeLLM:
    """Minimal stand-in for ChatOpenAI exercising only what the agent calls."""

    def __init__(self, text: str = "Here are some picks."):
        self._text = text

    async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        return AIMessage(content=self._text)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(text="Try Near Sushi for a quiet bite.")


@pytest.fixture
async def nominatim_client() -> httpx.AsyncClient:
    async with httpx.AsyncClient(base_url=NOMINATIM_BASE) as client:
        yield client


@pytest.fixture
async def overpass_client() -> httpx.AsyncClient:
    async with httpx.AsyncClient(base_url=OVERPASS_BASE) as client:
        yield client


def overpass_payload(*names_with_cuisine: tuple[str, str, float, float]) -> dict:
    """Build a minimal Overpass JSON response from (name, cuisine, lat, lon) tuples."""
    return {
        "elements": [
            {
                "type": "node",
                "lat": lat,
                "lon": lon,
                "tags": {"amenity": "restaurant", "name": name, "cuisine": cuisine},
            }
            for name, cuisine, lat, lon in names_with_cuisine
        ]
    }
