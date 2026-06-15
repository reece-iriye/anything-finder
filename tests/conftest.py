from typing import Any

import httpx
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest

from src.schemas.geo_search.intent import CravingIntent

# Stable base URLs so pytest-httpx can match requests deterministically.
NOMINATIM_BASE = "http://test-nominatim"
OVERPASS_BASE = "http://test-overpass"


class FakeLLM:
    """Minimal stand-in for ChatOpenAI exercising only what the nodes call.

    - ``with_structured_output`` returns a runnable yielding a preset object
      (used by parse_intent).
    - ``ainvoke`` returns an AIMessage with preset text (used by synthesize).
    The search node's model is never invoked directly (create_deep_agent is patched).
    """

    def __init__(self, structured: Any = None, text: str = "Here are some picks."):
        self._structured = structured
        self._text = text

    def with_structured_output(self, schema: Any, **kwargs: Any):
        async def _produce(_: Any) -> Any:
            return self._structured

        return RunnableLambda(_produce)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> AIMessage:
        return AIMessage(content=self._text)


@pytest.fixture
def craving_intent() -> CravingIntent:
    return CravingIntent(
        craving="sushi", vibe=["quiet"], location_phrase="The Colony, TX"
    )


@pytest.fixture
def fake_llms(craving_intent: CravingIntent) -> dict[str, FakeLLM]:
    return {
        "intent": FakeLLM(structured=craving_intent),
        "search": FakeLLM(),
        "synthesize": FakeLLM(text="Try Near Sushi for a quiet bite."),
    }


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
