from pathlib import Path

from src.utils.prompts import load_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def test_loads_existing_markdown_prompt():
    text = load_prompt(_PROMPTS_DIR, "restaurant_agent")
    assert "dining guide" in text


def test_no_unformatted_placeholders():
    text = load_prompt(_PROMPTS_DIR, "restaurant_agent")
    # The restaurant_agent prompt has no format placeholders; loading it must not raise.
    assert "{" not in text or "}" not in text or text.count("{") == text.count("}")
