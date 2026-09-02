"""Generate a synthetic eval set of Dallas food-search queries.

Produces ``data/eval/dallas_food_queries.csv`` with 200 rows. Each row pairs a
natural-language restaurant query (the kind the geo-search restaurant agent
consumes) with a self-contained ``context_data`` food-preferences profile — the
same shape as ``data/preferences/<user_id>.md``.

Downstream uses:
  * Drive the AI workflow once per row and capture (query, context_data, response)
    triples for LoRA fine-tuning of a target LLM.
  * Feed the same triples to an LLM-as-a-judge for scoring.

The set is deterministic (fixed seed) so re-runs are stable.
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

SEED = 20260902
N_ROWS = 200

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "dallas_food_queries.csv"

# --- Dallas neighbourhoods / landmarks the search should geocode against --------
NEIGHBORHOODS = [
    "Bishop Arts District",
    "Deep Ellum",
    "Lower Greenville",
    "Knox-Henderson",
    "Uptown Dallas",
    "Oak Cliff",
    "Trinity Groves",
    "the Design District",
    "Lakewood",
    "Downtown Dallas",
    "the Harwood District",
    "Victory Park",
    "the Cedars",
    "Exposition Park near Fair Park",
    "Sylvan Thirty in West Dallas",
    "Oak Lawn",
    "Cedar Springs",
    "Mockingbird Station",
    "Casa Linda",
    "the Katy Trail",
    "Klyde Warren Park",
    "the West End",
    "Greenville Avenue",
    "Henderson Avenue",
    "State-Thomas in Uptown",
    "Bryan Place",
    "Junius Heights",
    "Little Forest Hills",
    "the Bishop Arts strip on 7th Street",
    "the Dallas Farmers Market",
]

# --- Cravings: (phrase, cuisine hint for the tool) ----------------------------
CRAVINGS = [
    ("tacos", "mexican"),
    ("birria tacos", "mexican"),
    ("Tex-Mex and margaritas", "mexican"),
    ("wood-fired pizza", "pizza"),
    ("Neapolitan pizza", "pizza"),
    ("ramen", "ramen"),
    ("sushi", "sushi"),
    ("omakase sushi", "sushi"),
    ("Vietnamese pho", "vietnamese"),
    ("banh mi", "vietnamese"),
    ("Thai curry", "thai"),
    ("drunken noodles", "thai"),
    ("Texas barbecue brisket", "barbecue"),
    ("smoked ribs", "barbecue"),
    ("Southern comfort food", "american"),
    ("chicken and waffles", "american"),
    ("a great burger", "burger"),
    ("smash burgers", "burger"),
    ("Ethiopian food", "ethiopian"),
    ("Indian butter chicken", "indian"),
    ("dosa and South Indian food", "indian"),
    ("Korean BBQ", "korean"),
    ("Korean fried chicken", "korean"),
    ("dim sum", "chinese"),
    ("hand-pulled noodles", "chinese"),
    ("Sichuan hot pot", "chinese"),
    ("Mediterranean mezze", "mediterranean"),
    ("shawarma", "mediterranean"),
    ("a raw bar and oysters", "seafood"),
    ("Gulf seafood", "seafood"),
    ("New American tasting menu", "american"),
    ("farm-to-table dinner", "american"),
    ("French bistro fare", "french"),
    ("Italian pasta", "italian"),
    ("brunch", "american"),
    ("a good vegan spot", "vegan"),
    ("vegetarian food", "vegetarian"),
    ("late-night eats", None),
    ("coffee and pastries", "cafe"),
    ("a gastropub with craft beer", None),
]

VIBES = [
    "somewhere quiet for a date night",
    "a lively spot with a patio",
    "casual, no reservations needed",
    "upscale, dress-up kind of place",
    "family-friendly with room for kids",
    "a hole-in-the-wall the locals love",
    "good for a big group of 8",
    "a counter-service spot I can get in and out of",
    "cozy and dim, good for conversation",
    "a rooftop or patio with a view",
    "a chef-driven place worth the splurge",
    "cheap eats, under $15 a person",
    "somewhere I can bring my laptop and work",
    "a bar-forward place with good cocktails",
    "dog-friendly outdoor seating",
]

TIME_CONSTRAINTS = [
    "",
    " We're going tonight around 8pm.",
    " Looking for lunch tomorrow.",
    " It's for Saturday brunch.",
    " Need something open late, after 11pm.",
    " Early dinner, we like to eat by 5:30.",
    " Sunday afternoon, nothing too formal.",
]

RADIUS_HINTS = [
    "",
    " I'd rather not walk more than a few blocks.",
    " Willing to drive up to 15 minutes.",
    " Walking distance only.",
    " Anywhere within a couple miles is fine.",
]

TEMPLATES = [
    "Find me {craving} in {hood}. I want {vibe}.{time}{radius}",
    "Where should I go for {craving} near {hood}? Ideally {vibe}.{time}",
    "We're in {hood} and craving {craving}. Somewhere {vibe} would be perfect.{time}{radius}",
    "Looking for {vibe} around {hood} that does {craving}.{time}",
    "Best {craving} in {hood}? Prefer {vibe}.{radius}",
    "I'll be around {hood} later — need {craving}, {vibe}.{time}",
    "Recommend {craving} in {hood}. {vibe_cap}.{time}{radius}",
    "Date night in {hood}, thinking {craving}. Want {vibe}.{time}",
    "Group dinner near {hood}. Everyone wants {craving}. {vibe_cap}.{radius}",
    "Quick {craving} fix in {hood} — {vibe}.{time}",
]

# --- context_data profiles ---------------------------------------------------
FAV_CUISINES = [
    "Japanese (sushi, ramen), Thai, Neapolitan pizza",
    "Mexican, Tex-Mex, Southwestern",
    "Vietnamese, Korean, Sichuan",
    "Italian, French, New American",
    "Barbecue, Southern, Cajun",
    "Indian, Ethiopian, Middle Eastern",
    "Mediterranean, Greek, Lebanese",
    "American comfort food, burgers, diners",
    "Seafood, raw bars, coastal",
    "Farm-to-table, seasonal tasting menus",
]
DIET = [
    "None",
    "Vegetarian",
    "Vegan",
    "Pescatarian",
    "Gluten-free",
    "No pork",
    "Dairy-free",
    "Nut allergy",
]
DISLIKES = [
    "Fast-food chains",
    "Overly sweet sauces",
    "Loud sports bars",
    "Long waits without reservations",
    "Buffets",
    "Chain steakhouses",
    "Places that are all hype",
    "Cilantro-heavy dishes",
]
VIBE_PREF = [
    "Hole-in-the-walls preferred over giant chains",
    "Lively rooms with a bar scene",
    "Quiet, intimate spaces for conversation",
    "Patios and outdoor seating whenever possible",
    "Counter-service and casual is fine",
    "Happy to splurge for a standout meal",
    "Budget-conscious, looks for value",
]
SPICE = ["mild", "medium", "loves it hot", "no strong preference"]


def make_profile(rng: random.Random, idx: int) -> str:
    uid = f"eval-user-{idx:03d}"
    return (
        f"# Food Preferences for {uid}\n\n"
        f"**Favourite cuisines:** {rng.choice(FAV_CUISINES)}\n"
        f"**Dietary restrictions:** {rng.choice(DIET)}\n"
        f"**Dislikes:** {rng.choice(DISLIKES)}\n"
        f"**Spice tolerance:** {rng.choice(SPICE)}\n"
        f"**Vibe preferences:** {rng.choice(VIBE_PREF)}"
    )


def build_query(rng: random.Random) -> tuple[str, str, str | None, str]:
    hood = rng.choice(NEIGHBORHOODS)
    craving, cuisine = rng.choice(CRAVINGS)
    vibe = rng.choice(VIBES)
    time = rng.choice(TIME_CONSTRAINTS)
    radius = rng.choice(RADIUS_HINTS)
    template = rng.choice(TEMPLATES)
    query = template.format(
        craving=craving,
        hood=hood,
        vibe=vibe,
        vibe_cap=vibe[0].upper() + vibe[1:],
        time=time,
        radius=radius,
    ).strip()
    return query, hood, cuisine, vibe


def main() -> None:
    rng = random.Random(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    guard = 0
    while len(rows) < N_ROWS and guard < N_ROWS * 50:
        guard += 1
        query, hood, cuisine, vibe = build_query(rng)
        if query in seen:
            continue
        seen.add(query)
        idx = len(rows) + 1
        profile = make_profile(rng, idx)
        row_id = "q-" + hashlib.sha1(query.encode()).hexdigest()[:10]
        rows.append(
            {
                "id": row_id,
                "user_id": f"eval-user-{idx:03d}",
                "session_id": f"eval-session-{idx:03d}",
                "query": query,
                "context_data": profile,
                "city": "Dallas",
                "state": "TX",
                "target_neighborhood": hood,
                "target_cuisine": cuisine or "",
                "target_vibe": vibe,
            }
        )

    if len(rows) < N_ROWS:
        raise SystemExit(f"only generated {len(rows)} unique queries; widen the pools")

    fieldnames = list(rows[0].keys())
    with OUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
