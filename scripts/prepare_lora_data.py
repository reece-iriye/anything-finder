"""Turn captured Claude eval runs into a LoRA supervised-fine-tuning dataset.

    uv run scripts/prepare_lora_data.py
    uv run scripts/prepare_lora_data.py --input 'data/eval/*_runs.jsonl' --val-frac 0.1

Reads the ``transcript`` field of every matched ``*_runs.jsonl`` (see
``src/training/dataset.py`` for why only ``transcript`` is trusted), converts each
usable trajectory to an OpenAI tool-calling chat record, and writes:

    <out-dir>/train.jsonl
    <out-dir>/val.jsonl
    <out-dir>/tool_schemas.json

Prints kept/dropped counts with drop reasons. Hermetic: no torch, no network.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(path: Path) -> list[str]:
    """Minimal .env loader (no dependency). Existing env vars win."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="glob for run files (repeatable); default data/eval/*_runs.jsonl",
    )
    parser.add_argument("--out-dir", default="data/lora")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--dedup-by", default="id")
    parser.add_argument(
        "--include-builtin-tools",
        action="store_true",
        help="also emit the deepagents built-in tool schemas",
    )
    args = parser.parse_args()

    _load_dotenv(REPO_ROOT / ".env")

    from src.training.dataset import (
        ConversionStats,
        build_records,
        load_system_prompt,
        read_jsonl,
        split_records,
    )
    from src.training.tool_schemas import export_tool_schemas

    patterns = args.input or ["data/eval/*_runs.jsonl"]
    files: list[Path] = []
    for pat in patterns:
        p = pat if os.path.isabs(pat) else str(REPO_ROOT / pat)
        files.extend(sorted(Path(f) for f in glob.glob(p)))
    files = sorted(set(files))
    if not files:
        sys.exit(f"no run files matched: {patterns}")
    print("inputs:")
    for f in files:
        print(f"  {f.relative_to(REPO_ROOT)}")

    rows: list[dict] = []
    for f in files:
        rows.extend(read_jsonl(f))
    print(f"read {len(rows)} rows")

    stats = ConversionStats()
    records = build_records(
        rows,
        load_system_prompt(),
        dedup_by=args.dedup_by or None,
        stats=stats,
    )
    print(stats.report())

    train, val = split_records(records, val_frac=args.val_frac, seed=args.seed)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    schemas = export_tool_schemas(
        out_dir / "tool_schemas.json",
        include_builtin_tools=args.include_builtin_tools,
    )
    tool_names = [s["function"]["name"] for s in schemas]

    for name, part in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in part:
                fh.write(
                    json.dumps({**rec, "tools": schemas}, ensure_ascii=False) + "\n"
                )
        print(f"wrote {len(part):>3} -> {path.relative_to(REPO_ROOT)}")

    print(f"tool schemas ({len(tool_names)}): {', '.join(tool_names)}")


if __name__ == "__main__":
    main()
