"""Bundle the scratch telemetry tree into committable per-mode archives.

    make traces-freeze
    uv run scripts/freeze_traces.py --trace-dir telemetry --out-dir data/traces

``telemetry/`` is git-ignored working scratch: one JSON file per run, rewritten
after every span. This packs each mode into ``data/traces/<mode>.jsonl.gz`` —
one trace per line — which *is* committed. That makes the comparison corpus
reproducible: a fresh clone can run ``make compare`` and get the same report
with no GPU, no API key, and no re-running of the agent.

Existing bundles are merged, not replaced: a trace already in the bundle is
matched by ``trace_id`` and the newer copy wins. So freezing after each capture
run accumulates rather than clobbers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRACE_DIR = REPO_ROOT / "telemetry"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "traces"


def _read_bundle(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = doc.get("trace_id") or f"_{len(out)}"
            out[key] = doc
    return out


def freeze_mode(mode_dir: Path, out_path: Path) -> tuple[int, int]:
    """Merge ``mode_dir/*.json`` into ``out_path``. Returns (added, total)."""
    docs = _read_bundle(out_path)
    before = len(docs)
    for f in sorted(mode_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print(f"  skipped unreadable {f.name}", file=sys.stderr)
            continue
        docs[doc.get("trace_id") or f.stem] = doc

    ordered = sorted(docs.values(), key=lambda d: d.get("started_at") or "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for doc in ordered:
            fh.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
    tmp.replace(out_path)
    return len(docs) - before, len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=str(DEFAULT_TRACE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    if not trace_dir.is_dir():
        print(f"no trace dir: {trace_dir} — nothing to freeze", file=sys.stderr)
        raise SystemExit(2)

    mode_dirs = [p for p in sorted(trace_dir.iterdir()) if p.is_dir() and any(p.glob("*.json"))]
    if not mode_dirs:
        print(f"no traces under {trace_dir}/<mode>/ — capture some first (make compare-run)")
        raise SystemExit(2)

    for mode_dir in mode_dirs:
        out_path = out_dir / f"{mode_dir.name}.jsonl.gz"
        added, total = freeze_mode(mode_dir, out_path)
        size_kb = out_path.stat().st_size / 1024
        print(f"{mode_dir.name:<24} +{added:<4} = {total:>4} traces  ->  {out_path} ({size_kb:,.0f} KB)")

    print("\nThese are tracked by git — commit them alongside data/eval/.")


if __name__ == "__main__":
    main()
