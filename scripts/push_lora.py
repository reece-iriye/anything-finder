"""Push an already-trained LoRA adapter to the Hugging Face Hub.

    uv run scripts/push_lora.py --repo <hf-user>/anything-finder-qwen-lora
    uv run scripts/push_lora.py --repo ... --run-dir data/lora/runs/<run_name> --public
    uv run scripts/push_lora.py --repo ... --dry-run     # render the card, upload nothing

Separate from ``scripts/train_lora.py`` so a finished run can be (re)published
without a GPU and without retraining — useful after ``make lora-fetch`` pulls the
run dir back from the brev box, or when only the model card changed.

Uploads the adapter weights + tokenizer via ``huggingface_hub.HfApi``, plus the
metadata files from ``src/training/hub.py`` (README model card, tool schemas,
resolved config).

Auth: ``HF_TOKEN`` from the environment or ``.env`` (or a prior
``huggingface-cli login``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Adapter weights + tokenizer files, i.e. everything that is not run bookkeeping
# (checkpoints, tensorboard logs) and not already handled by EXTRA_FILES.
_WEIGHT_PATTERNS = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "added_tokens.json",
)


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
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded


def _resolve_config(run_dir: Path, config_arg: str | None):
    """Prefer the run's own resolved_config.json — it is what actually trained."""
    from src.training.config import config_from_dict, load_config

    resolved = run_dir / "resolved_config.json"
    if config_arg:
        return load_config(config_arg, [])
    if resolved.is_file():
        return config_from_dict(json.loads(resolved.read_text(encoding="utf-8")))
    raise SystemExit(
        f"no {resolved} — pass --config configs/lora/<the config you trained with>.yaml"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="<hf-user>/<repo-name>")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="adapter dir (default: output_dir from --config, else the newest under data/lora/runs)",
    )
    parser.add_argument("--config", default=None, help="config yaml the run used")
    parser.add_argument("--public", action="store_true", help="create a public repo")
    parser.add_argument(
        "--dry-run", action="store_true", help="write the card locally, upload nothing"
    )
    args = parser.parse_args()

    loaded = _load_dotenv(REPO_ROOT / ".env")
    if loaded:
        print(f".env: loaded {', '.join(sorted(loaded))}")

    if args.run_dir:
        run_dir = (REPO_ROOT / args.run_dir).resolve()
    elif args.config:
        from src.training.config import load_config

        run_dir = (REPO_ROOT / load_config(args.config, []).output_dir).resolve()
    else:
        runs = sorted(
            (p for p in (REPO_ROOT / "data" / "lora" / "runs").glob("*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not runs:
            raise SystemExit("no run dirs under data/lora/runs — pass --run-dir")
        run_dir = runs[0]
        print(f"run-dir (newest): {run_dir}")

    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")
    if not (run_dir / "adapter_config.json").is_file():
        raise SystemExit(f"{run_dir} has no adapter_config.json — is this a finished run?")

    cfg = _resolve_config(run_dir, args.config)
    from src.training.hub import upload_run_artifacts, write_run_artifacts

    written = write_run_artifacts(run_dir, cfg)
    print("wrote: " + ", ".join(p.name for p in written))

    if args.dry_run:
        print(f"\n--dry-run: nothing uploaded. Card at {run_dir / 'README.md'}")
        return

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print(
            "note: HF_TOKEN not set — relying on a previous `huggingface-cli login`",
            file=sys.stderr,
        )

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    private = not args.public
    api.create_repo(args.repo, private=private, repo_type="model", exist_ok=True)

    for name in _WEIGHT_PATTERNS:
        path = run_dir / name
        if not path.is_file():
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  uploaded {name}")

    for name in upload_run_artifacts(run_dir, args.repo, private):
        print(f"  uploaded {name}")

    print(f"\npushed -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
