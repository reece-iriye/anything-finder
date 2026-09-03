"""Adapter run-directory artifacts and the Hugging Face upload.

Two halves, deliberately separated so the first stays hermetic:

* ``write_run_artifacts`` — pure filesystem: renders the model card and copies
  the tool schemas next to the adapter weights. No network, no torch.
* ``upload_run_artifacts`` — pushes the non-weight files (README, schemas,
  resolved config) to the Hub. ``huggingface_hub`` is imported lazily so this
  module stays importable without the ``training`` dependency group.

The adapter weights and tokenizer are pushed by their own ``push_to_hub``
methods (see ``scripts/train_lora.py`` / ``scripts/push_lora.py``); this only
fills in what those leave out.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.training.config import LoraTrainingConfig, config_to_dict
from src.training.model_card import render_model_card

REPO_ROOT = Path(__file__).resolve().parents[2]

# Everything in the run dir that is metadata rather than weights.
EXTRA_FILES = ("README.md", "tool_schemas.json", "resolved_config.json")


def write_run_artifacts(
    run_dir: Path,
    cfg: LoraTrainingConfig,
    stats: dict[str, Any] | None = None,
) -> list[Path]:
    """Write README.md + tool_schemas.json (+ resolved_config.json if absent).

    Returns the paths written. Safe to call on an existing run dir — it
    overwrites the card so a re-push always reflects the current config.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    card = run_dir / "README.md"
    card.write_text(render_model_card(cfg, stats), encoding="utf-8")
    written.append(card)

    src_schemas = REPO_ROOT / cfg.data.tool_schemas
    if src_schemas.is_file():
        dst = run_dir / "tool_schemas.json"
        shutil.copyfile(src_schemas, dst)
        written.append(dst)

    resolved = run_dir / "resolved_config.json"
    if not resolved.exists():
        resolved.write_text(json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8")
        written.append(resolved)

    return written


def upload_run_artifacts(run_dir: Path, repo_id: str, private: bool = True) -> list[str]:
    """Upload the metadata files from ``run_dir`` to ``repo_id``. Returns filenames."""
    from huggingface_hub import HfApi  # lazy: only on the push path

    api = HfApi()
    api.create_repo(repo_id, private=private, repo_type="model", exist_ok=True)
    uploaded: list[str] = []
    for name in EXTRA_FILES:
        path = run_dir / name
        if not path.is_file():
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded.append(name)
    return uploaded
