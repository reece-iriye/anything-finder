#!/usr/bin/env bash
# Kick a LoRA run off on a remote GPU box and return immediately.
#
#   BREV_HOST=my-brev-box bash scripts/brev_train.sh
#   BREV_HOST=... HF_REPO=me/anything-finder-qwen-lora bash scripts/brev_train.sh
#   BREV_HOST=... LORA_CONFIG=configs/lora/qwen3_8-27b-qlora.yaml ARGS="--set train.learning_rate=5e-5" \
#     bash scripts/brev_train.sh
#
# This machine is likely a Mac with no CUDA, so training happens over there:
# rsync the repo, run scripts/brev_setup.sh once, then launch `make lora-train`
# under nohup. The run survives the SSH session closing; tail it with
# `make lora-logs`, pull the adapter back with `make lora-fetch`.
#
# Env:
#   BREV_HOST     (required) ssh target, e.g. user@1.2.3.4 or an ssh_config alias
#   BREV_DIR      remote checkout path       (default ~/anything-finder)
#   LORA_CONFIG   config yaml                (default configs/lora/qwen2_5-7b-qlora.yaml)
#   HF_REPO       push the adapter here when the run finishes (optional)
#   HF_TOKEN      forwarded for the huggingface login / push (optional)
#   ARGS          extra flags for scripts/train_lora.py (optional)
#   FORCE_SETUP   set to 1 to re-run brev_setup.sh even if .venv exists
set -euo pipefail

cd "$(dirname "$0")/.."

: "${BREV_HOST:?set BREV_HOST=<ssh target> (e.g. BREV_HOST=my-brev-box)}"
BREV_DIR="${BREV_DIR:-anything-finder}"
LORA_CONFIG="${LORA_CONFIG:-configs/lora/qwen2_5-7b-qlora.yaml}"
ARGS="${ARGS:-}"

if [ ! -f "$LORA_CONFIG" ]; then
  echo "ERROR: no such config: $LORA_CONFIG" >&2
  exit 1
fi
RUN_NAME=$(awk '/^run_name:/{print $2; exit}' "$LORA_CONFIG")
RUN_NAME="${RUN_NAME:-lora-run}"
LOG="$BREV_DIR/logs/$RUN_NAME.log"

if [ ! -f data/lora/train.jsonl ]; then
  echo "ERROR: data/lora/train.jsonl missing — run 'make lora-data' first." >&2
  exit 1
fi

echo "== sync -> $BREV_HOST:$BREV_DIR =="
# Send source + the SFT dataset. Excludes the OSM extracts (100+ MB, useless on
# the GPU box), the local venv, scratch telemetry, and git history.
ssh "$BREV_HOST" "mkdir -p '$BREV_DIR/logs'"
rsync -az --delete-after \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.egg-info/' \
  --exclude 'telemetry/' \
  --exclude 'data/*.osm*' \
  --exclude 'data/lora/runs/' \
  --exclude 'data/eval/' \
  --exclude 'logs/' \
  ./ "$BREV_HOST:$BREV_DIR/"

echo "== setup =="
if [ "${FORCE_SETUP:-0}" = "1" ] || ! ssh "$BREV_HOST" "test -d '$BREV_DIR/.venv'"; then
  ssh "$BREV_HOST" "cd '$BREV_DIR' && HF_TOKEN='${HF_TOKEN:-}' bash scripts/brev_setup.sh"
else
  echo "  .venv present — skipping (FORCE_SETUP=1 to re-run)"
fi

# `make lora-train` already appends the hub overrides when HF_REPO is set.
REMOTE_CMD="cd '$BREV_DIR' && export PATH=\$HOME/.local/bin:\$PATH && \
  HF_TOKEN='${HF_TOKEN:-}' HF_REPO='${HF_REPO:-}' \
  nohup make lora-train LORA_CONFIG='$LORA_CONFIG' ARGS='$ARGS' \
  > 'logs/$RUN_NAME.log' 2>&1 & echo \$!"

echo "== launch =="
PID=$(ssh "$BREV_HOST" "$REMOTE_CMD")

cat <<EOF

== running ==
  host:   $BREV_HOST
  config: $LORA_CONFIG
  pid:    $PID
  log:    $LOG
${HF_REPO:+  push:   https://huggingface.co/$HF_REPO (on completion)}

  make lora-logs      # tail it
  make lora-fetch     # pull data/lora/runs/$RUN_NAME back when it finishes
EOF
