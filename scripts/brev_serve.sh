#!/usr/bin/env bash
# Start vLLM on the remote GPU box so this machine can run the comparison
# captures against it.
#
#   BREV_HOST=my-brev-box bash scripts/brev_serve.sh                    # base model only
#   BREV_HOST=... LORA_DIR=data/lora/runs/qwen25-7b-restaurant-react-lora \
#     bash scripts/brev_serve.sh                                        # base + adapter
#   BREV_HOST=... bash scripts/brev_serve.sh --stop
#
# One server can hold the base model and the adapter at the same time, so
# `make compare-run-qwen` and `make compare-run-lora` both talk to this one
# endpoint — pick which by LLM_MODEL_AGENT / LLM_MODEL_LORA.
#
# Env:
#   BREV_HOST     (required) ssh target
#   BREV_DIR      remote checkout path   (default anything-finder)
#   MODEL         served base model      (default Qwen/Qwen2.5-7B-Instruct)
#   LORA_DIR      adapter dir, remote-relative; enables --enable-lora (optional)
#   LORA_NAME     adapter name to serve it under (default af-lora)
#   MAX_LORA_RANK LoRA rank cap          (default 32)
#   VLLM_PORT     remote + local port    (default 8000)
#   HF_TOKEN      forwarded for gated repos (optional)
#   FORCE_SETUP   set to 1 to re-run brev_setup.sh even if .venv exists
#   SKIP_SYNC     set to 1 to skip the rsync (e.g. right after lora-train-remote)
set -euo pipefail

cd "$(dirname "$0")/.."

: "${BREV_HOST:?set BREV_HOST=<ssh target>}"
BREV_DIR="${BREV_DIR:-anything-finder}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LORA_NAME="${LORA_NAME:-af-lora}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
VLLM_PORT="${VLLM_PORT:-8000}"
PIDFILE="$BREV_DIR/logs/vllm.pid"
LOG="$BREV_DIR/logs/vllm.log"

if [ "${1:-}" = "--stop" ]; then
  ssh "$BREV_HOST" "if [ -f '$PIDFILE' ]; then kill \$(cat '$PIDFILE') 2>/dev/null && echo stopped || echo 'not running'; rm -f '$PIDFILE'; else echo 'no pidfile'; fi"
  exit 0
fi

if [ "${SKIP_SYNC:-0}" != "1" ]; then
  echo "== sync -> $BREV_HOST:$BREV_DIR =="
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
fi

echo "== setup =="
if [ "${FORCE_SETUP:-0}" = "1" ] || ! ssh "$BREV_HOST" "test -d '$BREV_DIR/.venv'"; then
  ssh "$BREV_HOST" "cd '$BREV_DIR' && HF_TOKEN='${HF_TOKEN:-}' bash scripts/brev_setup.sh"
else
  echo "  .venv present — skipping (FORCE_SETUP=1 to re-run)"
fi

LORA_FLAGS=""
if [ -n "${LORA_DIR:-}" ]; then
  if ! ssh "$BREV_HOST" "test -f '$BREV_DIR/$LORA_DIR/adapter_config.json'"; then
    echo "ERROR: $BREV_HOST:$BREV_DIR/$LORA_DIR has no adapter_config.json." >&2
    echo "       Has the run finished? (make lora-logs)" >&2
    exit 1
  fi
  LORA_FLAGS="--enable-lora --lora-modules $LORA_NAME=$BREV_DIR/$LORA_DIR --max-lora-rank $MAX_LORA_RANK"
fi

echo "== starting vLLM on $BREV_HOST =="
ssh "$BREV_HOST" "mkdir -p '$BREV_DIR/logs'"
ssh "$BREV_HOST" "cd '$BREV_DIR' && export PATH=\$HOME/.local/bin:\$PATH && \
  HF_TOKEN='${HF_TOKEN:-}' nohup uv run --group training vllm serve '$MODEL' \
    --port $VLLM_PORT --api-key EMPTY --gpu-memory-utilization 0.90 --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    $LORA_FLAGS > 'logs/vllm.log' 2>&1 & echo \$! > '$PIDFILE'; cat '$PIDFILE'"

cat <<EOF

== vLLM starting (first boot downloads weights — watch the log) ==
  ssh $BREV_HOST tail -f $LOG

Open the tunnel, then the capture targets work unchanged:

  ssh -N -L $VLLM_PORT:localhost:$VLLM_PORT $BREV_HOST &

  make compare-run-qwen  QWEN_MODEL=$MODEL
${LORA_DIR:+  make compare-run-lora  LORA_ADAPTER=$LORA_NAME}

Stop it with: BREV_HOST=$BREV_HOST bash scripts/brev_serve.sh --stop
EOF
