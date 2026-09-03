#!/usr/bin/env bash
# Serve TensorBoard on the remote GPU box against a run's own log dir, so loss
# curves are visible live while `make lora-train-remote` is still running.
#
#   BREV_HOST=my-brev-box bash scripts/brev_tensorboard.sh
#   BREV_HOST=... LORA_CONFIG=configs/lora/qwen3_8-27b-qlora.yaml bash scripts/brev_tensorboard.sh
#   BREV_HOST=... bash scripts/brev_tensorboard.sh --stop
#
# TensorBoard reads the event files as the trainer writes them (its own
# polling, ~30s), so the dashboard updates while training is in progress —
# no need to wait for a run to finish.
#
# Env:
#   BREV_HOST   (required) ssh target
#   BREV_DIR    remote checkout path  (default anything-finder)
#   LORA_CONFIG config yaml, used only to find output_dir (default configs/lora/qwen2_5-7b-qlora.yaml)
#   TB_PORT     remote + local port   (default 6006)
set -euo pipefail

cd "$(dirname "$0")/.."

: "${BREV_HOST:?set BREV_HOST=<ssh target>}"
BREV_DIR="${BREV_DIR:-anything-finder}"
LORA_CONFIG="${LORA_CONFIG:-configs/lora/qwen2_5-7b-qlora.yaml}"
TB_PORT="${TB_PORT:-6006}"
PIDFILE="$BREV_DIR/logs/tensorboard.pid"
LOG="$BREV_DIR/logs/tensorboard.log"

if [ "${1:-}" = "--stop" ]; then
  ssh "$BREV_HOST" "if [ -f '$PIDFILE' ]; then kill \$(cat '$PIDFILE') 2>/dev/null && echo stopped || echo 'not running'; rm -f '$PIDFILE'; else echo 'no pidfile'; fi"
  exit 0
fi

OUTPUT_DIR=$(awk '/^output_dir:/{print $2; exit}' "$LORA_CONFIG")
OUTPUT_DIR="${OUTPUT_DIR:-data/lora/runs}"

echo "== starting TensorBoard on $BREV_HOST (logdir: $OUTPUT_DIR) =="
ssh "$BREV_HOST" "mkdir -p '$BREV_DIR/logs'"
ssh "$BREV_HOST" "cd '$BREV_DIR' && export PATH=\$HOME/.local/bin:\$PATH && \
  nohup uv run --group training tensorboard --logdir '$OUTPUT_DIR' --port $TB_PORT --host 0.0.0.0 \
  > 'logs/tensorboard.log' 2>&1 & echo \$! > '$PIDFILE'; cat '$PIDFILE'"

cat <<EOF

== TensorBoard starting ==
  ssh $BREV_HOST tail -f $LOG

Open the tunnel, then browse the dashboard locally:

  ssh -N -L $TB_PORT:localhost:$TB_PORT $BREV_HOST &
  open http://localhost:$TB_PORT

It picks up new events as the trainer writes them (~30s poll) — safe to open
while training is still running. Scalars: train/loss, eval/loss, learning_rate,
grad_norm, epoch.

Stop it with: BREV_HOST=$BREV_HOST bash scripts/brev_tensorboard.sh --stop
EOF
