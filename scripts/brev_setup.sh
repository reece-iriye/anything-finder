#!/usr/bin/env bash
# One-command bootstrap for a brev.dev (or any CUDA) GPU box.
#
#   git clone <this repo> && cd anything-finder
#   HF_TOKEN=hf_... bash scripts/brev_setup.sh
#
# Then launch training with one of the two commands it prints.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found — this needs a CUDA GPU box." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${VRAM_MIB:-0}" -lt 49152 ]; then
  echo "WARNING: ${VRAM_MIB} MiB VRAM. QLoRA on Qwen3.8-27B wants >= 48 GB." >&2
  echo "         Use configs/lora/smoke.yaml first, or a bigger box." >&2
fi

echo "== uv =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "== deps (training group) =="
uv sync --group training

echo "== huggingface login =="
if [ -n "${HF_TOKEN:-}" ]; then
  uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
else
  echo "HF_TOKEN not set — skipping. Gated Qwen repos will 401." >&2
fi

echo "== torch CUDA / bf16 check =="
uv run python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not visible to torch'
assert torch.cuda.is_bf16_supported(), 'bf16 not supported on this GPU'
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))
"

cat <<'EOF'

== ready ==
Validate the pipeline first (tiny model, 20 steps, a few dollars):
  uv run scripts/train_lora.py --config configs/lora/smoke.yaml

Then the real run:
  uv run scripts/train_lora.py --config configs/lora/qwen3_8-27b-qlora.yaml
  # multi-GPU: accelerate launch scripts/train_lora.py --config configs/lora/qwen3_8-27b-qlora.yaml
EOF
