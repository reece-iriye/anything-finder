.PHONY: add add-dev remove remove-dev update install dev trace eval-queries eval-run eval-run-sonnet osm-convert lora-data lora-smoke lora-train

LORA_CONFIG ?= configs/lora/qwen3_8-27b-qlora.yaml

TRACE_PORT ?= 7861
AF_TRACE_DIR ?= telemetry
AF_API_BASE ?= http://localhost:9022

# Local geo services as published by compose.claude.yaml (nominatim 8082:8080,
# overpass 8083:80). Override if you remapped the host ports.
EVAL_NOMINATIM_URL ?= http://localhost:8082
EVAL_OVERPASS_URL ?= http://localhost:8083
EVAL_SONNET_MODEL ?= claude-sonnet-5

# make add pkg=requests
# make add pkg="requests>=2.28"
# make add pkg="requests~=2.28"
# make add pkg="requests==2.28.0"
add:
	@test -n "$(pkg)" || { echo "Usage: make add pkg=<package[constraint]>"; exit 1; }
	pipenv install "$(pkg)"
	python3 scripts/sync_pyproject.py

# make add-dev pkg=pytest
# make add-dev pkg="pytest>=8.0"
add-dev:
	@test -n "$(pkg)" || { echo "Usage: make add-dev pkg=<package[constraint]>"; exit 1; }
	pipenv install --dev "$(pkg)"

# make remove pkg=requests
remove:
	@test -n "$(pkg)" || { echo "Usage: make remove pkg=<package>"; exit 1; }
	pipenv uninstall "$(pkg)"
	python3 scripts/sync_pyproject.py

# make remove-dev pkg=pytest
remove-dev:
	@test -n "$(pkg)" || { echo "Usage: make remove-dev pkg=<package>"; exit 1; }
	pipenv uninstall "$(pkg)"

# make update pkg=requests ver=">=2.32"
# make update pkg=requests ver="~=2.28"
# make update pkg=requests ver="==2.28.0"
update:
	@test -n "$(pkg)" || { echo "Usage: make update pkg=<package> ver=<constraint>"; exit 1; }
	@test -n "$(ver)" || { echo "Usage: make update pkg=<package> ver=<constraint>"; exit 1; }
	pipenv install "$(pkg)$(ver)"
	python3 scripts/sync_pyproject.py

install:
	pipenv install --deploy

dev:
	AF_TRACE_DIR=$(AF_TRACE_DIR) uvicorn src.main:app --host 127.0.0.1 --port 9022 --reload

# Telemetry console (FastAPI): query the agent and drill into the trace it produces.
# Needs the agent API running (make dev, or docker compose). `make dev` and compose
# set AF_TRACE_DIR so runs are captured under $(AF_TRACE_DIR)/<mode>/.
trace:
	@command -v open >/dev/null && ( sleep 3 && open "http://127.0.0.1:$(TRACE_PORT)" ) & \
	AF_TRACE_DIR=$(AF_TRACE_DIR) AF_API_BASE=$(AF_API_BASE) TRACE_PORT=$(TRACE_PORT) uv run scripts/trace_ui.py

# Regenerate the synthetic Dallas food-search eval set
# (data/eval/dallas_food_queries.csv): 200 query + context_data rows for
# LoRA training data and LLM-as-a-judge runs. Deterministic (fixed seed).
eval-queries:
	uv run scripts/gen_eval_queries.py

# Execute every eval row through the restaurant workflow with Claude as the
# inference engine -> data/eval/dallas_food_runs.jsonl (LoRA SFT + LLM-as-judge).
# Needs ANTHROPIC_API_KEY and reachable Nominatim / Overpass — set
# NOMINATIM_BASE_URL / OVERPASS_BASE_URL to your local containers, or export
# NOMINATIM_USE_EXTERNAL_API=true OVERPASS_USE_EXTERNAL_API=true for the public
# servers. Pass ARGS for flags, e.g. make eval-run ARGS="--limit 5 --resume".
eval-run:
	LLM_BACKEND=claude uv run scripts/run_eval_queries.py $(ARGS)

# Same, pinned to Claude Sonnet and wired to the local compose geo services.
# Needs ANTHROPIC_API_KEY and the compose stack up:
#   docker compose -f compose.claude.yaml up -d nominatim overpass
eval-run-sonnet:
	LLM_BACKEND=claude \
	LLM_MODEL_AGENT=$(EVAL_SONNET_MODEL) \
	NOMINATIM_BASE_URL=$(EVAL_NOMINATIM_URL) \
	OVERPASS_BASE_URL=$(EVAL_OVERPASS_URL) \
	uv run scripts/run_eval_queries.py $(ARGS)

# ─── LoRA fine-tuning ──────────────────────────────────────────────────────
# Distil the captured Claude trajectories (data/eval/*_runs.jsonl) into a LoRA
# adapter for Qwen3.8-27B. Install the stack with `uv sync --group training`
# (kept out of the container). See the README "Fine-tuning on brev.dev" section.

# jsonl transcripts -> data/lora/{train,val}.jsonl + tool_schemas.json (hermetic).
lora-data:
	uv run scripts/prepare_lora_data.py $(ARGS)

# Data + masking sanity check: no GPU. Reads the printed masked example end to
# end — system/user/tool spans must be masked, every assistant turn trained.
lora-smoke:
	uv run scripts/train_lora.py --config configs/lora/smoke.yaml --dry-run

# Full run (needs a GPU box + `uv sync --group training`). Pass ARGS for --set.
lora-train:
	uv run scripts/train_lora.py --config $(LORA_CONFIG) $(ARGS)

osm-convert:
	@test -f data/Dallas.osm.gz || { echo "data/Dallas.osm.gz not found"; exit 1; }
	@test -f data/Dallas.osm.pbf || { echo "data/Dallas.osm.pbf not found"; exit 1; }
	osmium cat data/Dallas.osm.gz -o data/Dallas.osm.bz2 --overwrite
	@echo "data/Dallas.osm.bz2 ready."
