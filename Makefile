.PHONY: add add-dev remove remove-dev update install dev-install dev trace up down data test \
        eval-queries eval-run eval-run-sonnet osm-convert \
        lora-data lora-smoke lora-train lora-train-remote lora-logs lora-fetch lora-push \
        vllm-remote vllm-stop \
        compare-run compare-run-sonnet compare-run-qwen compare-run-lora traces-freeze compare

LORA_CONFIG ?= configs/lora/qwen2_5-7b-qlora.yaml

TRACE_PORT ?= 7861
AF_TRACE_DIR ?= telemetry
AF_API_BASE ?= http://localhost:9022

COMPOSE ?= docker compose -f compose.claude.yaml

# Local geo services as published by compose.claude.yaml (nominatim 8082:8080,
# overpass 8083:80). Override if you remapped the host ports.
EVAL_NOMINATIM_URL ?= http://localhost:8082
EVAL_OVERPASS_URL ?= http://localhost:8083
EVAL_SONNET_MODEL ?= claude-sonnet-5

# ─── Three-way comparison (see `make compare`) ─────────────────────────────
# The same COMPARE_LIMIT rows of the eval set, run through each backend with
# telemetry on. The eval CSV is deterministic and read in file order, so
# --limit N is the *same* N queries every time — that is what makes it a
# comparison rather than three unrelated samples.
COMPARE_LIMIT ?= 25
COMPARE_ARGS ?= --limit $(COMPARE_LIMIT)
QWEN_MODEL ?= Qwen/Qwen2.5-7B-Instruct-AWQ
VLLM_BASE_URL ?= http://localhost:8000/v1
LORA_ADAPTER ?= af-lora

# ─── Remote GPU box (brev.dev) ─────────────────────────────────────────────
# BREV_HOST is an ssh target or ssh_config alias; HF_REPO enables the push.
BREV_DIR ?= anything-finder
LORA_RUN_NAME = $(shell awk '/^run_name:/{print $$2; exit}' $(LORA_CONFIG))

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

# The project is uv + uv.lock everywhere else (containerfile, every `uv run`
# target below); the pipenv helpers above are legacy dependency bookkeeping.
install:
	uv sync

dev-install:
	uv sync --group dev

test:
	uv run pytest

# ─── Local stack ───────────────────────────────────────────────────────────
# Bring up the geo services the agent needs. Nominatim's FIRST boot imports the
# Dallas extract and can take 10+ minutes; the poll below waits it out.
up:
	$(COMPOSE) up -d postgres osm-server osm-bz2-ready nominatim overpass
	@echo "waiting for nominatim ($(EVAL_NOMINATIM_URL)) and overpass ($(EVAL_OVERPASS_URL))…"
	@for i in $$(seq 1 120); do \
	  n=$$(curl -sf "$(EVAL_NOMINATIM_URL)/search?q=Dallas&format=json&limit=1" >/dev/null && echo ok || echo no); \
	  o=$$(curl -sf "$(EVAL_OVERPASS_URL)/api/interpreter?data=[out:json];out;" >/dev/null && echo ok || echo no); \
	  if [ "$$n" = ok ] && [ "$$o" = ok ]; then echo "geo stack ready."; exit 0; fi; \
	  printf "\r  nominatim=%s overpass=%s  (%ds)" "$$n" "$$o" $$((i*10)); sleep 10; \
	done; \
	printf "\nstill not healthy — check '$(COMPOSE) logs nominatim overpass'\n"; exit 1

down:
	$(COMPOSE) down

# Full teacher-data chain: query set -> Claude trajectories -> SFT train/val.
data: eval-queries eval-run-sonnet lora-data

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

# ─── Three-way comparison capture ──────────────────────────────────────────
# Each target replays the same COMPARE_LIMIT eval rows with AF_TRACE_DIR set,
# so every run leaves a full trace under $(AF_TRACE_DIR)/<mode>/. Student
# captures go to data/eval/runs/<mode>.jsonl — NOT data/eval/*_runs.jsonl,
# which prepare_lora_data.py globs for the teacher's SFT set.

compare-run: compare-run-sonnet compare-run-qwen compare-run-lora

# Teacher. Needs ANTHROPIC_API_KEY and `make up`.
compare-run-sonnet:
	AF_TRACE_DIR=$(AF_TRACE_DIR) \
	LLM_BACKEND=claude \
	LLM_MODEL_AGENT=$(EVAL_SONNET_MODEL) \
	NOMINATIM_BASE_URL=$(EVAL_NOMINATIM_URL) \
	OVERPASS_BASE_URL=$(EVAL_OVERPASS_URL) \
	uv run scripts/run_eval_queries.py --out data/eval/runs/claude.jsonl $(COMPARE_ARGS) $(ARGS)

# Student, before fine-tuning. Needs a reachable vLLM (make vllm-remote + tunnel).
compare-run-qwen:
	AF_TRACE_DIR=$(AF_TRACE_DIR) \
	LLM_BACKEND=vllm \
	LLM_MODEL_AGENT=$(QWEN_MODEL) \
	LLM_BASE_URL=$(VLLM_BASE_URL) \
	NOMINATIM_BASE_URL=$(EVAL_NOMINATIM_URL) \
	OVERPASS_BASE_URL=$(EVAL_OVERPASS_URL) \
	uv run scripts/run_eval_queries.py --out data/eval/runs/raw-open-source.jsonl $(COMPARE_ARGS) $(ARGS)

# Student, after fine-tuning. Same endpoint, the adapter selected by name.
compare-run-lora:
	AF_TRACE_DIR=$(AF_TRACE_DIR) \
	LLM_BACKEND=lora \
	LLM_MODEL_LORA=$(LORA_ADAPTER) \
	LLM_BASE_URL=$(VLLM_BASE_URL) \
	NOMINATIM_BASE_URL=$(EVAL_NOMINATIM_URL) \
	OVERPASS_BASE_URL=$(EVAL_OVERPASS_URL) \
	uv run scripts/run_eval_queries.py --out data/eval/runs/finetuned-open-source.jsonl $(COMPARE_ARGS) $(ARGS)

# telemetry/<mode>/*.json -> data/traces/<mode>.jsonl.gz (committed; merges).
traces-freeze:
	AF_TRACE_DIR=$(AF_TRACE_DIR) uv run scripts/freeze_traces.py --trace-dir $(AF_TRACE_DIR)

# One self-contained data/compare/report.html: leaderboard + per-query
# side-by-side + full trace drill-in. Opens from disk; no server needed.
compare:
	uv run scripts/build_compare_report.py --trace-dir $(AF_TRACE_DIR) $(ARGS)
	@command -v open >/dev/null && open data/compare/report.html || true

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
# Set HF_REPO to push the finished adapter + model card to the Hub.
lora-train:
	uv run scripts/train_lora.py --config $(LORA_CONFIG) \
	  $(if $(HF_REPO),--set hub.push_to_hub=true --set hub.hub_model_id=$(HF_REPO)) $(ARGS)

# ─── Remote GPU box ────────────────────────────────────────────────────────
# This machine has no CUDA, so training and vLLM live on a brev.dev box.
#   make lora-train-remote BREV_HOST=my-box HF_REPO=me/anything-finder-qwen-lora

lora-train-remote:
	@test -n "$(BREV_HOST)" || { echo "Usage: make lora-train-remote BREV_HOST=<ssh target> [HF_REPO=<user>/<repo>]"; exit 1; }
	BREV_HOST=$(BREV_HOST) BREV_DIR=$(BREV_DIR) LORA_CONFIG=$(LORA_CONFIG) \
	  HF_REPO=$(HF_REPO) ARGS="$(ARGS)" bash scripts/brev_train.sh

lora-logs:
	@test -n "$(BREV_HOST)" || { echo "Usage: make lora-logs BREV_HOST=<ssh target>"; exit 1; }
	ssh $(BREV_HOST) tail -f $(BREV_DIR)/logs/$(LORA_RUN_NAME).log

lora-fetch:
	@test -n "$(BREV_HOST)" || { echo "Usage: make lora-fetch BREV_HOST=<ssh target>"; exit 1; }
	mkdir -p data/lora/runs
	rsync -az --exclude 'checkpoint-*/' --exclude 'runs/' \
	  $(BREV_HOST):$(BREV_DIR)/data/lora/runs/$(LORA_RUN_NAME)/ data/lora/runs/$(LORA_RUN_NAME)/
	@echo "adapter -> data/lora/runs/$(LORA_RUN_NAME)"

# Push a finished run dir to Hugging Face — no GPU, no retraining.
lora-push:
	@test -n "$(HF_REPO)" || { echo "Usage: make lora-push HF_REPO=<user>/<repo>"; exit 1; }
	uv run scripts/push_lora.py --repo $(HF_REPO) --run-dir data/lora/runs/$(LORA_RUN_NAME) $(ARGS)

# Serve the base model (+ the adapter when LORA_DIR is set) on the GPU box.
vllm-remote:
	@test -n "$(BREV_HOST)" || { echo "Usage: make vllm-remote BREV_HOST=<ssh target> [LORA_DIR=data/lora/runs/<run>]"; exit 1; }
	BREV_HOST=$(BREV_HOST) BREV_DIR=$(BREV_DIR) MODEL="$(MODEL)" LORA_DIR="$(LORA_DIR)" \
	  LORA_NAME=$(LORA_ADAPTER) bash scripts/brev_serve.sh

vllm-stop:
	@test -n "$(BREV_HOST)" || { echo "Usage: make vllm-stop BREV_HOST=<ssh target>"; exit 1; }
	BREV_HOST=$(BREV_HOST) BREV_DIR=$(BREV_DIR) bash scripts/brev_serve.sh --stop

osm-convert:
	@test -f data/Dallas.osm.gz || { echo "data/Dallas.osm.gz not found"; exit 1; }
	@test -f data/Dallas.osm.pbf || { echo "data/Dallas.osm.pbf not found"; exit 1; }
	osmium cat data/Dallas.osm.gz -o data/Dallas.osm.bz2 --overwrite
	@echo "data/Dallas.osm.bz2 ready."
