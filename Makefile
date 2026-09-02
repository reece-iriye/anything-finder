.PHONY: add add-dev remove remove-dev update install dev trace osm-convert

TRACE_PORT ?= 7861
AF_TRACE_DIR ?= telemetry
AF_API_BASE ?= http://localhost:9022

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

osm-convert:
	@test -f data/Dallas.osm.gz || { echo "data/Dallas.osm.gz not found"; exit 1; }
	@test -f data/Dallas.osm.pbf || { echo "data/Dallas.osm.pbf not found"; exit 1; }
	osmium cat data/Dallas.osm.gz -o data/Dallas.osm.bz2 --overwrite
	@echo "data/Dallas.osm.bz2 ready."
