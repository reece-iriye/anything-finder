.PHONY: add add-dev remove remove-dev update install dev ui osm-convert

AF_GRADIO_PORT ?= 7860

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
	uvicorn src.main:app --host 127.0.0.1 --port 9022 --reload

# Gradio query UI for the restaurant endpoint (talks to a running API over HTTP).
# Override target API / port: make ui AF_API_BASE=http://localhost:9022 AF_GRADIO_PORT=7860
ui:
	@command -v open >/dev/null && ( sleep 4 && open "http://127.0.0.1:$(AF_GRADIO_PORT)" ) & \
	AF_GRADIO_PORT=$(AF_GRADIO_PORT) uv run --with gradio --with httpx scripts/query_ui.py

osm-convert:
	@test -f data/Dallas.osm.gz || { echo "data/Dallas.osm.gz not found"; exit 1; }
	@test -f data/Dallas.osm.pbf || { echo "data/Dallas.osm.pbf not found"; exit 1; }
	osmium cat data/Dallas.osm.gz -o data/Dallas.osm.bz2 --overwrite
	@echo "data/Dallas.osm.bz2 ready."
