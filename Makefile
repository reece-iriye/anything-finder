.PHONY: add add-dev remove remove-dev update install dev

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
