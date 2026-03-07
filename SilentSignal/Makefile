SHELL := /bin/zsh
PYTHONPATH := .
PYTHON := python3
PIP := $(PYTHON) -m pip

.PHONY: setup check-deps generate-data api dashboard dev lint test verify

setup:
	$(PIP) install -r requirements.txt

check-deps:
	@$(PYTHON) -c "import fastapi, uvicorn" >/dev/null 2>&1 || (echo "Missing dependencies. Run: make setup"; exit 1)

generate-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) data/generate_synthetic_data.py

api:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m uvicorn services.api.app.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	$(PYTHON) -m http.server 5173 -d apps/dashboard

dev:
	./scripts/run_local.sh

lint:
	PYTHONPYCACHEPREFIX=.pycache $(PYTHON) -m compileall -q services agents data

test:
	@$(MAKE) check-deps
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

verify: lint check-deps generate-data test
