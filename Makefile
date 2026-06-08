PYTHON ?= python

.PHONY: install lint format-check test test-unit ci-local ingest run-api api run-ui ui eval-retrieval eval-answers eval-enterprise eval-gate docker-infra

install:
	$(PYTHON) scripts/dev.py install

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) scripts/dev.py run-tests

test-unit:
	$(PYTHON) -m pytest -m "not integration" -q

ci-local: lint format-check test-unit

ingest:
	$(PYTHON) scripts/dev.py ingest-data

run-api api:
	$(PYTHON) scripts/dev.py run-api

run-ui ui:
	$(PYTHON) scripts/dev.py run-ui

eval-retrieval:
	$(PYTHON) scripts/dev.py benchmark-retrieval

eval-answers:
	$(PYTHON) scripts/dev.py benchmark-answers

eval-enterprise:
	$(PYTHON) eval/evaluate_enterprise_support.py --dry-run

eval-gate:
	$(PYTHON) eval/evaluate_enterprise_support.py --dry-run --fail-under --min-recall-at-5 0.65 --min-source-hit-rate 0.90

docker-infra:
	docker compose up -d qdrant redis
