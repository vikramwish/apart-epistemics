.PHONY: help install lint format test check smoke reproduce paper paper-figures paper-audit paper-reanalyze

PYTHON ?= .venv/bin/python
BOOTSTRAP_PYTHON ?= python3

help: ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST)

install: ## Install Python package and research dependencies
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PYTHON) -m pip install -e ".[dev,paper]"

lint: ## Check source style
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Format non-frozen source
	$(PYTHON) -m ruff format .

test: ## Run offline tests
	$(PYTHON) -m pytest -v

smoke: ## Validate configurations without API calls
	$(PYTHON) -m apart_epistemics run --config configs/polyphony_p0_p4.yaml --dry-run
	$(PYTHON) -m apart_epistemics run --config configs/polyphony_replication_2025.yaml --dry-run

check: lint test smoke ## Run offline acceptance checks

reproduce: ## Reproduce the discovery-study audit
	$(PYTHON) scripts/audit_and_reproduce_p0_p4.py

paper-audit: ## Check mathematics and task-level sensitivity
	$(PYTHON) research/manuscript/verify.py
	$(PYTHON) research/manuscript/audit_math.py
	$(PYTHON) research/manuscript/audit_statistics.py

paper-figures: ## Regenerate manuscript figures
	cd research/manuscript && $(abspath $(PYTHON)) fig0.py
	cd research/manuscript && $(abspath $(PYTHON)) figs.py
	cd research/manuscript && $(abspath $(PYTHON)) fig_main_v3.py
	cd research/manuscript && $(abspath $(PYTHON)) fig_replication.py

paper: paper-figures ## Rebuild retained manuscript source without replacing the submitted PDF
	cd research/manuscript && latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex

paper-reanalyze: ## Recompute supplements from separately supplied raw logs
	$(PYTHON) research/manuscript/s2_analysis.py
	$(PYTHON) research/manuscript/s2_extra.py
