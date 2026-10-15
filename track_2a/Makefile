# SmartStart — AI job interview coach (Hack Apertus 2026, Track 2A FHGR)
# Judges: `make run` from this directory on a clean checkout.
#   It builds and starts the LangGraph/Aegra API (port 2026) + Postgres + Redis in Docker.
#   Apertus is reached via LLM_BASE_URL (OpenAI-compatible). See README.md.

.PHONY: run run-gpu down env cli test smoke eval-assess eval-sim eval-judge eval-report eval

PY ?= python3
export LLM_BASE_URL ?= http://localhost:8080/v1

env:
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example — set LLM_BASE_URL / LLM_API_KEY if needed")

run: env  ## localhost endpoints are rewritten to host.docker.internal for the container
	LLM_BASE_URL=$(subst 127.0.0.1,host.docker.internal,$(subst localhost,host.docker.internal,$(LLM_BASE_URL))) docker compose up --build

run-gpu: env  ## also serve Apertus with vLLM on an NVIDIA GPU
	LLM_BASE_URL=http://vllm:8000/v1 docker compose --profile gpu up --build

down:
	docker compose --profile gpu down

cli:  ## chat in the terminal without Docker (needs a running Apertus endpoint)
	cd src && LLM_BASE_URL=$${LLM_BASE_URL:-http://localhost:8080/v1} $(PY) -m smartstart.cli $(ARGS)

test:
	cd src && LLM_BASE_URL=mock $(PY) -m pytest -q ../tests

smoke:  ## end-to-end run of all 28 scenarios with the mock backend (no GPU)
	LLM_BASE_URL=mock $(PY) eval/simulate.py --out eval/results/smoke --limit 28

eval-assess:  ## assessment accuracy vs. FHGR expert annotations (per language)
	$(PY) eval/eval_assess.py --per-lang 25 --out eval/results

eval-sim:  ## simulated interviews for the FHGR scenarios
	$(PY) eval/simulate.py --out eval/results/sim

eval-judge:  ## LLM-as-judge on simulated transcripts + judge validation
	$(PY) eval/judge.py --sim eval/results/sim --out eval/results

eval-report:
	$(PY) eval/report.py --results eval/results --figures docs/figures

eval: eval-assess eval-sim eval-judge eval-report
