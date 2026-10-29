SHELL := /bin/bash
PYTHON := .venv/bin/python
PY := PYTHONPATH=src $(PYTHON)

.PHONY: help setup index api web test data audit eval eval-baseline eval-rag compare redteam perf docker-up docker-down

help:
	@echo "make setup      创建虚拟环境并安装核心依赖"
	@echo "make index      构建知识库索引"
	@echo "make api        启动 FastAPI（默认 offline 生成器）"
	@echo "make web        启动 Next.js 前端"
	@echo "make data       校验数据并生成 data card / 训练数据"
	@echo "make test       运行 pytest"
	@echo "make audit      生成 20 条人工抽检包"
	@echo "make eval       跑 baseline / RAG 两轮评测并对比"
	@echo "make redteam    跑 50 条红队回归"
	@echo "make perf       压测本地 API"

setup:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e .
	cd app/web && npm install
	@test -f .env || cp .env.example .env
	@echo "完成。运行 make index && make api && make web"

index:
	$(PY) -m heart_say.rag.ingest --json

api:
	PYTHONPATH=src .venv/bin/uvicorn app.api.main:app --host 0.0.0.0 --port 8010 --reload

web:
	cd app/web && npm run dev

test:
	$(PYTHON) -m pytest -q

data:
	$(PY) scripts/data_report.py --check
	$(PY) scripts/build_training_data.py

audit:
	$(PY) scripts/build_audit.py

eval-baseline:
	QDRANT_PATH= QDRANT_URL= $(PY) -m heart_say.evaluation.runner --name baseline --no-rag

eval-rag:
	QDRANT_PATH= QDRANT_URL= $(PY) -m heart_say.evaluation.runner --name rag

eval: eval-baseline eval-rag compare

compare:
	$(PY) -m heart_say.evaluation.compare --baseline reports/eval_baseline.json --rag reports/eval_rag.json

redteam:
	QDRANT_PATH= QDRANT_URL= $(PY) scripts/run_redteam.py --name offline

perf:
	$(PY) scripts/benchmark_inference.py --mode app --url http://localhost:8010 --label offline-api

docker-up:
	docker compose -f deploy/docker-compose.yml up --build

docker-down:
	docker compose -f deploy/docker-compose.yml down
