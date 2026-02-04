# Makefile for NL-to-SQL Verified
# Provides convenient commands for development and deployment

.PHONY: help install install-dev test lint format type-check build up down logs clean

# Default target
help:
	@echo "NL-to-SQL Verified - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run tests with coverage"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make lint           Run linter (ruff)"
	@echo "  make format         Format code (ruff)"
	@echo "  make type-check     Run type checker (mypy)"
	@echo "  make check          Run all checks (lint, format, type-check, test)"
	@echo ""
	@echo "Docker:"
	@echo "  make build          Build Docker image"
	@echo "  make up             Start local stack (API + observability)"
	@echo "  make down           Stop local stack"
	@echo "  make logs           View API logs"
	@echo "  make shell          Open shell in API container"
	@echo ""
	@echo "API:"
	@echo "  make run            Run API locally (without Docker)"
	@echo "  make query          Test query endpoint"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts and caches"
	@echo "  make clean-docker   Remove Docker images and volumes"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e ".[api,observability]"

install-dev:
	pip install -e ".[all]"

# =============================================================================
# Development
# =============================================================================

test:
	pytest tests/ -v --cov=src/nl_to_sql --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v --cov=src/nl_to_sql --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v -m integration

lint:
	ruff check src/ tests/ api/ observability/

format:
	ruff format src/ tests/ api/ observability/

type-check:
	mypy src/

check: lint format type-check test

# =============================================================================
# Docker
# =============================================================================

DOCKER_COMPOSE = docker compose -f infra/docker/docker-compose.yml

build:
	docker build -t nl-to-sql-verified:latest -f infra/docker/Dockerfile .

build-dev:
	docker build -t nl-to-sql-verified:dev --target development -f infra/docker/Dockerfile .

up:
	$(DOCKER_COMPOSE) up -d
	@echo ""
	@echo "Services started:"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  Jaeger:     http://localhost:16686"

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f api

logs-all:
	$(DOCKER_COMPOSE) logs -f

shell:
	$(DOCKER_COMPOSE) exec api /bin/bash

ps:
	$(DOCKER_COMPOSE) ps

restart:
	$(DOCKER_COMPOSE) restart api

# =============================================================================
# API
# =============================================================================

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

query:
	@curl -s -X POST http://localhost:8000/api/v1/query \
		-H "Content-Type: application/json" \
		-H "X-API-Key: dev-key-12345" \
		-d '{"query": "Show me all premium customers"}' | python -m json.tool

query-audit:
	@curl -s -X POST http://localhost:8000/api/v1/query \
		-H "Content-Type: application/json" \
		-H "X-API-Key: dev-key-12345" \
		-d '{"query": "Show me all premium customers", "include_audit": true}' | python -m json.tool

health:
	@curl -s http://localhost:8000/health | python -m json.tool

metrics:
	@curl -s http://localhost:8000/metrics

# =============================================================================
# Kubernetes
# =============================================================================

k8s-apply:
	kubectl apply -f infra/k8s/namespace.yaml
	kubectl apply -f infra/k8s/configmap.yaml
	kubectl apply -f infra/k8s/deployment.yaml
	kubectl apply -f infra/k8s/service.yaml

k8s-delete:
	kubectl delete -f infra/k8s/ --ignore-not-found

k8s-logs:
	kubectl logs -f -l app.kubernetes.io/name=nl-to-sql-api -n nl-to-sql

k8s-status:
	kubectl get all -n nl-to-sql

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-docker:
	$(DOCKER_COMPOSE) down -v --rmi local
	docker image prune -f

# =============================================================================
# MLOps
# =============================================================================

mlflow-ui:
	mlflow ui --port 5000

benchmark:
	python -m mlops.eval.benchmark

# =============================================================================
# Documentation
# =============================================================================

docs-serve:
	@echo "Documentation available at docs/quickstart.md"
