# Mission-Oriented Agentic AI with Last-Mile Verification

[![CI](https://github.com/berkunis/nl-to-sql-verified/actions/workflows/ci.yml/badge.svg)](https://github.com/berkunis/nl-to-sql-verified/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready demonstration of **verified agentic AI** for mission-critical environments. This project showcases enterprise AI engineering patterns including CI/CD, containerization, observability, security, and MLOps.

## Core Thesis

LLM-generated outputs are probabilistic and may contain errors—syntax mistakes, schema violations, semantic mismatches, or unsafe operations. In mission-oriented environments (defense, healthcare, finance, critical infrastructure), **unverified AI outputs are unacceptable**.

This project demonstrates:
1. Generating structured outputs (SQL) from natural language
2. Verifying outputs through a deterministic pipeline
3. Automatically correcting failures with bounded retries
4. Maintaining full traceability of all decisions

## Features

| Capability | Implementation |
|------------|----------------|
| **Verification Chain** | 4 verifiers: Syntax, Schema, Safety, Semantic |
| **REST API** | FastAPI with OpenAPI docs |
| **Observability** | Prometheus metrics, OpenTelemetry tracing, structured logging |
| **Security** | API key auth, PII detection, NIST-compliant audit |
| **MLOps** | MLflow tracking, versioned datasets, benchmarking |
| **Containerization** | Docker, Kubernetes manifests |
| **CI/CD** | GitHub Actions pipeline |
| **RAG** | Schema-aware retrieval for improved generation |

## Quick Start

### Option 1: Run Locally

```bash
# Install
pip install -e ".[api,observability]"

# Start API
make run

# Test
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-12345" \
  -d '{"query": "Show me all premium customers"}'
```

### Option 2: Docker Compose (Recommended)

```bash
# Start full stack (API + Prometheus + Grafana + Jaeger)
make up

# Access services
# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
# Jaeger:     http://localhost:16686
```

### Option 3: Python Library

```python
from nl_to_sql import NLToSQLAgent, MockLLM

llm = MockLLM(responses={"customers": ["SELECT * FROM customers"]})
agent = NLToSQLAgent(llm=llm, max_retries=3)

result = agent.process("Show me all customers")
print(f"SQL: {result.sql}")  # SELECT * FROM customers
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      NLToSQLAgent Controller                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Query ──▶ LLM Generation ──▶ Verification Chain          │
│                        ▲               │                        │
│                        │               ▼                        │
│                   Correction ◀── Pass/Fail Decision             │
│                   (bounded)                                     │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              Verification Chain (Last Mile)              │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │  │
│   │  │ Syntax   │→│ Schema   │→│ Safety   │→│ Semantic   │  │  │
│   │  │ Verifier │ │ Verifier │ │ Verifier │ │ Verifier   │  │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Audit Logger                          │  │
│   │  • Timestamps every step • Full traceability             │  │
│   │  • NIST-compliant export • Integrity hashing             │  │
│   └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
nl-to-sql-verified/
├── src/nl_to_sql/           # Core agent library
│   ├── agent.py             # Main controller
│   ├── verifiers/           # Verification chain
│   ├── llm/                 # LLM interfaces
│   └── rag/                 # RAG components
├── api/                     # FastAPI REST API
├── observability/           # Prometheus, OpenTelemetry, logging
├── security/                # Auth, PII detection, audit export
├── mlops/                   # MLflow tracking, benchmarks
├── infra/                   # Docker, Kubernetes
├── tests/                   # Unit & integration tests
└── docs/                    # Documentation
```

## API Reference

### POST /api/v1/query

Convert natural language to verified SQL.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-12345" \
  -d '{"query": "What is the total revenue?", "include_audit": true}'
```

Response:
```json
{
  "success": true,
  "sql": "SELECT SUM(amount) FROM orders",
  "original_query": "What is the total revenue?",
  "attempts": 1,
  "message": "Query verified successfully after 1 attempt(s)",
  "request_id": "abc123",
  "processing_time_ms": 15.2
}
```

See [API Documentation](docs/api/openapi.yaml) for full reference.

## Verification Chain

| Verifier | Purpose | Method |
|----------|---------|--------|
| `SyntaxVerifier` | Valid SQL syntax | SQLite EXPLAIN |
| `SchemaVerifier` | Tables/columns exist | Regex + schema lookup |
| `SafetyVerifier` | No destructive ops | Pattern matching |
| `SemanticVerifier` | Matches user intent | Heuristic analysis |
| `PIIVerifier` | No PII exposure | Pattern + column check |

## Observability

### Metrics (Prometheus)

- `nl_to_sql_queries_total` - Total queries by status
- `nl_to_sql_query_duration_seconds` - Processing time histogram
- `nl_to_sql_verification_failures_total` - Failures by verifier

### Tracing (OpenTelemetry)

Distributed tracing through Jaeger for request flow visualization.

### Logging (Structured)

JSON-formatted logs with context propagation.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run linter
make lint

# Type check
make type-check

# All checks
make check
```

## Deployment

### Docker

```bash
make build                    # Build image
make up                       # Start stack
make logs                     # View logs
```

### Kubernetes

```bash
kubectl apply -f infra/k8s/   # Deploy to cluster
kubectl get all -n nl-to-sql  # Check status
```

## Extension Points

### Custom Verifier

```python
from nl_to_sql.verifiers import Verifier, VerificationResult, VerificationStatus

class BusinessRuleVerifier(Verifier):
    @property
    def name(self) -> str:
        return "BusinessRuleVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        if "customers" in sql.lower() and "tenant_id" not in sql.lower():
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.FAILED,
                message="Customer queries must include tenant_id filter"
            )
        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.PASSED,
            message="Business rules satisfied"
        )
```

### Custom LLM Provider

```python
from nl_to_sql.llm import LLMInterface, LLMResponse

class MyLLM(LLMInterface):
    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        # Your LLM integration
        return LLMResponse(content="SELECT ...", model="my-model")
```

## Skills Demonstrated

This project demonstrates enterprise AI engineering skills:

- **CI/CD Pipelines**: GitHub Actions with linting, type checking, testing, security scanning
- **Containerization**: Multi-stage Docker builds, Kubernetes manifests with HPA
- **API Architecture**: FastAPI with Pydantic, OpenAPI spec, versioned endpoints
- **Full-Stack Observability**: Prometheus metrics, OpenTelemetry tracing, structured logging
- **Testing**: pytest with fixtures, unit/integration tests, coverage reporting
- **MLOps**: MLflow experiment tracking, versioned datasets, benchmarking
- **Security**: API authentication, PII detection, NIST-compliant audit trails
- **RAG Integration**: Schema-aware retrieval for improved SQL generation

## Documentation

- [Quick Start Guide](docs/quickstart.md) - Get running in 5 minutes
- [Architecture Guide](docs/architecture.md) - System design and components
- [Design Decisions](docs/design-decisions.md) - Detailed rationale for all architectural choices
- [API Reference](docs/api/openapi.yaml) - Full OpenAPI specification

## License

MIT - Use freely, but verify your outputs.
