# Quick Start Guide

Get the NL-to-SQL Verified agent running in 5 minutes.

## Prerequisites

- Python 3.10+
- Docker (optional, for full stack)

## Installation

### Option 1: Basic Installation

```bash
# Clone the repository
git clone https://github.com/berkunis/nl-to-sql-verified.git
cd nl-to-sql-verified

# Install core dependencies
pip install -e .

# Run the evaluation demo
python nl_to_sql_agent.py
```

### Option 2: Full Stack with API

```bash
# Install with API and observability
pip install -e ".[api,observability]"

# Start the API
make run

# Or use uvicorn directly
uvicorn api.main:app --reload
```

### Option 3: Docker Compose (Recommended)

```bash
# Start everything (API + Prometheus + Grafana + Jaeger)
make up

# Check status
make ps

# View logs
make logs
```

## Verify Installation

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Generate SQL
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-12345" \
  -d '{"query": "Show me all premium customers"}'
```

### Expected Response

```json
{
  "success": true,
  "sql": "SELECT name, email FROM customers WHERE tier = 'premium'",
  "original_query": "Show me all premium customers",
  "attempts": 1,
  "message": "Query verified successfully after 1 attempt(s)",
  "request_id": "abc123",
  "processing_time_ms": 15.2
}
```

## Access Observability Stack

After `make up`:

| Service    | URL                      | Credentials  |
|------------|--------------------------|--------------|
| API Docs   | http://localhost:8000/docs | -          |
| Prometheus | http://localhost:9090    | -            |
| Grafana    | http://localhost:3000    | admin/admin  |
| Jaeger     | http://localhost:16686   | -            |

## Basic Usage (Python)

```python
from nl_to_sql import NLToSQLAgent, MockLLM

# Create agent with mock LLM (for demo)
llm = MockLLM(responses={
    "customers": ["SELECT * FROM customers"],
    "premium": ["SELECT name FROM customers WHERE tier = 'premium'"]
})
agent = NLToSQLAgent(llm=llm)

# Process a query
result = agent.process("Show me all premium customers")

if result.success:
    print(f"SQL: {result.sql}")
    print(f"Attempts: {result.attempts}")
else:
    print(f"Failed: {result.final_message}")

# Inspect audit trail
for entry in result.audit_trail:
    print(f"{entry.step}: {entry.output_data}")
```

## Run Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
make test

# Run unit tests only
make test-unit
```

## Next Steps

- [Architecture Guide](./architecture.md) - Understand the system design
- [Design Decisions](./design-decisions.md) - Detailed rationale for architectural choices
- [API Reference](./api/openapi.yaml) - Full API documentation

## Troubleshooting

### "ModuleNotFoundError: No module named 'nl_to_sql'"

Make sure you've installed the package:
```bash
pip install -e .
```

### Docker containers not starting

Check Docker is running:
```bash
docker ps
```

Clean up and restart:
```bash
make clean-docker
make build
make up
```

### API returns 401 Unauthorized

Include the API key header:
```bash
-H "X-API-Key: dev-key-12345"
```
