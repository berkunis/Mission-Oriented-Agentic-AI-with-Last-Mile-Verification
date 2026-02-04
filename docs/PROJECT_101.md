# Project 101: Mission-Oriented Agentic AI with Last-Mile Verification

## TL;DR

This project demonstrates a **verification-first architecture** for LLM-powered agents that convert natural language to SQL. Instead of blindly trusting LLM output, every generated query passes through a chain of verifiers (syntax, schema, safety, semantic) before execution. Failed queries trigger automatic correction loops with error context. The result: auditable, safe SQL generation with full traceability.

---

## The Problem

LLMs are **probabilistic systems**—they generate plausible output, not guaranteed-correct output:

- **Syntax errors**: Typos like `SELEC` instead of `SELECT`, missing commas, unbalanced parentheses
- **Schema violations**: References to tables or columns that don't exist
- **Unsafe operations**: Accidental `DROP TABLE`, `DELETE` without `WHERE`, SQL injection vectors
- **Semantic drift**: Query that runs but doesn't match user intent (e.g., missing `ORDER BY` when user asked for "top 10")

In mission-critical environments (financial systems, healthcare, infrastructure), **"trust the AI" is not acceptable**. The "last-mile" gap between LLM generation and safe execution must be closed with deterministic verification.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
│                  "Show me top 5 customers"                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       NLToSQLAgent                              │
│                   (src/nl_to_sql/agent.py)                      │
│                                                                 │
│   ┌──────────────┐    ┌──────────────────────────────────────┐ │
│   │  LLM Interface│───▶│  Generate SQL from prompt            │ │
│   └──────────────┘    └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Verification Chain                           │
│               (src/nl_to_sql/verifiers/base.py)                 │
│                                                                 │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│   │ Syntax   │──▶│ Schema   │──▶│ Safety   │──▶│ Semantic │   │
│   │ Verifier │   │ Verifier │   │ Verifier │   │ Verifier │   │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                                 │
│            Fail-fast: stops at first failure                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       ┌──────────┐                   ┌──────────────┐
       │  PASS    │                   │    FAIL      │
       └──────────┘                   └──────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│  Verified SQL           │     │  Correction Loop            │
│  + Full Audit Trail     │     │  (max 3 retries)            │
│                         │     │  Error context → LLM → Retry│
└─────────────────────────┘     └─────────────────────────────┘
```

---

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Agent Controller** | `src/nl_to_sql/agent.py` | Orchestrates generation, verification, and correction loops |
| **Verification Chain** | `src/nl_to_sql/verifiers/base.py` | Sequential fail-fast pipeline of verifiers |
| **SyntaxVerifier** | `src/nl_to_sql/verifiers/syntax.py` | SQLite `EXPLAIN` validation on in-memory DB |
| **SchemaVerifier** | `src/nl_to_sql/verifiers/schema.py` | Table/column existence checks via regex extraction |
| **SafetyVerifier** | `src/nl_to_sql/verifiers/safety.py` | Blocks DROP, TRUNCATE, DELETE without WHERE |
| **SemanticVerifier** | `src/nl_to_sql/verifiers/semantic.py` | Intent matching (aggregation, ordering, limits) |
| **Data Models** | `src/nl_to_sql/models.py` | `VerificationResult`, `AuditEntry`, `AgentResult`, `LLMResponse` |
| **LLM Interface** | `src/nl_to_sql/llm/base.py` | Abstract base for LLM providers |
| **Mock LLM** | `src/nl_to_sql/llm/mock.py` | Testing implementation with canned responses |

---

## Usage Example

```python
from nl_to_sql import NLToSQLAgent, MockLLM

# Configure LLM with expected responses
llm = MockLLM(responses={
    "customers": ["SELECT * FROM customers"]
})

# Initialize agent with retry limit
agent = NLToSQLAgent(llm=llm, max_retries=3)

# Process natural language query
result = agent.process("Show me all customers")

# Access results
print(result.success)        # True
print(result.sql)            # SELECT * FROM customers
print(result.attempts)       # 1
print(result.final_message)  # "Query verified successfully after 1 attempt(s)"

# Full audit trail available
for entry in result.audit_trail:
    print(f"{entry.step}: {entry.output_data}")
```

---

## Verification Chain Detail

### 1. SyntaxVerifier

**How it works:**
1. Creates an in-memory SQLite database
2. Creates empty tables matching the schema (with correct column types)
3. Runs `EXPLAIN {sql}` to validate syntax
4. Returns PASS if no SQLite error, FAIL with error message otherwise

```python
# Internal flow (simplified)
conn = sqlite3.connect(":memory:")
cursor.execute("CREATE TABLE customers (id INTEGER, name TEXT, ...)")
cursor.execute(f"EXPLAIN {sql}")  # Validates syntax without executing
```

### 2. SchemaVerifier

**How it works:**
1. Extracts table names using regex: `FROM|JOIN|INTO|UPDATE` patterns
2. Validates each table exists in the known schema
3. Extracts column references from SELECT clause
4. Validates columns exist (accounting for aliases and SQL keywords)

**Blocked patterns:**
- Unknown tables: `SELECT * FROM nonexistent_table`
- Unknown columns: `SELECT bad_column FROM customers`

### 3. SafetyVerifier

**How it works:**
Pattern matching against dangerous SQL operations:

| Pattern | Detection |
|---------|-----------|
| `DROP TABLE/DATABASE/INDEX` | Destructive DDL |
| `TRUNCATE` | Data deletion |
| `DELETE FROM table;` | DELETE without WHERE |
| `UPDATE table SET ... ;` | UPDATE without WHERE |
| `; --` | Potential SQL injection |
| `EXEC(` | Dynamic SQL execution |

### 4. SemanticVerifier

**How it works:**
Heuristic keyword matching between user intent and generated SQL:

| User Intent Keywords | Expected SQL |
|---------------------|--------------|
| "total", "sum", "count", "average", "how many" | `COUNT(`, `SUM(`, `AVG(` |
| "highest", "lowest", "top", "most" | `ORDER BY` |
| "top", "first", "only" | `LIMIT` |

---

## Correction Loop

When verification fails, the agent automatically retries with error context:

### Example Flow

**Attempt 1:**
```
User: "Show me all customers"
LLM generates: "SELEC * FROM customers"
                 ^^^^^ typo
SyntaxVerifier: FAIL - "near 'SELEC': syntax error"
```

**Correction Prompt Sent to LLM:**
```
The previous SQL query had an error.

Original question: Show me all customers
Previous SQL: SELEC * FROM customers
Error: SQL syntax error: near "SELEC": syntax error

Please generate a corrected SQL query that fixes this issue.
Return ONLY the SQL query, no explanations.
```

**Attempt 2:**
```
LLM generates: "SELECT * FROM customers"
SyntaxVerifier: PASS
SchemaVerifier: PASS
SafetyVerifier: PASS
SemanticVerifier: PASS

Result: SUCCESS after 2 attempts
```

---

## Audit Trail Structure

Every operation is logged with full traceability:

```python
@dataclass
class AuditEntry:
    timestamp: str           # ISO format: "2025-02-03T10:15:30.123456"
    step: str                # "generation_attempt_1", "verification_1", etc.
    input_data: dict         # What went into this step
    output_data: dict        # What came out
    verification_results: list[VerificationResult]  # If verification step

@dataclass
class VerificationResult:
    verifier_name: str       # "SyntaxVerifier", "SchemaVerifier", etc.
    status: VerificationStatus  # PASSED, FAILED, SKIPPED
    message: str             # Human-readable result
    details: dict            # Additional context (errors, violations)
```

**Example audit trail output:**
```python
[
    AuditEntry(
        timestamp="2025-02-03T10:15:30",
        step="generation_attempt_1",
        input_data={"prompt": "You are a SQL query generator..."},
        output_data={}
    ),
    AuditEntry(
        timestamp="2025-02-03T10:15:31",
        step="llm_response_1",
        input_data={},
        output_data={"sql": "SELECT * FROM customers", "model": "mock-llm-v1"}
    ),
    AuditEntry(
        timestamp="2025-02-03T10:15:31",
        step="verification_1",
        input_data={"sql": "SELECT * FROM customers"},
        output_data={"passed": True},
        verification_results=[
            VerificationResult(verifier_name="SyntaxVerifier", status=PASSED, ...),
            VerificationResult(verifier_name="SchemaVerifier", status=PASSED, ...),
            VerificationResult(verifier_name="SafetyVerifier", status=PASSED, ...),
            VerificationResult(verifier_name="SemanticVerifier", status=PASSED, ...)
        ]
    )
]
```

---

## Extension Points

### Custom Verifiers

Implement the `Verifier` base class:

```python
from nl_to_sql.verifiers.base import Verifier
from nl_to_sql.models import VerificationResult, VerificationStatus

class MyCustomVerifier(Verifier):
    @property
    def name(self) -> str:
        return "MyCustomVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        # Your verification logic here
        if self._check_passes(sql):
            return VerificationResult(
                verifier_name=self.name,
                status=VerificationStatus.PASSED,
                message="Custom check passed"
            )
        return VerificationResult(
            verifier_name=self.name,
            status=VerificationStatus.FAILED,
            message="Custom check failed",
            details={"reason": "..."}
        )
```

### Custom LLM Providers

Implement the `LLMInterface` abstract class:

```python
from nl_to_sql.llm.base import LLMInterface
from nl_to_sql.models import LLMResponse

class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens
        )
```

### Custom Verification Chain

Configure which verifiers run and in what order:

```python
from nl_to_sql import NLToSQLAgent, VerificationChain, SyntaxVerifier, SafetyVerifier

# Minimal chain: only syntax and safety
chain = VerificationChain(verifiers=[
    SyntaxVerifier(),
    SafetyVerifier(),
])

agent = NLToSQLAgent(llm=my_llm, verification_chain=chain)
```

### RAG Integration

The project includes RAG components for schema-aware prompting:
- `src/nl_to_sql/rag/schema_retriever.py` - Schema retrieval
- `src/nl_to_sql/rag/rag_enhanced.py` - RAG-enhanced agent

---

## Project Status

**V0 Research Prototype** - Demonstrates verification-first architecture principles.

### In Scope (Current)
- Core agent with generation/verification loop
- Four verifiers: Syntax, Schema, Safety, Semantic
- Full audit trail with traceability
- Unit tests for verifiers and agent
- Mock LLM for testing
- FastAPI endpoints (`api/`)
- Security utilities (`security/`)
- Observability hooks (`observability/`)
- LangChain adapters (`adapters/`)

### Future Work
- Multi-version Python testing (3.10, 3.11, 3.12)
- Strict linting enforcement
- Docker builds and containerization
- Integration tests with real databases
- Production LLM provider implementations
- Performance benchmarking

---

## Key Files Reference

Quick reference for engineers getting started:

```
src/nl_to_sql/
├── __init__.py          # Public API exports
├── agent.py             # Main NLToSQLAgent class
├── models.py            # Data models (VerificationResult, AuditEntry, etc.)
├── llm/
│   ├── base.py          # LLMInterface abstract class
│   └── mock.py          # MockLLM for testing
└── verifiers/
    ├── base.py          # Verifier ABC + VerificationChain
    ├── syntax.py        # SyntaxVerifier (SQLite EXPLAIN)
    ├── schema.py        # SchemaVerifier (table/column checks)
    ├── safety.py        # SafetyVerifier (dangerous operations)
    └── semantic.py      # SemanticVerifier (intent matching)

tests/
├── conftest.py          # Pytest fixtures
└── unit/
    ├── test_verifiers.py
    └── test_agent.py

api/                     # FastAPI application
adapters/                # LangChain integration
security/                # Auth, PII detection, audit export
observability/           # Metrics, tracing, logging
```

---

## Running the Example

```bash
# Install dependencies
pip install -e .

# Run tests to verify setup
pytest tests/unit/ -v

# Try the example
python -c "
from nl_to_sql import NLToSQLAgent, MockLLM

llm = MockLLM(responses={'customers': ['SELECT * FROM customers']})
agent = NLToSQLAgent(llm=llm, max_retries=3)
result = agent.process('Show me all customers')

print(f'Success: {result.success}')
print(f'SQL: {result.sql}')
print(f'Attempts: {result.attempts}')
"
```
