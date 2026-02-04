# Architecture Guide

Technical architecture of the NL-to-SQL Verified agent system.

> **For detailed rationale behind each architectural decision, see [Design Decisions](./design-decisions.md).**

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  REST API (FastAPI)  │  Python SDK  │  LangChain Adapters                   │
└───────────────────────────────────────────────────────────────────────────────
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API Gateway Layer                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Authentication │ Rate Limiting │ Telemetry │ Request Routing               │
└───────────────────────────────────────────────────────────────────────────────
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Agent Core Layer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                         NLToSQLAgent Controller                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────┐    │
│  │   LLM       │───▶│   Verification  │───▶│   Correction Loop        │    │
│  │  Interface  │    │     Chain       │    │   (bounded retries)      │    │
│  └─────────────┘    └─────────────────┘    └──────────────────────────┘    │
│        │                    │                         │                     │
│        ▼                    ▼                         ▼                     │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────┐    │
│  │   RAG       │    │    Verifiers    │    │     Audit Logger         │    │
│  │  Retriever  │    │  (4 built-in)   │    │  (full traceability)     │    │
│  └─────────────┘    └─────────────────┘    └──────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────────
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Observability Layer                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│   Prometheus Metrics  │  OpenTelemetry Tracing  │  Structured Logging       │
└───────────────────────────────────────────────────────────────────────────────
```

## Core Components

### 1. NLToSQLAgent Controller

The central orchestrator that manages the NL-to-SQL conversion process.

```python
class NLToSQLAgent:
    def process(self, query: str) -> AgentResult:
        for attempt in range(max_retries + 1):
            sql = llm.generate(prompt)
            passed, results = verification_chain.run(sql)
            if passed:
                return success_result
            prompt = correction_prompt(sql, error)
        return failure_result
```

**Key Design Decisions:**
- Explicit loop (not framework magic) for auditability
- Bounded retries for predictable behavior
- Full audit trail for compliance

### 2. Verification Chain

Sequential verification with fail-fast semantics.

| Verifier | Purpose | Method |
|----------|---------|--------|
| SyntaxVerifier | Valid SQL syntax | SQLite EXPLAIN |
| SchemaVerifier | Tables/columns exist | Regex + lookup |
| SafetyVerifier | No destructive ops | Pattern matching |
| SemanticVerifier | Matches intent | Heuristics |
| PIIVerifier | No PII exposure | Pattern + column check |

```
Query → Syntax → Schema → Safety → Semantic → PII → PASS/FAIL
         ↓        ↓        ↓         ↓        ↓
        FAIL     FAIL     FAIL      FAIL     FAIL
         ↓        ↓        ↓         ↓        ↓
       (stop)  (stop)   (stop)    (stop)   (stop)
```

### 3. LLM Interface

Pluggable interface for different LLM providers.

```python
class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        pass
```

**Implementations:**
- `MockLLM` - Testing and demonstration
- `RAGEnhancedLLM` - Schema-aware generation
- (External) Claude, OpenAI, local models

### 4. RAG System

Schema-aware retrieval for improved generation.

```
Query: "Show total spending by customer"
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Schema Retriever              │
│  1. Embed query                         │
│  2. Find similar schema chunks          │
│  3. Return relevant tables/columns      │
└─────────────────────────────────────────┘
                    │
                    ▼
Context: [customers.name, orders.amount, orders.customer_id]
                    │
                    ▼
Enhanced Prompt → LLM → Better SQL
```

## Data Flow

### Request Flow

```
1. Client Request
   POST /api/v1/query {"query": "Show premium customers"}

2. Authentication
   X-API-Key validation → Rate limit check

3. Agent Processing
   NLToSQLAgent.process(query)

4. Generation Loop
   LLM.generate() → Verification → [Correction] → Result

5. Response
   {success: true, sql: "SELECT...", attempts: 1}
```

### Verification Flow

```
SQL: "SELECT name, email FROM customers WHERE tier = 'premium'"

1. SyntaxVerifier
   - EXPLAIN query in SQLite → PASS

2. SchemaVerifier
   - Check: customers table exists? → YES
   - Check: name, email columns exist? → YES
   - → PASS

3. SafetyVerifier
   - Check: No DROP/TRUNCATE/DELETE? → YES
   - → PASS

4. SemanticVerifier
   - Query mentions "premium" → WHERE clause present → PASS

Result: All verifiers passed ✓
```

## Module Structure

```
nl-to-sql-verified/
├── src/nl_to_sql/           # Core agent library
│   ├── agent.py             # Main controller
│   ├── models.py            # Data structures
│   ├── verifiers/           # Verification chain
│   │   ├── base.py          # Verifier interface
│   │   ├── syntax.py        # SQL syntax check
│   │   ├── schema.py        # Schema validation
│   │   ├── safety.py        # Dangerous ops check
│   │   └── semantic.py      # Intent matching
│   ├── llm/                 # LLM interfaces
│   │   ├── base.py          # Abstract interface
│   │   └── mock.py          # Testing mock
│   └── rag/                 # RAG components
│       ├── schema_retriever.py
│       └── rag_enhanced.py
│
├── api/                     # REST API
│   ├── main.py              # FastAPI app
│   ├── routes/              # Endpoints
│   ├── middleware/          # Auth, telemetry
│   └── schemas.py           # Pydantic models
│
├── observability/           # Monitoring
│   ├── metrics.py           # Prometheus
│   ├── tracing.py           # OpenTelemetry
│   └── logging_config.py    # Structured logs
│
├── security/                # Security
│   ├── auth.py              # API authentication
│   ├── pii_detector.py      # PII detection
│   └── audit_export.py      # NIST compliance
│
├── mlops/                   # MLOps
│   ├── tracking.py          # MLflow integration
│   └── eval/                # Benchmarking
│
└── infra/                   # Infrastructure
    ├── docker/              # Containerization
    └── k8s/                 # Kubernetes
```

## Deployment Architecture

### Docker Compose (Development)

```
┌──────────────────────────────────────────────────────────────┐
│                    Docker Network                             │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌────────────┐  ┌────────────┐             │
│  │   API       │  │ Prometheus │  │  Grafana   │             │
│  │  :8000      │  │   :9090    │  │   :3000    │             │
│  └──────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
│         │               │               │                     │
│         └───────────────┴───────────────┘                     │
│                         │                                     │
│                   ┌─────┴──────┐                              │
│                   │   Jaeger   │                              │
│                   │   :16686   │                              │
│                   └────────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

### Kubernetes (Production)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                            │
├─────────────────────────────────────────────────────────────────┤
│  Namespace: nl-to-sql                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Deployment: nl-to-sql-api (2+ replicas)                    ││
│  │  ┌─────────┐  ┌─────────┐                                   ││
│  │  │ Pod 1   │  │ Pod 2   │  ... (HPA: 2-10 replicas)        ││
│  │  └────┬────┘  └────┬────┘                                   ││
│  │       └──────┬─────┘                                        ││
│  │              ▼                                              ││
│  │       ┌────────────┐                                        ││
│  │       │  Service   │ (ClusterIP)                            ││
│  │       └──────┬─────┘                                        ││
│  └──────────────┼──────────────────────────────────────────────┘│
│                 ▼                                                │
│          ┌────────────┐                                          │
│          │  Ingress   │ (nginx)                                  │
│          └──────┬─────┘                                          │
└─────────────────┼────────────────────────────────────────────────┘
                  ▼
            External Traffic
```

## Security Architecture

### Authentication Flow

```
Request
   │
   ▼
┌──────────────────┐
│ Extract API Key  │
│ (X-API-Key)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────┐
│ Validate Key     │────▶│ Rate Limit   │
│ (hash lookup)    │     │ Check        │
└────────┬─────────┘     └──────┬───────┘
         │                      │
         ▼                      ▼
    [Invalid]              [Exceeded]
        │                      │
        ▼                      ▼
    401 Error              429 Error
```

### Data Protection

```
Input Query
     │
     ▼
┌─────────────────┐
│ PII Detector    │ → Log: [redacted]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PIIVerifier     │ → Block sensitive column access
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audit Export    │ → NIST-compliant records
└─────────────────┘
```

## Performance Characteristics

### Latency Budget (Target: <500ms)

| Component | Target | Notes |
|-----------|--------|-------|
| API overhead | <10ms | Routing, auth |
| RAG retrieval | <50ms | Schema lookup |
| LLM generation | <300ms | Mock: <1ms |
| Verification | <50ms | All verifiers |
| Response | <10ms | Serialization |

### Scaling

- Horizontal: Add more API pods (stateless)
- LLM: Rate limit aware (configurable)
- Metrics: Prometheus scraping (15s interval)

## Extension Points

### Custom Verifier

```python
class BusinessRuleVerifier(Verifier):
    @property
    def name(self) -> str:
        return "BusinessRuleVerifier"

    def verify(self, sql: str, context: dict) -> VerificationResult:
        # Your custom logic
        pass
```

### Custom LLM Provider

```python
class MyLLM(LLMInterface):
    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        # Your LLM integration
        pass
```

### Custom Metrics

```python
MY_METRIC = Counter(
    "my_custom_metric",
    "Description",
    ["label"],
    registry=REGISTRY,
)
```
