# Design Decisions

This document explains the architectural choices and their rationale for the NL-to-SQL Verified Agent system. Each decision is framed in the context of mission-critical AI systems where reliability, auditability, and security are paramount.

---

## Table of Contents

1. [Core Agent Design](#core-agent-design)
2. [Verification Architecture](#verification-architecture)
3. [API Design](#api-design)
4. [Observability Strategy](#observability-strategy)
5. [Security Architecture](#security-architecture)
6. [MLOps Approach](#mlops-approach)
7. [RAG Integration](#rag-integration)
8. [Infrastructure Choices](#infrastructure-choices)
9. [Testing Strategy](#testing-strategy)

---

## Core Agent Design

### Decision 1: Explicit Agent Loop vs. Framework Abstractions

**Choice**: Implement an explicit `for` loop with clear state transitions rather than using LangChain's AgentExecutor or similar framework abstractions.

```python
for attempt in range(max_retries + 1):
    sql = llm.generate(prompt)
    passed, results = verification_chain.run(sql, context)
    if passed:
        return success_result
    prompt = build_correction_prompt(sql, error)
return failure_result
```

**Rationale**:

| Consideration | Framework Approach | Explicit Loop |
|---------------|-------------------|---------------|
| **Auditability** | Hidden state machines, callbacks | Every decision point is visible and logged |
| **Debugging** | Stack traces through framework internals | Direct code path, clear breakpoints |
| **Control** | Framework decides retry logic | Full control over each step |
| **Compliance** | Harder to explain to auditors | "Show me the code" is straightforward |
| **Portability** | Locked to framework version | Maps to any orchestration framework |

**Trade-offs Accepted**:
- More boilerplate code
- No automatic tool calling or agent planning
- Must implement features that frameworks provide "for free"

**Why This Matters for Mission-Critical Systems**:
In defense, healthcare, and financial systems, auditors and compliance officers need to understand exactly what the system does. Framework magic creates "black boxes" that are difficult to certify. An explicit loop can be reviewed line-by-line.

---

### Decision 2: Bounded Deterministic Retries

**Choice**: Hard limit of `max_retries` (default: 3) with no exponential backoff or adaptive retry logic.

```python
max_retries: int = 3  # Hard upper bound, not configurable at runtime
```

**Rationale**:

1. **Predictable Behavior**: Mission environments cannot tolerate unbounded loops. A query that would take 3 attempts should always take exactly 3 attempts (or succeed earlier).

2. **Cost Control**: Each LLM call has associated costs (API fees, latency, compute). Unbounded retries can lead to runaway costs during edge cases.

3. **Failure Escalation**: After N attempts, the system explicitly fails and returns control to a human operator. This is preferable to infinite retry loops that mask underlying issues.

4. **Testing**: Fixed retry counts make testing deterministic. We can write tests that verify behavior at attempt 1, 2, and 3.

**Alternatives Considered**:

| Approach | Problem |
|----------|---------|
| Exponential backoff | Delays are unpredictable; may exceed SLAs |
| Adaptive retries based on error type | Adds complexity; harder to reason about |
| No retries (fail fast) | Too aggressive; many errors are correctable |
| Unlimited retries | Unbounded cost and latency |

**Configuration**:
The `max_retries` is set at agent initialization, not per-request. This prevents callers from accidentally (or maliciously) requesting unlimited retries.

---

### Decision 3: Stateless Agent with Per-Request Audit Trail

**Choice**: The agent is stateless between requests. Each `process()` call resets the audit trail.

```python
def process(self, natural_language_query: str) -> AgentResult:
    self.audit_trail = []  # Reset for new query
    # ... processing ...
```

**Rationale**:

1. **Thread Safety**: Stateless design allows concurrent request processing without locks or synchronization.

2. **Horizontal Scaling**: Any instance can handle any request. No sticky sessions or state replication needed.

3. **Clear Boundaries**: Each request is independent. Audit trails are self-contained and can be exported/archived per-request.

4. **Memory Management**: No accumulating state that could lead to memory leaks in long-running services.

**Trade-offs**:
- No conversation history or multi-turn interactions
- No learning from previous requests (each request starts fresh)
- Context must be passed in each request

---

## Verification Architecture

### Decision 4: Sequential Fail-Fast Verification Chain

**Choice**: Verifiers run in a fixed sequence, stopping at the first failure.

```
Query → Syntax → Schema → Safety → Semantic → PASS
           ↓        ↓        ↓        ↓
         FAIL     FAIL     FAIL     FAIL
           ↓        ↓        ↓        ↓
        (stop)  (stop)   (stop)   (stop)
```

**Rationale**:

1. **Efficiency**: No point checking semantics if syntax is invalid. Fail-fast prevents unnecessary computation.

2. **Clear Error Messages**: The error returned is specific to the first failure point. Users don't see a confusing list of cascading errors.

3. **Logical Dependencies**: Later verifiers may depend on earlier ones passing. SchemaVerifier assumes valid SQL syntax.

4. **Predictable Latency**: Worst case is running all verifiers. Best case (failure on first) is very fast.

**Verifier Order Rationale**:

| Order | Verifier | Why This Position |
|-------|----------|-------------------|
| 1 | Syntax | Can't analyze structure if syntax is broken |
| 2 | Schema | Can't check safety of non-existent tables |
| 3 | Safety | Security before semantics |
| 4 | Semantic | Only matters if query is safe and valid |
| 5 | PII (optional) | Final compliance check |

**Alternatives Considered**:

| Approach | Problem |
|----------|---------|
| Parallel verification | Can't short-circuit; wastes resources |
| Run all, aggregate errors | Confusing error messages; cascading failures |
| Adaptive ordering | Complexity without clear benefit |

---

### Decision 5: Verifier Interface Design

**Choice**: Abstract base class with `name` property and `verify(sql, context)` method.

```python
class Verifier(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def verify(self, sql: str, context: dict) -> VerificationResult:
        pass
```

**Rationale**:

1. **Simplicity**: Two methods is the minimum viable interface. Easy to implement custom verifiers.

2. **Context Flexibility**: The `context` dict allows passing arbitrary data (schema, original query, user info) without changing the interface.

3. **Named Results**: The `name` property enables meaningful audit trails and metrics ("SafetyVerifier failed" vs "Verifier 3 failed").

4. **No Side Effects**: Verifiers are pure functions—same input always produces same output. This enables caching and testing.

**Why Not Use Decorators or Middleware Pattern?**

Decorators and middleware patterns are harder to test in isolation and create implicit ordering. The explicit chain makes dependencies visible.

---

### Decision 6: SQLite for Syntax Verification

**Choice**: Use SQLite's `EXPLAIN` command to verify SQL syntax rather than a parser library.

```python
def verify(self, sql: str, context: dict) -> VerificationResult:
    conn = sqlite3.connect(":memory:")
    self._create_schema_tables(conn, schema)
    cursor.execute(f"EXPLAIN {sql}")  # Validates syntax
```

**Rationale**:

1. **Zero Dependencies**: SQLite is built into Python. No external parser libraries needed.

2. **Real Database Validation**: SQLite's parser is battle-tested and handles edge cases that regex-based parsers miss.

3. **Schema-Aware**: By creating tables matching the schema, we catch references to non-existent tables at the syntax level.

4. **Performance**: In-memory SQLite is extremely fast for validation.

**Limitations Accepted**:
- SQLite syntax may differ from target database (PostgreSQL, MySQL)
- Some valid PostgreSQL features may fail SQLite validation
- EXPLAIN doesn't execute the query, so runtime errors aren't caught

**Mitigation**: For production, implement dialect-specific verifiers or use the target database for validation.

---

## API Design

### Decision 7: Single Versioned Endpoint

**Choice**: One primary endpoint (`/api/v1/query`) rather than multiple specialized endpoints.

```
POST /api/v1/query
{
  "query": "Show me all premium customers",
  "max_retries": 3,
  "include_audit": false
}
```

**Rationale**:

1. **Simplicity**: One endpoint is easier to document, test, and maintain.

2. **Version Prefix**: `/api/v1/` allows breaking changes in `/api/v2/` without affecting existing clients.

3. **Flexible Options**: Query parameters (`include_audit`, `max_retries`) allow customization without endpoint proliferation.

4. **RESTful**: POST is appropriate because the operation has side effects (LLM calls, audit logging).

**Alternatives Considered**:

| Approach | Problem |
|----------|---------|
| GET with query param | URL length limits; not idiomatic for complex requests |
| Multiple endpoints per feature | API sprawl; harder to version |
| GraphQL | Overkill for single-purpose API |

---

### Decision 8: Pydantic for Request/Response Validation

**Choice**: Use Pydantic models for all API schemas with strict validation.

```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    max_retries: int | None = Field(default=None, ge=0, le=10)
    include_audit: bool = Field(default=False)
```

**Rationale**:

1. **Automatic Validation**: Invalid requests are rejected before reaching business logic.

2. **Documentation**: Pydantic models generate OpenAPI schema automatically.

3. **Type Safety**: IDE autocomplete and type checking work with Pydantic models.

4. **Serialization**: Consistent JSON serialization without custom encoders.

**Validation Rules**:

| Field | Constraint | Reason |
|-------|------------|--------|
| `query` | 1-1000 chars | Prevent empty and excessively long queries |
| `max_retries` | 0-10 | Prevent infinite retries |
| `include_audit` | boolean | Explicit opt-in for verbose output |

---

### Decision 9: Request ID Propagation

**Choice**: Generate UUID for each request; propagate through logs, traces, and responses.

```python
class TelemetryMiddleware:
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response.headers["X-Request-ID"] = request_id
```

**Rationale**:

1. **Traceability**: Any log line can be correlated to a specific request.

2. **Client Correlation**: Clients can provide their own request ID for end-to-end tracing.

3. **Support**: "What's your request ID?" enables quick issue investigation.

4. **Compliance**: Audit requirements often mandate request-level traceability.

---

## Observability Strategy

### Decision 10: Three Pillars with Optional Dependencies

**Choice**: Implement metrics, tracing, and logging as optional dependencies that gracefully degrade.

```python
try:
    from prometheus_client import Counter
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
```

**Rationale**:

1. **Zero-Dependency Core**: The core agent works without any observability libraries.

2. **Gradual Adoption**: Teams can add observability incrementally.

3. **Testing Simplicity**: Unit tests don't need observability infrastructure.

4. **Production Flexibility**: Some deployments may use different observability stacks.

**Graceful Degradation**:

| Component | If Unavailable |
|-----------|----------------|
| Prometheus | Metrics silently not collected |
| OpenTelemetry | Tracing disabled; no spans created |
| Structlog | Falls back to standard Python logging |

---

### Decision 11: Semantic Metric Names

**Choice**: Use domain-specific metric names rather than generic HTTP metrics.

```python
QUERIES_TOTAL = Counter(
    "nl_to_sql_queries_total",
    "Total number of NL-to-SQL queries processed",
    ["status"],
)

VERIFICATION_FAILURES = Counter(
    "nl_to_sql_verification_failures_total",
    "Total verification failures by verifier",
    ["verifier"],
)
```

**Rationale**:

1. **Business Meaning**: "nl_to_sql_queries_total" is more meaningful than "http_requests_total" for dashboards and alerts.

2. **Granular Labels**: The `verifier` label enables alerts like "SyntaxVerifier failures > 10% in 5 minutes".

3. **SLI/SLO Alignment**: Metrics directly map to service level indicators.

**Key Metrics Chosen**:

| Metric | Type | Why |
|--------|------|-----|
| `queries_total` | Counter | Track throughput and success rate |
| `query_duration_seconds` | Histogram | Latency SLOs |
| `query_attempts` | Histogram | Detect queries requiring many retries |
| `verification_failures_total` | Counter | Identify problematic verifiers |

---

## Security Architecture

### Decision 12: API Key Authentication

**Choice**: Simple API key in header (`X-API-Key`) rather than OAuth2/JWT.

```python
api_key = request.headers.get("X-API-Key")
if not validate_key(api_key):
    raise HTTPException(status_code=401)
```

**Rationale**:

1. **Simplicity**: API keys are easy to generate, distribute, and revoke.

2. **Stateless**: No token refresh flows or session management.

3. **Appropriate Scope**: This is a single-purpose API, not a user-facing application.

4. **Easy Integration**: Clients just add a header; no OAuth dance.

**Security Measures**:

| Measure | Implementation |
|---------|----------------|
| Key hashing | Keys stored as SHA-256 hashes |
| Rate limiting | Per-key request limits |
| Expiration | Optional TTL per key |
| Scopes | Keys can be limited to specific operations |

**When to Use OAuth2 Instead**:
- Multi-tenant SaaS with user delegation
- Integration with identity providers
- Fine-grained user permissions

---

### Decision 13: PII Detection as Verifier

**Choice**: Implement PII detection as a verifier in the chain rather than a separate preprocessing step.

```python
class PIIVerifier(Verifier):
    def verify(self, sql: str, context: dict) -> VerificationResult:
        if self.detector.detect(sql):
            return VerificationResult(status=FAILED, message="PII detected")
```

**Rationale**:

1. **Consistency**: PII check follows the same pattern as other verifications.

2. **Ordering Control**: PII check runs after safety check (security before compliance).

3. **Audit Trail**: PII detection results are logged with other verification results.

4. **Configurability**: Can be added/removed from chain based on deployment requirements.

**PII Detection Strategy**:

| Type | Detection Method | Confidence |
|------|------------------|------------|
| Email | Regex pattern | High |
| Phone | Regex pattern | Medium |
| SSN | Regex pattern | High |
| Sensitive columns | Column name matching | High |
| Names | (Not implemented) | Would need NER |

---

### Decision 14: NIST-Compliant Audit Records

**Choice**: Structure audit records according to NIST SP 800-53 AU-3 requirements.

```python
@dataclass
class NISTAuditRecord:
    event_id: str           # AU-3(1)
    event_type: str         # AU-3
    timestamp: str          # AU-8
    source_component: str   # AU-3
    outcome: str            # AU-3
    subject_id: str         # AU-3
    record_hash: str        # AU-9 (integrity)
    previous_hash: str      # AU-10 (non-repudiation)
```

**Rationale**:

1. **Compliance Ready**: Records meet federal audit requirements out of the box.

2. **Integrity Verification**: Hash chain allows detection of tampering.

3. **Forensics**: Records contain all information needed for incident investigation.

4. **Export Formats**: JSON and JSONL support common SIEM ingestion.

**NIST Mapping**:

| NIST Control | Implementation |
|--------------|----------------|
| AU-3 (Content) | All required fields present |
| AU-8 (Timestamps) | ISO 8601 format |
| AU-9 (Protection) | SHA-256 hash per record |
| AU-10 (Non-repudiation) | Hash chain linking |

---

## MLOps Approach

### Decision 15: MLflow for Experiment Tracking

**Choice**: Use MLflow as the experiment tracking backend with optional availability.

```python
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
```

**Rationale**:

1. **Industry Standard**: MLflow is widely adopted and well-documented.

2. **Self-Hosted Option**: Can run locally or on-premises for sensitive environments.

3. **Comprehensive**: Tracks parameters, metrics, and artifacts in one place.

4. **Integration**: Works with existing ML tools and frameworks.

**What We Track**:

| Category | Examples |
|----------|----------|
| Parameters | `max_retries`, `llm_model`, `verifier_count` |
| Metrics | `success_rate`, `avg_attempts`, `processing_time` |
| Artifacts | Test datasets, benchmark reports |

---

### Decision 16: Versioned Test Datasets

**Choice**: Store test cases in versioned JSON files with metadata.

```json
{
  "version": "1.0.0",
  "schema_version": "sample_schema_v1",
  "test_cases": [
    {
      "id": "basic_001",
      "category": "basic_select",
      "difficulty": "easy",
      "query": "Show me all customers"
    }
  ]
}
```

**Rationale**:

1. **Reproducibility**: Same version always produces same test set.

2. **Comparison**: Can compare agent performance across dataset versions.

3. **Categorization**: Categories enable focused testing (e.g., only aggregation queries).

4. **Difficulty Levels**: Can measure performance by complexity.

**Dataset Design**:

| Category | Count | Purpose |
|----------|-------|---------|
| basic_select | 2 | Baseline functionality |
| filtering | 2 | WHERE clause handling |
| aggregation | 3 | COUNT, SUM, AVG |
| grouping | 2 | GROUP BY |
| ordering | 2 | ORDER BY, LIMIT |
| joins | 2 | Multi-table queries |
| complex | 2 | Combined operations |
| safety | 1 | Safety verifier test |
| semantic | 1 | Semantic verifier test |

---

## RAG Integration

### Decision 17: Schema-Aware Retrieval

**Choice**: Index schema information (tables, columns, types) for semantic retrieval.

```python
class SchemaRetriever:
    def index_schema(self, schema: dict):
        for table_name, table_info in schema.items():
            # Create table chunk
            self.chunks.append(SchemaChunk(
                content=f"Table '{table_name}' with columns: {columns}",
                chunk_type="table"
            ))
            # Create column chunks
            for col in columns:
                self.chunks.append(SchemaChunk(
                    content=f"Column '{col}' in table '{table_name}'",
                    chunk_type="column"
                ))
```

**Rationale**:

1. **Focused Context**: Only relevant schema information is included in prompts.

2. **Token Efficiency**: Large schemas don't consume entire context window.

3. **Semantic Matching**: "total spending" retrieves "amount" column.

4. **Scalability**: Works with schemas of any size.

**Chunking Strategy**:

| Chunk Type | Content | Why Separate |
|------------|---------|--------------|
| Table | Table name + all columns | High-level overview |
| Column | Column name + type + hints | Granular matching |

---

### Decision 18: Graceful Degradation for Embeddings

**Choice**: Fall back to keyword matching if sentence-transformers is unavailable.

```python
if EMBEDDINGS_AVAILABLE:
    return self._retrieve_with_embeddings(query, top_k)
else:
    return self._retrieve_with_keywords(query, top_k)
```

**Rationale**:

1. **No Hard Dependencies**: Core functionality works without ML libraries.

2. **Testing**: Tests can run without downloading embedding models.

3. **Deployment Flexibility**: Some environments may not support sentence-transformers.

4. **Progressive Enhancement**: Add embeddings for better retrieval when available.

---

## Infrastructure Choices

### Decision 19: Multi-Stage Docker Build

**Choice**: Use multi-stage Dockerfile with separate builder and production stages.

```dockerfile
FROM python:3.11-slim as builder
# Install build dependencies, create venv

FROM python:3.11-slim as production
COPY --from=builder /opt/venv /opt/venv
# Copy only runtime files
```

**Rationale**:

1. **Smaller Images**: Production image doesn't include build tools.

2. **Security**: Fewer packages = smaller attack surface.

3. **Layer Caching**: Build dependencies cached separately from application code.

4. **Reproducibility**: Pinned base image version.

**Image Size Comparison**:

| Stage | Approximate Size |
|-------|------------------|
| Builder | ~800MB |
| Production | ~200MB |

---

### Decision 20: Non-Root Container User

**Choice**: Run container as non-root user `appuser` (UID 1000).

```dockerfile
RUN useradd --create-home appuser
USER appuser
```

**Rationale**:

1. **Security Best Practice**: Container escape attacks are limited to non-root user.

2. **Kubernetes Compliance**: Many clusters enforce non-root via PodSecurityPolicy.

3. **Principle of Least Privilege**: Application doesn't need root.

**Kubernetes Security Context**:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
```

---

### Decision 21: Kubernetes Resource Limits

**Choice**: Set explicit CPU and memory requests/limits.

```yaml
resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"
```

**Rationale**:

1. **Scheduling**: Kubernetes can make informed scheduling decisions.

2. **Protection**: Limits prevent runaway processes from affecting other pods.

3. **Cost Control**: Resource limits enable capacity planning.

4. **HPA Compatibility**: HPA needs resource metrics to scale.

**Sizing Rationale**:

| Resource | Request | Limit | Reasoning |
|----------|---------|-------|-----------|
| CPU | 100m | 500m | Python is CPU-light; bursts for LLM calls |
| Memory | 256Mi | 512Mi | FastAPI + observability libraries |

---

## Testing Strategy

### Decision 22: Fixture-Based Test Architecture

**Choice**: Use pytest fixtures for all reusable test components.

```python
@pytest.fixture
def syntax_verifier() -> SyntaxVerifier:
    return SyntaxVerifier()

@pytest.fixture
def agent_simple(mock_llm_simple) -> NLToSQLAgent:
    return NLToSQLAgent(llm=mock_llm_simple, max_retries=2)
```

**Rationale**:

1. **DRY**: Common setup code is defined once.

2. **Isolation**: Each test gets fresh instances.

3. **Composability**: Fixtures can depend on other fixtures.

4. **Readability**: Test functions focus on assertions, not setup.

**Fixture Hierarchy**:
```
sample_schema
    └── verification_context
            └── syntax_verifier, schema_verifier, ...
                    └── verification_chain
                            └── agent_simple, agent_with_correction
```

---

### Decision 23: Mock LLM for Testing

**Choice**: Use `MockLLM` with configurable response sequences for testing.

```python
llm = MockLLM(responses={
    "customers": ["SELECT * FROM customers"],
    "revenue": [
        "SELEC SUM(amount)",  # First attempt: typo
        "SELECT SUM(amount) FROM orders"  # Corrected
    ]
})
```

**Rationale**:

1. **Deterministic**: Tests produce same results every run.

2. **Fast**: No network calls or API latency.

3. **Free**: No API costs for running tests.

4. **Scenario Testing**: Can simulate specific failure/correction sequences.

**Mock Behavior**:
- Matches query keywords to response lists
- Returns successive responses on repeated calls (simulates correction)
- Falls back to default error response for unmatched queries

---

## Summary

These design decisions prioritize:

1. **Auditability** over convenience
2. **Predictability** over flexibility
3. **Security** over ease of use
4. **Simplicity** over feature richness
5. **Explicit** over implicit behavior

For mission-critical AI systems, these trade-offs are appropriate. The system is designed to be understood, debugged, and certified by humans who may not be ML experts.
