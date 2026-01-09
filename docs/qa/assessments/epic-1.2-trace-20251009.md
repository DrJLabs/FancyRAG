# Requirements Traceability Matrix

## Story: 1.2 - Hybrid MCP Server

### Coverage Summary

- Total Requirements: 3
- Fully Covered: 3 (100%)
- Partially Covered: 0 (0%)
- Not Covered: 0 (0%)

### Requirement Mappings

#### AC1: Server exposes `search` (hybrid) and `fetch` tools using a shared Neo4j driver.

**Coverage: FULL**

- `tests/servers/test_runtime.py::test_search_sync_returns_scores`
  - Given: A stubbed `HybridCypherRetriever` and Neo4j driver return deterministic hybrid records and scores.
  - When: `search_sync` executes with `top_k=2` and ratio 1.
  - Then: Each result contains text, metadata without embeddings, and normalized vector/full-text scores.
- `tests/servers/test_runtime.py::test_fetch_sync_found`
  - Given: A stubbed driver maps the fetch query to a hydrated node.
  - When: `fetch_sync` runs with a known element id.
  - Then: The response flags the node as found and surfaces metadata with labels and element id.
- `tests/servers/test_runtime.py::test_fetch_sync_not_found`
  - Given: The driver returns no rows for the fetch query.
  - When: `fetch_sync` executes.
  - Then: The response indicates the node is not found and echoes the element id.
- `tests/servers/test_runtime.py::test_build_server_registers_tools_with_shared_state`
  - Given: `build_server` wires a FastMCP instance with a shared driver and retriever stubs.
  - When: The test invokes `search` and `fetch` via the registered tools.
  - Then: Both tools run successfully, reuse the stub driver, and produce normalized scores and metadata.

#### AC2: Query embeddings use an environment-configured OpenAI-compatible service.

**Coverage: FULL**

- `tests/test_config.py::test_load_config_success`
  - Given: Required environment variables point to a temporary Cypher file.
  - When: `load_config` runs without overrides.
  - Then: The resulting config carries the embedding base URL, API key, timeout defaults, and OAuth scopes.
- `tests/test_config.py::test_missing_variable_raises`
  - Given: `NEO4J_PASSWORD` is removed after seeding the environment.
  - When: `load_config` executes.
  - Then: A `ConfigurationError` is raised, proving fail-fast validation.
- `tests/test_embeddings.py::test_retrying_embeddings_respects_configuration`
  - Given: A stub OpenAI client tracks initialization parameters and transient failures.
  - When: `RetryingOpenAIEmbeddings` requests embeddings with retry/backoff.
  - Then: The configured base URL, API key, timeout, and retries are honored and telemetry captured.
- `tests/test_embeddings.py::test_retrying_embeddings_raises_after_max_attempts`
  - Given: The stub client keeps failing beyond the retry budget.
  - When: `RetryingOpenAIEmbeddings` exhausts retries.
  - Then: The last error surfaces, proving fail-fast behavior on persistent outages.

#### AC3: Retrieval query returns text, metadata, vector and full-text scores.

**Coverage: FULL**

- `tests/servers/test_runtime.py::test_search_sync_returns_scores`
  - Given: Stubbed vector and full-text score queries return deterministic values.
  - When: `search_sync` composes results from the hybrid retriever.
  - Then: Each item exposes `text`, metadata, `score_vector`, and `score_fulltext`, and strips the raw embedding from metadata.

### Critical Gaps

No critical gaps remain. All acceptance criteria now have deterministic unit coverage. Retain performance smoke as a longer-term improvement outside this story's scope.

### Test Design Recommendations

1. Expand coverage to include unauthorized FastMCP requests once stateless HTTP harness is available.
2. Add an integration smoke test (possibly with Testcontainers Neo4j) that asserts returned payloads include both score types under real Cypher execution.
3. Track documentation-sensitive requirements by verifying `.env.example` and README updates via snapshot or schema tests to guard future regressions.

### Risk Assessment

- **High Risk**: None identified — hybrid scoring is covered by deterministic tests.
- **Medium Risk**: OAuth negative-path coverage pending.
- **Low Risk**: Configuration loading and fetch semantics.
