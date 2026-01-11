# Semantic Retrieval Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend hybrid retrieval to surface semantic graph context in MCP search responses when semantic enrichment is enabled.

**Architecture:** Keep SimpleKGPipeline as the ingestion path, but enhance retrieval by expanding the Cypher query to fetch semantic nodes/relationships linked to chunks (via `chunk_uid`/`chunk_uids`) and pass those fields through the MCP response. No worktrees (project rule); work on a normal branch.

**Tech Stack:** Python 3.12, Neo4j, neo4j-graphrag, FastMCP, pytest.

## Task 1: Pass through semantic fields in MCP search responses

**Files:**
- Modify: `src/fancryrag/mcp/runtime.py` (search response mapping)
- Test: `tests/servers/test_runtime.py`

### Step 1: Write the failing test

Add a test case to `tests/servers/test_runtime.py` that supplies a retriever record with `semantic_nodes` and `semantic_relationships` fields and asserts they appear in the response items.

```python
# in tests/servers/test_runtime.py

def test_search_sync_includes_semantic_fields(base_config):
    records = [
        {
            "node": FakeNode("1", text="Doc 1", embedding=[0.1]),
            "score": 0.9,
            "text": "Doc 1",
            "semantic_nodes": [{"id": "n1"}],
            "semantic_relationships": [{"type": "RELATED_TO"}],
        }
    ]
    metadata = {"query_vector": [0.11, 0.22]}
    driver = StubDriver({
        runtime.VECTOR_SCORE_QUERY: ([{"element_id": "1", "score": 0.5}], None, None),
        runtime.FULLTEXT_SCORE_QUERY: ([{"element_id": "1", "score": 2.0}], None, None),
    })
    state = _state_with(driver, FakeRetriever(records, metadata), base_config)

    response = runtime.search_sync(state, "graph", top_k=1, effective_ratio=1)

    item = response["results"][0]
    assert item["semantic_nodes"] == [{"id": "n1"}]
    assert item["semantic_relationships"] == [{"type": "RELATED_TO"}]
```

### Step 2: Run test to verify it fails

Run: `pytest tests/servers/test_runtime.py::test_search_sync_includes_semantic_fields -v`
Expected: FAIL (missing keys in response).

### Step 3: Write minimal implementation

Update `src/fancryrag/mcp/runtime.py` `search_sync` to include optional fields when present:

```python
semantic_nodes = record.get("semantic_nodes")
semantic_relationships = record.get("semantic_relationships")
if semantic_nodes is not None:
    item["semantic_nodes"] = semantic_nodes
if semantic_relationships is not None:
    item["semantic_relationships"] = semantic_relationships
```

### Step 4: Run test to verify it passes

Run: `pytest tests/servers/test_runtime.py::test_search_sync_includes_semantic_fields -v`
Expected: PASS.

### Step 5: Commit

```bash
git add tests/servers/test_runtime.py src/fancryrag/mcp/runtime.py
git commit -m "feat: include semantic fields in search responses"
```

## Task 2: Enhance hybrid retrieval query to include semantic context

**Files:**
- Modify: `queries/hybrid_retrieval.cypher`
- Test: `tests/test_queries.py` (new)
- Docs: `README.md` (add semantic retrieval note)

### Step 1: Write the failing test

Create `tests/test_queries.py` with a check that the hybrid query exposes semantic fields:

```python
from pathlib import Path


def test_hybrid_query_exposes_semantic_fields():
    query = Path("queries/hybrid_retrieval.cypher").read_text(encoding="utf-8")
    assert "semantic_nodes" in query
    assert "semantic_relationships" in query
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_queries.py::test_hybrid_query_exposes_semantic_fields -v`
Expected: FAIL (fields absent).

### Step 3: Write minimal implementation

Update `queries/hybrid_retrieval.cypher` to collect semantic nodes/relationships tied to the chunk `uid` and return them with the existing `node` and `score` fields. Keep `RETURN node, score` intact and add two new fields:
- `semantic_nodes`: projected node metadata (labels + core props)
- `semantic_relationships`: projected relationship metadata (type + endpoints)

Example shape (final query should be valid Cypher):

```cypher
WITH node, score
OPTIONAL MATCH (sem)
WHERE sem.semantic_source = 'kg_build.semantic_enrichment.v1'
  AND sem.chunk_uid = node.uid
WITH node, score, collect(sem) AS sems
CALL {
  WITH sems
  UNWIND sems AS s
  OPTIONAL MATCH (s)-[rel]->(other)
  WHERE rel.semantic_source = 'kg_build.semantic_enrichment.v1'
  RETURN collect(DISTINCT {
    type: type(rel),
    from: elementId(s),
    to: elementId(other),
    properties: properties(rel)
  }) AS semantic_relationships
}
RETURN
  node,
  score,
  [s IN sems | {id: elementId(s), labels: labels(s), properties: properties(s)}] AS semantic_nodes,
  semantic_relationships
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_queries.py::test_hybrid_query_exposes_semantic_fields -v`
Expected: PASS.

### Step 5: Update docs

Add a short note to `README.md` under MCP search output to mention the optional `semantic_nodes` and `semantic_relationships` fields (populated only when `--enable-semantic` ingestion is used).

### Step 6: Commit

```bash
git add queries/hybrid_retrieval.cypher tests/test_queries.py README.md
git commit -m "feat: add semantic context to hybrid retrieval"
```

## Task 3: Verify end-to-end local workflow (manual)

**Files:**
- No code changes

### Step 1: Run ingestion with semantic extraction enabled

Run: `uv run python scripts/kg_build.py --source-dir docs --profile markdown --enable-semantic`
Expected: Successful run with semantic section in output log.

### Step 2: Query MCP search and confirm semantic fields

Run (example):
```bash
curl -H "Authorization: Bearer $MCP_STATIC_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"example", "top_k":3}' \
  http://localhost:8080/mcp/search
```
Expected: Each result includes `semantic_nodes`/`semantic_relationships` when semantic data exists.

### Step 3: Note results in `TESTING_SUMMARY.md`

Append a short note describing the manual verification.

### Step 4: Commit

```bash
git add TESTING_SUMMARY.md
git commit -m "docs: record semantic retrieval verification"
```

---

Plan complete and saved to `docs/completed/plans/2026-01-09-semantic-retrieval-enhancement.md`.
