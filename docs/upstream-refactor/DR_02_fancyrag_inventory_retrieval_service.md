---
title: "FancyRAG Inventory — Retrieval Runtime + MCP Server"
doc_id: "DR_02_fancyrag_inventory_retrieval_service"
generated_on: "2026-01-11"
repos:
  upstream: "https://github.com/neo4j/neo4j-graphrag-python.git"
  custom: "https://github.com/DrJLabs/FancyRAG.git"
repo_shas:
  upstream_default_branch_head: "UNVERIFIED (not resolved in this chat)"
  custom_default_branch_head: "UNVERIFIED (not resolved in this chat)"
evidence_note: "Citations/permalinks are preserved exactly as they appeared in the originating Deep Research answer."
---

# FancyRAG Inventory — Retrieval Runtime + MCP Server

## 1) Server/MCP Entrypoints

### Entrypoint Functions & Initialization

The main server process is launched via `servers/mcp_hybrid_google.py`. Its `main()` function configures logging, loads environment config, initializes the Neo4j connection and retriever state, and then starts the FastMCP server【13†L19-L27】【13†L84-L92】. On startup, it loads `.env.local` (if present) for secrets【13†L19-L27】 and builds an `AppConfig` via `load_config()` (which validates required env vars or raises `ConfigurationError`)【13†L23-L30】. Using this config, it creates a `ServerState` object via `create_state()`, which opens the Neo4j driver and constructs the `HybridCypherRetriever` with the configured vector index, fulltext index, retrieval Cypher template, and embedder【6†L73-L82】【6†L89-L97】. The `ServerState` holds the loaded config, Neo4j `Driver`, and the `retriever` instance【6†L41-L49】 for use by the request handlers.

### Tool Registration

After initializing state, the server is built by `build_server(state)` in `fancryrag.mcp.runtime`. This creates a FastMCP app and registers the retrieval tools. In `build_server`, two asynchronous tool endpoints are defined using decorators: `@server.tool async def search(...)` and `@server.tool async def fetch(...)`【31†L284-L293】【31†L297-L303】. The `search` tool takes parameters `query: str`, `top_k: int = 5`, and `effective_search_ratio: int = 1`【31†L284-L292】, and it executes a hybrid search by calling the synchronous `search_sync` function in a background thread【31†L289-L295】. The `fetch` tool takes an `element_id: str` and similarly offloads to the `fetch_sync` function for database retrieval【31†L297-L303】. These decorated functions automatically become HTTP POST endpoints (`/mcp/search` and `/mcp/fetch`) exposed by FastMCP. Input validation is enforced in-code (e.g. `top_k` > 0)【31†L288-L292】【31†L298-L302】 before performing the operations.

### Health Check Route

The server defines a custom health endpoint for container orchestration. In `main()`, after building the server, a `@server.custom_route("/mcp/health")` is added for GET requests【13†L54-L62】. This handler asynchronously verifies Neo4j connectivity (`driver.verify_connectivity()`) in a thread executor and returns `{"status":"ok"}` if the database is reachable, or a 503 JSON response if not【13†L56-L64】【13†L65-L73】. This health check is used by Docker (and `make up`) to ensure the MCP service is ready【1†L109-L117】.

### Server Startup

Finally, `main()` registers an `atexit` hook to close the Neo4j driver on shutdown【13†L71-L74】 and invokes `server.run(...)` to launch the FastMCP server HTTP listener【13†L75-L83】【13†L84-L92】. The server runs on the host/port from config (default `0.0.0.0:8080`) under the base path `/mcp`【13†L75-L83】. Notably, it uses `stateless_http=True`【13†L84-L90】, meaning the OAuth flow will not rely on server-side sessions (appropriate for integration with ChatGPT). At this point the MCP server is live, exposing the `/mcp/search`, `/mcp/fetch`, and `/mcp/health` endpoints (with OAuth enforcement if enabled).

### Server/MCP Entrypoints Table

| Entrypoint / Component | Description & Code Reference |
|---|---|
| `mcp_hybrid_google.py main()` | Loads logging and env config (`.env.local`), initializes `ServerState` (Neo4j driver + retriever), sets up optional auth (static token or Google OAuth), and starts the FastMCP HTTP server【13†L19-L27】【13†L75-L83】. If config is invalid or initialization fails, it logs an error and exits non-zero【13†L25-L33】【13†L31-L39】. |
| `ServerState` / `create_state` | Establishes runtime state: opens Neo4j driver to `config.neo4j.uri` (with credentials) and verifies connectivity【6†L73-L82】, instantiates the `RetryingOpenAIEmbeddings` and `HybridCypherRetriever` with the configured index names, retrieval Cypher, and embedder【6†L81-L89】【6†L89-L97】. Returns a dataclass holding `config`, `driver`, and `retriever`【6†L41-L49】【6†L99-L107】. |
| `build_server(state)` | Constructs the FastMCP app. If auth is required, attaches an OAuth provider (Google or static token)【31†L273-L282】. Registers tool routes: `search` (POST `/mcp/search`) and `fetch` (POST `/mcp/fetch`), by decorating inner async functions【31†L284-L293】【31†L297-L303】. Returns the configured `FastMCP` server instance【31†L281-L289】【31†L297-L304】. |
| `/mcp/search` tool | `search(query:str, top_k:int=5, effective_search_ratio:int=1)` – Executes hybrid retrieval over Neo4j【31†L284-L292】. Validates inputs【31†L288-L292】, then calls `search_sync` in a thread【31†L289-L295】. |
| `/mcp/fetch` tool | `fetch(element_id:str)` – Fetches a node by its Neo4j element ID【31†L297-L303】. Validates input【31†L298-L302】, then calls `fetch_sync` in a thread. |
| `/mcp/health` route | GET route that runs Neo4j connectivity probe and returns OK/503 accordingly【13†L54-L62】【13†L56-L64】. |

Runtime startup: Docker `fancryrag-mcp` uses `python servers/mcp_hybrid_google.py` as the entrypoint【10†L34-L37】, invoking `main()`. It initializes and calls `server.run()`【13†L75-L83】【13†L84-L92】. It logs a `"server.starting"` event with host/port/path【13†L75-L83】 and closes the driver via `atexit` on shutdown【13†L71-L74】.

---

## 2) MCP Contract Spec (Agent-readable)

### Tool: `search` (POST `/mcp/search`)

**Parameters**

| Name | Type | Required | Default | Notes |
|---|---|---:|---:|---|
| `query` | string | Yes | — | Natural language query text. |
| `top_k` | integer | No | 5 | Must be `> 0`【31†L288-L292】. |
| `effective_search_ratio` | integer | No | 1 | Must be `> 0`【31†L288-L292】. Multiplies candidate pool size. |

**Response Schema**

Top-level response:

| Field | Type | Description |
|---|---|---|
| `query` | string | Echoes query text【31†L223-L228】. |
| `top_k` | integer | Echoes requested `top_k`【31†L223-L228】. |
| `effective_search_ratio` | integer | Echoes ratio used【31†L223-L228】. |
| `results` | array | List of result objects. |

Per-result object:

| Field | Type | Description |
|---|---|---|
| `text` | string | Result text: record `text` if present, else node `text` property【31†L192-L200】【31†L200-L208】. |
| `metadata` | object | Node properties + `labels` and `element_id`, with `embedding` removed【6†L62-L70】. |
| `score` | float | Primary hybrid score from retriever record【31†L202-L210】. |
| `score_vector` | float | Normalized vector score (0–1) from `_vector_scores`【31†L202-L210】. |
| `score_fulltext` | float | Normalized fulltext score (0–1) from `_fulltext_scores`【31†L204-L212】. |
| `semantic_nodes` | array | Optional semantic entity nodes returned by retrieval query template【25†L11-L19】. |
| `semantic_relationships` | array | Optional semantic entity relationships returned by retrieval query template【25†L19-L27】. |

**Example (abridged)**

```json
{
  "query": "example question",
  "top_k": 3,
  "effective_search_ratio": 1,
  "results": [
    {
      "text": "…",
      "metadata": { "element_id": "1234", "labels": ["Chunk"], "uid": "chunk-1", "text": "…" },
      "score": 0.9876,
      "score_vector": 1.0,
      "score_fulltext": 0.975,
      "semantic_nodes": [],
      "semantic_relationships": []
    }
  ]
}
```

### Tool: `fetch` (POST `/mcp/fetch`)

**Parameters**

| Name | Type | Required | Default | Notes |
|---|---|---:|---:|---|
| `element_id` | string | Yes | — | Neo4j elementId of node. Must be non-empty【31†L298-L302】. |

**Response Schema**

If found:

| Field | Type | Description |
|---|---|---|
| `found` | boolean | `true` if node exists【31†L262-L267】. |
| `element_id` | string | Echoes requested element id【31†L262-L267】. |
| `metadata` | object | Node properties + labels + element_id, with embedding removed【31†L258-L266】. |
| `text` | string | Node text property (or empty)【31†L262-L267】. |

If not found:

| Field | Type | Description |
|---|---|---|
| `found` | boolean | `false`【31†L252-L254】. |
| `element_id` | string | Echoed requested element id【31†L252-L254】. |

If error:

| Field | Type | Description |
|---|---|---|
| `found` | boolean | `false`【31†L247-L250】. |
| `element_id` | string | Echoed requested element id【31†L247-L250】. |
| `error` | string | Error message【31†L247-L250】. |

---

## 3) Retrieval Implementation Inventory

| Component | Type | Responsibility | Evidence |
|---|---|---|---|
| `HybridCypherRetriever.get_search_results` | Upstream | Executes hybrid vector + fulltext search with a retrieval query appended. Returns records containing `node` and `score` at minimum. | 【31†L179-L187】 (called); upstream behavior described in 【20†L483-L492】【20†L495-L502】 |
| `search_sync(state, query, top_k, ratio)` | Custom | Orchestrates search: calls retriever; computes score breakdown (`score_vector`, `score_fulltext`); shapes response. | 【31†L179-L187】【31†L200-L209】 |
| `_vector_scores(state, query_vector, top_k, ratio)` | Custom | Runs `db.index.vector.queryNodes` to compute vector scores, normalizes via `_normalize_scores`. | 【6†L115-L123】【6†L131-L139】 |
| `_fulltext_scores(state, query_text, top_k)` | Custom | Runs `db.index.fulltext.queryNodes` to compute fulltext scores, normalizes via `_normalize_scores`. | 【6†L144-L152】【6†L158-L164】 |
| `_normalize_scores(records)` | Custom | Normalizes scores to 0–1 based on max score. | 【6†L51-L56】 |
| `_node_metadata(node)` | Custom | Converts Neo4j `Node` to dict, strips `embedding`, adds `labels` and `element_id`. | 【6†L62-L70】 |
| `fetch_sync(state, element_id)` | Custom | Node lookup by `elementId`, returns found/metadata/text payload. | 【31†L234-L242】【31†L261-L267】 |

---

## 4) Cypher + Index Assumptions

### Inline Cypher constants

**VECTOR_SCORE_QUERY** (vector index breakdown)【6†L25-L33】:
```cypher
CALL db.index.vector.queryNodes($index_name, $limit, $query_vector)
YIELD node, score
RETURN elementId(node) AS element_id, score
ORDER BY score DESC
```

**FULLTEXT_SCORE_QUERY** (fulltext breakdown)【6†L30-L38】:
```cypher
CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $limit})
YIELD node, score
RETURN elementId(node) AS element_id, score
ORDER BY score DESC
```

**FETCH_NODE_QUERY** (fetch by elementId)【6†L35-L38】:
```cypher
MATCH (n) 
WHERE elementId(n) = $element_id 
RETURN n AS node, labels(n) AS labels
```

### Retrieval query template (config-driven)

Loaded from `HYBRID_RETRIEVAL_QUERY_PATH` (default `queries/hybrid_retrieval.cypher`)【22†L41-L44】【29†L159-L167】. Default template returns semantic enrichment data alongside `node, score`:

```cypher
WITH node, score
OPTIONAL MATCH (sem:__Entity__)
  WHERE sem.semantic_source = 'kg_build.semantic_enrichment.v1'
    AND sem.chunk_uid = node.uid
OPTIONAL MATCH (sem)-[rel]->(other:__Entity__)
  WHERE rel.semantic_source = 'kg_build.semantic_enrichment.v1'
WITH node, score,
     collect(DISTINCT sem) AS semantic_nodes,
     collect(DISTINCT rel) AS semantic_relationships
RETURN node, score,
       [sem IN semantic_nodes |
          { id: elementId(sem), labels: labels(sem), properties: properties(sem) }
       ] AS semantic_nodes,
       [rel IN semantic_relationships |
          { type: type(rel), from: elementId(startNode(rel)), to: elementId(endNode(rel)), properties: properties(rel) }
       ] AS semantic_relationships
```  
【25†L1-L9】【25†L11-L19】

### Expected indexes

- Vector index name from `INDEX_NAME` (env). Used by `db.index.vector.queryNodes`【6†L115-L123】. Defaults described in env example【22†L27-L33】.
- Fulltext index name from `FULLTEXT_INDEX_NAME` (env). Used by `db.index.fulltext.queryNodes`【6†L144-L152】. Defaults described in env example【22†L27-L33】.
- Fulltext label/property defaults: `FULLTEXT_LABEL=Chunk`, `FULLTEXT_PROPERTY=text`【22†L27-L33】.

### Database selection

Neo4j database is selected via `NEO4J_DATABASE` in config; all `execute_query` calls include `database_=` parameter (read routing for retrieval)【31†L234-L242】.

---

## 5) Auth Modes

FancyRAG’s MCP server supports:

1. **Google OAuth (default when auth required)**  
   When `MCP_AUTH_REQUIRED=true` (default)【22†L33-L40】, `build_server` attaches a `GoogleProvider` configured with `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, scopes, and base URL【31†L273-L282】.

2. **Static token verifier (dev/test override)**  
   If `MCP_AUTH_REQUIRED=true` and `MCP_STATIC_TOKEN` is set, the entrypoint constructs a `StaticTokenVerifier` and passes it into `build_server`, overriding Google OAuth provider selection【13†L37-L46】【31†L273-L282】.

3. **No auth**  
   If `MCP_AUTH_REQUIRED=false`, `build_server` passes `auth=None`, leaving tools unprotected【31†L273-L282】.

Notes:
- Default required scopes configured via env example include `openid` and `userinfo.email`【22†L39-L44】【29†L135-L143】.
- Health endpoint is implemented as a custom route for orchestration; auth behavior for this route is not specified in the excerpted code beyond route registration【13†L54-L62】.

---

## 6) Upstream vs Custom Boundary

### Upstream (neo4j-graphrag-python)

- `HybridCypherRetriever` provides hybrid retrieval execution (vector + fulltext), embedding integration, and record shaping.
- `OpenAIEmbeddings` base embedder class (extended by FancyRAG)【27†L17-L25】.
- Upstream hybrid procedure usage `db.index.vector.queryNodes` / `db.index.fulltext.queryNodes` described in upstream docs【20†L483-L492】【20†L495-L502】.

### Upstream (FastMCP)

- `FastMCP` server framework provides tool decorator surface and run lifecycle.
- `GoogleProvider` and `StaticTokenVerifier` supply OAuth/token verification primitives【13†L10-L18】.

### Custom (FancyRAG)

- MCP server wrapper and entrypoint (`servers/mcp_hybrid_google.py`) including auth mode selection, health check route, and server startup behavior【13†L54-L62】【13†L84-L92】.
- Retrieval orchestration and score transparency logic: `search_sync`, `_vector_scores`, `_fulltext_scores`, `_normalize_scores`, and response shaping with `score_vector`/`score_fulltext`【31†L179-L187】【6†L115-L123】.
- Domain-specific retrieval query template (`queries/hybrid_retrieval.cypher`) that enriches results with `semantic_nodes` and `semantic_relationships`【25†L11-L19】.
- Robust embedding wrapper `RetryingOpenAIEmbeddings` adding retry/backoff instrumentation【27†L61-L69】.
- Configuration loader and environment modeling (Pydantic) including query template loading via `HYBRID_RETRIEVAL_QUERY_PATH`【29†L159-L167】.

---

## Sources

1. FancyRAG README – environment setup, auth config, and usage【1†L119-L127】【1†L133-L142】  
2. FancyRAG README – FastMCP server description (Google OAuth, HybridCypherRetriever)【2†L177-L185】【2†L193-L201】  
3. FancyRAG `servers/mcp_hybrid_google.py` – server entrypoint (init, auth setup, health route, server.run call)【13†L37-L46】【13†L84-L92】  
4. FancyRAG `fancryrag/mcp/runtime.py` – server builder and runtime logic (tool registration, search/fetch implementation)【31†L273-L282】【31†L284-L293】  
5. FancyRAG `fancryrag/mcp/runtime.py` – search_sync implementation (calls retriever, augments scores, assembles results)【31†L179-L187】【31†L200-L209】  
6. FancyRAG `fancryrag/mcp/runtime.py` – vector/fulltext score queries and normalization【6†L115-L123】【6†L131-L139】  
7. FancyRAG `queries/hybrid_retrieval.cypher` – Cypher template for semantic enrichment in hybrid results【25†L1-L9】【25†L11-L19】  
8. FancyRAG `fancryrag/embeddings.py` – RetryingOpenAIEmbeddings (extends GraphRAG OpenAIEmbeddings with retries)【27†L17-L25】【27†L61-L69】  
9. Neo4j GraphRAG documentation – HybridCypherRetriever search execution and result structure【20†L483-L492】【20†L495-L502】  
10. FancyRAG `.env.example` – default settings for indexes, auth, and query path【22†L27-L35】【22†L37-L44】  
