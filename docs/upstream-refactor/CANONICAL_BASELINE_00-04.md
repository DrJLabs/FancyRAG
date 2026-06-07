# CANONICAL_BASELINE_00–04 (FancyRAG ↔ Upstream GraphRAG)

Confirmed readable inputs (all five required files present):
- `DR_00_overview_and_baselines.md`
- `DR_01_fancyrag_inventory_ingestion.md`
- `DR_02_fancyrag_inventory_retrieval_service.md`
- `DR_03_fancyrag_inventory_qa_and_ops.md`
- `DR_04_upstream_capabilities_map.md`

## A) Canonical SHAs & Baseline Scope

### FancyRAG baseline (canonical)
- **Repo:** DrJLabs/FancyRAG
- **Default branch:** `main`
- **Canonical HEAD SHA:** `0ff7b46717da4310b1626b0756551b7bd175d81b`
- **Commit date:** January 10, 2026
- **Repo-root permalink (@ SHA):** https://github.com/DrJLabs/FancyRAG/tree/0ff7b46717da4310b1626b0756551b7bd175d81b/

### Upstream baseline (canonical)
- **Repo:** neo4j/neo4j-graphrag-python
- **Default branch:** `main`
- **Canonical HEAD SHA:** `9e6b58827caa642539c8cb15dcde36c3e4d6d2c6`
- **Commit date:** December 17, 2025
- **Repo-root permalink (@ SHA):** https://github.com/neo4j/neo4j-graphrag-python/tree/9e6b58827caa642539c8cb15dcde36c3e4d6d2c6/

### Observed SHA conflicts (table)

| input_file | repo | sha | notes |
|---|---|---|---|
| DR_00_overview_and_baselines.md | FancyRAG | 0ff7b46717da4310b1626b0756551b7bd175d81b | Declared as default-branch `main` HEAD; includes commit date + repo-root permalink. |
| DR_00_overview_and_baselines.md | Upstream | 9e6b58827caa642539c8cb15dcde36c3e4d6d2c6 | Declared as default-branch `main` HEAD; includes commit date + repo-root permalink. |
| DR_01_fancyrag_inventory_ingestion.md | FancyRAG | 0ff7b46717da4310b1626b0756551b7bd175d81b | “Repo version pins (as reported)”. |
| DR_01_fancyrag_inventory_ingestion.md | Upstream | b8408185ca0df6cc141a29d5d28ad006c9844c5b | Reported as “v1.12.0 / commit …”【61†L84-L88】【65†L201-L209】; conflicts with DR_00 upstream HEAD SHA. |
| DR_02_fancyrag_inventory_retrieval_service.md | FancyRAG | UNVERIFIED | Front matter explicitly: `custom_default_branch_head: "UNVERIFIED (not resolved in this chat)"`. |
| DR_02_fancyrag_inventory_retrieval_service.md | Upstream | UNVERIFIED | Front matter explicitly: `upstream_default_branch_head: "UNVERIFIED (not resolved in this chat)"`. |
| DR_03_fancyrag_inventory_qa_and_ops.md | FancyRAG | UNVERIFIED | Metadata explicitly: “Default branch + HEAD SHAs used: … UNVERIFIED” (not recorded in this chat). |
| DR_03_fancyrag_inventory_qa_and_ops.md | Upstream | UNVERIFIED | Metadata explicitly: “Default branch + HEAD SHAs used: … UNVERIFIED” (not recorded in this chat). |
| DR_04_upstream_capabilities_map.md | Upstream | UNVERIFIED | Metadata explicitly: “HEAD commit SHA: UNVERIFIED (not resolved in this run)”. |

### Canonical choice rationale
- **SHA authority:** `DR_00_overview_and_baselines.md` is the only input that explicitly claims “verified baseline metadata” for both repos and provides **default branch**, **HEAD SHA**, **commit date**, and a **repo-root permalink** for each.
- **Conflict handling:** `DR_01_fancyrag_inventory_ingestion.md` reports an upstream “v1.12.0 / commit …” value【61†L84-L88】【65†L201-L209】. This appears to be a *dependency pin / version reference* captured during ingestion analysis rather than the upstream repo’s default-branch HEAD. Per rules, this is recorded as a delta, not blended.

### Scope statement
**Covered by inputs (in-scope):**
- FancyRAG ingestion/KG build entrypoints, core orchestration, config surfaces, and upstream touchpoints (DR_01).
- FancyRAG retrieval runtime + MCP server entrypoints, tool contracts (`/mcp/search`, `/mcp/fetch`, `/mcp/health`), Cypher/index assumptions, and auth modes (DR_02).
- FancyRAG QA evaluator + query helpers, index scripts + idempotence behaviors, and ops/devex surfaces (Makefile, compose files, smoke tests, Dockerfile, CI guardrails) (DR_03).
- High-level upstream capability map: retrievers, embedders, GraphRAG loop, experimental pipeline builder + injection model, and drift notes (DR_04).
- Canonical repo baselines (default branch + HEAD SHA + dates + repo-root permalinks) (DR_00).

**Missing / out-of-scope (explicitly not covered by the five inputs):**
- Complete, SHA-pinned file/line permalinks for *all* cited items (only repo-root permalinks are provided in DR_00; many other citations are “local” placeholders like `【31†L288-L292】`).
- Full symbol-level mapping between FancyRAG and upstream (this baseline is a prerequisite, not the mapping itself).
- Performance benchmarks, load characteristics, and cost profiles.
- Deployment hardening beyond what is reflected in Makefile/compose/Dockerfile notes.
- Any historical diffing across tags/releases (DR_04 explicitly states “no historical version diffing”).

## B) Unified Repo Topography (Source-of-Truth Index)

### Repo: DrJLabs/FancyRAG

| Subsystem (ingestion &#124; retrieval &#124; mcp/server &#124; qa &#124; ops/devex &#124; config &#124; utilities &#124; docs/examples) | Key paths (files/dirs) | Entrypoints (CLI/script/module) and how invoked | Primary symbols/classes/functions | Evidence (citations from inputs, or UNVERIFIED) |
|---|---|---|---|---|
| ingestion | `scripts/kg_build.py`; `fancyrag/cli/kg_build_main.py` (module path per call graph); `fancyrag/kg/pipeline.py`; `fancyrag/kg/phases.py`; legacy: `tools/run_pipeline.py`, `pipelines/kg_ingest.yaml` | **Current (intended):** run `scripts/kg_build.py` which calls `fancyrag.cli.kg_build_main.main()` then `run_pipeline(options)`. **Legacy:** run `tools/run_pipeline.py` which invokes upstream `PipelineRunner` on `pipelines/kg_ingest.yaml`. | `PipelineOptions`; `run_pipeline`; `resolve_settings`; `discover_sources` + `SourceSpec`; `build_clients` + `ClientBundle`; `ingest_source` + `IngestionArtifacts`; `perform_qa` + `IngestionQaEvaluator`; `rollback_ingest`; legacy `PipelineRunner`. | Current entrypoint chain【10†L11-L19】【31†L25-L34】; legacy runner + YAML【2†L23-L27】【1†L15-L18】; core symbols inventory【19†L371-L379】【16†L1362-L1370】【5†L417-L425】【55†L129-L137】 |
| retrieval | `fancryrag/mcp/runtime.py` (path spelling per DR_02 Sources; **UNVERIFIED** exact path); `queries/hybrid_retrieval.cypher` | MCP tool calls: POST `/mcp/search` and POST `/mcp/fetch` call into sync implementations via thread wrapper. | `search_sync`; `fetch_sync`; `_vector_scores`; `_fulltext_scores`; `_normalize_scores`; `_node_metadata`; retrieval query template returns semantic payloads. | MCP endpoints + tool wrappers【31†L289-L295】【31†L298-L302】; retrieval implementation inventory【31†L179-L187】【6†L115-L123】【6†L62-L70】; Cypher template snippet【25†L1-L9】【25†L11-L19】 |
| mcp/server | `servers/mcp_hybrid_google.py`; `fancryrag/mcp/runtime.py` (build/create state) | Runtime start via `python servers/mcp_hybrid_google.py`; exposes `/mcp/search`, `/mcp/fetch`, plus `/mcp/health`. | `main()`; `ServerState`/`create_state`; `build_server`; auth providers (Google OAuth / static token / no auth). | Entrypoints table + health route【13†L25-L33】【13†L54-L62】; server builder【31†L281-L289】【31†L273-L282】 |
| qa | `fancyrag/qa/*`; `fancyrag/db/neo4j_queries.py` | Called from ingestion via `perform_qa(...)`; includes destructive rollback/reset helpers. | `IngestionQaEvaluator`; `QaResult`; `QaThresholds`; `QaSourceRecord`; QA query helpers (`collect_counts`, `count_missing_embeddings`, etc.); `rollback_ingest`; `reset_database`. | QA inventory + notes【5†L249-L257】【4†L125-L132】; rollback on QA failure【5†L417-L425】【55†L129-L137】 |
| ops/devex | `Makefile`; `docker-compose.yml`; `docker-compose.smoke.yml`; MCP `Dockerfile`; CI workflows | Operator commands via `make ...` targets; smoke compose runs pytest; CI performs secret scanning. | Make targets (`up/down/logs/index/fulltext-index/ingest/smoke/...`); smoke harness; CI scanning workflow. | Make targets + smoke + CI scanning【23†L7-L15】【41†L138-L144】【45†L39-L47】 |
| config | `.env.example`; env var settings (Pydantic); legacy YAML pipeline config | Env vars configure Neo4j/OpenAI/indexes; legacy YAML config configures upstream pipeline runner; retrieval loads Cypher template via `HYBRID_RETRIEVAL_QUERY_PATH`. | `Neo4jSettings`; `OpenAISettings`; env vars (`NEO4J_*`, `OPENAI_*`, `INDEX_NAME`, `FULLTEXT_*`, `HYBRID_RETRIEVAL_QUERY_PATH`). | Env var surface table【25†L615-L623】【23†L31-L39】【2†L16-L24】; YAML keys table【29†L3-L11】【29†L68-L76】; query template path config【29†L159-L167】 |
| utilities | `cli/utils.py`; `scrub_object` utility module; `fancryrag/embeddings.py` (path spelling per DR_02 Sources; **UNVERIFIED**); OpenAI shared wrappers | Used for embedding-dimension validation, secret scrubbing, and retry/backoff instrumentation. | `ensure_embedding_dimensions`; `scrub_object`; `RetryingOpenAIEmbeddings`; `SharedOpenAIClient`/`SharedOpenAIEmbedder`/`SharedOpenAILLM`. | Dimension validator【61†L33-L41】【61†L58-L66】; scrubbing utility【50†L19-L27】【50†L128-L136】; retrying embeddings wrapper【27†L61-L69】; shared OpenAI wrappers【40†L1142-L1150】【40†L1189-L1197】 |
| docs/examples | README | README provides env setup + MCP description. | — | README references【1†L119-L127】【2†L177-L185】 |

### Repo: neo4j/neo4j-graphrag-python (Upstream)

| Subsystem (ingestion &#124; retrieval &#124; mcp/server &#124; qa &#124; ops/devex &#124; config &#124; utilities &#124; docs/examples) | Key paths (files/dirs) | Entrypoints (CLI/script/module) and how invoked | Primary symbols/classes/functions | Evidence (citations from inputs, or UNVERIFIED) |
|---|---|---|---|---|
| ingestion | `neo4j_graphrag.experimental.pipeline.*` | Python-native pipeline composition; component injection via construction or `Pipeline.add_component()`; examples referenced as `examples/pipeline/*`. | Experimental pipeline builder and KG writers/extractors; schema objects referenced by FancyRAG include `GraphSchema`, `Neo4jGraph`. | Pipeline injection model【22†L159-L168】; examples reference【48†L420-L428】; schema objects【19†L456-L462】【35†L27-L35】 |
| retrieval | `neo4j_graphrag.retrievers.*`; `neo4j_graphrag.generation.graphrag.*` | Library calls; examples referenced as `examples/retrieve/*`. | `HybridCypherRetriever`; `Text2CypherRetriever`; `GraphRAG`; `RagTemplate`. | Retriever + docs references【27†L423-L431】【20†L483-L492】; prompt control【33†L129-L137】; drift note【51†L7-L15】 |
| mcp/server | — | Not indicated as provided by upstream. | — | UNVERIFIED (not present in inputs) |
| qa | — | Not indicated as provided by upstream. | — | UNVERIFIED (not present in inputs) |
| ops/devex | — | Not indicated as provided by upstream. | — | UNVERIFIED (not present in inputs) |
| config | Interface-driven injection model (LLM + embedder abstractions); any YAML/config runner presence requires verification. | Inject providers at construction time; no YAML/JSON layer described in DR_04 for stable modules. | `LLMInterface`; `Embedder`; `RateLimitHandler`. | Interfaces + customization notes【20†L69-L71】【46†L45-L53】【27†L353-L362】 |
| utilities | `neo4j_graphrag.indexes.*` | Index helper functions used by FancyRAG scripts. | `create_vector_index`; `create_fulltext_index`; `retrieve_vector_index`. | Index helper semantics【26†L141-L150】【26†L159-L168】【36†L223-L232】 |
| docs/examples | `examples/pipeline/*`; `examples/retrieve/*`; official docs | Reference guides for retrievers and embedders. | — | Docs references【27†L423-L431】【27†L514-L522】 |


eferenced in DR_04【27†L423-L431】【27†L514-L522】 | — | 【27†L423-L431】【27†L514-L522】 |


## C) Canonical Contracts (agent-readable)

### 1) Retrieval/MCP Contract (if present)

**Server surface**
- **Tools (HTTP):**
  - `search` → POST `/mcp/search`【31†L289-L295】
  - `fetch` → POST `/mcp/fetch`【31†L298-L302】
- **Health route (HTTP):**
  - GET `/mcp/health` (returns OK vs 503 based on Neo4j connectivity probe)【13†L54-L62】【13†L56-L64】

#### Tool: `search` (POST `/mcp/search`)
**Parameters**【31†L288-L292】
- `query` (string, required)
- `top_k` (int, optional, default 5) — must be `> 0`【31†L288-L292】
- `effective_search_ratio` (int, optional, default 1) — must be `> 0`【31†L288-L292】

**Response schema**
Top-level fields【31†L223-L228】
- `query` (string) — echo
- `top_k` (int) — echo
- `effective_search_ratio` (int) — echo
- `results` (array) — list of result objects

Per-result object fields
- `text` (string) — record `text` if present, else node `text` property【31†L192-L200】【31†L200-L208】
- `metadata` (object) — node properties plus `labels` and `element_id`, with `embedding` removed【6†L62-L70】
- `score` (float) — primary hybrid score【31†L202-L210】
- `score_vector` (float) — normalized vector score 0–1【31†L202-L210】
- `score_fulltext` (float) — normalized fulltext score 0–1【31†L204-L212】
- `semantic_nodes` (array) — optional semantic entity nodes from query template【25†L11-L19】
- `semantic_relationships` (array) — optional semantic entity relationships from query template【25†L19-L27】

**Error modes**
- Parameter validation exists for `top_k > 0` and `effective_search_ratio > 0`【31†L288-L292】.
- **UNVERIFIED:** exact HTTP status codes and error response body for invalid parameters or Neo4j failures.  
  **Verification steps:** inspect `build_server` tool wrapper + validation in `fancryrag/mcp/runtime.py` around `search(...)` registration【31†L281-L289】【31†L288-L292】.

#### Tool: `fetch` (POST `/mcp/fetch`)
**Parameters**
- `element_id` (string, required) — must be non-empty【31†L298-L302】

**Response schema**
- If found:
  - `found` (boolean) — `true` if node exists【31†L262-L267】
  - `element_id` (string) — echo【31†L262-L267】
  - `metadata` (object) — node props + labels + element_id, `embedding` removed【31†L258-L266】
  - `text` (string) — node text property (or empty)【31†L262-L267】
- If not found:
  - `found` (boolean) — `false`【31†L252-L254】
  - `element_id` (string) — echo【31†L252-L254】
- If error:
  - `found` (boolean) — `false`【31†L247-L250】
  - `element_id` (string) — echo【31†L247-L250】
  - `error` (string) — error message【31†L247-L250】

#### Semantic enrichment payload contract (as returned in `search.results[*]`)
Returned from `queries/hybrid_retrieval.cypher`:
- Semantic nodes: `__Entity__` nodes where `semantic_source = 'kg_build.semantic_enrichment.v1'` and `chunk_uid = node.uid`【25†L1-L9】.
- Semantic relationships: relationships with `semantic_source = 'kg_build.semantic_enrichment.v1'`【25†L11-L19】.
- Returned as:
  - `semantic_nodes`: array of `{ id, labels, properties }`【25†L1-L9】【25†L11-L19】
  - `semantic_relationships`: array of `{ type, from, to, properties }`【25†L11-L19】

#### Auth contract (MCP)
Supported modes (selected by env/config):
1. **Google OAuth (default when auth required)** — when `MCP_AUTH_REQUIRED=true` (default)【22†L33-L40】; uses `GOOGLE_OAUTH_CLIENT_ID/SECRET`, scopes, base URL【31†L273-L282】.
2. **Static token** — when auth required and `MCP_STATIC_TOKEN` is set【13†L37-L46】【31†L273-L282】.
3. **No auth** — when `MCP_AUTH_REQUIRED=false`, `auth=None`【31†L273-L282】.

Notes:
- Default scopes include `openid` and `userinfo.email`【22†L39-L44】【29†L135-L143】.
- **UNVERIFIED:** exact OAuth redirect/callback endpoints and token validation error semantics (not specified in inputs).

---

### 2) Ingestion Contract

#### Canonical invocation path(s)
**Current (intended canonical):**
- `scripts/kg_build.py` → `fancyrag.cli.kg_build_main.main()` → `PipelineOptions` → `run_pipeline(options)`【10†L11-L19】【31†L25-L34】.
- Rationale: explicitly described as intended canonical path “replacing the older YAML/PipelineRunner method”【68†L47-L55】.

**Legacy (still present / still used by Makefile today):**
- `tools/run_pipeline.py` uses upstream `PipelineRunner` to execute `pipelines/kg_ingest.yaml`【2†L23-L27】【1†L15-L18】.
- Makefile `ingest` target still routes through YAML runner for consistency with “newer defaults”【10†L3-L11】【10†L13-L21】.

#### Required inputs (env/config/files)
**Neo4j connectivity**
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`【25†L615-L623】
- `NEO4J_DATABASE` (optional; passed via `database_=` to driver/query)【25†L618-L626】【55†L51-L59】

**LLM/Embeddings**
- `OPENAI_API_KEY`【27†L600-L608】
- `OPENAI_MODEL` default `gpt-5-mini`【22†L17-L25】【23†L31-L39】
- `OPENAI_EMBEDDING_MODEL` / `EMBEDDING_MODEL` default shown as `text-embe...3-small` (exact literal truncated in input)【23†L43-L51】 (**UNVERIFIED exact string**)
- `OPENAI_EMBEDDING_DIMENSIONS` / `EMBEDDING_DIMENSIONS` (used for vector index sizing + embedding length validation)【23†L73-L81】【2†L16-L24】
- Retry/behavior knobs: `OPENAI_MAX_ATTEMPTS` (default 3)【23†L55-L63】; `OPENAI_BACKOFF_SECONDS` (default 0.5)【23†L73-L81】; `OPENAI_TEMPERATURE` (default 0.3)【23†L91-L99】; `OPENAI_ENABLE_FALLBACK` (default true)【23†L109-L117】
- Optional proxying/routing: `OPENAI_BASE_URL` / `OPENAI_ALLOW_INSECURE_BASE_URL`【22†L75-L83】【22†L95-L104】; `EMBEDDING_API_BASE_URL` / `EMBEDDING_API_KEY`【22†L39-L47】

**Index naming inputs (used across ingestion/retrieval/indexing)**
- `INDEX_NAME` default `text_embeddings`【2†L16-L24】【2†L16-L24】
- `FULLTEXT_INDEX_NAME`, `FULLTEXT_LABEL`, `FULLTEXT_PROPERTY` (defaults shown as `Chunk`, `text`)【2†L19-L27】

**Source inputs**
- `PipelineOptions` includes a `source: ...` argument (full signature truncated in inputs)【19†L371-L379】.  
  **UNVERIFIED:** exact CLI flags and supported source types for code path; verify `fancyrag/cli/kg_build_main.py` and `PipelineOptions` definition.

**Legacy YAML config (legacy path only)**
- `pipelines/kg_ingest.yaml` defines `neo4j_config`, `llm_config`, `embedder_config`, and `component_config.*` including the exact-match resolver【29†L3-L11】【29†L68-L76】.

#### Expected outputs / artifacts
**Neo4j side effects**
- Chunk/document linkage is explicitly ensured by `ensure_document_relationships(...)`【55†L85-L93】【55†L97-L105】; QA queries assume `Document` and `Chunk` labels with `HAS_CHUNK` relationships【4†L125-L132】【11†L51-L55】.
- Embedding presence is QA-checked via `count_missing_embeddings(...)` over `Chunk.embedding`【4†L125-L132】.
- Ingestion run boundary tagging: `ingest_run_key` tags all new nodes/relationships for run-scoped QA and rollback【64†L67-L75】【64†L79-L82】.
- Optional semantic enrichment supports retrieval-time `semantic_nodes` / `semantic_relationships` keyed to `Chunk.uid` and `semantic_source` values【25†L1-L9】【25†L11-L19】.

**Filesystem/artifact side effects**
- `IngestionQaEvaluator` writes a detailed JSON/Markdown report【5†L249-L257】【5†L258-L266】; reports are timestamped/versioned (e.g., `ingestion-qa-report/v1`)【6†L35-L43】.
- Sensitive values are scrubbed before writing reports/logs via `scrub_object(...)`【50†L19-L27】【50†L128-L136】.
- Index scripts output structured JSON logs under an artifacts path with sensitive values scrubbed【32†L308-L316】【30†L411-L418】.

#### Success criteria (Neo4j + filesystem)
A run is considered successful when:
1. **Pipeline completes without rollback:** `run_pipeline` ingests all `SourceSpec` entries and `perform_qa` does not trigger `rollback_ingest(...)`【5†L417-L425】【55†L129-L137】.
2. **Required relationships exist:** `ensure_document_relationships` has established doc→chunk links consistent with QA assumptions【55†L85-L93】【4†L125-L132】.
3. **Embeddings exist and match expected dimensions:** missing embeddings and dimension mismatches are not present (dimension checks exist)【61†L33-L41】【4†L125-L132】.
4. **QA artifacts are generated:** QA report artifacts are written and scrubbed【5†L249-L257】【50†L19-L27】.

---

### 3) Index Contract

#### Required indexes and how they are used
**Vector index (required for retrieval)**
- Retrieval uses `db.index.vector.queryNodes($index_name, ...)` via `VECTOR_SCORE_QUERY`【6†L25-L33】.
- Index name sourced from env `INDEX_NAME` (defaults described in env examples; `INDEX_NAME` shown as `text_embeddings`)【22†L27-L33】【2†L16-L24】.
- Makefile legacy index creation indicates vector index on property `'embedding'`, dimension `${EMBEDDING_DIMENSIONS:-1024}`, similarity `'cosine'`【23†L16-L24】. (**Note:** default dimension value conflicts with script defaults; see Issues Log.)

**Fulltext index (required for retrieval)**
- Retrieval uses `db.index.fulltext.queryNodes($index_name, ...)` via `FULLTEXT_SCORE_QUERY`【6†L30-L38】.
- Fulltext index name from `FULLTEXT_INDEX_NAME` env var【22†L27-L33】; label/property defaults `FULLTEXT_LABEL=Chunk`, `FULLTEXT_PROPERTY=text`【22†L27-L33】.

#### Idempotent creation / verification (canonical scripts)
**Vector index creation — canonical script**
- `scripts/create_vector_index.py` ensures a vector index exists【28†L52-L60】【28†L69-L77】.
- Existence check uses GraphRAG `retrieve_vector_index(...)` and compares config【36†L223-L232】【36†L235-L244】.
- If mismatch: raises `VectorIndex...` (fails fast; manual action required)【36†L249-L258】【36†L259-L267】.
- Creation uses GraphRAG `create_vector_index(...)` with `CREATE INDEX ... IF NOT EXISTS` semantics【26†L141-L150】【26†L143-L151】.
- Wrapped in retry/backoff (3 attempts) for Neo4j startup resilience【30†L387-L396】【30†L389-L397】.
- Emits scrubbed, structured logs under artifacts path【32†L308-L316】【30†L411-L418】.

**Fulltext index creation — canonical script**
- `scripts/create_fulltext_index.py` ensures fulltext index exists; checks with `SHOW INDEXES ... type='FULLTEXT'` and skips if already present【20†L173-L181】【20†L185-L193】.
- Creates via GraphRAG `create_fulltext_index(...)` using `CREATE INDEX ... IF NOT EXISTS` semantics【20†L199-L208】【26†L159-L168】.
- Waits for Neo4j readiness; exits non-zero on timeout【22†L253-L262】【22†L263-L272】.
- **Known limitation:** does not detect “mismatch” when name matches but label/property differs【20†L93-L101】【20†L95-L103】.

#### Runtime validation
- `cli.utils.ensure_embedding_dimensions()` validates expected vs actual embedding dimensions and provides operator guidance on mismatch【61†L33-L41】【61†L58-L66】.

## D) Normalized Taxonomy & Glossary

### Canonical category taxonomy (exact allowed values)
- ingestion
- retrieval
- mcp/server
- qa
- ops/devex
- config
- utilities
- docs/examples

### Decision enum definitions (for later symbol mapping)
- **REPLACE:** Remove the current FancyRAG implementation and use upstream (or a new canonical implementation) as the primary source of truth.
- **WRAP:** Keep upstream (or existing) core behavior, but enforce FancyRAG-specific policy/telemetry/guards via a thin wrapper.
- **ADAPT:** Keep the component but refactor interfaces/inputs/outputs to align with canonical contracts and reduce drift.
- **KEEP:** Retain the component with minimal changes; only fix bugs and add SHA-pinned references/tests.
- **SPLIT:** Divide a component into multiple smaller components because responsibilities are currently conflated or contracts differ.

### Evidence minimum rule for future mapping rows
For each future **symbol-level** mapping row (FancyRAG → upstream target):

**FancyRAG-side evidence (minimum)**
- Must cite the defining symbol and at least one call-site or entrypoint reference using existing citations (e.g., `run_pipeline` callers) **or** (preferred in follow-on work) a SHA-pinned permalink generated from the canonical FancyRAG SHA.
- Must specify category taxonomy value (one of the allowed set above).

**Upstream-side evidence (minimum)**
- Must cite the upstream symbol definition or official doc/example reference (as captured in DR_04) and, when relevant, usage semantics (e.g., retriever behavior)【27†L423-L431】【20†L483-L492】.
- If relying on `neo4j_graphrag.experimental.*`, explicitly label as experimental/unstable【44†L388-L396】 and require version pinning.

**When `upstream_target` is null**
- Row must be marked “custom-only” and include:
  - Evidence for FancyRAG implementation (as above), and
  - An explicit verification step: repo search terms to confirm no upstream equivalent (e.g., search upstream for the symbol name or functional keywords), plus expected module paths to check.

## E) Consolidated Component Map (top-level only)

| Component | Upstream overlap (High/Med/Low) | Likely posture (REPLACE/WRAP/ADAPT/KEEP/SPLIT) | Evidence pointers (from inputs) | Non-negotiables (contracts/fields/ops guarantees that must not change) |
|---|---|---|---|---|
| **Canonical ingestion entrypoint** (`scripts/kg_build.py` → `run_pipeline`) | Med | KEEP + ADAPT | Code path intended canonical【68†L47-L55】; entrypoint chain【10†L11-L19】【31†L25-L34】 | Must continue to produce run-scoped QA artifacts + enforce rollback on failure【5†L417-L425】【55†L129-L137】 |
| **Legacy YAML pipeline runner** (`tools/run_pipeline.py` + `pipelines/kg_ingest.yaml`) | High | REPLACE | Still used by Makefile ingest【2†L23-L27】【10†L13-L21】 | If retained during transition, must remain functionally equivalent (dedupe resolver parity, QA parity)【29†L68-L76】【5†L417-L425】 |
| **Upstream experimental pipeline core** (SimpleKGPipeline etc) | High | WRAP/KEEP | FancyRAG calls upstream pipeline under `_execute_pipeline`【41†L1316-L1324】【16†L1321-L1329】 | Version pin required; isolate usage behind a stable FancyRAG facade【41†L1316-L1324】【44†L388-L396】 |
| **Text splitting + caching splitter** | Med | KEEP + ADAPT | `CachingFixedSizeSplitter` inventory【50†L94-L102】【50†L159-L165】 | Must preserve determinism/perf characteristics; chunk UID/metadata stability affects QA + semantic joins【19†L401-L409】【4†L125-L132】 |
| **Neo4j writing w/ sanitization** | Med | KEEP + WRAP | `SanitizingNeo4jWriter` subclass of upstream writer【43†L741-L749】【43†L749-L757】 | Must preserve schema assumptions used by QA + retrieval (Chunk/Document labels, uid fields, etc.)【4†L125-L132】 |
| **Semantic enrichment extraction + query template** | Med | KEEP | Retrieval template expects `semantic_source='kg_build.semantic_enrichment.v1'` and `chunk_uid=node.uid`【25†L1-L9】 | `semantic_nodes`/`semantic_relationships` payload shape must remain stable for agents/clients【25†L11-L19】 |
| **MCP server runtime** (`servers/mcp_hybrid_google.py` + runtime builder) | Low | KEEP + ADAPT | Entrypoints + endpoints【13†L25-L33】【31†L281-L289】 | HTTP tool names/paths and response fields must remain stable (`/mcp/search`, `/mcp/fetch`, `/mcp/health`)【31†L289-L295】【13†L54-L62】 |
| **Retrieval scoring transparency** (vector/fulltext breakdown + normalization) | Med | KEEP | Score normalization + per-score fields【6†L51-L56】【31†L202-L210】 | Must preserve `score`, `score_vector`, `score_fulltext`, plus metadata stripping of `embedding`【6†L62-L70】 |
| **Index management (scripts)** | High | KEEP (scripts) / REPLACE (Makefile cypher) | Script idempotence + mismatch detection【36†L249-L258】【20†L173-L181】; Makefile drop/recreate note【23†L16-L24】 | Vector + fulltext indexes must exist, match configured dimensions/labels, and be created idempotently【26†L141-L150】【26†L159-L168】 |
| **QA evaluator + rollback** | Low | KEEP | QA inventory + destructive rollback utilities【5†L249-L257】【55†L129-L137】 | Rollback must remain safe and run-scoped; report scrubbing must remain in place【50†L19-L27】【64†L67-L75】 |
| **Ops/DevEx stack** (compose, smoke tests, secret scanning) | Low | KEEP | Make targets + smoke tests + CI scanning【23†L30-L38】【41†L138-L144】【45†L39-L47】 | `make smoke` should remain a reliable regression gate for stack health【41†L138-L144】 |

## F) Issues Log (blocking gaps + contradictions)

Prioritized issues (highest impact first). Verification steps must be executed against the canonical SHAs in Section A.

1) **Upstream SHA conflict: DR_00 (HEAD) vs DR_01 (v1.12.0 commit)**
- **Severity:** High  
- **Impact:** Blocks symbol mapping + refactor PR plan (cannot decide “align to upstream HEAD” vs “align to pinned upstream version”).  
- **Verification steps:**
  - In FancyRAG repo, search dependency pinning:
    - Search terms: `neo4j-graphrag-python`, `neo4j_graphrag`, `b8408185`, `v1.12.0`
    - Likely paths: `pyproject.toml`, `uv.lock`, `requirements*.txt`, `poetry.lock`, `pip-tools` lockfiles.
  - In upstream repo, confirm `9e6b58827caa642...` is current `main` HEAD and whether a tag/release maps to `b8408185...`.

2) **Path spelling inconsistency: `fancryrag/...` vs `fancyrag/...`**
- **Severity:** High  
- **Impact:** Blocks automated repo search + symbol mapping; could invalidate contract references.  
- **Verification steps:**
  - In FancyRAG repo:
    - Search terms: `mcp/runtime.py`, `build_server(`, `search_sync(`, `FastMCP`, `ServerState`
    - Expected paths: `fancyrag/mcp/runtime.py` and/or `fancryrag/mcp/runtime.py`; `servers/mcp_hybrid_google.py`.

3) **Canonical ingestion path vs Makefile drift (Makefile still uses YAML runner)**
- **Severity:** High  
- **Impact:** Blocks refactor PR plan + atomic task breakdown (must choose single operational entrypoint).  
- **Verification steps:**
  - Search in FancyRAG repo:
    - Search terms: `make ingest`, `tools/run_pipeline.py`, `scripts/kg_build.py`, `pipelines/kg_ingest.yaml`
    - Expected paths: `Makefile`, `tools/run_pipeline.py`, `scripts/kg_build.py`, `pipelines/kg_ingest.yaml`.
  - Compare features: ensure code path includes resolver parity (`SinglePropertyExactMatchResolver`) or document intentional removal【29†L68-L76】.

4) **Index dimension/default mismatches across scripts vs Makefile**
- **Severity:** High  
- **Impact:** Blocks verification checklist + ops reliability; can break retrieval at runtime (vector index dimension mismatch).  
- **Evidence of mismatch:**
  - Script default embedding dimensions “default 1024”【28†L52-L60】【28†L69-L77】
  - Makefile vector index creation defaults to `${EMBEDDING_DIMENSIONS:-1024}`【23†L16-L24】
- **Verification steps:**
  - Search terms: `EMBEDDING_DIMENSIONS`, `OPENAI_EMBEDDING_DIMENSIONS`, `text_embeddings`, `create_vector_index`
  - Expected paths: `.env.example`, `.env.local`, `scripts/create_vector_index.py`, `Makefile`, retrieval runtime config loader.

5) **Search tool error semantics are not fully specified**
- **Severity:** Med  
- **Impact:** Blocks agent contract hardening and client integration tests (clients need predictable error responses).  
- **Verification steps:**
  - Inspect `/mcp/search` tool wrapper in `build_server(state)` for validation/exception handling【31†L288-L292】.
  - Search terms: `HTTPException`, `raise`, `try:`, `except`, `top_k`, `effective_search_ratio`.

6) **Fulltext index mismatch handling is explicitly limited**
- **Severity:** Med  
- **Impact:** Risk register + ops guardrails (silent drift possible if name collides but label/property changes).  
- **Verification steps:**
  - Inspect `scripts/create_fulltext_index.py` logic:
    - Search terms: `SHOW INDEXES`, `FULLTEXT_LABEL`, `FULLTEXT_PROPERTY`, `FULLTEXT_INDEX_NAME`
    - Expected path: `scripts/create_fulltext_index.py`.
  - Decide whether to add “mismatch detection” analogous to vector index script.

7) **Upstream “no config-file layer” claim vs legacy `PipelineRunner.from_config_file(...)` usage**
- **Severity:** Med  
- **Impact:** Blocks alignment strategy for ingestion (whether to retain YAML runner).  
- **Verification steps:**
  - In upstream repo:
    - Search terms: `class PipelineRunner`, `from_config_file`, `yaml`, `config_file`
    - Expected paths: `neo4j_graphrag/experimental/...` (exact path UNVERIFIED).
  - Reconcile with DR_04 statement: “not config-file driven” (may apply to stable modules only)【22†L159-L168】.

8) **Schema assumptions that must be enforced are partially implicit**
- **Severity:** Med  
- **Impact:** Blocks risk register refinement + verification checklist (must lock schema invariants).  
- **Verification steps:**
  - Search terms: `(:Chunk`, `(:Document`, `HAS_CHUNK`, `uid`, `checksum`, `embedding`, `ingest_run_key`
  - Expected paths: `fancyrag/db/neo4j_queries.py`, ingestion pipeline writer/executor modules, `queries/hybrid_retrieval.cypher`.

## G) Baseline Handoff Summary

- Canonical FancyRAG baseline: `main` @ `0ff7b46717da4310b1626b0756551b7bd175d81b` (commit date January 10, 2026) — repo root permalink: https://github.com/DrJLabs/FancyRAG/tree/0ff7b46717da4310b1626b0756551b7bd175d81b/
- Canonical upstream baseline: `main` @ `9e6b58827caa642539c8cb15dcde36c3e4d6d2c6` (commit date December 17, 2025) — repo root permalink: https://github.com/neo4j/neo4j-graphrag-python/tree/9e6b58827caa642539c8cb15dcde36c3e4d6d2c6/
- Observed upstream SHA conflict: DR_01 reports “v1.12.0 / commit `b8408185...`”【61†L84-L88】【65†L201-L209】; DR_00 reports upstream HEAD `9e6b588...`.
- Canonical ingestion entrypoint is intended to be the code path (`scripts/kg_build.py` → `run_pipeline`)【68†L47-L55】【10†L11-L19】, but Makefile still invokes legacy YAML runner【10†L13-L21】.
- Ingestion core orchestration revolves around `PipelineOptions`, `run_pipeline`, `ingest_source`, and `perform_qa` with rollback support【16†L1362-L1370】【55†L129-L137】.
- QA is run-scoped and produces scrubbed artifacts; run boundaries use `ingest_run_key` tagging【64†L67-L75】【50†L19-L27】.
- MCP contract exposes POST `/mcp/search` and `/mcp/fetch` plus GET `/mcp/health` for orchestration readiness【31†L289-L295】【13†L54-L62】.
- `/mcp/search` responses include score transparency (`score`, `score_vector`, `score_fulltext`) and optional semantic payloads (`semantic_nodes`, `semantic_relationships`)【31†L202-L210】【25†L11-L19】.
- Retrieval depends on Neo4j vector and fulltext indexes; names are env-driven (`INDEX_NAME`, `FULLTEXT_INDEX_NAME`, `FULLTEXT_LABEL`, `FULLTEXT_PROPERTY`)【22†L27-L33】.
- Canonical index creation should be via scripts with idempotence + mismatch detection (vector) rather than Makefile “drop and recreate” Cypher【36†L249-L258】【23†L16-L24】.
- Top components to preserve (evidence-backed): QA evaluator + rollback layer, MCP contract surface, retrieval score transparency fields, and secret scrubbing in artifacts/logs【5†L249-L257】【6†L62-L70】【50†L19-L27】.
- First gaps to resolve: upstream SHA alignment decision; path spelling inconsistency; Makefile ingestion drift; embedding dimension defaults mismatch; search-tool error semantics (standardize).