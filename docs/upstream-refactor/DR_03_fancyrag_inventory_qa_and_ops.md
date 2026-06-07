# Deep Research Report — FancyRAG Inventory: QA, Indexing Scripts, Ops/DevEx

## Metadata

- Report file: `DR_03_fancyrag_inventory_qa_and_ops.md`
- Generated: 2026-01-11 (America/New_York)
- Authoritative code sources (as provided):
  - Upstream: `https://github.com/neo4j/neo4j-graphrag-python.git`
  - Custom: `https://github.com/DrJLabs/FancyRAG.git`
- Default branch + HEAD SHAs used:
  - Upstream: **UNVERIFIED** (not recorded in this chat)
  - Custom: **UNVERIFIED** (not recorded in this chat)

### UNVERIFIED — How to confirm SHAs (exact steps)

1. Determine default branches:
   - `git ls-remote --symref https://github.com/neo4j/neo4j-graphrag-python.git HEAD`
   - `git ls-remote --symref https://github.com/DrJLabs/FancyRAG.git HEAD`
2. Record HEAD commit SHAs:
   - `git ls-remote https://github.com/neo4j/neo4j-graphrag-python.git <default-branch>`
   - `git ls-remote https://github.com/DrJLabs/FancyRAG.git <default-branch>`

---

## QA System Inventory

**Component** | **Description** | **Scope & Metrics** | **Thresholds & Outcome**
---|---|---|---
`IngestionQaEvaluator` (class) | Gathers QA metrics after an ingestion run and enforces gating thresholds【4†L115-L123】【5†L218-L226】. The evaluator is initialized with a Neo4j driver, target database, a list of ingested source files, QA thresholds, a report directory, and optional semantic enrichment summary【4†L95-L104】. | **Per-Run Focus:** It collects both **global graph counts** and **per-run stats**. It queries total counts of `Document`, `Chunk`, and relationships【4†L125-L132】【4†L129-L137】, then focuses on the current run’s chunks via their unique UIDs. For those chunks, it counts missing embeddings (chunks where the vector property `embedding` is null/empty)【13†L210-L218】, orphan chunks (ingested chunks with no `HAS_CHUNK` link to any Document)【13†L228-L236】, and checksum mismatches (chunks whose stored `checksum` differs from the provided source checksum)【13†L253-L260】. If semantic enrichment was performed, it also pulls counts of semantic nodes/relationships tagged to this run’s `source_tag` and how many of those are “orphan” (no relations)【13†L274-L283】【13†L286-L294】. Totals like files processed, chunks count, character lengths, and token estimates are computed across the run’s chunks for reporting【5†L301-L310】【5†L311-L320】. | **Configured via** `QaThresholds` (max allowed counts for each anomaly type)【4†L60-L68】. **Threshold Enforcement:** After metrics are gathered, each count is compared against its max threshold. If a count exceeds the limit, an anomaly string is added (e.g. `"missing_embeddings=X exceeds max Y"`)【5†L218-L227】. Semantic enrichment failures and orphaned entities are similarly checked against their own thresholds【4†L189-L197】【4†L198-L205】. If *any* anomalies were recorded, the run **fails QA** (status `"fail"`); otherwise it passes【5†L231-L239】. The evaluator returns a `QaResult` with status `"pass"` or `"fail"`, a human-readable summary, the full metrics dict, the list of anomaly messages, the thresholds used, and file paths to a detailed JSON/Markdown report【5†L249-L257】【5†L258-L266】.
`QaResult` (dataclass) | Container for QA evaluation output【4†L70-L78】. Captures the outcome of an evaluator run. | **Fields:** `status` (`"pass"`/`"fail"`), `summary` (short text summary), `metrics` (nested dict of all measured values), `anomalies` (list of anomaly descriptions), `thresholds` (the `QaThresholds` used), `report_json` & `report_markdown` (relative file paths of the saved report), timestamp, version, and duration_ms【4†L70-L78】【5†L237-L246】. Also has a convenience property `passed` that is True if status is “pass”【4†L85-L90】. | **Report Generation:** After evaluation, the raw result payload is scrubbed of sensitive info and written to a timestamped folder as `quality_report.json` and `quality_report.md`【5†L249-L257】. The Markdown report includes a header with version, timestamp, status, summary, then a **Metrics** section (files and chunks processed, token counts, missing embeddings, orphans, mismatches, etc.), a **Findings** section listing each anomaly (or a ✅ all-clear if none), a **Thresholds** listing, and a **Files** table detailing each ingested file (relative path, chunk count, checksum, git commit)【9†L74-L82】【9†L99-L107】. This provides both a human-friendly QA summary and a machine-readable JSON for further analysis.
`QaThresholds` (dataclass) | Configuration for gating criteria【4†L60-L68】. All thresholds default to 0 (requiring a perfect ingestion unless overridden). | Contains `max_missing_embeddings`, `max_orphan_chunks`, `max_checksum_mismatches`, plus semantic-related `max_semantic_failures` and `max_semantic_orphans`【4†L60-L68】. These define the **tolerance** for anomalies. The evaluator compares each measured value against the corresponding max; any value above the allowed max is flagged【5†L218-L227】【4†L197-L205】. For example, setting `max_missing_embeddings=5` would allow up to 5 chunks without embeddings before failing QA. In practice, defaults are 0 (no anomalies tolerated) unless configured otherwise. | Used by `IngestionQaEvaluator` to decide pass/fail. After computing metrics, the evaluator iterates over each threshold and appends an anomaly message for every metric that exceeds its max【5†L218-L227】. If the anomalies list remains empty, the status is “pass”, otherwise “fail”【5†L231-L239】. This gating mechanism ensures an ingestion run is rolled back or halted if data quality issues exceed the defined limits.
`QaSourceRecord` (dataclass) | Metadata record for one source document in an ingestion run【4†L47-L56】. The pipeline produces one `QaSourceRecord` per file ingested. | Holds identifying info: file `path`, `relative_path` within source, the file’s `document_checksum`, optional `git_commit` hash if from a repo, and the list of emitted chunk records (`chunks`, each a `QaChunkRecord` with `uid`, `checksum`, `text`)【4†L47-L56】【4†L39-L46】. May also carry an `ingest_run_key` identifying the run that ingested this file【4†L51-L56】. | Used as input to QA and rollback. The evaluator uses the collected chunk `uid`s and checksums from each `QaSourceRecord` to scope its queries (focusing on just those chunks)【4†L121-L129】【4†L155-L164】. The presence of an `ingest_run_key` (a unique run ID, typically a UUID with prefix) ties all these records to a specific ingestion session. That run key is applied to created nodes/relationships during ingestion for traceability【64†L67-L75】【64†L79-L82】 and later used to undo the ingestion if QA fails. In sum, `QaSourceRecord` defines the boundary of an ingestion run for QA purposes (by listing all chunk UIDs and a common run identifier).

### Notes on QA boundaries and artifacts

- QA evaluation is performed at the **ingestion run level**, not as a whole-of-database audit.
- The evaluator expects a set of recently ingested sources and checks that those sources’ chunks are correctly embedded and linked.
- It includes overall graph counts for context【4†L125-L132】【4†L129-L137】, but critical metrics like missing embeddings or orphan chunks are computed only for the provided chunk UIDs.
- Ingestion run boundaries are tracked via `ingest_run_key`: each run generates a unique key and tags all new nodes/relationships with it【64†L67-L75】【64†L79-L82】.
- The output of QA is a `QaResult` with references to saved artifacts. These artifacts are timestamped/versioned (e.g. under `ingestion-qa-report/v1`)【6†L35-L43】.
- Sensitive info is scrubbed from reports via `scrub_object` before saving【5†L249-L257】, using patterns that mask secrets like API keys/tokens/passwords with `***`【50†L19-L27】【50†L128-L136】.

---

## QA Queries Inventory

**Function** | **Cypher Query & Purpose** | **Graph Schema Assumptions**
---|---|---
`collect_counts(driver, database)` → **returns** `{documents, chunks, relationships}` | Runs three small COUNT queries to get high-level graph stats【11†L185-L193】. It counts all nodes labeled `Document`, all nodes labeled `Chunk`, and all `(:Document)-[:HAS_CHUNK]->(:Chunk)` relationships【11†L185-L193】. Each query is executed via a helper that extracts the integer value【11†L189-L197】【13†L299-L307】. If any query fails, a warning is logged and that count is left as 0【11†L191-L197】. | Assumes the model: `Document` nodes, `Chunk` nodes, and `HAS_CHUNK` relationships. Returns total (global) counts, not run-scoped.
`count_missing_embeddings(driver, database, chunk_uids)` → **returns** *int* | Counts how many of the given chunk UIDs lack an embedding vector【13†L210-L218】. Cypher: `UNWIND $uids AS uid MATCH (c:Chunk {uid: uid}) WHERE c.embedding IS NULL OR size(c.embedding) = 0 RETURN count(*)`【13†L210-L218】. Returns 0 if input list is empty【13†L203-L210】. | Assumes each `Chunk` has `uid` and `embedding` (list/array) properties.
`count_orphan_chunks(driver, database, chunk_uids)` → **returns** *int* | Counts chunks (from given UIDs) that have no incoming `HAS_CHUNK` relationship from any Document【13†L230-L238】. Cypher: `UNWIND $uids AS uid MATCH (c:Chunk {uid: uid}) WHERE NOT ( (:Document)-[:HAS_CHUNK]->(c) ) RETURN count(*)`【13†L230-L238】. Returns 0 if no UIDs【13†L223-L231】. | Assumes `HAS_CHUNK` exists and is required for each ingested chunk.
`count_checksum_mismatches(driver, database, chunk_rows)` → **returns** *int* | Finds chunks whose stored checksum doesn’t match expected checksum from ingestion metadata【13†L253-L260】. Cypher: `UNWIND $rows AS row MATCH (c:Chunk {uid: row.uid}) WHERE coalesce(c.checksum, "") <> row.checksum RETURN count(*)`【13†L253-L260】. Returns 0 if no rows【13†L248-L256】. | Assumes `Chunk.checksum` is present and comparable against expected.
`collect_semantic_counts(driver, database, source_tag)` → **returns** `{nodes_in_db, relationships_in_db, orphan_entities}` | Tallies semantic enrichment counts tagged to `semantic_source = $source`【13†L271-L279】【13†L286-L294】: nodes count, relationships count, and count of tagged nodes with no relationships (`AND NOT (n)--()`)【13†L286-L294】. Returns zeros if `source_tag` absent【13†L269-L277】. | Assumes semantic enrichment tags nodes/relationships with `semantic_source` and that semantic nodes should generally be connected.
`ensure_document_relationships(driver, ..., chunks_metadata)` → **writes** | Ensures a `Document` exists and attaches chunk nodes to it via `HAS_CHUNK`【11†L83-L92】【11†L95-L103】. Uses `MERGE` on `(:Document {source_path:$source_path})` and updates metadata fields; for each chunk, `MATCH`es `(:Chunk {uid: meta.uid})`, sets provenance fields, and `MERGE`s `(doc)-[:HAS_CHUNK]->(chunk)`【11†L97-L105】【11†L101-L108】. | Assumes `Document.source_path` is a stable identity; `Chunk.uid` is unique; `HAS_CHUNK` links doc→chunk.
`reset_database(driver, database)` → **destructive** | Deletes everything: `MATCH (n) DETACH DELETE n`【11†L51-L55】. | No schema assumptions; wipes DB.
`rollback_ingest(driver, sources)` → **destructive** | Deletes relationships and nodes created in a run via `ingest_run_key`, deletes chunks by UID, and deletes empty Documents by `relative_path`【11†L121-L129】【11†L133-L142】【11†L143-L151】【11†L155-L163】【11†L169-L177】. | Assumes relationships and non-core nodes carry `ingest_run_key`; chunks are uniquely identifiable by UID; documents can be deleted if no `HAS_CHUNK` remain.

---

## Index Scripts + Idempotence

### Vector index creation (`scripts/create_vector_index.py`)

- Ensures a Neo4j **vector index** exists for similarity search.
- Configuration is driven by env variables (with `.env.local` loading), including index name, label (default `Chunk`), embedding property (default `embedding`), and embedding dimensions (default 1024; set 1536 for OpenAI `text-embedding-3-small`)【28†L52-L60】【28†L69-L77】.
- Existence check is performed using GraphRAG’s `retrieve_vector_index_info` to compare the existing index config (if found) to the requested config【36†L223-L232】【36†L235-L244】.
- If a mismatched config is found, it raises a `VectorIndexMismatchError` rather than silently dropping/recreating【36†L249-L258】【36†L259-L267】.
- Creation uses `neo4j_graphrag.indexes.create_vector_index(...)` with dimensions and similarity metric (cosine)【36†L275-L283】, which internally uses `CREATE VECTOR INDEX ... IF NOT EXISTS` semantics【26†L141-L150】【26†L143-L151】.
- Creation is wrapped in retry/backoff (3 attempts) for resilience when Neo4j is still starting【30†L387-L396】【30†L389-L397】.
- Produces structured JSON logs saved to an artifacts path and printed to stdout, with sensitive values scrubbed before writing【32†L308-L316】【30†L411-L418】.

**Idempotence behavior:**
- If the index exists and matches config: no changes, status “exists”.
- If absent: creates index once.
- If exists but mismatched: fails fast (manual action required).

**Makefile note:** A Make target `index` exists that drops and recreates a vector index via Cypher shell:
- `DROP INDEX text_embeddings IF EXISTS; CALL db.index.vector.createNodeIndex('text_embeddings','Chunk','embedding',${EMBEDDING_DIMENSIONS:-1024},'cosine');`【23†L16-L24】

### Fulltext index creation (`scripts/create_fulltext_index.py`)

- Ensures fulltext index exists on chunk text.
- Requires `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` (fails early if missing)【20†L83-L92】【22†L225-L233】.
- Defaults: `NEO4J_DATABASE=neo4j`, `FULLTEXT_INDEX_NAME=chunk_text_fulltext`, `FULLTEXT_LABEL=Chunk`, `FULLTEXT_PROPERTY=text`【20†L93-L101】【20†L95-L103】.
- Waits for Neo4j readiness (`verify_connectivity`, `RETURN 1`) with retries/timeouts【20†L127-L136】【20†L139-L148】; exits non-zero on timeout【22†L253-L262】【22†L263-L272】.
- Checks existence via `SHOW INDEXES ... type='FULLTEXT'` and skips if exists【20†L173-L181】【20†L185-L193】.
- Creates via GraphRAG `create_fulltext_index`, which uses `CREATE FULLTEXT INDEX ... IF NOT EXISTS` semantics【20†L199-L208】【26†L159-L168】.

**Idempotence behavior:**
- If index exists: “skipped” (exit 0).
- If missing: creates index and logs “created” (exit 0).
- Does not currently detect “mismatch” if properties/label differ but name is the same.

### Embedding dimensions validation (runtime)

- `cli.utils.ensure_embedding_dimensions()` checks actual embedding length equals expected dimensions and gives operator guidance if not【61†L33-L41】【61†L53-L61】【61†L58-L66】.
- Smoke testing uses an embedding stub returning a 3D vector (`[0.42, 0.13, 0.88]`)【41†L43-L51】, so tests rely on configuration to match that dimension.

---

## DevEx Surface Map

**Surface/Tool** | **Purpose & Key Commands/Configs** | **Details & Workflow**
---|---|---
Makefile Targets | Convenience commands for common dev/ops tasks【23†L6-L14】. | Core targets include: `up`, `down`, `logs`, `index`, `fulltext-index`, `ingest`, `counts`, `smoke`, `smoke-logs`, `scan-image`【23†L7-L15】【23†L13-L18】【23†L16-L24】【23†L19-L23】【23†L23-L27】【23†L25-L28】【23†L30-L38】【23†L37-L40】【23†L41-L48】.
Docker Compose (Primary – `docker-compose.yml`) | Orchestrates local `neo4j` and `mcp` services with health-based dependency【40†L0-L9】【40†L22-L30】. | Neo4j uses `neo4j:5.26.12` and publishes 22010/22011; MCP builds `fancryrag-mcp:local`, publishes 22012, loads `.env.local`, and exposes `/mcp/health` for healthcheck【40†L1-L9】【40†L28-L36】【40†L39-L47】.
Docker Compose (Smoke – `docker-compose.smoke.yml`) | Isolated stack for end-to-end smoke tests: `neo4j`, `embedding-stub`, `mcp`, `smoke-tests`【41†L1-L9】【41†L113-L121】. | Embedding stub mimics `/v1/embeddings` and returns a fixed 3D vector【41†L42-L50】. Smoke tests mount repo and run `pytest -q tests/smoke/test_stack.py` after installing deps with `uv`【41†L133-L137】【41†L138-L144】.
Dockerfile (MCP) | Builds the MCP app image with Python 3.12 slim and `uv` for pinned deps via `uv.lock`【44†L0-L8】【44†L18-L27】. | Entrypoint runs `python servers/mcp_hybrid_google.py` and exposes 8080【44†L32-L37】.
CI (GitHub Actions) | Container secret scanning workflow builds image and scans via Trivy【45†L0-L9】【45†L33-L41】【45†L39-L47】. | Trivy is configured for secrets scanning and fails on medium/high/critical findings, providing a guardrail against leaked credentials【45†L39-L47】.

---

## Command List (from the analysis above)

### Makefile (primary operational interface)
- Start stack: `make up`【23†L7-L15】
- Stop/cleanup: `make down`【23†L9-L17】
- Tail logs: `make logs`【23†L13-L18】
- Create vector index (Makefile method): `make index`【23†L16-L24】
- Ensure fulltext index: `make fulltext-index`【23†L19-L23】
- Run ingestion pipeline: `make ingest f=<path>`【23†L23-L27】
- Quick node count: `make counts`【23†L25-L28】
- Run smoke tests: `make smoke`【23†L30-L38】
- Fetch smoke logs: `make smoke-logs`【23†L37-L40】
- Scan built image for secrets: `make scan-image`【23†L41-L48】

### Direct Cypher command embedded in Makefile
- Vector index drop/create:
  - `DROP INDEX text_embeddings IF EXISTS; CALL db.index.vector.createNodeIndex('text_embeddings','Chunk','embedding',${EMBEDDING_DIMENSIONS:-1024},'cosine');`【23†L16-L24】

### Smoke test execution (inside `docker-compose.smoke.yml`)
- Install deps and run tests:
  - `uv pip install --system ".[dev]"` then `pytest -q tests/smoke/test_stack.py`【41†L138-L144】

---

## Keep vs Replace Recommendations (bounded)

### Keep
1. **QA evaluator/reporting (`fancyrag/qa/*`)**: This is a core value-add (run-scoped integrity checks + gating + artifacts). Keep as a first-class layer.
2. **QA query helpers (`fancyrag/db/neo4j_queries.py`)**: Keep; they are aligned with FancyRAG’s schema and enable robust rollback and integrity checks.
3. **Smoke test harness**: Keep; embedding stub + isolated compose stack is strong coverage for ops regressions【41†L1-L9】【41†L42-L50】.
4. **CI container secret scanning**: Keep; effective guardrail against leaked credentials【45†L39-L47】.

### Replace / Consolidate
1. **Vector index management via Makefile Cypher**: Replace/retire `make index`’s hard-coded drop/create logic in favor of the Python script to gain config validation and mismatch detection【23†L16-L24】【36†L249-L258】.
2. **Fulltext index mismatch handling** (optional enhancement): The fulltext script currently “skips if exists” without verifying the label/property; consider adding config comparison similar to the vector script (only if you expect to change fulltext indexing parameters).

### Align with upstream (selectively)
- Continue using upstream GraphRAG index helpers (`create_vector_index`, `create_fulltext_index`) as the primary mechanism for index creation rather than custom Cypher, to remain forward-compatible and benefit from upstream validation【36†L275-L283】【20†L199-L208】.
- Keep FancyRAG’s QA gating and rollback behavior as a separate “operator layer,” unless upstream provides an equivalent with feature parity.

---

## UNVERIFIED Items & Verification Steps

Because this chat did not capture SHA-pinned permalinks or repo HEAD SHAs, the following are **UNVERIFIED** within this report:

- Default branch names and HEAD SHAs for both repos.
- Any “exact file path” mapping implied by citations.

### Verification steps
1. Clone both repos locally.
2. Record default branch + head SHA:
   - `git remote show origin | sed -n 's/  HEAD branch: //p'`
   - `git rev-parse HEAD`
3. Replace placeholder citations with SHA-pinned GitHub permalinks:
   - `https://github.com/<org>/<repo>/blob/<SHA>/<path>#Lx-Ly`

---

_End of report._
