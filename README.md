# FancyRAG Workspace

FancyRAG delivers a local-first GraphRAG playground backed by Neo4j, Qdrant, and OpenAI. The current release (Epics 1–2) equips a solo developer with everything needed to ingest sample content and retrieve grounded answers on a laptop.

## Current Stack Snapshot
- **Environment & Workspace (Epic 1):** `scripts/bootstrap.sh` provisions a Python 3.12 virtualenv with pinned dependencies and scaffolds `.env` defaults.
- **Local GraphRAG Minimal Path (Epic 2):** `docker-compose.neo4j-qdrant.yml` launches Neo4j 5.26.12 + Qdrant 1.15.4, while scripts under `scripts/` create vector indexes, run `SimpleKGPipeline`, export embeddings, and query with `GraphRAG`.
- **Automation:** Unit tests cover the ingestion/export scripts; `tests/integration/local_stack/test_minimal_path_smoke.py` exercises the end-to-end flow when Docker and OpenAI credentials are available.

## Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12.x | Virtualenv bootstrapped via `scripts/bootstrap.sh` |
| Neo4j (APOC Core) | 5.26.12 | Pinned in `docker-compose.neo4j-qdrant.yml` |
| Qdrant | 1.15.4 | Pinned in `docker-compose.neo4j-qdrant.yml` |
| neo4j-graphrag | latest main (2025-09 snapshot) | Installed via `pip install -r requirements.txt` |
| OpenAI Models | `gpt-4.1-mini`, `text-embedding-3-small` | Configure via `.env`; audited by `scripts/audit_openai_allowlist.py` |

> All documentation references these canonical versions; update this table first, then propagate changes.

## Run the Minimal Path Locally
1. **Bootstrap tooling**
   ```bash
   bash scripts/bootstrap.sh
   source .venv/bin/activate
   ```
2. **Configure credentials** – copy `.env.example` to `.env`, supply `OPENAI_API_KEY`, and keep Neo4j/Qdrant defaults for local usage (`bolt://localhost:7687`, `http://localhost:6333`).
3. **Verify the workspace**
   ```bash
   PYTHONPATH=src python -m cli.diagnostics workspace --write-report
   PYTHONPATH=src python -m cli.diagnostics openai-probe
   ```
4. **Start the stack**
   ```bash
   scripts/check_local_stack.sh --up
   scripts/check_local_stack.sh --status --wait
   ```
5. **Ingest and query**
   ```bash
   PYTHONPATH=src python scripts/create_vector_index.py --index-name chunks_vec --label Chunk --dimensions 1536
   PYTHONPATH=src python scripts/kg_build.py --source docs/samples/pilot.txt
   PYTHONPATH=src python scripts/export_to_qdrant.py --collection chunks_main
   PYTHONPATH=src python scripts/ask_qdrant.py --question "What did Acme launch?" --top-k 5
   ```
   - Use `--profile text|markdown|code` to apply tuned chunk sizes/overlaps. For example, `PYTHONPATH=src python scripts/kg_build.py --source-dir docs --profile markdown --include-pattern "**/*.md"` ingests every Markdown file with metadata captured for Neo4j and Qdrant payloads (relative paths, git commit, chunk checksums).
   - The ingestion step emits a QA report (`ingestion-qa-report/v1`) under `artifacts/ingestion/<timestamp>/quality_report.{json,md}` and enforces thresholds for missing embeddings, orphaned chunks, and checksum mismatches. Override limits with `--qa-max-missing-embeddings`, `--qa-max-orphan-chunks`, or `--qa-max-checksum-mismatches` as needed.
   - The QA section of the run log now records `duration_ms` alongside `metrics.qa_evaluation_ms`, giving a quick view into the overhead added by telemetry so you can track runtime against your baselines.
6. **Tear down when finished**
   ```bash
   scripts/check_local_stack.sh --down --destroy-volumes
   ```

Each script emits sanitized JSON logs under `artifacts/local_stack/`, making the flow automation-friendly.

## Adaptive Chunking & Metadata

- Chunking presets (`text`, `markdown`, `code`) apply tuned splitter defaults; override via `--chunk-size` / `--chunk-overlap` when needed.
- Directory ingestion (`--source-dir` + `--include-pattern`) walks files deterministically, skips binary inputs, and records per-chunk metadata (relative path, git commit, checksum, indices) in Neo4j and Qdrant payloads to simplify traceability.
- Structured logs now expose `files`/`chunks` arrays so operators can diff runs and monitor ingestion quality over time.

## Documentation Map
- [Architecture Overview](docs/architecture/overview.md) – workflow sequencing, environmental guardrails, and change history.
- [Source Tree Blueprint](docs/architecture/source-tree.md) – file locations for scripts, CLI modules, and tests.
- [PRD Shards](docs/prd/) – goals, requirements, and epic catalog.
- [GraphRAG Quickstart](docs/graphrag/QUICKSTART.md) – condensed setup checklist for newcomers.
- Story close-outs live in `docs/stories/` with QA evidence under `docs/qa/`.

For managed-service work, see open follow-ups in QA assessments; the mainline backlog currently ends with the local minimal path.
