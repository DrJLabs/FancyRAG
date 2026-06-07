# DR_08 Verification Checklist

**Purpose:** Agent-executable checklist to verify FancyRAG ingestion, indexing, retrieval/MCP contracts, QA gating, regressions, and secret/log safety **before and after every PR/phase**.

**Grounding:** Commands and expectations are derived from the repo’s documented DevEx surface (Makefile targets, docker-compose stacks, and `scripts/*` utilities) and the canonical contracts captured in DR_00–DR_04.

---

## How to run this checklist

### Minimum gate (recommended on every PR)
1) Preflight (Section A)  
2) Setup → Bring up services → Create indexes (idempotent)  
3) Contract checks (`/mcp/health`, `/mcp/search`, `/mcp/fetch`)  
4) `make smoke`  
5) Secret/log safety checks (`make scan-image` + artifact scrubbing scan)

### Full gate (recommended before releases / risky PRs)
- Minimum gate, plus:
  - Run ingestion on a representative fixture (`make ingest`)
  - Index verification via Neo4j `SHOW INDEXES`
  - QA gating verification (including rollback behavior) — see Section B.6

---

## Conventions

- **Host context**: commands run on your workstation/CI runner shell (repo root).
- **Container context**: commands run inside containers via `docker compose exec`.
- All command blocks are **copy/paste friendly** and fail fast via `set -euo pipefail`.
- Any item marked **UNVERIFIED** means the exact command/flag/target was not explicitly confirmed in the attached baseline docs; the block includes exact discovery steps to resolve it without needing more documentation.

---

# A) Preflight

## A.1 Record state (required)

**Context:** Host

```bash
set -euo pipefail

git rev-parse --show-toplevel
git status --porcelain
git rev-parse HEAD
git log -1 --oneline
```

**Expected assertions (pass):**
- You are in the repo root (or a subdir of it).
- Working tree is clean for “after PR” validation (or you deliberately accept diffs when running “before PR” checks).
- You have a concrete commit SHA recorded for the run.

---

## A.2 Tooling checks (required)

**Context:** Host

```bash
set -euo pipefail

# Core tools used by Make targets and smoke stack
docker --version
docker compose version

# FancyRAG baseline uses uv for Python dependency execution ("uv run ...")
uv --version

# Basic network tooling for contract checks
curl --version
python --version
```

**Expected assertions (pass):**
- `docker` and `docker compose` commands succeed.
- `uv` is installed (Makefile targets use it).
- `curl` and `python` are present.

---

## A.3 Repo-required files (required)

**Context:** Host

```bash
set -euo pipefail

test -f Makefile
test -f docker-compose.yml
test -f docker-compose.smoke.yml

# Index scripts (idempotent path)
test -f scripts/create_vector_index.py
test -f scripts/create_fulltext_index.py

# MCP server entrypoint
test -f servers/mcp_hybrid_google.py

# Retrieval Cypher template (used by MCP server config by default)
test -f queries/hybrid_retrieval.cypher

# Example env template
test -f .env.example
```

**Expected assertions (pass):**
- All `test -f ...` checks succeed.

---

## A.4 Environment file hygiene (required)

**Context:** Host

```bash
set -euo pipefail

# If you use a local env file (recommended), it should exist and must not be committed.
if [ -f .env.local ]; then
  echo ".env.local: present"
else
  echo ".env.local: missing (OK if you export env vars another way)"
fi

# Ensure local secrets are not tracked
git ls-files --error-unmatch .env.local >/dev/null 2>&1 && {
  echo "FAIL: .env.local is tracked by git (should be ignored)"; exit 1;
} || echo ".env.local is not tracked (OK)"
```

**Expected assertions (pass):**
- `.env.local` is either present or intentionally absent.
- `.env.local` is **not** tracked by git.

---

## A.5 Required secrets/configs (names only)

> Do **not** print values during verification. This section lists **names only**.

### Neo4j connectivity
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (optional; defaults vary by environment)

### OpenAI / LLM / Embeddings
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL` (or `EMBEDDING_MODEL`)
- `OPENAI_EMBEDDING_DIMENSIONS` (or `EMBEDDING_DIMENSIONS`)
- `OPENAI_MAX_ATTEMPTS` (optional)
- `OPENAI_BACKOFF_SECONDS` (optional)
- `OPENAI_TEMPERATURE` (optional)
- `OPENAI_ENABLE_FALLBACK` (optional)
- `OPENAI_BASE_URL` (optional)
- `OPENAI_ALLOW_INSECURE_BASE_URL` (optional)
- `EMBEDDING_API_BASE_URL` (optional; used by smoke stack embedding stub)
- `EMBEDDING_API_KEY` (optional; if embedding endpoint requires auth)

### Index names and fulltext settings
- `INDEX_NAME`
- `FULLTEXT_INDEX_NAME`
- `FULLTEXT_LABEL`
- `FULLTEXT_PROPERTY`

### Retrieval query template
- `HYBRID_RETRIEVAL_QUERY_PATH`

### MCP auth and OAuth (depending on mode)
- `MCP_AUTH_REQUIRED`
- `MCP_STATIC_TOKEN` (optional; dev/test auth mode)
- `GOOGLE_OAUTH_CLIENT_ID` (required if Google OAuth enabled)
- `GOOGLE_OAUTH_CLIENT_SECRET` (required if Google OAuth enabled)
- `GOOGLE_OAUTH_SCOPES` (optional; if configurable via env)
- `MCP_BASE_URL` (optional; used for OAuth redirects in some deployments)

---

## A.6 Non-secret env presence check (safe)

**Context:** Host

This checks **presence** only (no values printed).

```bash
set -euo pipefail

# Load .env.local if present (do NOT echo values)
if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

python - <<'PY'
import os, sys

required = [
  # Neo4j
  "NEO4J_URI","NEO4J_USERNAME","NEO4J_PASSWORD",
  # Retrieval config
  "INDEX_NAME","FULLTEXT_INDEX_NAME","FULLTEXT_LABEL","FULLTEXT_PROPERTY","HYBRID_RETRIEVAL_QUERY_PATH",
]

missing = [k for k in required if not os.getenv(k)]
if missing:
  print("FAIL: missing required env var(s):")
  for k in missing:
    print(f"  - {k}")
  sys.exit(1)

print("OK: required env vars present (values not shown)")
PY
```

**Expected assertions (pass):**
- Script prints `OK: required env vars present...` and exits 0.

**Notes:**
- `OPENAI_*` vars may be optional in smoke mode (embedding stub), but are required for real ingestion.

---

# B) Commands

## B.1 Setup

### B.1.1 Python environment sanity

**Context:** Host

```bash
set -euo pipefail

# Smoke check: uv can resolve project and run python.
uv run python -c "import sys; print(sys.version)"

# Import sanity for the known module-name inconsistency (fancryrag vs fancyrag).
uv run python - <<'PY'
mods = ["fancyrag", "fancryrag"]
found = []
for m in mods:
  try:
    __import__(m)
    found.append(m)
  except Exception:
    pass

if not found:
  raise SystemExit("FAIL: neither 'fancyrag' nor 'fancryrag' is importable (packaging/import path issue)")
print("OK: importable package(s):", ", ".join(found))
PY
```

**Expected assertions (pass):**
- `uv run` works (dependency resolution is functional).
- At least one of `fancyrag` or `fancryrag` imports successfully.

---

### B.1.2 MCP config validation (optional but recommended)

**Context:** Host

```bash
set -euo pipefail

# Load env if available
if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

# Attempt to call load_config() if present (works pre/post the fancryrag→fancyrag rename).
uv run python - <<'PY'
import importlib

candidates = [
  ("fancryrag.config", "load_config"),
  ("fancyrag.config", "load_config"),
]
for mod_name, fn_name in candidates:
  try:
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
      continue
    fn()
    print(f"OK: {mod_name}.{fn_name}() succeeded")
    raise SystemExit(0)
  except SystemExit:
    raise
  except Exception as e:
    # Try next candidate
    last = (mod_name, str(e))

print("UNVERIFIED/FAIL: could not validate config via load_config().")
print("Next steps:")
print("  1) Inspect servers/mcp_hybrid_google.py for the config import and the function name.")
print("  2) Run: uv run python servers/mcp_hybrid_google.py (it should fail fast with ConfigurationError if env is missing).")
raise SystemExit(1)
PY
```

**Expected assertions (pass):**
- Prints `OK: ...load_config() succeeded` and exits 0.

**If this fails:**
- Treat as **UNVERIFIED** and use the discovery steps printed by the script.

---

## B.2 Bring up services

### B.2.1 Start the local stack (Neo4j + MCP)

**Context:** Host

```bash
set -euo pipefail

make up
```

**Expected assertions (pass):**
- Docker Compose starts services successfully.
- MCP becomes healthy (see contract checks).

**Operator notes:**
- If this hangs/fails due to ports already in use, stop conflicting processes or update compose port bindings.

---

### B.2.2 Tail logs (optional)

**Context:** Host

```bash
set -euo pipefail

make logs
```

**Expected assertions (pass):**
- Neo4j reaches a “ready” state (Bolt/HTTP listening).
- MCP logs indicate server start.

---

## B.3 Create indexes (idempotent)

> Prefer these scripts over `make index` because the vector script detects config mismatches and avoids silent destructive recreation.

### B.3.1 Vector index ensure (idempotent, mismatch-safe)

**Context:** Host

```bash
set -euo pipefail

# Load env if available
if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

uv run python scripts/create_vector_index.py
```

**Expected assertions (pass):**
- Exit code 0.
- Output indicates either “created” or “exists” (exact wording may vary).
- If the existing index config does not match, the script **fails fast** (this is an intentional safety feature).

---

### B.3.2 Fulltext index ensure (idempotent)

**Context:** Host

```bash
set -euo pipefail

# Load env if available
if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

uv run python scripts/create_fulltext_index.py
```

**Expected assertions (pass):**
- Exit code 0.
- Output indicates either “created” or “skipped/exists” (exact wording may vary).
- Note: baseline behavior does **not** guarantee mismatch detection when the index name exists but label/property differs.

---

### B.3.3 Legacy Makefile index targets (destructive; use only intentionally)

**Context:** Host

```bash
set -euo pipefail

# WARNING: Destructive (drops and recreates vector index).
# Use only if you have decided to recreate the index.
make index

# Fulltext index creation via Makefile helper
make fulltext-index
```

**Expected assertions (pass):**
- `make index` exits 0 and recreates the vector index.
- `make fulltext-index` exits 0 and ensures fulltext index exists.

**Safety note:**
- The Makefile vector index target has historically used a default dimension different from the Python script default; avoid relying on defaults—set `EMBEDDING_DIMENSIONS` explicitly.

---

## B.4 Index verification (direct Neo4j inspection)

**Context:** Container (Neo4j)

```bash
set -euo pipefail

# Load env if available (for credentials)
if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

# Show relevant indexes (name/type/state/properties)
docker compose exec -T neo4j cypher-shell \
  -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" -d "${NEO4J_DATABASE:-neo4j}" \
  "SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties
   WHERE name IN ['$INDEX_NAME', '$FULLTEXT_INDEX_NAME']
   RETURN name, type, state, labelsOrTypes, properties
   ORDER BY name;"
```

**Expected assertions (pass):**
- Both indexes appear.
- `state` is `ONLINE`.
- Vector index `type` is `VECTOR`, fulltext index `type` is `FULLTEXT`.
- The vector index properties include the embedding property (typically `embedding`), and fulltext includes the configured text property.

**If this fails:**
- Confirm the compose service name is `neo4j`:
  - `docker compose ps`
- If `cypher-shell` is not in PATH in the container (unlikely for Neo4j image), discovery step:
  - `docker compose exec -T neo4j bash -lc 'ls -la /var/lib/neo4j/bin && which cypher-shell || true'`

---

## B.5 Run ingest

### B.5.1 Baseline Makefile ingest (legacy YAML runner)

**Context:** Host

```bash
set -euo pipefail

# Choose a small, representative input file or directory.
# Examples (pick one):
#   export INGEST_PATH="README.md"
#   export INGEST_PATH="docs/"
#   export INGEST_PATH="/absolute/path/to/your/test-corpus"
: "${INGEST_PATH:?Set INGEST_PATH to a file/dir to ingest}"

make ingest f="$INGEST_PATH"
```

**Expected assertions (pass):**
- Command exits 0.
- Neo4j graph counts increase (see B.5.3).

**Important note:**
- This Makefile ingest path is legacy and may not include the full FancyRAG QA gating step (depending on implementation). Treat smoke tests and/or the code pipeline path as the QA gate unless you have confirmed otherwise.

---

### B.5.2 Canonical code ingestion path (recommended) — **UNVERIFIED CLI flags**

**Context:** Host

```bash
set -euo pipefail

# Discovery step 1: inspect CLI usage/flags
uv run python scripts/kg_build.py --help

# Discovery step 2: once flags are known, run an ingestion against a fixture.
# Example shape (FLAGS UNVERIFIED):
# uv run python scripts/kg_build.py --source "$INGEST_PATH" --profile markdown --qa
```

**Expected assertions (pass):**
- Ingestion completes successfully.
- QA evaluation runs and returns PASS (or exits non-zero and triggers rollback on FAIL).
- QA report artifacts are written (see B.6.2).

**Why UNVERIFIED:**
- The baseline documents confirm this script exists and is intended to be canonical, but do not provide the exact CLI flags in the excerpts.

---

### B.5.3 Basic ingestion verification queries (global)

**Context:** Container (Neo4j)

```bash
set -euo pipefail

# Load env if available (for credentials)
if [ -f .env.local ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

docker compose exec -T neo4j cypher-shell \
  -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" -d "${NEO4J_DATABASE:-neo4j}" \
  "MATCH (d:Document) RETURN count(d) AS documents;
   MATCH (c:Chunk) RETURN count(c) AS chunks;
   MATCH (:Document)-[r:HAS_CHUNK]->(:Chunk) RETURN count(r) AS has_chunk_rels;
   MATCH (c:Chunk) WHERE c.embedding IS NULL OR size(c.embedding)=0 RETURN count(c) AS missing_embeddings;"
```

**Expected assertions (pass):**
- `documents > 0` and `chunks > 0` (for any non-empty ingestion).
- `has_chunk_rels >= chunks` is typical if every chunk is linked (exact relationship count depends on model).
- `missing_embeddings == 0` for production ingestion; for smoke stack or partial ingests, confirm expected behavior.

---

## B.6 Run QA gating verification

### B.6.1 QA gating via smoke tests (recommended baseline gate)

**Context:** Host

```bash
set -euo pipefail

make smoke
```

**Expected assertions (pass):**
- Smoke stack comes up and `pytest -q tests/smoke/test_stack.py` exits 0.
- This provides a deterministic end-to-end regression gate using the repo’s smoke compose stack.

---

### B.6.2 QA report artifact presence + scrubbing check (safe)

**Context:** Host

```bash
set -euo pipefail

# Locate QA reports produced by ingestion (path may vary; this is robust).
REPORT_JSON="$(find . -type f -name 'quality_report.json' -path '*ingestion-qa-report*' | sort | tail -n 1 || true)"
REPORT_MD="$(find . -type f -name 'quality_report.md'   -path '*ingestion-qa-report*' | sort | tail -n 1 || true)"

if [ -z "${REPORT_JSON}" ] || [ -z "${REPORT_MD}" ]; then
  echo "UNVERIFIED: QA report artifacts not found."
  echo "Next steps:"
  echo "  1) Confirm you ran the code ingestion path that triggers perform_qa()."
  echo "  2) Search for alternate report directory names:"
  echo "     find . -maxdepth 5 -type f -name '*report*.json' -o -name '*report*.md'"
  exit 1
fi

echo "Found QA artifacts:"
echo "  JSON: ${REPORT_JSON}"
echo "  MD:   ${REPORT_MD}"

# Scrubbing scan: fail if obvious secret patterns appear (does not print matches).
python - <<PY
import pathlib, re, sys

paths = [pathlib.Path(r"${REPORT_JSON}"), pathlib.Path(r"${REPORT_MD}")]
patterns = [
  re.compile(r"sk-[A-Za-z0-9]{20,}"),            # OpenAI-style keys (heuristic)
  re.compile(r"OPENAI_API_KEY", re.I),
  re.compile(r"NEO4J_PASSWORD", re.I),
  re.compile(r"-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----"),
]

bad = []
for p in paths:
  data = p.read_text(errors="ignore")
  for pat in patterns:
    if pat.search(data):
      bad.append((str(p), pat.pattern))

if bad:
  print("FAIL: potential secret material found in QA artifacts (pattern match only):")
  for f, pat in bad:
    print(f"  - file={f} pattern={pat}")
  sys.exit(1)

print("OK: QA artifacts present and no obvious secret patterns detected")
PY
```

**Expected assertions (pass):**
- A `quality_report.json` and `quality_report.md` exist (latest run).
- Scrubbing scan prints OK and exits 0.

---

### B.6.3 QA failure + rollback behavior (deep check) — **UNVERIFIED harness**

**Context:** Host + Container

This is a deliberate negative test to prove rollback scoping. The baseline confirms `rollback_ingest(...)` exists and is triggered on QA failure in the code pipeline, but the exact “how to force fail deterministically” harness is not standardized in Make targets.

**UNVERIFIED discovery steps (do these once; then codify in CI):**
1. Identify the ingestion run key tagging mechanism:
   - Search in repo: `git grep -n "ingest_run_key" src/ fancyrag/ fancryrag/`
2. Identify how to set QA thresholds via CLI/env:
   - `uv run python scripts/kg_build.py --help`
   - Search: `git grep -n "QaThresholds" src/ fancyrag/`
3. Create a deterministic failure mode:
   - Example options: set thresholds to 0 and inject a known missing-embedding chunk, or temporarily point embedder at a non-working endpoint in a throwaway DB.

**Expected assertions (pass):**
- QA fails (non-zero exit).
- Rollback removes only nodes/relationships from the failing run (previous ingests remain).

---

## B.7 Start MCP / retrieval service

### B.7.1 MCP via docker-compose (baseline)

**Context:** Host

```bash
set -euo pipefail

# If already started, this is safe to re-run.
make up
```

**Expected assertions (pass):**
- MCP is running and reachable at the published port (verify via B.8 contract checks).

---

### B.7.2 MCP as a host-run process (optional)

**Context:** Host

```bash
set -euo pipefail

# Runs the server directly (useful for debugging outside containers).
# NOTE: this will block the terminal; run in a separate shell if needed.
uv run python servers/mcp_hybrid_google.py
```

**Expected assertions (pass):**
- Server logs show startup.
- `/mcp/health` returns ok (B.8.1).

---

## B.8 Contract checks (search / fetch / health)

**Context:** Host

> These checks validate the public MCP contract:
> - `GET /mcp/health`
> - `POST /mcp/search` with `query`, `top_k`, `effective_search_ratio`
> - `POST /mcp/fetch` with `element_id`
>
> **Auth note:** If `MCP_AUTH_REQUIRED=true`, requests may require OAuth or a static token. For local contract testing, set `MCP_AUTH_REQUIRED=false` in `.env.local` (recommended) or supply the required auth header (header format may be UNVERIFIED).

### B.8.1 Health check

```bash
set -euo pipefail

MCP_BASE_URL="${MCP_BASE_URL:-http://localhost:22012}"

curl -sS "${MCP_BASE_URL}/mcp/health" | python - <<'PY'
import json, sys
data = json.load(sys.stdin)
assert data.get("status") == "ok", f"unexpected health payload: {data}"
print("OK: /mcp/health")
PY
```

**Expected assertions (pass):**
- Response JSON includes `{"status":"ok"}`.

---

### B.8.2 Search → schema assertions (no secret leakage)

```bash
set -euo pipefail

MCP_BASE_URL="${MCP_BASE_URL:-http://localhost:22012}"

# Minimal query; adjust as needed.
REQ='{"query":"test query","top_k":2,"effective_search_ratio":1}'

curl -sS -X POST "${MCP_BASE_URL}/mcp/search" \
  -H 'Content-Type: application/json' \
  -d "${REQ}" \
| python - <<'PY'
import json, sys

data = json.load(sys.stdin)

# Top-level contract fields
for k in ["query","top_k","effective_search_ratio","results"]:
  assert k in data, f"missing top-level field: {k}"

assert isinstance(data["results"], list), "results must be a list"
if data["results"]:
  r0 = data["results"][0]
  for k in ["text","metadata","score","score_vector","score_fulltext","semantic_nodes","semantic_relationships"]:
    assert k in r0, f"missing result field: {k}"
  assert isinstance(r0["metadata"], dict), "metadata must be object"
  assert "embedding" not in r0["metadata"], "metadata must not include embedding"
  assert "element_id" in r0["metadata"], "metadata must include element_id"
  assert "labels" in r0["metadata"], "metadata must include labels"

print("OK: /mcp/search contract")
PY
```

**Expected assertions (pass):**
- Response contains required top-level fields.
- Each result contains required fields, and `metadata` excludes `embedding`.

---

### B.8.3 Fetch round-trip using `element_id` from search

```bash
set -euo pipefail

MCP_BASE_URL="${MCP_BASE_URL:-http://localhost:22012}"

ELEMENT_ID="$(curl -sS -X POST "${MCP_BASE_URL}/mcp/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"test query","top_k":1,"effective_search_ratio":1}' \
| python - <<'PY'
import json, sys
data=json.load(sys.stdin)
if not data.get("results"):
  print("")
  raise SystemExit(0)
print(data["results"][0]["metadata"].get("element_id",""))
PY
)"

if [ -z "${ELEMENT_ID}" ]; then
  echo "UNVERIFIED: no results returned from /mcp/search; cannot run fetch round-trip"
  exit 1
fi

curl -sS -X POST "${MCP_BASE_URL}/mcp/fetch" \
  -H 'Content-Type: application/json' \
  -d "{\"element_id\":\"${ELEMENT_ID}\"}" \
| python - <<'PY'
import json, sys
data = json.load(sys.stdin)

assert data.get("element_id"), "missing element_id echo"
assert data.get("found") in [True, False], "found must be boolean"
assert data["found"] is True, f"expected found=true, got {data}"
assert "metadata" in data and isinstance(data["metadata"], dict), "missing metadata"
assert "embedding" not in data["metadata"], "fetch metadata must not include embedding"
print("OK: /mcp/fetch contract")
PY
```

**Expected assertions (pass):**
- `found=true`
- `metadata` present and does not include `embedding`

---

### B.8.4 Invalid parameter behavior (optional hardening check)

**Context:** Host

```bash
set -euo pipefail

MCP_BASE_URL="${MCP_BASE_URL:-http://localhost:22012}"

# top_k must be > 0; expect a non-2xx response.
HTTP_CODE="$(curl -sS -o /dev/null -w '%{http_code}' \
  -X POST "${MCP_BASE_URL}/mcp/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"x","top_k":0,"effective_search_ratio":1}')"

echo "HTTP ${HTTP_CODE}"
test "${HTTP_CODE}" -ge 400
```

**Expected assertions (pass):**
- HTTP status is >= 400.

**UNVERIFIED detail:**
- Exact status code and error body are not standardized in the baseline excerpts; if you need strict 400 semantics, enforce with a contract test once error handling is pinned.

---

## B.9 Run tests / smoke

### B.9.1 Smoke test (authoritative end-to-end gate)

**Context:** Host

```bash
set -euo pipefail

make smoke
```

**Expected assertions (pass):**
- Exits 0.
- Smoke stack health is good.
- No failing tests.

---

### B.9.2 Smoke logs (on failure)

**Context:** Host

```bash
set -euo pipefail

make smoke-logs
```

**Expected assertions (pass):**
- Logs are available for diagnosis.

---

## B.10 Secret / log safety checks

### B.10.1 Container image secret scanning (baseline CI parity)

**Context:** Host

```bash
set -euo pipefail

make scan-image
```

**Expected assertions (pass):**
- Exits 0.
- No medium/high/critical secret findings.

---

### B.10.2 Artifact/log scrubbing spot-check (safe, no values printed)

**Context:** Host

```bash
set -euo pipefail

# Scan common artifact/log locations for secret-like patterns WITHOUT printing matches.
python - <<'PY'
import os, re, sys, pathlib

ROOT = pathlib.Path(".")
patterns = [
  re.compile(r"sk-[A-Za-z0-9]{20,}"),
  re.compile(r"-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----"),
  re.compile(r"NEO4J_PASSWORD\s*=", re.I),
  re.compile(r"OPENAI_API_KEY\s*=", re.I),
]

# Limit scan to known artifact-y areas to avoid scanning huge repos.
paths = []
for p in ["artifacts", "logs", "ingestion-qa-report"]:
  d = ROOT / p
  if d.exists():
    paths.append(d)

if not paths:
  print("OK: no artifact/log directories found to scan (nothing to do)")
  sys.exit(0)

bad = []
for base in paths:
  for file in base.rglob("*"):
    if not file.is_file():
      continue
    # Skip very large files
    try:
      if file.stat().st_size > 5_000_000:
        continue
      data = file.read_text(errors="ignore")
    except Exception:
      continue
    for pat in patterns:
      if pat.search(data):
        bad.append((str(file), pat.pattern))

if bad:
  print("FAIL: potential secret material found (pattern match only):")
  for f, pat in bad[:50]:
    print(f"  - file={f} pattern={pat}")
  sys.exit(1)

print("OK: artifact/log scan found no obvious secret patterns")
PY
```

**Expected assertions (pass):**
- Prints OK and exits 0.

---

## B.11 Teardown

**Context:** Host

```bash
set -euo pipefail

make down
```

**Expected assertions (pass):**
- Stack stops cleanly.

---

# C) Troubleshooting Matrix

| Symptom | Likely cause | Next steps (with anchors) |
|---|---|---|
| `make up` fails immediately | Docker daemon not running, compose not installed, or port conflicts | 1) `docker info` 2) `docker compose version` 3) `docker compose ps` 4) If port conflict, inspect `docker-compose.yml` port mappings and free ports. |
| `/mcp/health` returns 503 | Neo4j not reachable from MCP (startup lag, wrong credentials, wrong `NEO4J_URI`) | 1) `make logs` (look for Neo4j ready + MCP connectivity errors) 2) Confirm `.env.local` has correct `NEO4J_*` names 3) From container: `docker compose exec -T neo4j cypher-shell ... "RETURN 1"` |
| `/mcp/search` returns 401/403 | Auth required (`MCP_AUTH_REQUIRED=true`) and request lacks OAuth/token | For local testing set `MCP_AUTH_REQUIRED=false` in `.env.local` and restart (`make down && make up`). If using static token: inspect `servers/mcp_hybrid_google.py` auth selection logic (header format may be UNVERIFIED). |
| Vector index script fails with mismatch error | Existing vector index config (dimensions/property/label) doesn’t match current env | 1) Run B.4 `SHOW INDEXES` and compare 2) Fix `EMBEDDING_DIMENSIONS` and/or `INDEX_NAME` in env 3) If intentional change, drop/recreate index (destructive) via `make index` or manual `DROP INDEX ...` + rerun script. |
| Fulltext index exists but search yields poor/no lexical hits | Fulltext index name/label/property mismatch (script may “skip” without validating config) | 1) Use B.4 `SHOW INDEXES` to verify `labelsOrTypes` + `properties` 2) If mismatch, drop/recreate fulltext index manually, then rerun `scripts/create_fulltext_index.py` 3) Consider adding mismatch detection (planned hardening). |
| Smoke tests fail with embedding dimension errors | Smoke embedding stub returns a fixed 3D vector; config expects different dimension | 1) Inspect `docker-compose.smoke.yml` for `EMBEDDING_DIMENSIONS`/embedding model settings 2) Ensure smoke env sets the dimension to 3 3) Re-run `make smoke`. |
| Ingestion completes but `missing_embeddings > 0` | Embedding service down, invalid API key, rate limiting, or pipeline didn’t set embeddings | 1) Check OpenAI/embedding env vars (`OPENAI_API_KEY`, `EMBEDDING_API_BASE_URL`) 2) Inspect ingestion logs 3) If using smoke stub, ensure it’s reachable in smoke stack. |
| `make ingest` succeeds but QA artifacts absent | You used the legacy YAML ingestion path which may not run QA | Run B.6.1 `make smoke` as gate, or switch to the code ingestion path (`scripts/kg_build.py`) once CLI flags confirmed. Locate QA report directory via B.6.2 `find ... quality_report.*`. |
| `/mcp/fetch` returns `{found:false}` for an `element_id` from search | DB changed between calls, or elementId not stable across DB, or parsing bug | 1) Re-run search + fetch round-trip immediately 2) Verify Neo4j is not being reset between calls (ingest/reset scripts) 3) Inspect `fancryrag/mcp/runtime.py` fetch query and `elementId()` usage. |
| `make scan-image` fails | Trivy finds secrets in image layers or scanning tooling issues | 1) Review Trivy output 2) Remove secrets from build context / layers (avoid copying `.env.local` into image) 3) Rebuild image and rescan. |
| Import errors: `ModuleNotFoundError: fancyrag` or `fancryrag` | Package rename drift or incorrect `src/` layout | Run B.1.1 import sanity. If neither imports, inspect `pyproject.toml` package config and `src/` directory layout. PR1 in the refactor plan explicitly targets this normalization. |

---

## Appendix: “UNVERIFIED” discovery playbook

When a command/flag is marked **UNVERIFIED**, resolve it by discovery (do not guess):

1. List Make targets:
   ```bash
   make -n help 2>/dev/null || true
   make -qp | awk -F':' '/^[a-zA-Z0-9][^$#\/\t=]*:([^=]|$)/{print $1}' | sort -u
   ```
2. Discover CLI flags:
   ```bash
   uv run python scripts/kg_build.py --help
   ```
3. Find symbol or env usage:
   ```bash
   git grep -n "INGEST_PATH\|ingest_run_key\|QaThresholds\|HYBRID_RETRIEVAL_QUERY_PATH"
   ```

