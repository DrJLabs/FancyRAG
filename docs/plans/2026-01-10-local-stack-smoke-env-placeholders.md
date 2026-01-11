## Goal
- Prevent placeholder `.env.example` values from causing `local-stack-smoke` failures while keeping CI and local workflows deterministic.

## Assumptions / constraints
- `.env.example` remains documentation-oriented and may keep `YOUR_*` placeholders.
- No secrets are written to repo files; CI injects secrets via GitHub Actions.
- Changes should be scoped to the smoke workflow/scripts and avoid altering runtime app behavior.

## Research (current state)
- Relevant files/entrypoints:
  - `.github/workflows/local-stack-smoke.yml` (copies `.env.example`, exports every key to `GITHUB_ENV`)
  - `.env.example` (contains `YOUR_*` placeholders for embedding and MCP settings)
  - `src/config/settings.py` (validates `EMBEDDING_API_BASE_URL` as http/https)
  - `tests/integration/local_stack/test_minimal_path_smoke.py` (loads `.env` via `load_project_dotenv`)
  - `src/fancyrag/utils/env.py` (loads `.env` from repo root in CI container)
- Existing patterns to follow:
  - CI already mutates `.env` for OpenAI settings in `local-stack-smoke.yml`.
  - Settings validation treats invalid URLs as hard errors.

## Analysis
### Options
1) Extend the workflow’s `.env` rewrite to sanitize all `YOUR_*` placeholders and only export a whitelist of required keys.
2) Add a CI-only env file (e.g., `.env.ci`) with safe defaults and point the workflow + tests at it.
3) Modify settings loaders to ignore `YOUR_*` placeholder values globally.

### Decision
- Chosen: Option 1 (workflow sanitize + whitelist export).
- Why: minimal surface area, avoids runtime behavior changes, and keeps `.env.example` as documentation while protecting CI from placeholders.

### Risks / edge cases
- Missing a required env key in the allowlist could break smoke tests; needs a clear inventory of required keys.
- Sanitizing `YOUR_*` values might inadvertently clear legitimate values if someone uses that prefix intentionally.
- `.env` and `GITHUB_ENV` could drift if the workflow logic diverges.

### Open questions
- None; proceed with the allowlist + placeholder sanitization approach.

## Q&A (answer before implementation)
- None.

## Implementation plan
1) Inventory the exact env keys consumed by the smoke path (OpenAI, embedding, Neo4j, Qdrant, and any script-specific flags).
2) Update `.github/workflows/local-stack-smoke.yml` to:
   - sanitize `YOUR_*` placeholders for all exported keys (not just embedding vars),
   - export only the allowlisted keys to `GITHUB_ENV`,
   - keep the OpenAI secret injection and embed base URL fallback.
3) Add a short local preflight note or helper command in docs to mirror the CI env sanitization for developers (optional but aligns with the issue recommendation).
4) Re-run the local smoke workflow or targeted containerized smoke to validate.

## Tests to run
- `scripts/check_local_stack.sh --config`
- `docker compose -f docker-compose.neo4j-qdrant.yml run --rm --no-deps smoke-tests ... pytest tests/integration/local_stack/test_minimal_path_smoke.py`
