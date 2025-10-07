# GraphRAG Quickstart (Project Workspace)

This is the one-page handoff for running FancyRAG’s local GraphRAG workflows. Use it alongside the [official docs index](./OFFICIAL_LINKS.md) whenever you need deeper detail.

## 1. Bootstrap the Workspace
```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
```
This installs the pinned dependencies (including optional extras) and activates the project virtualenv.

## 2. Configure Environment Variables
Copy the template and fill in service credentials before running CLI commands:
```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY, NEO4J_*, QDRANT_* values
```
Keep the real `.env` file uncommitted.

## 3. Verify the Environment
Run the workspace diagnostics to confirm imports, dependency versions, and git metadata:
```bash
PYTHONPATH=src python -m cli.diagnostics workspace --write-report
```
Reports land in `artifacts/environment/versions.json` (see README for sharing guidance).

## 4. Validate OpenAI Connectivity (Optional but Recommended)
```bash
PYTHONPATH=src python -m cli.diagnostics openai-probe
```
This writes sanitized metrics to `artifacts/openai/`. Use `--skip-live` if the API is intentionally blocked.

## 5. Run the Stack Automation (Story 5.3)
Once Story 5.3 ships, the Make targets expose a one-command onboarding flow:

```bash
make service-run                       # bootstrap → ingest → export → eval → teardown
FANCYRAG_PRESET=full make service-run  # override the preset profile
make service-rollback                  # rollback partial ingest, keep volumes
make service-reset                     # rollback + destroy volumes when you need a clean slate
make service-rollback                  # undo a failed run without destroying volumes
```

The targets wrap the pipeline helpers plus `scripts/check_local_stack.sh`. Artifacts land under `artifacts/local_stack/`, and the smoke preset remains the default for CI and local validation. When you need manual control, fall back to the scripted steps in the README.

## 6. Next Steps
- Review the [official documentation links](./OFFICIAL_LINKS.md) for the feature you plan to use.
- For ingestion, follow the Knowledge Graph Builder guide.
- For retrieval and generation, align with the retriever/GraphRAG references.

Need more? Ping the team or extend this quickstart with scenario-specific scripts.
