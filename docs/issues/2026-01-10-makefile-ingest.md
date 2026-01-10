# Makefile ingest target should align with Responses API pipeline

## Summary
The current `make ingest` target runs `tools/run_pipeline.py`, which uses
`pipelines/kg_ingest.yaml` and a `gpt-4o-mini` LLM config. This bypasses the
Responses API path and `gpt-5-mini` defaults added to the FancyRAG pipeline
(`scripts/kg_build.py`).

## Recommendation
Update the Makefile so `make ingest` exercises the same code path used in local
MCP + semantic enrichment workflows.

Two safe options:
1) Replace `make ingest` with the FancyRAG pipeline entrypoint:
   `PYTHONPATH=src python scripts/kg_build.py --enable-semantic ...`
2) Keep `make ingest` as-is and add a new target (e.g. `ingest-semantic`) that
   uses `scripts/kg_build.py --enable-semantic` to validate the Responses API
   configuration.

## Impact
- Ensures local ingestion tests the new `gpt-5-mini` Responses API path.
- Keeps semantic enrichment behavior consistent with MCP retrieval.
- Avoids confusion between the YAML pipeline (legacy) and the FancyRAG pipeline.

## References
- `Makefile` (ingest target)
- `tools/run_pipeline.py`
- `pipelines/kg_ingest.yaml`
- `scripts/kg_build.py`
