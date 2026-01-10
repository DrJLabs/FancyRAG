# local-stack-smoke should not export placeholder embedding envs

## Summary
The local-stack smoke workflow copies `.env.example` to `.env` and exports every
non-commented value to the CI environment. Placeholder entries such as
`EMBEDDING_API_BASE_URL="YOUR_EMBEDDING_API_BASE_URL"` then get validated by
`OpenAISettings` and can fail the smoke test when the values are not a valid
http(s) URL. This makes the job brittle and can appear flaky when the template
values change or when reruns reuse different env content.

## Recommendation
- Add a CI-specific env template or sanitize placeholder values before exporting
  (e.g., clear `YOUR_*` entries for embedding-related variables).
- Limit exported variables to the ones required by the smoke test (OpenAI +
  Neo4j), instead of exporting the full `.env` file.
- Add a local preflight command in docs (or a script) that mirrors the CI smoke
  invocation so issues can be caught before CI.

## Impact
- Makes `local-stack-smoke` deterministic and less brittle.
- Avoids validation failures caused by placeholder values in `.env.example`.
- Improves developer confidence by aligning local checks with CI.

## References
- `.github/workflows/local-stack-smoke.yml`
- `.env.example`
- `tests/integration/local_stack/test_minimal_path_smoke.py`
- `src/config/settings.py`
