# OpenAI Responses API alignment plan

## Scope
- Use OpenAI Responses API for chat completions.
- Default model: gpt-5-mini with reasoning effort set to minimal; keep embeddings on local model.
- Keep Qdrant configuration optional in env template.
- Update tests to use OpenAI error signatures and responses output shape.

## Architecture notes
- Client uses `responses.create` and reads `output_text` or `output` list.
- Embeddings remain unchanged (local base URL).
- No database schema changes, no new endpoints, no auth changes.

## Implementation steps
1. Update client + stubs for Responses API signature and usage mapping.
2. Update tests to build `httpx.Response` for OpenAI error types.
3. Normalize env template placeholders for secrets; allow safe defaults.
4. Run targeted unit tests.

## Assumptions
- No new security or SLA requirements beyond existing defaults.
