# Risk checklist: OpenAI Responses API alignment

- [x] OpenAI error signature mismatch (httpx.Response required). Mitigated by updating tests to use httpx responses.
- [x] Probe clients using old `_chat` hook. Mitigated by switching tests to `_response`.
- [x] Env template secrets using unsafe placeholders. Mitigated by `YOUR_*` placeholders.
- [ ] gpt-5-mini rejects `temperature`; semantic extraction now sends `temperature=0.3` and may error in production.
- [ ] Behavior drift in OpenAI response parsing (`output_text` vs `output`). Monitor integration tests.
- [ ] Env template safe defaults may need documentation update if values change.

Notes:
- No auth changes, no DB schema changes.
- No new external endpoints.
