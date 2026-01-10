# Decision log: OpenAI Responses API alignment

- Use Responses API (`responses.create`) for chat to match OpenAI guidance.
- Keep default chat model at `gpt-5-mini` with reasoning effort set to `minimal`.
- Set semantic extraction temperature default to `0.3`.
- Leave embeddings on local server; no migration to OpenAI embeddings.
- Make QDRANT_* optional by commenting in `.env.example`.
- Normalize secret placeholders to `YOUR_*` while allowing safe local defaults.
