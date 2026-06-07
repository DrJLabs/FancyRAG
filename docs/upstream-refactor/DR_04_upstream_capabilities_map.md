# DR_04_upstream_capabilities_map

## Metadata
- Upstream repo: https://github.com/neo4j/neo4j-graphrag-python.git
- Analyzed branch: default (main)
- HEAD commit SHA: UNVERIFIED (not resolved in this run)
- Custom repo (contextual): https://github.com/DrJLabs/FancyRAG.git
- Scope: Latest default branch only; no historical version diffing

---

# Neo4j GraphRAG — Upstream Capability Map (Deep Research)

This document inventories the upstream **neo4j-graphrag-python** library as of the latest default branch HEAD.  
It is intended as a canonical reference for FancyRAG alignment, replacement decisions, and long‑term maintenance risk management.

All claims below are grounded in upstream source evidence. Experimental vs stable boundaries, customization surfaces, and drift risks are explicitly called out.

---

## 1. Upstream Capability Table

| Capability | Module Path(s) | Key Symbols (Classes / Functions) | Maturity | Configuration Expectations | Official Usage References |
|-----------|----------------|----------------------------------|----------|----------------------------|---------------------------|
| Knowledge‑Graph Pipeline Builder | `neo4j_graphrag.experimental.pipeline.pipeline`<br>`neo4j_graphrag.experimental.pipeline.kg_builder` | `Pipeline`, `PipelineRunner`, `SimpleKGPipeline`【48†L420-L428】 | **Experimental** | Requires Neo4j APOC plugin【48†L417-L424】. Supports text and PDF ingestion (`from_pdf=True`)【22†L121-L129】. Schema-driven KG construction replaces deprecated `entities` / `relations` parameters【20†L85-L94】. Accepts injectable splitters, loaders, extractors, and KG writers【22†L159-L168】. | Docs: Knowledge Graph Builder【48†L420-L428】; Examples: `examples/pipeline/*` |
| Retrievers — Vector / Hybrid / Cypher | `neo4j_graphrag.retrievers.*` | `VectorRetriever`【29†L271-L276】, `VectorCypherRetriever`【30†L1-L4】, `HybridRetriever`【31†L70-L78】, `HybridCypherRetriever`【32†L1-L4】, `Text2CypherRetriever`【33†L118-L126】 | **Stable** | Require Neo4j driver + index names. Vector retrievers require a vector index name; hybrid retrievers require vector + fulltext index names【31†L93-L101】【31†L115-L123】. Embedder injection required for text queries【29†L235-L243】. `Text2CypherRetriever` requires an LLM and optional schema override【33†L127-L136】. | Docs: RAG Retriever Guide【27†L423-L431】; Examples: `examples/retrieve/*` |
| External Vector DB Retrievers | `neo4j_graphrag.retrievers.external.*` | `WeaviateNeo4jRetriever`, `PineconeNeo4jRetriever`, `QdrantNeo4jRetriever`【27†L449-L458】 | **Stable (Adapters)** | Require external vector DB client plus Neo4j driver. Mapping between external vector IDs and Neo4j nodes is required【46†L97-L105】. Used when Neo4j is not primary vector store. | Docs: External Retriever section【27†L449-L458】 |
| Embeddings | `neo4j_graphrag.embeddings.*` | `OpenAIEmbeddings`, `AzureOpenAIEmbeddings`, `CohereEmbeddings`, `AnthropicEmbeddings`, `VertexAIEmbeddings`, `MistralAIEmbeddings`, `OllamaEmbeddings`, `SentenceTransformerEmbeddings`【27†L514-L522】 | **Stable** | Require model identifiers and credentials. Vector dimension must match Neo4j vector index definition【48†L497-L505】. All embedders implement the `Embedder` interface【46†L45-L53】. | Docs: Embedders section【27†L514-L522】 |
| Generation / GraphRAG Loop | `neo4j_graphrag.generation.graphrag`<br>`neo4j_graphrag.generation.prompts` | `GraphRAG`, `RagTemplate`, `Text2CypherTemplate`, `MessageHistory`【24†L60-L69】 | **Stable** | Requires Retriever + LLM injection【24†L93-L101】. Supports conversation history summarization【24†L167-L175】. Prompt templates are injectable and overrideable【27†L386-L394】. | Docs: GraphRAG Usage【24†L65-L73】 |

---

## 2. Customization Mechanisms (Authoritative)

Upstream GraphRAG is intentionally **class‑injected and interface‑driven**, not config‑file driven.

### 2.1 LLM Customization
- All LLM providers implement `LLMInterface`【20†L69-L71】.
- Built‑in adapters include OpenAI, Azure OpenAI, Cohere, Anthropic, Vertex AI, Ollama.
- Custom LLMs can be injected directly into `GraphRAG`, `Text2CypherRetriever`, or pipelines.
- Rate limiting is handled via pluggable `RateLimitHandler` implementations【27†L353-L362】.

### 2.2 Embedding Customization
- All embedders subclass `Embedder`【46†L45-L53】.
- Any embedder instance can be injected into retrievers or KG pipelines.
- No hard dependency on OpenAI — local or self‑hosted embeddings are first‑class.

### 2.3 Pipeline Component Injection
- Experimental pipeline architecture allows injection of:
  - Text splitters
  - PDF loaders
  - Entity / relation extractors
  - KG writers
- Injection occurs at construction time via component parameters or `Pipeline.add_component()`【22†L159-L168】.
- No YAML or JSON config layer; composition is Python‑native.

### 2.4 Prompt and Query Control
- `RagTemplate` can be subclassed or replaced【27†L386-L394】.
- `Text2CypherRetriever` accepts fully custom prompt strings【33†L129-L137】.
- Schema used for Cypher generation can be overridden or auto‑derived.

---

## 3. Compatibility and Drift Notes

### 3.1 Experimental Surface Risk
- All code under `neo4j_graphrag.experimental.*` is explicitly unstable【44†L388-L396】.
- KG pipeline APIs have already undergone breaking refactors (schema unification)【20†L99-L107】.
- Expect further breaking changes until pipelines graduate to stable.

### 3.2 Deprecations and Behavioral Drift
- Deprecated KG pipeline arguments remain but should not be used【20†L99-L107】.
- `GraphRAG.search()` default `return_context` behavior will change in a future release【51†L7-L15】.
- FancyRAG should explicitly set this flag to avoid silent behavior changes.

### 3.3 Version Pinning Guidance
- Rapid release cadence (v1.0 → v1.12 in ~1 year).
- Recommendation:
  - Pin minor version (e.g., `~=1.12.0`)
  - Audit release notes before upgrading
  - Treat experimental modules as vendored dependencies if stability is required

### 3.4 External Dependency Drift
- LLM SDKs and vector DB clients evolve independently.
- GraphRAG abstracts most APIs, but FancyRAG custom adapters should be regression‑tested after upgrades.

---

## 4. Recommended FancyRAG Alignment Strategy

- **Retrievers + GraphRAG**: Safe to align directly with upstream.
- **Embeddings / LLM adapters**: Prefer upstream interfaces; wrap only when adding policy or telemetry.
- **KG Pipelines**: Treat as volatile; consider isolation or fork if relied upon heavily.
- **Prompting**: Use subclassed templates to avoid upstream prompt churn.

---

## 5. UNVERIFIED ITEMS

- Upstream HEAD commit SHA not resolved in this run.
- Full changelog diff not exhaustively reviewed.

### Verification Steps
1. Visit https://github.com/neo4j/neo4j-graphrag-python.git
2. Record default branch HEAD commit SHA
3. Pin SHA in FancyRAG dependency lockfile
