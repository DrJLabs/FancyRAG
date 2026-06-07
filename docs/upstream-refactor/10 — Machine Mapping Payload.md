{
  "metadata": {
    "generated_at": "2026-01-12T23:03:56Z",
    "canonical_shas": {
      "fancyrag": {
        "repo": "DrJLabs/FancyRAG",
        "sha": "0ff7b46717da4310b1626b0756551b7bd175d81b"
      },
      "upstream": {
        "repo": "neo4j/neo4j-graphrag-python",
        "sha": "9e6b58827caa642539c8cb15dcde36c3e4d6d2c6"
      }
    },
    "input_shas": {
      "dr05_baseline": {
        "fancyrag_sha": "0ff7b46717da4310b1626b0756551b7bd175d81b",
        "upstream_sha": "62bbea3fe3dca4c6441ec037bfdcb9df6591a798",
        "note": "DR_05 mapping matrix upstream SHA differs from canonical upstream SHA in baseline."
      }
    },
    "inputs": {
      "baseline_file": "CANONICAL_BASELINE_00-04.md",
      "mapping_matrix_file": "DR_05_symbol_mapping_matrix.md",
      "pr_plan_file": "DR_06 Phased Refactor Plan.md"
    },
    "artifact": {
      "file_name": "DR_10_machine_mapping_payload.json",
      "sandbox_path": "sandbox:/mnt/data/DR_10_machine_mapping_payload.json"
    },
    "evidence": [
      {
        "repo": "local-docs",
        "file_path": "CANONICAL_BASELINE_00-04.md",
        "line_range": "L15-L15",
        "quote": "- **Canonical HEAD SHA:** `0ff7b46717da4310b1626b0756551b7bd175d81b`"
      },
      {
        "repo": "local-docs",
        "file_path": "CANONICAL_BASELINE_00-04.md",
        "line_range": "L22-L22",
        "quote": "- **Canonical HEAD SHA:** `9e6b58827caa642539c8cb15dcde36c3e4d6d2c6`"
      },
      {
        "repo": "local-docs",
        "file_path": "DR_05_symbol_mapping_matrix.md",
        "line_range": "L6-L6",
        "quote": "- **FancyRAG baseline:** commit `0ff7b46717da4310b1626b0756551b7bd175d81b` (as per CANONICAL_BASELINE_00-04)."
      },
      {
        "repo": "local-docs",
        "file_path": "DR_05_symbol_mapping_matrix.md",
        "line_range": "L8-L8",
        "quote": "- **Upstream baseline:** commit `62bbea3fe3dca4c6441ec037bfdcb9df6591a798` (tag v1.10)."
      }
    ]
  },
  "mapping": [
    {
      "id": "fancyrag/config.py::ConfigurationError",
      "fancyrag_symbol": "ConfigurationError",
      "fancyrag_file": "fancyrag/config.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L319-L319",
          "quote": "| `ConfigurationError`                        | `fancyra...iguration` | Configuration | *null* (no equivalent) |"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "ConfigurationError",
          "neo4j_graphrag ConfigurationError"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/config.py::Neo4jSettings",
      "fancyrag_symbol": "Neo4jSettings",
      "fancyrag_file": "fancyrag/config.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L320-L320",
          "quote": "| `Neo4jSettings` (FancyRAG version) | `fancyra...nfiguration | `*neo4j_graphrag*` – *none* (similarly named) |"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "Neo4jSettings",
          "neo4j_graphrag Neo4jSettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/config.py::IndexSettings",
      "fancyrag_symbol": "IndexSettings",
      "fancyrag_file": "fancyrag/config.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L321-L321",
          "quote": "| `IndexSettings`                            | `fancyra...iguration` | Configuration | *null* |"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "IndexSettings",
          "neo4j_graphrag IndexSettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/config.py::EmbeddingSettings",
      "fancyrag_symbol": "EmbeddingSettings",
      "fancyrag_file": "fancyrag/config.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L322-L322",
          "quote": "| `EmbeddingSettings`                        | `fancyra...iguration` | Configuration | *null* |"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "EmbeddingSettings",
          "neo4j_graphrag EmbeddingSettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/config.py::OAuthSettings",
      "fancyrag_symbol": "OAuthSettings",
      "fancyrag_file": "fancyrag/config.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L330-L330",
          "quote": "- OAuthSettings"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L339-L339",
          "quote": "- fancyrag/config.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "OAuthSettings",
          "neo4j_graphrag OAuthSettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/config.py::ServerSettings",
      "fancyrag_symbol": "ServerSettings",
      "fancyrag_file": "fancyrag/config.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L331-L331",
          "quote": "- ServerSettings"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L340-L340",
          "quote": "- fancyrag/config.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "ServerSettings",
          "neo4j_graphrag ServerSettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/config.py::QuerySettings",
      "fancyrag_symbol": "QuerySettings",
      "fancyrag_file": "fancyrag/config.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L332-L332",
          "quote": "- QuerySettings"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L341-L341",
          "quote": "- fancyrag/config.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "QuerySettings",
          "neo4j_graphrag QuerySettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/config.py::AppConfig",
      "fancyrag_symbol": "AppConfig",
      "fancyrag_file": "fancyrag/config.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L333-L333",
          "quote": "- AppConfig"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L342-L342",
          "quote": "- fancyrag/config.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "AppConfig",
          "neo4j_graphrag AppConfig"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "config/utils.py::parse_boolean",
      "fancyrag_symbol": "parse_boolean",
      "fancyrag_file": "config/utils.py",
      "category": "utilities",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L359-L359",
          "quote": "- parse_boolean"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L368-L368",
          "quote": "- config/utils.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "parse_boolean",
          "neo4j_graphrag parse_boolean"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "config/utils.py::format_allowed_boolean_values",
      "fancyrag_symbol": "format_allowed_boolean_values",
      "fancyrag_file": "config/utils.py",
      "category": "utilities",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L360-L360",
          "quote": "- format_allowed_boolean_values"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L369-L369",
          "quote": "- config/utils.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "format_allowed_boolean_values",
          "neo4j_graphrag format_allowed_boolean_values"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "config/settings.py::OpenAISettings",
      "fancyrag_symbol": "OpenAISettings",
      "fancyrag_file": "config/settings.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L361-L361",
          "quote": "- OpenAISettings"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L370-L370",
          "quote": "- config/settings.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "OpenAISettings",
          "neo4j_graphrag OpenAISettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "config/settings.py::Neo4jSettings (duplicate in settings.py)",
      "fancyrag_symbol": "Neo4jSettings (duplicate in settings.py)",
      "fancyrag_file": "config/settings.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "ADAPT",
      "decision_rationale": "ADAPT: Keep local symbol but adjust to align with canonical interfaces and remove duplication.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L362-L362",
          "quote": "- Neo4jSettings (duplicate in settings.py)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L371-L371",
          "quote": "- config/settings.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "Neo4jSettings (duplicate in settings.py)",
          "neo4j_graphrag Neo4jSettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "config/settings.py::QdrantSettings",
      "fancyrag_symbol": "QdrantSettings",
      "fancyrag_file": "config/settings.py",
      "category": "config",
      "category_source": "Configuration",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L388-L388",
          "quote": "- QdrantSettings"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L397-L397",
          "quote": "- config/settings.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "QdrantSettings",
          "neo4j_graphrag QdrantSettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "cli/openai_client.py::SharedOpenAIClient",
      "fancyrag_symbol": "SharedOpenAIClient",
      "fancyrag_file": "cli/openai_client.py",
      "category": "utilities",
      "category_source": "LLM/Embeddings",
      "upstream_target": "OpenAILLM (neo4j_graphrag.llm)",
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Prefer upstream OpenAILLM (neo4j_graphrag.llm); remove or significantly reduce local implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L402-L402",
          "quote": "- SharedOpenAIClient"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L406-L406",
          "quote": "- cli/openai_client.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L426-L426",
          "quote": "- OpenAILLM (neo4j_graphrag.llm)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L734-L734",
          "quote": "- SharedOpenAIClient and result wrappers – Replaced by upstream OpenAILLM/OpenAIEmbeddings, removing ~400 lines."
        }
      ]
    },
    {
      "id": "cli/openai_client.py::ChatResult / EmbeddingResult",
      "fancyrag_symbol": "ChatResult / EmbeddingResult",
      "fancyrag_file": "cli/openai_client.py",
      "category": "utilities",
      "category_source": "LLM/Embeddings",
      "upstream_target": null,
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Remove local implementation as redundant; behavior covered by upstream/canonical flow.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L403-L403",
          "quote": "- ChatResult / EmbeddingResult"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L407-L407",
          "quote": "- cli/openai_client.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L734-L734",
          "quote": "- SharedOpenAIClient and result wrappers – Replaced by upstream OpenAILLM/OpenAIEmbeddings, removing ~400 lines."
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "ChatResult / EmbeddingResult",
          "neo4j_graphrag ChatResult"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/embeddings.py::OpenAIClientError",
      "fancyrag_symbol": "OpenAIClientError",
      "fancyrag_file": "fancyrag/embeddings.py",
      "category": "utilities",
      "category_source": "LLM/Embeddings",
      "upstream_target": null,
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Remove local implementation as redundant; behavior covered by upstream/canonical flow.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L434-L434",
          "quote": "- OpenAIClientError"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L435-L435",
          "quote": "fancyrag_file: fancyrag/embeddings.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L733-L733",
          "quote": "- OpenAIClientError exception – Removed to rely on upstream errors."
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "OpenAIClientError",
          "neo4j_graphrag OpenAIClientError"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "cli/telemetry.py::OpenAIMetrics & get_metrics",
      "fancyrag_symbol": "OpenAIMetrics & get_metrics",
      "fancyrag_file": "cli/telemetry.py",
      "category": "utilities",
      "category_source": "LLM/Embeddings",
      "upstream_target": null,
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Remove local implementation as redundant; behavior covered by upstream/canonical flow.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L440-L440",
          "quote": "- OpenAIMetrics & get_metrics"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L441-L441",
          "quote": "fancyrag_file: cli/telemetry.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L732-L732",
          "quote": "- OpenAIMetrics telemetry – Dropped in favor of simpler logging."
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "OpenAIMetrics & get_metrics",
          "neo4j_graphrag OpenAIMetrics"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "cli/sanitizer.py::mask_base_url & scrub_object",
      "fancyrag_symbol": "mask_base_url & scrub_object",
      "fancyrag_file": "cli/sanitizer.py",
      "category": "utilities",
      "category_source": "LLM/Embeddings",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L446-L446",
          "quote": "- mask_base_url & scrub_object"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L447-L447",
          "quote": "fancyrag_file: cli/sanitizer.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "mask_base_url & scrub_object",
          "neo4j_graphrag mask_base_url"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/embeddings.py::RetryingOpenAIEmbeddings",
      "fancyrag_symbol": "RetryingOpenAIEmbeddings",
      "fancyrag_file": "fancyrag/embeddings.py",
      "category": "utilities",
      "category_source": "LLM/Embeddings",
      "upstream_target": "OpenAIEmbeddings (neo4j_graphrag.embeddings)",
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Prefer upstream OpenAIEmbeddings (neo4j_graphrag.embeddings); remove or significantly reduce local implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L452-L452",
          "quote": "- RetryingOpenAIEmbeddings"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L453-L453",
          "quote": "fancyrag_file: fancyrag/embeddings.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L455-L455",
          "quote": "upstream_target (neo4j_graphrag): OpenAIEmbeddings (neo4j_graphrag.embeddings)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L734-L734",
          "quote": "- SharedOpenAIClient and result wrappers – Replaced by upstream OpenAILLM/OpenAIEmbeddings, removing ~400 lines."
        }
      ]
    },
    {
      "id": "fancyrag/splitters/caching_fixed_size.py::CachingSplitterConfig",
      "fancyrag_symbol": "CachingSplitterConfig",
      "fancyrag_file": "fancyrag/splitters/caching_fixed_size.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L465-L465",
          "quote": "| CachingSplitterConfig | fancyrag/splitt...haching_fixed_size.py | Ingestion Pipeline | null (upstream has FixedSizeSplitter) |"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "CachingSplitterConfig",
          "neo4j_graphrag CachingSplitterConfig"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/splitters/caching_fixed_size.py::CachingFixedSizeSplitter",
      "fancyrag_symbol": "CachingFixedSizeSplitter",
      "fancyrag_file": "fancyrag/splitters/caching_fixed_size.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": "FixedSizeSplitter (neo4j_graphrag.experimental.config.splitters)",
      "decision": "WRAP",
      "decision_rationale": "WRAP: Use FixedSizeSplitter (neo4j_graphrag.experimental.config.splitters) as core implementation; keep a thin FancyRAG wrapper for added behavior.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L466-L466",
          "quote": "| CachingFixedSizeSplitter | fancyrag/splitt...haching_fixed_size.py | Ingestion Pipeline | FixedSizeSplitter (neo4j_graphrag.experimental.config.splitters) |"
        }
      ]
    },
    {
      "id": "fancyrag/splitters/caching_fixed_size.py::build_caching_splitter(config)",
      "fancyrag_symbol": "build_caching_splitter(config)",
      "fancyrag_file": "fancyrag/splitters/caching_fixed_size.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L467-L467",
          "quote": "| build_caching_splitter(config) | fancyrag/splitt...haching_fixed_size.py | Ingestion Pipeline | null |"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "build_caching_splitter(config)",
          "neo4j_graphrag build_caching_splitter"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/phases.py::ResolvedSettings",
      "fancyrag_symbol": "ResolvedSettings",
      "fancyrag_file": "fancyrag/kg/phases.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L487-L487",
          "quote": "- ResolvedSettings"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L496-L496",
          "quote": "- fancyrag/kg/phases.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "ResolvedSettings",
          "neo4j_graphrag ResolvedSettings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/phases.py::SourceDiscoveryResult",
      "fancyrag_symbol": "SourceDiscoveryResult",
      "fancyrag_file": "fancyrag/kg/phases.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L488-L488",
          "quote": "- SourceDiscoveryResult"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L497-L497",
          "quote": "- fancyrag/kg/phases.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "SourceDiscoveryResult",
          "neo4j_graphrag SourceDiscoveryResult"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/phases.py::ClientBundle",
      "fancyrag_symbol": "ClientBundle",
      "fancyrag_file": "fancyrag/kg/phases.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L489-L489",
          "quote": "- ClientBundle"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L498-L498",
          "quote": "- fancyrag/kg/phases.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "ClientBundle",
          "neo4j_graphrag ClientBundle"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/phases.py::IngestionArtifacts",
      "fancyrag_symbol": "IngestionArtifacts",
      "fancyrag_file": "fancyrag/kg/phases.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L490-L490",
          "quote": "- IngestionArtifacts"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L499-L499",
          "quote": "- fancyrag/kg/phases.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "IngestionArtifacts",
          "neo4j_graphrag IngestionArtifacts"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/phases.py::QaOutcome",
      "fancyrag_symbol": "QaOutcome",
      "fancyrag_file": "fancyrag/kg/phases.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L519-L519",
          "quote": "- QaOutcome"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L520-L520",
          "quote": "fancyrag_file: fancyrag/kg/phases.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "QaOutcome",
          "neo4j_graphrag QaOutcome"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/phases.py::resolve_settings(...)",
      "fancyrag_symbol": "resolve_settings(...)",
      "fancyrag_file": "fancyrag/kg/phases.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L524-L524",
          "quote": "- resolve_settings(...)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L525-L525",
          "quote": "fancyrag_file: fancyrag/kg/phases.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "resolve_settings(...)",
          "neo4j_graphrag resolve_settings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/phases.py::discover_sources(...)",
      "fancyrag_symbol": "discover_sources(...)",
      "fancyrag_file": "fancyrag/kg/phases.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L529-L529",
          "quote": "- discover_sources(...)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L530-L530",
          "quote": "fancyrag_file: fancyrag/kg/phases.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "discover_sources(...)",
          "neo4j_graphrag discover_sources"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/phases.py::build_clients(...)",
      "fancyrag_symbol": "build_clients(...)",
      "fancyrag_file": "fancyrag/kg/phases.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Remove local implementation as redundant; behavior covered by upstream/canonical flow.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L534-L534",
          "quote": "- build_clients(...)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L535-L535",
          "quote": "fancyrag_file: fancyrag/kg/phases.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L731-L731",
          "quote": "- build_clients factory – Removed now-extraneous client assembly code."
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "build_clients(...)",
          "neo4j_graphrag build_clients"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/pipeline.py::SimpleKGPipeline (FancyRAG wrapper usage)",
      "fancyrag_symbol": "SimpleKGPipeline (FancyRAG wrapper usage)",
      "fancyrag_file": "fancyrag/kg/pipeline.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": "SimpleKGPipeline (neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline)",
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Prefer upstream SimpleKGPipeline (neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline); remove or significantly reduce local implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L549-L549",
          "quote": "- SimpleKGPipeline (FancyRAG wrapper usage)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L558-L558",
          "quote": "- fancyrag/kg/pipeline.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L562-L562",
          "quote": "- SimpleKGPipeline (neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline)"
        }
      ]
    },
    {
      "id": "fancyrag/db/neo4j_queries.py::reset_database (FancyRAG)",
      "fancyrag_symbol": "reset_database (FancyRAG)",
      "fancyrag_file": "fancyrag/db/neo4j_queries.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L550-L550",
          "quote": "- reset_database (FancyRAG)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L559-L559",
          "quote": "- fancyrag/db/neo4j_queries.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "reset_database (FancyRAG)",
          "neo4j_graphrag reset_database"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/pipeline.py::ensure_document_relationships",
      "fancyrag_symbol": "ensure_document_relationships",
      "fancyrag_file": "fancyrag/kg/pipeline.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Remove local implementation as redundant; behavior covered by upstream/canonical flow.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L551-L551",
          "quote": "- ensure_document_relationships"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L560-L560",
          "quote": "- fancyrag/kg/pipeline.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L730-L730",
          "quote": "- ensure_document_relationships query – Removed duplicate graph linking logic (upstream writer covers it)."
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "ensure_document_relationships",
          "neo4j_graphrag ensure_document_relationships"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/kg/pipeline.py::rollback_ingest",
      "fancyrag_symbol": "rollback_ingest",
      "fancyrag_file": "fancyrag/kg/pipeline.py",
      "category": "ingestion",
      "category_source": "Ingestion Pipeline",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L552-L552",
          "quote": "- rollback_ingest"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L561-L561",
          "quote": "- fancyrag/kg/pipeline.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "rollback_ingest",
          "neo4j_graphrag rollback_ingest"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/evaluator.py::collect_counts",
      "fancyrag_symbol": "collect_counts",
      "fancyrag_file": "fancyrag/qa/evaluator.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L588-L588",
          "quote": "- fancyrag_symbol: `collect_counts`"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L589-L589",
          "quote": "- fancyrag_file: `fancyrag/qa/evaluator.py`"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "collect_counts",
          "neo4j_graphrag collect_counts"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/evaluator.py::count_missing_embeddings / count_orphan_chunks",
      "fancyrag_symbol": "count_missing_embeddings / count_orphan_chunks",
      "fancyrag_file": "fancyrag/qa/evaluator.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L593-L593",
          "quote": "- fancyrag_symbol: `count_missing_embeddings` / `count_orphan_chunks`"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L594-L594",
          "quote": "- fancyrag_file: `fancyrag/qa/evaluator.py`"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "count_missing_embeddings / count_orphan_chunks",
          "neo4j_graphrag count_missing_embeddings"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/evaluator.py::count_checksum_mismatches",
      "fancyrag_symbol": "count_checksum_mismatches",
      "fancyrag_file": "fancyrag/qa/evaluator.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L598-L598",
          "quote": "- fancyrag_symbol: `count_checksum_mismatches`"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L599-L599",
          "quote": "- fancyrag_file: `fancyrag/qa/evaluator.py`"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "count_checksum_mismatches",
          "neo4j_graphrag count_checksum_mismatches"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/evaluator.py::collect_semantic_counts",
      "fancyrag_symbol": "collect_semantic_counts",
      "fancyrag_file": "fancyrag/qa/evaluator.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L603-L603",
          "quote": "- fancyrag_symbol: `collect_semantic_counts`"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L604-L604",
          "quote": "- fancyrag_file: `fancyrag/qa/evaluator.py`"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "collect_semantic_counts",
          "neo4j_graphrag collect_semantic_counts"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/evaluator.py::QaThresholds",
      "fancyrag_symbol": "QaThresholds",
      "fancyrag_file": "fancyrag/qa/evaluator.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L608-L608",
          "quote": "- fancyrag_symbol: `QaThresholds`"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L609-L609",
          "quote": "- fancyrag_file: `fancyrag/qa/evaluator.py`"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "QaThresholds",
          "neo4j_graphrag QaThresholds"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/evaluator.py::QaChunkRecord / QaSourceRecord",
      "fancyrag_symbol": "QaChunkRecord / QaSourceRecord",
      "fancyrag_file": "fancyrag/qa/evaluator.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L627-L627",
          "quote": "- QaChunkRecord / QaSourceRecord"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L628-L628",
          "quote": "fancyrag/qa/evaluator.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "QaChunkRecord / QaSourceRecord",
          "neo4j_graphrag QaChunkRecord"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/evaluator.py::QaResult",
      "fancyrag_symbol": "QaResult",
      "fancyrag_file": "fancyrag/qa/evaluator.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L632-L632",
          "quote": "- QaResult"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L633-L633",
          "quote": "fancyrag/qa/evaluator.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "QaResult",
          "neo4j_graphrag QaResult"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/evaluator.py::IngestionQaEvaluator",
      "fancyrag_symbol": "IngestionQaEvaluator",
      "fancyrag_file": "fancyrag/qa/evaluator.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L637-L637",
          "quote": "- IngestionQaEvaluator"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L638-L638",
          "quote": "fancyrag/qa/evaluator.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "IngestionQaEvaluator",
          "neo4j_graphrag IngestionQaEvaluator"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/qa/report.py::render_markdown & write_ingestion_report",
      "fancyrag_symbol": "render_markdown & write_ingestion_report",
      "fancyrag_file": "fancyrag/qa/report.py",
      "category": "qa",
      "category_source": "Pipeline QA",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L642-L642",
          "quote": "- render_markdown & write_ingestion_report"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L643-L643",
          "quote": "fancyrag/qa/report.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "render_markdown & write_ingestion_report",
          "neo4j_graphrag write_ingestion_report"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "servers/mcp_hybrid_google.py (implicitly in runtime)::HybridCypherRetriever usage",
      "fancyrag_symbol": "HybridCypherRetriever usage",
      "fancyrag_file": "servers/mcp_hybrid_google.py (implicitly in runtime)",
      "category": "retrieval",
      "category_source": "Retrieval Server",
      "upstream_target": "HybridCypherRetriever (neo4j_graphruntime.HybridCypherRetriever)",
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Prefer upstream HybridCypherRetriever (neo4j_graphruntime.HybridCypherRetriever); remove or significantly reduce local implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L665-L665",
          "quote": "- HybridCypherRetriever usage"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L674-L674",
          "quote": "- servers/mcp_hybrid_google.py (implicitly in runtime)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L681-L681",
          "quote": "- HybridCypherRetriever (neo4j_graphruntime.HybridCypherRetriever)"
        }
      ]
    },
    {
      "id": "fancyrag/mcp/runtime.py::RetryingOpenAIEmbeddings in retriever",
      "fancyrag_symbol": "RetryingOpenAIEmbeddings in retriever",
      "fancyrag_file": "fancyrag/mcp/runtime.py",
      "category": "mcp/server",
      "category_source": "Retrieval Server",
      "upstream_target": "OpenAIEmbeddings",
      "decision": "REPLACE",
      "decision_rationale": "REPLACE: Prefer upstream OpenAIEmbeddings; remove or significantly reduce local implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L666-L666",
          "quote": "- RetryingOpenAIEmbeddings in retriever"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L675-L675",
          "quote": "- fancyrag/mcp/runtime.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L682-L682",
          "quote": "- OpenAIEmbeddings"
        }
      ]
    },
    {
      "id": "fancyrag/mcp/runtime.py::FastMCP server startup",
      "fancyrag_symbol": "FastMCP server startup",
      "fancyrag_file": "fancyrag/mcp/runtime.py",
      "category": "mcp/server",
      "category_source": "Retrieval Server",
      "upstream_target": "FastMCP (fastmcp.FastMCP) & ...",
      "decision": "ADAPT",
      "decision_rationale": "ADAPT: Keep local symbol but adjust to align with canonical interfaces and remove duplication.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L667-L667",
          "quote": "- FastMCP server startup"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L676-L676",
          "quote": "- fancyrag/mcp/runtime.py"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L683-L683",
          "quote": "- FastMCP (fastmcp.FastMCP) & ..."
        }
      ]
    },
    {
      "id": "fancyrag/mcp/runtime.py::ServerState",
      "fancyrag_symbol": "ServerState",
      "fancyrag_file": "fancyrag/mcp/runtime.py",
      "category": "mcp/server",
      "category_source": "Retrieval Server",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L691-L691",
          "quote": "- ServerState"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L699-L699",
          "quote": "- fancyrag/mcp/runtime.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "ServerState",
          "neo4j_graphrag ServerState"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/mcp/runtime.py::search_sync(query) (MCP handler)",
      "fancyrag_symbol": "search_sync(query) (MCP handler)",
      "fancyrag_file": "fancyrag/mcp/runtime.py",
      "category": "mcp/server",
      "category_source": "Retrieval Server",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L692-L692",
          "quote": "- search_sync(query) (MCP handler)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L700-L700",
          "quote": "- fancyrag/mcp/runtime.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "search_sync(query) (MCP handler)",
          "neo4j_graphrag search_sync"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    },
    {
      "id": "fancyrag/mcp/runtime.py::fetch_node(fetch_sync)",
      "fancyrag_symbol": "fetch_node(fetch_sync)",
      "fancyrag_file": "fancyrag/mcp/runtime.py",
      "category": "mcp/server",
      "category_source": "Retrieval Server",
      "upstream_target": null,
      "decision": "KEEP",
      "decision_rationale": "KEEP: No upstream equivalent mapped; retain FancyRAG implementation.",
      "evidence": [
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L693-L693",
          "quote": "- fetch_node(fetch_sync)"
        },
        {
          "repo": "local-docs",
          "file_path": "DR_05_symbol_mapping_matrix.md",
          "line_range": "L701-L701",
          "quote": "- fancyrag/mcp/runtime.py"
        }
      ],
      "verification": {
        "upstream_search_terms": [
          "fetch_node(fetch_sync)",
          "neo4j_graphrag fetch_node"
        ],
        "notes": "No upstream target listed in DR_05 mapping matrix; treat as FancyRAG-local unless future upstream adds equivalent."
      }
    }
  ],
  "pr_plan": [
    {
      "pr_number": 1,
      "title": "Stabilize Baseline & Interfaces",
      "intent": "Normalize naming, imports, and config shape; lock down baseline behavior for safe refactors.",
      "key_changes": "- Fix path spelling inconsistency (`fancryrag` vs `fancyrag`).<br>- Consolidate duplicate config definitions.<br>- Add small compatibility shims where needed.<br>- Update README/dev docs to match.",
      "files_touched": "`src/fancryrag/**`, `src/fancyrag/**`, `config/**`, docs",
      "acceptance_criteria": "- All tests pass.<br>- No import path mismatches.<br>- Config loads identically before/after.<br>- CI green.",
      "validation_steps": "- Run unit tests and basic ingestion + retrieval smoke tests.<br>- Verify server starts and responds to a basic query.",
      "rollback_plan": "Revert PR; no data migrations expected.",
      "risks_and_mitigations": "Risk: breaking imports. Mitigate with shim module + deprecation warnings.",
      "evidence": {
        "repo": "local-docs",
        "file_path": "DR_06 Phased Refactor Plan.md",
        "line_range": "L13-L13",
        "quote": "Stabilize Baseline & Interfaces"
      }
    },
    {
      "pr_number": 2,
      "title": "QA Layer Decoupling",
      "intent": "Extract QA evaluator/report generation into a stable layer; reduce coupling to pipeline internals.",
      "key_changes": "- Move evaluator/report to dedicated package/module boundary.<br>- Standardize QA result types.<br>- Add CLI entry for QA checks.",
      "files_touched": "`src/fancyrag/qa/**`, `scripts/**`",
      "acceptance_criteria": "- QA report generation works against existing DB.<br>- No ingestion behavior changes.",
      "validation_steps": "- Run QA report against sample DB; diff expected stable output.<br>- Run ingestion to ensure no regressions.",
      "rollback_plan": "Revert PR; QA code is additive/refactor-only.",
      "risks_and_mitigations": "Risk: downstream imports break. Mitigate with re-export stubs for one release.",
      "evidence": {
        "repo": "local-docs",
        "file_path": "DR_06 Phased Refactor Plan.md",
        "line_range": "L14-L14",
        "quote": "QA Layer Decoupling"
      }
    },
    {
      "pr_number": 3,
      "title": "Ingestion Path Consolidation",
      "intent": "Simplify ingestion flow; align phases and artifacts; reduce bespoke orchestration.",
      "key_changes": "- Normalize phases (`ResolvedSettings`, `ClientBundle`, `IngestionArtifacts`).<br>- Tighten ingestion artifacts contract.<br>- Reduce bespoke source discovery where upstream supports it.",
      "files_touched": "`src/fancyrag/kg/**`, `src/fancyrag/db/**`",
      "acceptance_criteria": "- Ingestion succeeds end-to-end on fixture corpus.<br>- Graph and embeddings populated correctly.<br>- QA passes within thresholds.",
      "validation_steps": "- Run ingestion on fixture corpus; run QA; verify counts/relationships.",
      "rollback_plan": "Revert PR; ingestion is deterministic and can be re-run.",
      "risks_and_mitigations": "Risk: subtle ingestion output diffs. Mitigate by snapshot testing + QA thresholds.",
      "evidence": {
        "repo": "local-docs",
        "file_path": "DR_06 Phased Refactor Plan.md",
        "line_range": "L15-L15",
        "quote": "Ingestion Path Consolidation"
      }
    },
    {
      "pr_number": 4,
      "title": "LLM & Embeddings Upstream Integration",
      "intent": "Adopt upstream neo4j_graphrag LLM/embedding components; remove custom OpenAI client/telemetry where redundant.",
      "key_changes": "- Replace `SharedOpenAIClient` usage with upstream `OpenAILLM` + `OpenAIEmbeddings`.<br>- Remove custom result wrappers.<br>- Reduce/replace telemetry and custom retry logic.<br>- Keep sanitizer where necessary.",
      "files_touched": "`src/fancyrag/embeddings.py`, `src/cli/openai_client.py`, `src/cli/telemetry.py`, `src/cli/sanitizer.py`",
      "acceptance_criteria": "- Ingestion and retrieval still work with OpenAI keys.<br>- Equivalent embeddings produced (within tolerance).<br>- No missing metrics required for ops.",
      "validation_steps": "- Run embedding generation smoke test.<br>- Run ingestion and a retrieval query; compare answers with baseline.",
      "rollback_plan": "Revert PR; revert to old client wrappers.",
      "risks_and_mitigations": "Risk: API changes between upstream versions. Mitigate with pinned upstream SHA and adapter layer.",
      "evidence": {
        "repo": "local-docs",
        "file_path": "DR_06 Phased Refactor Plan.md",
        "line_range": "L16-L16",
        "quote": "LLM & Embeddings Upstream Integration"
      }
    },
    {
      "pr_number": 5,
      "title": "Pipeline Orchestration Adoption",
      "intent": "Shift orchestration to upstream pipeline runner patterns; remove redundant client assembly and relationship enforcement.",
      "key_changes": "- Replace bespoke `build_clients` and relationship enforcement with upstream pipeline patterns.<br>- Standardize DB lifecycle/reset behavior.<br>- Simplify rollback handling.",
      "files_touched": "`src/fancyrag/kg/phases.py`, `src/fancyrag/kg/pipeline.py`, `src/fancyrag/db/**`",
      "acceptance_criteria": "- Pipeline runs successfully with fewer custom steps.<br>- Relationship integrity preserved.<br>- No orphan chunks/embeddings per QA.",
      "validation_steps": "- Run pipeline end-to-end; run QA; validate relationship counts and checksums.",
      "rollback_plan": "Revert PR; re-run pipeline if needed.",
      "risks_and_mitigations": "Risk: upstream pipeline assumptions mismatch FancyRAG schema. Mitigate by adapters + schema docs.",
      "evidence": {
        "repo": "local-docs",
        "file_path": "DR_06 Phased Refactor Plan.md",
        "line_range": "L17-L17",
        "quote": "Pipeline Orchestration Adoption"
      }
    },
    {
      "pr_number": 6,
      "title": "Retrieval API Surface Alignment",
      "intent": "Align retrieval server and MCP handlers to upstream retriever semantics; reduce bespoke query assembly.",
      "key_changes": "- Use upstream `HybridCypherRetriever` patterns.<br>- Standardize MCP request/response models.<br>- Simplify runtime state management.",
      "files_touched": "`src/fancyrag/mcp/runtime.py`, `servers/**`",
      "acceptance_criteria": "- MCP server responds to search/fetch requests.<br>- Retrieval quality comparable to baseline.<br>- No runtime regressions.",
      "validation_steps": "- Start server; run MCP tool calls; run integration tests.",
      "rollback_plan": "Revert PR; restore old runtime handler logic.",
      "risks_and_mitigations": "Risk: retriever behavior drift. Mitigate with fixed test fixtures and golden queries.",
      "evidence": {
        "repo": "local-docs",
        "file_path": "DR_06 Phased Refactor Plan.md",
        "line_range": "L18-L18",
        "quote": "Retrieval API Surface Alignment"
      }
    },
    {
      "pr_number": 7,
      "title": "Cleanup, Dead Code Removal, and Packaging",
      "intent": "Remove unused code paths; clean exports; ensure package boundaries are clear.",
      "key_changes": "- Delete unused telemetry/clients.<br>- Remove legacy paths and shims no longer needed.<br>- Update package exports and docs.",
      "files_touched": "Multiple",
      "acceptance_criteria": "- Lint + tests pass.<br>- No unused modules remain.<br>- Packaging metadata correct.",
      "validation_steps": "- Run full test suite; run import graph check; run packaging build.",
      "rollback_plan": "Revert PR.",
      "risks_and_mitigations": "Risk: removing code still referenced. Mitigate with static analysis + grep.",
      "evidence": {
        "repo": "local-docs",
        "file_path": "DR_06 Phased Refactor Plan.md",
        "line_range": "L19-L19",
        "quote": "Cleanup, Dead Code Removal, and Packaging"
      }
    },
    {
      "pr_number": 8,
      "title": "Documentation & Ops Hardening",
      "intent": "Finalize docs and operational guidance; ensure smooth developer experience.",
      "key_changes": "- Update README and deployment docs.<br>- Add runbooks for ingestion/retrieval/QA.<br>- Add version pinning guidance for upstream dependencies.",
      "files_touched": "docs, README, scripts",
      "acceptance_criteria": "- Docs reflect current behavior.<br>- Runbooks validated end-to-end.<br>- Release notes included.",
      "validation_steps": "- Follow docs from scratch in a clean env; validate success.",
      "rollback_plan": "N/A (docs only).",
      "risks_and_mitigations": "Risk: doc drift. Mitigate with CI doc checks and examples.",
      "evidence": {
        "repo": "local-docs",
        "file_path": "DR_06 Phased Refactor Plan.md",
        "line_range": "L20-L20",
        "quote": "Documentation & Ops Hardening"
      }
    }
  ]
}
