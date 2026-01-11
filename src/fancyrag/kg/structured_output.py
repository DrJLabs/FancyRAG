"""Schema helpers for structured semantic output."""

from __future__ import annotations

from typing import Any


def _strict_schema(schema: Any) -> Any:
    """Recursively set additionalProperties=false on object schemas with explicit properties."""

    if isinstance(schema, list):
        return [_strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    updated = {key: _strict_schema(value) for key, value in schema.items()}
    if updated.get("type") == "object":
        if "properties" in updated:
            updated["additionalProperties"] = False
        if "additionalProperties" in updated:
            updated["additionalProperties"] = _strict_schema(updated["additionalProperties"])
    return updated


def build_neo4j_graph_schema() -> dict[str, Any]:
    """Return a strict JSON schema for Neo4jGraph structured output."""

    try:
        from neo4j_graphrag.experimental.components.types import Neo4jGraph
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "neo4j_graphrag is required for structured semantic output"
        ) from exc

    schema = Neo4jGraph.model_json_schema()
    return _strict_schema(schema)


__all__ = ["build_neo4j_graph_schema"]
