from pathlib import Path


def test_hybrid_query_exposes_semantic_fields() -> None:
    query = Path("queries/hybrid_retrieval.cypher").read_text(encoding="utf-8")
    assert "semantic_nodes" in query
    assert "semantic_relationships" in query
