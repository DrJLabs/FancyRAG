WITH node, score
OPTIONAL MATCH (sem:__Entity__)
WHERE sem.semantic_source = 'kg_build.semantic_enrichment.v1'
  AND sem.chunk_uid = node.uid
OPTIONAL MATCH (sem)-[rel]->(other:__Entity__)
WHERE rel.semantic_source = 'kg_build.semantic_enrichment.v1'
WITH
    node,
    score,
    collect(DISTINCT sem) AS semantic_nodes,
    collect(DISTINCT rel) AS semantic_relationships
RETURN
    node,
    score,
    [sem IN semantic_nodes |
        {
            id: elementId(sem),
            labels: labels(sem),
            properties: properties(sem)
        }
    ] AS semantic_nodes,
    [rel IN semantic_relationships |
        {
            type: type(rel),
            from: elementId(startNode(rel)),
            to: elementId(endNode(rel)),
            properties: properties(rel)
        }
    ] AS semantic_relationships
