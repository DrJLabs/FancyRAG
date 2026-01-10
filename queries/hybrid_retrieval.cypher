WITH node, score
OPTIONAL MATCH (sem)
WHERE sem.semantic_source = 'kg_build.semantic_enrichment.v1'
  AND sem.chunk_uid = node.uid
OPTIONAL MATCH (sem)-[rel]->(other)
WHERE rel.semantic_source = 'kg_build.semantic_enrichment.v1'
WITH
    node,
    score,
    collect(DISTINCT sem) AS semantic_nodes,
    collect(DISTINCT rel) AS semantic_relationships
RETURN
    node,
    score,
    [sem IN semantic_nodes WHERE sem IS NOT NULL |
        {
            id: elementId(sem),
            labels: labels(sem),
            properties: properties(sem)
        }
    ] AS semantic_nodes,
    [rel IN semantic_relationships WHERE rel IS NOT NULL |
        {
            type: type(rel),
            from: elementId(startNode(rel)),
            to: elementId(endNode(rel)),
            properties: properties(rel)
        }
    ] AS semantic_relationships
