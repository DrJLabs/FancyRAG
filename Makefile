.PHONY: up down index index-recreate fulltext-index ingest counts logs smoke smoke-logs scan-image

PROJECT_NAME ?= fancryrag
SMOKE_PROJECT_NAME ?= fancryrag-smoke-$(shell date +%s)
SMOKE_COMPOSE := docker-compose.smoke.yml

# Start Neo4j container and supporting services.
up:
	docker compose up -d neo4j mcp

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f neo4j mcp

index: up
	uv run python scripts/create_vector_index.py

index-recreate: up
	docker compose exec neo4j cypher-shell -u $${NEO4J_USERNAME:-neo4j} -p $${NEO4J_PASSWORD:-password} \
		"DROP INDEX text_embeddings IF EXISTS; CREATE VECTOR INDEX text_embeddings IF NOT EXISTS FOR (n:Chunk) ON (n.embedding) OPTIONS {indexConfig: {\`vector.dimensions\`: $${EMBEDDING_DIMENSIONS:-1024}, \`vector.similarity_function\`: 'cosine'}};"

# Run the Python helper to create or verify the full-text index.
fulltext-index: up
	uv run python scripts/create_fulltext_index.py

ingest:
	uv run python tools/run_pipeline.py $(f)

counts:
	docker compose exec neo4j cypher-shell -u $${NEO4J_USERNAME:-neo4j} -p $${NEO4J_PASSWORD:-password} "MATCH (n) RETURN count(n);"

smoke:
	@echo "Running smoke tests with project $(SMOKE_PROJECT_NAME)"
	COMPOSE_PROJECT_NAME="$(SMOKE_PROJECT_NAME)" docker compose -f $(SMOKE_COMPOSE) down --remove-orphans --volumes >/dev/null 2>&1 || true
	COMPOSE_PROJECT_NAME="$(SMOKE_PROJECT_NAME)" docker compose -f $(SMOKE_COMPOSE) build mcp
	COMPOSE_PROJECT_NAME="$(SMOKE_PROJECT_NAME)" docker compose -f $(SMOKE_COMPOSE) up -d --wait neo4j mcp embedding-stub
	COMPOSE_PROJECT_NAME="$(SMOKE_PROJECT_NAME)" docker compose -f $(SMOKE_COMPOSE) run --rm smoke-tests
	COMPOSE_PROJECT_NAME="$(SMOKE_PROJECT_NAME)" docker compose -f $(SMOKE_COMPOSE) down --remove-orphans --volumes

smoke-logs:
	COMPOSE_PROJECT_NAME="$(SMOKE_PROJECT_NAME)" docker compose -f $(SMOKE_COMPOSE) logs neo4j mcp embedding-stub || true

scan-image:
	@echo "Building MCP image for scanning"
	docker compose build mcp
	@echo "Running Trivy secret scan"
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:0.56.2 image --scanners secret --exit-code 1 --format table fancryrag-mcp:local
	@echo "Image digest:"
	docker image inspect fancryrag-mcp:local --format '{{.Id}}'
