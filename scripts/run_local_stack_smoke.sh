#!/usr/bin/env bash
# Mirror the local-stack-smoke CI workflow for local verification.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_OUTPUT="artifacts/local_stack/.env.local-stack-smoke"
COMPOSE_FILE="${ROOT}/docker-compose.neo4j-qdrant.yml"

usage() {
  cat <<'USAGE'
Usage: scripts/run_local_stack_smoke.sh [--env-output <path>]

Options:
  --env-output <path>  Relative path for the generated env file
                       (default: artifacts/local_stack/.env.local-stack-smoke).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-output)
      ENV_OUTPUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${ENV_OUTPUT}" = /* ]]; then
  echo "error: --env-output must be a relative path inside the repo" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY must be set to mirror the CI smoke workflow" >&2
  exit 1
fi

PYTHON_CMD=${PYTHON_CMD:-python3}
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  PYTHON_CMD=python
fi
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  echo "error: python interpreter not found (set PYTHON_CMD if needed)" >&2
  exit 1
fi

cd "${ROOT}"

${PYTHON_CMD} scripts/prepare_local_stack_env.py \
  --input .env.example \
  --output "${ENV_OUTPUT}"

cleanup() {
  docker compose -f "${COMPOSE_FILE}" down --volumes
}
trap cleanup EXIT

scripts/check_local_stack.sh --config
docker compose -f "${COMPOSE_FILE}" up -d --wait neo4j

docker compose -f "${COMPOSE_FILE}" run --rm --no-deps \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e LOCAL_STACK_SKIP_DOCKER_CHECK=1 \
  -e LOCAL_STACK_SKIP_QDRANT=1 \
  -e FANCYRAG_DOTENV_PATH="/workspace/${ENV_OUTPUT}" \
  smoke-tests bash -lc "set -euo pipefail; pip install --upgrade pip >/dev/null; pip install --no-cache-dir -r requirements.lock; pytest tests/integration/local_stack/test_minimal_path_smoke.py"
