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

Notes:
  - OPENAI_API_KEY is read from the environment, then .env.local or .env if unset.
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

PYTHON_CMD=${PYTHON_CMD:-python3}
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  PYTHON_CMD=python
fi
if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
  echo "error: python interpreter not found (set PYTHON_CMD if needed)" >&2
  exit 1
fi

read_openai_key() {
  local dotenv_path="$1"
  "${PYTHON_CMD}" - "$dotenv_path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    sys.exit(1)

for raw_line in path.read_text(encoding="utf-8").splitlines():
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        continue
    key, _, remainder = stripped.partition("=")
    if key.strip() != "OPENAI_API_KEY":
        continue
    value = remainder.split("#", 1)[0].strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]
    if value:
        print(value)
        sys.exit(0)
sys.exit(1)
PY
}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  dotenv_candidates=()
  if [[ -n "${FANCYRAG_DOTENV_PATH:-}" ]]; then
    dotenv_candidates+=("${FANCYRAG_DOTENV_PATH}")
  fi
  dotenv_candidates+=(".env.local" ".env")
  for candidate in "${dotenv_candidates[@]}"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      if openai_key=$(read_openai_key "${candidate}"); then
        OPENAI_API_KEY="${openai_key}"
        export OPENAI_API_KEY
        break
      fi
    fi
  done
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "error: OPENAI_API_KEY must be set (export it or add to .env.local/.env)" >&2
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
