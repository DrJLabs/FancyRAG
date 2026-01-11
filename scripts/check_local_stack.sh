#!/usr/bin/env bash
# Wrapper for managing the local Neo4j + Qdrant Docker Compose stack.

set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${COMPOSE_FILE}" = /* ]]; then
  COMPOSE_PATH="${COMPOSE_FILE}"
else
  COMPOSE_PATH="${PROJECT_ROOT}/${COMPOSE_FILE}"
fi
SCRIPT_NAME=$(basename "$0")

# default_mcp_env_file ensures MCP_ENV_FILE points at .env when .env.local is missing.
default_mcp_env_file() {
  if [[ -n "${MCP_ENV_FILE:-}" ]]; then
    return
  fi
  local env_local="${PROJECT_ROOT}/.env.local"
  local env_file="${PROJECT_ROOT}/.env"
  if [[ ! -f "${env_local}" && -f "${env_file}" ]]; then
    export MCP_ENV_FILE=".env"
  fi
}

# usage prints the script usage help text describing commands and options, then exits with status 1.
usage() {
  cat <<USAGE
Usage: ${SCRIPT_NAME} [--config|--up|--status|--down] [options]

Commands:
  --config             Render the resolved docker compose configuration.
  --up                 Start the stack in detached mode.
  --status [--wait]    Show service status; optionally wait until all health checks pass.
  --down [--destroy-volumes]
                       Stop the stack; optionally remove data volumes.

Environment:
  COMPOSE_FILE         Override compose file path (default: docker-compose.yml).
USAGE
  exit 1
}

# require_command checks that the specified command exists in PATH and, if missing, prints an error to stderr and exits with status 127.
require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '\033[31merror:\033[0m required command '\''%s'\'' not found\n' "$1" >&2
    exit 127
  fi
}

# ensure_compose_file verifies that the resolved compose file exists and prints an error and exits with status 1 if it is missing.
ensure_compose_file() {
  if [[ ! -f "${COMPOSE_PATH}" ]]; then
    printf '\033[31merror:\033[0m compose file '\''%s'\'' not found\n' "${COMPOSE_PATH}" >&2
    exit 1
  fi
}

# compose runs Docker Compose using the resolved COMPOSE_PATH as the compose file.
compose() {
  docker compose -f "${COMPOSE_PATH}" "$@"
}

# status_table prints a tab-separated table of Docker Compose services showing service name, state, and health (if present).
status_table() {
  compose ps --format '{{.Service}}\t{{.State}}\t{{if .Health}}{{.Health}}{{end}}'
}

# qdrant_present reports success when the qdrant service appears in the compose
# status table. When absent it returns a non-zero status so callers can skip
# readiness checks until the container is scheduled.
qdrant_present() {
  status_table | awk -F'\t' '$1 == "qdrant" { exit 0 } END { exit 1 }' >/dev/null
}

# qdrant_ready probes the HTTP readiness endpoint using curl when the qdrant
# service is part of the compose stack. If curl is unavailable the helper exits
# with an error so callers know to install it locally.
qdrant_ready() {
  if ! qdrant_present; then
    return 0
  fi
  require_command curl
  local base_url
  base_url=${QDRANT_HEALTH_URL:-${QDRANT_URL:-http://localhost:6333}}
  local readyz="${base_url%/}/readyz"
  curl --fail --silent --show-error "$readyz" >/dev/null 2>&1
}

# neo4j_port_ready attempts to open a TCP connection to the Neo4j Bolt port
# exposed by the compose stack. Returns success when the socket accepts
# connections so clients can proceed safely.
neo4j_port_ready() {
  local python_cmd=${PYTHON_CMD:-python3}
  if ! command -v "${python_cmd}" >/dev/null 2>&1; then
    python_cmd=python
  fi
  if ! command -v "${python_cmd}" >/dev/null 2>&1; then
    return 1
  fi
  "${python_cmd}" <<'PY'
import os
import socket
import sys
from urllib.parse import urlparse

def parse_endpoint(raw):
    if '://' in raw:
        parsed = urlparse(raw)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 7687
        return host, port
    if ':' in raw:
        host, port = raw.split(':', 1)
        host = host or 'localhost'
        return host, int(port or '7687')
    return raw or 'localhost', 7687

uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
host, port = parse_endpoint(uri)

try:
    with socket.create_connection((host, int(port)), timeout=2):
        pass
except OSError:
    sys.exit(1)
sys.exit(0)
PY
}

# all_healthy checks whether every service reported by status_table is in a running state and, when a health value is present, that it is `healthy`.
# Exits with status 0 if all services are acceptable, 1 if there are no services or any service is not running/healthy.
all_healthy() {
  local rows
  rows=$(status_table)
  [[ -z "${rows}" ]] && return 1
  while IFS=$'\t' read -r service state health; do
    # docker compose reports "running" or "running (healthy)" depending on version.
    if [[ ! "${state}" =~ ^running ]]; then
      return 1
    fi
    if [[ -n "${health}" && "${health}" != healthy ]]; then
      return 1
    fi
  done <<< "${rows}"
  return 0
}

# wait_for_health polls the compose services until every service is running and healthy, then prints the status table and exits successfully.
# On timeout it prints an error, outputs the status table, and returns a non-zero status.
wait_for_health() {
  local max_attempts=${LOCAL_STACK_WAIT_ATTEMPTS:-60}
  local sleep_seconds=${LOCAL_STACK_WAIT_INTERVAL:-5}
  if [[ ${max_attempts} -le 0 ]]; then
    max_attempts=60
  fi
  if [[ ${sleep_seconds} -le 0 ]]; then
    sleep_seconds=5
  fi
  local attempt=1
  while (( attempt <= max_attempts )); do
    if all_healthy && qdrant_ready && neo4j_port_ready; then
      status_table
      return 0
    fi
    sleep "${sleep_seconds}"
    (( attempt++ ))
  done
  printf '\033[31merror:\033[0m services failed to reach healthy state\n' >&2
  status_table
  if ! neo4j_port_ready; then
    local uri=${NEO4J_URI:-bolt://localhost:7687}
    printf '\033[31merror:\033[0m Neo4j Bolt endpoint %s unreachable\n' "${uri}" >&2
  fi
  return 1
}

# main parses command-line arguments, ensures Docker and the compose file are available, and dispatches subcommands (--config, --up, --status [--wait], --down [--destroy-volumes]) to control the local Docker Compose stack.
main() {
  [[ $# -eq 0 ]] && usage

  require_command docker
  ensure_compose_file
  default_mcp_env_file

  case "$1" in
    --config)
      compose config
      ;;
    --up)
      shift
      compose up -d "$@"
      ;;
    --status)
      shift
      local wait=false
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --wait)
            wait=true
            ;;
          *)
            echo "Unknown option: $1" >&2
            usage
            ;;
        esac
        shift
      done
      if [[ "${wait}" == true ]]; then
        wait_for_health
      else
        status_table
      fi
      ;;
    --down)
      shift
      local destroy=false
      while [[ $# -gt 0 ]]; do
        case "$1" in
          --destroy-volumes)
            destroy=true
            ;;
          *)
            echo "Unknown option: $1" >&2
            usage
            ;;
        esac
        shift
      done
      if [[ "${destroy}" == true ]]; then
        compose down --volumes
      else
        compose down
      fi
      ;;
    *)
      echo "Unknown command: $1" >&2
      usage
      ;;
  esac
}

main "$@"
