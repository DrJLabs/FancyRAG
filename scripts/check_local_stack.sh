#!/usr/bin/env bash
# Wrapper for managing the local Neo4j + Qdrant Docker Compose stack.

set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.neo4j-qdrant.yml}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "${COMPOSE_FILE}" = /* ]]; then
  COMPOSE_PATH="${COMPOSE_FILE}"
else
  COMPOSE_PATH="${PROJECT_ROOT}/${COMPOSE_FILE}"
fi
SCRIPT_NAME=$(basename "$0")

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
  COMPOSE_FILE         Override compose file path (default: docker-compose.neo4j-qdrant.yml).
USAGE
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "\u001b[31merror:\u001b[0m required command '$1' not found" >&2
    exit 127
  fi
}

ensure_compose_file() {
  if [[ ! -f "${COMPOSE_PATH}" ]]; then
    echo "\u001b[31merror:\u001b[0m compose file '${COMPOSE_PATH}' not found" >&2
    exit 1
  fi
}

compose() {
  docker compose -f "${COMPOSE_PATH}" "$@"
}

status_table() {
  compose ps --format '{{.Service}}\t{{.State}}\t{{if .Health}}{{.Health}}{{end}}'
}

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

wait_for_health() {
  local max_attempts=30
  local sleep_seconds=5
  local attempt=1
  while (( attempt <= max_attempts )); do
    if all_healthy; then
      status_table
      return 0
    fi
    sleep "${sleep_seconds}"
    (( attempt++ ))
  done
  echo "\u001b[31merror:\u001b[0m services failed to reach healthy state" >&2
  status_table
  return 1
}

main() {
  [[ $# -eq 0 ]] && usage

  require_command docker
  ensure_compose_file

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
