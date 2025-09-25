#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VENV=".venv"
VENV_PATH="$DEFAULT_VENV"
FORCE_RECREATE=0
SKIP_INSTALL="${BOOTSTRAP_SKIP_INSTALL:-0}"
CUSTOM_PYTHON_BIN="${BOOTSTRAP_PYTHON_BIN:-}"
# Test hook: allow CI/tests to bypass local interpreter check when shimmed python is provided.
ASSUME_PY312="${BOOTSTRAP_ASSUME_PY312:-0}"
PACKAGES=(
  "neo4j-graphrag[openai,qdrant]"
  "neo4j>=5,<6"
  "qdrant-client>=1.10"
  "openai>=1,<2"
  "structlog>=24,<25"
  "pytest>=8,<9"
)
TEST_LOCK_CONTENT=$(cat <<'EOF'
# Generated in test mode (no packages installed)
neo4j-graphrag==0.9.0
neo4j==5.23.0
qdrant-client==1.10.4
openai==1.40.3
structlog==24.1.0
pytest==8.3.2
EOF
)
# Test hook: when set, skip real import and simulate success/failure outcome.
TEST_IMPORT="${BOOTSTRAP_TEST_IMPORT:-}"

log() {
  local level="$1"; shift
  if [[ "$level" == "ERROR" ]]; then
    printf '[%s] %s\n' "$level" "$*" >&2
  else
    printf '[%s] %s\n' "$level" "$*"
  fi
}

usage() {
  cat <<USAGE
Usage: scripts/bootstrap.sh [--venv-path PATH] [--force]

Options:
  --venv-path PATH   Override virtual environment directory (default: .venv)
  --force            Recreate the virtual environment if it already exists
  -h, --help         Show this help message
USAGE
}

fail_trap() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log ERROR "Bootstrap failed. Review the log above for details."
    log ERROR "Troubleshooting tips: ensure Python 3.12 is installed (via pyenv/asdf/system package) and rerun with --force if dependencies are partially installed."
  fi
  exit $rc
}

trap fail_trap EXIT

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --venv-path)
        shift
        [[ $# -gt 0 ]] || {
          log ERROR "--venv-path requires a value";
          usage;
          exit 1;
        }
        VENV_PATH="$1"
        ;;
      --force)
        FORCE_RECREATE=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        log ERROR "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
}

require_python() {
  if [[ -n "$CUSTOM_PYTHON_BIN" ]]; then
    if [[ ! -x "$CUSTOM_PYTHON_BIN" ]]; then
      log ERROR "BOOTSTRAP_PYTHON_BIN is set but not executable: $CUSTOM_PYTHON_BIN"
      return 1
    fi
    if [[ "$ASSUME_PY312" == "1" ]]; then
      echo "$CUSTOM_PYTHON_BIN"
      return 0
    fi
    local version
    version="$($CUSTOM_PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
    if [[ "$version" == 3.12.* ]]; then
      echo "$CUSTOM_PYTHON_BIN"
      return 0
    fi
    log ERROR "BOOTSTRAP_PYTHON_BIN must point to a Python 3.12 interpreter (reported $version)."
    return 1
  fi

  local candidates=(python3.12 python3 python)
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      local resolved
      resolved="$(command -v "$candidate")"
      local version
      version="$($resolved -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
      if [[ "$version" == 3.12.* ]]; then
        echo "$resolved"
        return 0
      fi
    fi
  done

  if command -v pyenv >/dev/null 2>&1; then
    local pyenv_version
    pyenv_version="$(pyenv versions --bare 2>/dev/null | awk '/^3\.12/{print $1; exit}')"
    if [[ -n "$pyenv_version" ]]; then
      local pyenv_path
      pyenv_path="$(pyenv root)/versions/$pyenv_version/bin/python"
      if [[ -x "$pyenv_path" ]]; then
        echo "$pyenv_path"
        return 0
      fi
    fi
  fi

  if command -v asdf >/dev/null 2>&1; then
    local asdf_path
    asdf_path="$(asdf which python 2>/dev/null || true)"
    if [[ -n "$asdf_path" ]]; then
      local version
      version="$($asdf_path -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
      if [[ "$version" == 3.12.* ]]; then
        echo "$asdf_path"
        return 0
      fi
    fi
  fi

  return 1
}

activate_venv() {
  local venv_path="$1"
  # shellcheck disable=SC1090
  source "$venv_path/bin/activate"
}

main() {
  parse_args "$@"

  cd "$REPO_ROOT"

  local python_bin
  if ! python_bin="$(require_python)"; then
    log ERROR "Python 3.12 interpreter not found. Install Python 3.12 (e.g., via pyenv 'pyenv install 3.12.5' && pyenv local 3.12.5) or system package manager, then rerun."
    return 1
  fi
  log INFO "Using Python interpreter: $python_bin"

  if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="$REPO_ROOT/$VENV_PATH"
  fi
  local venv_display="${VENV_PATH#$REPO_ROOT/}"

  if [[ -d "$VENV_PATH" ]]; then
    if [[ $FORCE_RECREATE -eq 1 ]]; then
      log INFO "--force supplied; removing existing virtual environment at $venv_display"
      rm -rf "$VENV_PATH"
      "$python_bin" -m venv "$VENV_PATH"
    else
      log INFO "Virtual environment already exists at $venv_display; reusing"
    fi
  else
    log INFO "Creating virtual environment at $venv_display"
    mkdir -p "$(dirname "$VENV_PATH")"
    "$python_bin" -m venv "$VENV_PATH"
  fi

  activate_venv "$VENV_PATH"
  local lockfile="$REPO_ROOT/requirements.lock"

  if [[ "$SKIP_INSTALL" == "1" ]]; then
    log WARN "BOOTSTRAP_SKIP_INSTALL=1 detected; skipping dependency installation (test mode)."
    printf "%s" "$TEST_LOCK_CONTENT" > "$lockfile"
    if [[ -n "$TEST_IMPORT" ]]; then
      log INFO "Running import validation (test mode)"
      if [[ "$TEST_IMPORT" == "fail" ]]; then
        log ERROR "neo4j_graphrag import failed within the virtual environment."
        log ERROR "Review installation logs above; you may need to rerun with --force after resolving the issue."
        return 1
      fi
    fi
  else
    log INFO "Upgrading pip and build tooling"
    python -m pip install --upgrade pip setuptools wheel

    log INFO "Installing runtime dependencies"
    python -m pip install "${PACKAGES[@]}"

    log INFO "Writing dependency lockfile to requirements.lock"
    if command -v pip-compile >/dev/null 2>&1; then
      local req_in="$REPO_ROOT/.bootstrap-requirements.in"
      log INFO "pip-compile detected; generating lockfile via pip-tools"
      : > "$req_in"
      for pkg in "${PACKAGES[@]}"; do
        printf '%s\n' "$pkg" >> "$req_in"
      done
      if ! pip-compile "$req_in" --output-file "$lockfile" --quiet </dev/null; then
        log WARN "pip-compile failed; falling back to pip freeze."
        python -m pip freeze --exclude-editable > "$lockfile"
      fi
      rm -f "$req_in"
    else
      python -m pip freeze --exclude-editable > "$lockfile"
    fi

    log INFO "Running import validation"
    if ! python -c 'import neo4j_graphrag' >/dev/null 2>&1; then
      log ERROR "neo4j_graphrag import failed within the virtual environment."
      log ERROR "Review installation logs above; you may need to rerun with --force after resolving the issue."
      return 1
    fi
  fi

  log INFO "Bootstrap complete."
  cat <<NEXT
Next steps:
  1. Activate the environment: source "$venv_display/bin/activate"
  2. Populate environment variables using the upcoming .env template story.
  3. Run project commands (ingest, vectors, search) from within the activated environment.
NEXT
}

main "$@"
