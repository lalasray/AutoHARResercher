#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_autohar_backend.sh <openai|opencode> [args...]"
  exit 1
fi

BACKEND="$1"
shift

if [[ "$BACKEND" != "openai" && "$BACKEND" != "opencode" ]]; then
  echo "Unsupported backend: $BACKEND"
  echo "Supported backends: openai, opencode"
  exit 1
fi

if [[ ! -f ".autohar.env" ]]; then
  echo "Missing .autohar.env in $ROOT_DIR"
  echo "Create it from .autohar.env.example before running."
  exit 1
fi

set -a
source ".autohar.env"
set +a

export AUTOHAR_LLM_BACKEND="$BACKEND"

ROUNDS="${AUTOHAR_ROUNDS:-20}"
MATERIALIZE_GIT="${AUTOHAR_MATERIALIZE_GIT:-1}"
AUTO_PUSH_NON_MAIN="${AUTOHAR_AUTO_PUSH_NON_MAIN:-0}"
NEVER_STOP="${AUTOHAR_NEVER_STOP:-false}"
TIME_BUDGET="${AUTOHAR_TIME_BUDGET_SECONDS:-}"
MISSION_FILE="${AUTOHAR_MISSION_FILE:-}"

ARGS=()
HAS_ROUNDS_ARG=0
HAS_MISSION_ARG=0
for arg in "$@"; do
  if [[ "$arg" == "--rounds" ]]; then
    HAS_ROUNDS_ARG=1
  fi
  if [[ "$arg" == "--mission-file" ]]; then
    HAS_MISSION_ARG=1
  fi
done

if [[ "$NEVER_STOP" == "true" ]]; then
  ARGS+=(--never-stop)
elif [[ "$HAS_ROUNDS_ARG" == "0" ]]; then
  ARGS+=(--rounds "$ROUNDS")
fi

if [[ "$MATERIALIZE_GIT" == "1" ]]; then
  ARGS+=(--materialize-git)
fi

if [[ "$AUTO_PUSH_NON_MAIN" == "1" ]]; then
  ARGS+=(--auto-push-non-main)
fi

if [[ -n "$TIME_BUDGET" ]]; then
  ARGS+=(--time-budget "$TIME_BUDGET")
fi

if [[ -n "$MISSION_FILE" && "$HAS_MISSION_ARG" == "0" ]]; then
  ARGS+=(--mission-file "$MISSION_FILE")
fi

ARGS+=("$@")

echo "Running AutoHARResearcher from $ROOT_DIR"
echo "Backend: $BACKEND"
echo "Never-stop: $NEVER_STOP"
echo "Time budget: ${TIME_BUDGET:-none}"
echo "Materialize git: $MATERIALIZE_GIT"
echo "Mission file: ${MISSION_FILE:-none}"
echo "Command: python3 -m autohar_researcher.main ${ARGS[*]}"

python3 -m autohar_researcher.main "${ARGS[@]}"
