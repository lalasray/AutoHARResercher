#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".autohar.env" ]]; then
  echo "Missing .autohar.env in $ROOT_DIR"
  echo "Create it from .autohar.env.example before running."
  exit 1
fi

export AUTOHAR_LLM_BACKEND="lmstudio"
if [[ -n "${AUTOHAR_LMSTUDIO_MODEL:-}" ]]; then
  export AUTOHAR_LLM_MODEL="${AUTOHAR_LMSTUDIO_MODEL}"
fi

ROUNDS="${AUTOHAR_ROUNDS:-20}"
MATERIALIZE_GIT="${AUTOHAR_MATERIALIZE_GIT:-1}"
AUTO_PUSH_NON_MAIN="${AUTOHAR_AUTO_PUSH_NON_MAIN:-0}"
NEVER_STOP="${AUTOHAR_NEVER_STOP:-false}"
TIME_BUDGET="${AUTOHAR_TIME_BUDGET_SECONDS:-}"

ARGS=()

if [[ "$NEVER_STOP" == "true" ]]; then
  ARGS+=(--never-stop)
elif [[ "$HAS_ROUNDS_ARG:=0" == "0" ]]; then
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

ARGS+=("$@")

echo "Running AutoHARResearcher from $ROOT_DIR"
echo "Backend: lmstudio"
echo "Never-stop: $NEVER_STOP"
echo "Time budget: ${TIME_BUDGET:-none}"
echo "Materialize git: $MATERIALIZE_GIT"
echo "Command: python -m autohar_researcher.main ${ARGS[*]}"

python -m autohar_researcher.main "${ARGS[@]}"
