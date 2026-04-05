#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".autohar.env" ]]; then
  echo "Missing .autohar.env in $ROOT_DIR"
  echo "Create it from .autohar.env.example before running."
  exit 1
fi

export AUTOHAR_LLM_BACKEND="openai"
if [[ -n "${AUTOHAR_OPENAI_MODEL:-}" ]]; then
  export AUTOHAR_LLM_MODEL="${AUTOHAR_OPENAI_MODEL}"
fi

ROUNDS="${AUTOHAR_ROUNDS:-20}"
MATERIALIZE_GIT="${AUTOHAR_MATERIALIZE_GIT:-1}"
AUTO_PUSH_NON_MAIN="${AUTOHAR_AUTO_PUSH_NON_MAIN:-0}"

HAS_ROUNDS_ARG=0
for arg in "$@"; do
  if [[ "$arg" == "--rounds" ]]; then
    HAS_ROUNDS_ARG=1
    break
  fi
done

ARGS=()
if [[ "$HAS_ROUNDS_ARG" == "0" ]]; then
  ARGS+=(--rounds "$ROUNDS")
fi

if [[ "$MATERIALIZE_GIT" == "1" ]]; then
  ARGS+=(--materialize-git)
fi

if [[ "$AUTO_PUSH_NON_MAIN" == "1" ]]; then
  ARGS+=(--auto-push-non-main)
fi

ARGS+=("$@")

echo "Running AutoHARResearcher from $ROOT_DIR"
if [[ "$HAS_ROUNDS_ARG" == "1" ]]; then
  echo "Rounds: from CLI"
else
  echo "Rounds: $ROUNDS"
fi
echo "Backend: openai"
echo "Materialize git: $MATERIALIZE_GIT"
echo "Auto-push non-main: $AUTO_PUSH_NON_MAIN"
echo "Command: python3 -m autohar_researcher.main ${ARGS[*]}"

python3 -m autohar_researcher.main "${ARGS[@]}"
