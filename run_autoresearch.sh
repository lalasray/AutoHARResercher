#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".autohar.env" ]]; then
  set -a
  source ".autohar.env"
  set +a
fi

echo "Running AutoHARResearcher in autoresearch loop mode"
echo "From: $ROOT_DIR"
echo ""

MAX_EXPERIMENTS="${AUTOHAR_MAX_EXPERIMENTS:-}"
TIME_BUDGET="${AUTOHAR_TIME_BUDGET_SECONDS:-}"

ARGS=()

if [[ -n "$MAX_EXPERIMENTS" ]]; then
  ARGS+=(--max-experiments "$MAX_EXPERIMENTS")
fi

if [[ -n "$TIME_BUDGET" ]]; then
  ARGS+=(--time-budget "$TIME_BUDGET")
fi

ARGS+=("$@")

echo "Command: python3 -m autohar_researcher.autoresearch_loop ${ARGS[*]}"
echo ""

python3 -m autohar_researcher.autoresearch_loop "${ARGS[@]}"
