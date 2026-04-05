#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export AUTOHAR_LLM_BACKEND="openai"
if [[ -n "${AUTOHAR_OPENAI_MODEL:-}" ]]; then
  export AUTOHAR_LLM_MODEL="${AUTOHAR_OPENAI_MODEL}"
fi

exec "$ROOT_DIR/run_autohar.sh" "$@"
