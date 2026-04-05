#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export AUTOHAR_LLM_BACKEND="lmstudio"
if [[ -n "${AUTOHAR_LMSTUDIO_MODEL:-}" ]]; then
  export AUTOHAR_LLM_MODEL="${AUTOHAR_LMSTUDIO_MODEL}"
fi

exec "$ROOT_DIR/run_autohar.sh" "$@"
