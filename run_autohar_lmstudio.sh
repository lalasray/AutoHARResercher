#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "run_autohar_lmstudio.sh is deprecated."
echo "Forwarding to run_autohar_opencode.sh so local runs use OpenCode instead of LM Studio."

exec "$ROOT_DIR/run_autohar_opencode.sh" "$@"
