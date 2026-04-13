#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".autohar.env" ]]; then
  echo "Missing .autohar.env in $ROOT_DIR"
  echo "Create it from .autohar.env.example before running."
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run_mission.sh path/to/mission.md [extra args]"
  exit 1
fi

MISSION_FILE="$1"
shift

set -a
source ".autohar.env"
set +a

echo "Running AutoHARResearcher mission from $ROOT_DIR"
echo "Mission file: $MISSION_FILE"
echo "Command: python -m autohar_researcher.main --mission-file $MISSION_FILE $*"

python -m autohar_researcher.main --mission-file "$MISSION_FILE" "$@"
