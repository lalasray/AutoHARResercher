#!/bin/bash
# run_autohar_opencode.sh
# Run AutoHAR using OpenCode as the agent layer (no API key required)
#
# This script:
# 1. Runs AutoHAR which writes tasks to Git
# 2. Uses OpenCode CLI to process tasks
# 3. Reads results back from Git
#
# Prerequisites:
# - OpenCode installed (https://opencode.ai)
# - Git repo initialized with remote

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
ROUNDS=${ROUNDS:-5}
POLL_INTERVAL=${POLL_INTERVAL:-10}
TIMEOUT=${TIMEOUT:-300}

echo "=============================================="
echo "AutoHAR with OpenCode (Git-based)"
echo "=============================================="
echo "Rounds: $ROUNDS"
echo "Poll interval: ${POLL_INTERVAL}s"
echo "=============================================="

# Set environment
export AUTOHAR_LLM_BACKEND=opencode
export AUTOHAR_SANDBOX_MODE=thread

# Create research directories if they don't exist
mkdir -p research/tasks/{orchestrator,model_agent,augmentation_agent,data_agent,judge}
mkdir -p research/results/{orchestrator,model_agent,augmentation_agent,data_agent,judge}

# Initialize git if needed
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git add -A
    git commit -m "Initial commit"
fi

# Run AutoHAR
echo ""
echo "Starting AutoHAR..."
python -m autohar_researcher.main \
    --rounds "$ROUNDS" \
    --materialize-git

echo ""
echo "=============================================="
echo "AutoHAR completed!"
echo "Check research/tasks/ and research/results/"
echo "=============================================="
