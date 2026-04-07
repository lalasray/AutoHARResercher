#!/bin/bash
# run_opencode_agent.sh
# Process AutoHAR tasks using OpenCode CLI
#
# This script watches for new tasks in research/tasks/
# and processes them using OpenCode.
#
# Usage:
#   ./run_opencode_agent.sh                 # Process all pending tasks once
#   ./run_openocode_agent.sh --watch        # Watch and process continuously
#   ./run_opencode_agent.sh --agent model_agent  # Process specific agent tasks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

AGENT=${1:-}
WATCH_MODE=false

if [ "$1" == "--watch" ]; then
    WATCH_MODE=true
    AGENT=${2:-}
elif [ "$1" == "--agent" ] && [ -n "$2" ]; then
    AGENT=$2
fi

POLL_INTERVAL=${POLL_INTERVAL:-10}

get_pending_tasks() {
    local agent=$1
    if [ -n "$agent" ]; then
        find "research/tasks/$agent" -name "*.json" -type f 2>/dev/null | while read -r task_file; do
            if grep -q '"status": "pending"' "$task_file" 2>/dev/null; then
                echo "$task_file"
            fi
        done
    else
        find research/tasks -name "*.json" -type f 2>/dev/null | while read -r task_file; do
            if grep -q '"status": "pending"' "$task_file" 2>/dev/null; then
                echo "$task_file"
            fi
        done
    fi
}

process_task() {
    local task_file=$1
    echo "Processing task: $task_file"
    
    # Read task details
    TASK_ID=$(grep -o '"task_id": "[^"]*"' "$task_file" | head -1 | cut -d'"' -f4)
    AGENT_NAME=$(grep -o '"agent_name": "[^"]*"' "$task_file" | head -1 | cut -d'"' -f4)
    TASK_TYPE=$(grep -o '"task_type": "[^"]*"' "$task_file" | head -1 | cut -d'"' -f4)
    
    if [ -z "$TASK_ID" ] || [ -z "$AGENT_NAME" ]; then
        echo "  Error: Could not parse task file"
        return 1
    fi
    
    # Find the content file
    CONTENT_FILE="research/tasks/$AGENT_NAME/${TASK_ID}_content.md"
    if [ ! -f "$CONTENT_FILE" ]; then
        echo "  Warning: Content file not found: $CONTENT_FILE"
        return 1
    fi
    
    echo "  Agent: $AGENT_NAME"
    echo "  Type: $TASK_TYPE"
    
    # Run OpenCode on the task
    echo "  Running OpenCode..."
    RESULT=$(opencode analyze "$CONTENT_FILE" 2>&1) || {
        echo "  OpenCode failed: $RESULT"
        return 1
    }
    
    # Write result
    RESULT_DIR="research/results/$AGENT_NAME"
    mkdir -p "$RESULT_DIR"
    
    # Create result JSON
    cat > "$RESULT_DIR/${TASK_ID}.json" << EOF
{
  "task_id": "$TASK_ID",
  "agent_name": "$AGENT_NAME",
  "round_id": $(grep -o '"round_id": [0-9]*' "$task_file" | head -1 | cut -d':' -f2 | tr -d ' '),
  "task_type": "$TASK_TYPE",
  "status": "completed",
  "result": $(echo "$RESULT" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" 2>/dev/null || echo '""'),
  "created_at": $(grep -o '"created_at": "[^"]*"' "$task_file" | head -1 | cut -d'"' -f4 | xargs -I{} echo "\"{}\""),
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    # Update task status
    sed -i 's/"status": "pending"/"status": "completed"/' "$task_file"
    
    # Git commit
    git add -A
    git commit -m "Result: $AGENT_NAME $TASK_ID completed" || true
    
    echo "  Done!"
}

echo "=============================================="
echo "AutoCode Agent Processor"
echo "=============================================="
if [ -n "$AGENT" ]; then
    echo "Agent: $AGENT"
else
    echo "Agent: all"
fi
echo "Watch mode: $WATCH_MODE"
echo "=============================================="

if [ "$WATCH_MODE" == "true" ]; then
    echo "Watching for tasks... (Ctrl+C to stop)"
    while true; do
        TASKS=$(get_pending_tasks "$AGENT")
        if [ -n "$TASKS" ]; then
            echo "Found $(echo "$TASKS" | wc -l) pending tasks"
            while IFS= read -r task_file; do
                [ -z "$task_file" ] && continue
                process_task "$task_file"
            done <<< "$TASKS"
        else
            echo "No pending tasks found"
        fi
        sleep "$POLL_INTERVAL"
    done
else
    TASKS=$(get_pending_tasks "$AGENT")
    if [ -z "$TASKS" ]; then
        echo "No pending tasks found"
        exit 0
    fi
    echo "Found $(echo "$TASKS" | wc -l) pending tasks"
    while IFS= read -r task_file; do
        [ -z "$task_file" ] && continue
        process_task "$task_file"
        echo ""
    done <<< "$TASKS"
    echo "All tasks processed!"
fi
