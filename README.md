# AutoHARResearcher

Multi-agent autoresearch prototype for IMU-based human activity recognition (HAR).

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), this project gives AI agents a small but real HAR training setup and lets them experiment autonomously overnight. It modifies code, trains models, checks if results improved, keeps or discards, and repeats.

## Features

- **Multi-agent architecture** - ModelAgent, AugmentationAgent, DataAgent with LLM-based orchestration
- **Never-stop autonomous loop** - Run indefinitely until manually stopped (like karpathy/autoresearch)
- **Fixed time budget** - Each experiment runs for a configurable time budget
- **Results tracking** - Tab-separated `results.tsv` with commit, score, status, description
- **Crash-repair loop** - Automatic retry with exponential backoff and heuristic fallback
- **Free-form file patching** - Agents can propose arbitrary file modifications
- **Hill-climbing acceptance** - Promotes only if beats current baseline
- **Git-native workflow** - Branches and worktrees for materialized runs

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
cp .autohar.env.example .autohar.env
```

### 2. Configure

Edit `.autohar.env` with your preferred backend:

```bash
# Option A: LM Studio (no API key)
AUTOHAR_LLM_BACKEND=lmstudio
AUTOHAR_LLM_BASE_URL=http://127.0.0.1:1234/v1

# Option B: OpenAI
OPENAI_API_KEY=sk-...

# Option C: Heuristic (no LLM)
AUTOHAR_LLM_BACKEND=heuristic
AUTOHAR_ALLOW_LLM_FALLBACK=true
```

### 3. Run

```bash
# Basic run (3 rounds)
python -m autohar_researcher.main --rounds 10

# Never-stop mode (like karpathy/autoresearch)
python -m autohar_researcher.main --never-stop

# With time budget (5 minutes per experiment)
python -m autohar_researcher.main --rounds 10 --time-budget 300

# Full autonomy with git materialization
python -m autohar_researcher.main --never-stop --materialize-git --auto-push-non-main
```

## Never-Stop Autoresearch Mode

Inspired by karpathy/autoresearch, run indefinitely:

```bash
# Run forever until Ctrl+C
python -m autohar_researcher.main --never-stop

# With improvement limit
python -m autohar_researcher.main --never-stop --max-rounds-without-improvement 10

# With results tracking
python -m autohar_researcher.main --never-stop --results-path ./results.tsv
```

The never-stop mode:
- Runs until manually interrupted or max rounds without improvement
- Logs all experiments to `results.tsv`
- Includes simplification criterion (keeps results within threshold of best)
- Auto-restarts on crash

## Fixed Time Budget

Like karpathy/autoresearch, experiments can run with a fixed time budget:

```bash
# 5 minutes per experiment
python -m autohar_researcher.main --rounds 10 --time-budget 300

# Or via environment
AUTOHAR_TIME_BUDGET_ENABLED=true AUTOHAR_TIME_BUDGET_SECONDS=300 python -m autohar_researcher.main --never-stop
```

## Architecture

### Agents

- **ModelAgent** - Proposes model architecture and learning rate
- **AugmentationAgent** - Proposes data augmentation strategies  
- **DataAgent** - Proposes preprocessing and window size
- **Orchestrator** - Plans strategy and combines proposals

### Each Round

1. Plan round strategy (LLM orchestrator)
2. Get one scoped proposal from each specialist agent
3. Evaluate specialist candidates and combinations
4. Promote new baseline if it beats current best
5. Sync next round from accepted baseline

### Branch Model

```
main                      # Stable baseline
agent-model-main          # ModelAgent experiments
agent-augmentation-main   # AugmentationAgent experiments
agent-data-main          # DataAgent experiments
orchestrator-main        # Combined proposals
```

## Configuration

### Environment Variables

```bash
# Backend selection
AUTOHAR_LLM_BACKEND=lmstudio|openai|heuristic

# OpenAI
OPENAI_API_KEY=sk-...
AUTOHAR_OPENAI_MODEL=gpt-4o-mini

# LM Studio
AUTOHAR_LLM_BASE_URL=http://127.0.0.1:1234/v1
AUTOHAR_LLM_API_KEY=lm-studio

# Role-specific models
AUTOHAR_ORCHESTRATOR_MODEL=gpt-4o
AUTOHAR_MODEL_AGENT_MODEL=gpt-4o-mini
AUTOHAR_AUGMENTATION_AGENT_MODEL=gpt-4o-mini
AUTOHAR_DATA_AGENT_MODEL=gpt-4o-mini

# Judge thresholds
AUTOHAR_JUDGE_PRIOR_BEST_DELTA=0.015
AUTOHAR_JUDGE_RECENT_DELTA=0.01

# Sandbox
AUTOHAR_SANDBOX_TIMEOUT=5
AUTOHAR_SANDBOX_MODE=thread

# Crash-repair loop
AUTOHAR_REPAIR_MAX_RETRIES=3
AUTOHAR_REPAIR_BACKOFF_BASE=2.0
AUTOHAR_REPAIR_FALLBACK_HEURISTIC=true

# Time budget (karpathy/autoresearch style)
AUTOHAR_TIME_BUDGET_ENABLED=true
AUTOHAR_TIME_BUDGET_SECONDS=300
AUTOHAR_TIME_BUDGET_STARTUP_OVERHEAD=10

# Never-stop mode
AUTOHAR_NEVER_STOP=true
AUTOHAR_SIMPLIFICATION_THRESHOLD=0.001
AUTOHAR_AUTO_RESTART_ON_CRASH=true
```

## Results

### Results.tsv Format

```
commit  val_score  memory_gb  status  description  training_seconds  ...
a1b2c3d  0.8543    2.1       keep    improved_augmentation  120.5
b2c3d4e  0.8512    2.1       keep    simplification         118.3
c3d4e5f  0.0000    0.0       crash   OOM_error               0.0
```

### Status Values

- **keep** - Result improved or within simplification threshold
- **discard** - Result worse than baseline
- **crash** - Experiment failed (OOM, bug, etc.)

### Artifacts

```
artifacts/
  round_1/
    planner_decision.json      # LLM strategy
    promotion.json            # Who won and why
    results/                  # Per-agent results
    agent_proposals/          # Agent proposals
  round_2/
    ...
  reports/
    score_history.csv
    accepted_history.csv
```

## Scripts

```bash
./run_autohar_lmstudio.sh --rounds 10
./run_autohar_lmstudio.sh --never-stop
./run_autohar_openai.sh --rounds 10
```

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=autohar_researcher
```

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `agents.py` | Multi-agent orchestration |
| `planner.py` | LLM-based planning |
| `heuristic_planner.py` | Rule-based fallback |
| `harness.py` | HAR evaluation |
| `experiment.py` | Configuration |
| `git_runtime.py` | Git worktree management |
| `datasets.py` | UCI HAR dataset |
| `sandbox.py` | Safe code execution |
| `config.py` | Configuration |
| `reporting.py` | SVG/CSV output |
| `repair_loop.py` | Crash-repair retry |
| `file_patcher.py` | File patching |
| `time_budget.py` | Time budget wrapper |
| `results_tracker.py` | Results.tsv tracking |

## Design Philosophy

Like karpathy/autoresearch:
- **Single metric** - Higher macro-F1 is always better
- **Fixed time budget** - Results comparable across runs
- **Simplification criterion** - Prefer simpler when equal
- **Never stop** - Run overnight, check in morning
- **Git-native** - Every change is a commit
