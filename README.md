# AutoHARResearcher

Multi-agent autoresearch prototype for IMU-based human activity recognition.

## Overview

This project runs a repeated search loop over HAR experiments using:

- `ModelAgent` - proposes model architecture and learning rate
- `AugmentationAgent` - proposes data augmentation strategies
- `DataAgent` - proposes preprocessing and window size
- `Orchestrator` - plans strategy and combines proposals
- a real HAR evaluation harness on UCI HAR dataset
- Git branches and worktrees for materialized runs

Each round:
1. plans a round strategy
2. gets one scoped proposal from each specialist
3. evaluates specialist candidates and orchestrator combinations
4. promotes a new baseline only if it beats the current accepted baseline
5. syncs the next round from that accepted baseline

The search style:
- hill-climbing style acceptance
- multi-agent guided exploration
- orchestrator combination search
- bounded code evolution inside agent-owned modules
- crash-repair loop with automatic recovery and heuristic fallback
- free-form file patching for arbitrary file modifications

## Quick Start

### Option 1: LM Studio (Local LLM - No API Key)

```bash
# 1. Install and start LM Studio (https://lmstudio.ai)
# 2. Download a model (e.g., llama3.2, mistral)
# 3. Start the local server (click "Start Server" in UI)

pip install -e ".[dev]"
./run_autohar_lmstudio.sh --rounds 10
```

### Option 2: OpenAI (Requires API Key)

```bash
pip install -e ".[dev]"
echo 'OPENAI_API_KEY=your_key_here' > .autohar.env
./run_autohar_openai.sh --rounds 10
```

### Option 3: Heuristic Fallback (No LLM at all)

```bash
pip install -e ".[dev]"
AUTOHAR_ALLOW_LLM_FALLBACK=true python -m autohar_researcher.main --rounds 10
```

## Planner Backends

Choose one by setting `AUTOHAR_LLM_BACKEND` in `.autohar.env`:

| Backend | API Key | Setup Required |
|---------|---------|----------------|
| `lmstudio` | None | Install LM Studio |
| `openai` | Yes | Get API key |
| `heuristic` | None | None |

## Running

### With Scripts

```bash
./run_autohar_lmstudio.sh --rounds 20     # LM Studio
./run_autohar_openai.sh --rounds 20       # OpenAI
```

### With Environment Variables

```bash
AUTOHAR_ROUNDS=50 ./run_autohar_lmstudio.sh
AUTOHAR_MATERIALIZE_GIT=0 ./run_autohar_lmstudio.sh
AUTOHAR_AUTO_PUSH_NON_MAIN=1 ./run_autohar_lmstudio.sh
```

### Resume Mode

```bash
./run_autohar_lmstudio.sh --rounds 10 --resume
```

Resume continues from the next round already present in `artifacts/`.

### Direct Python

```bash
python -m autohar_researcher.main --rounds 20 --materialize-git
python -m autohar_researcher.main --rounds 10 --materialize-git --resume
```

## Environment Configuration

Create `.autohar.env` from the example:

```bash
cp .autohar.env.example .autohar.env
```

### LM Studio Configuration

```bash
AUTOHAR_LLM_BACKEND=lmstudio
AUTOHAR_LLM_BASE_URL=http://127.0.0.1:1234/v1
AUTOHAR_LLM_API_KEY=lm-studio
AUTOHAR_LLM_MODEL=your-model-id
```

### OpenAI Configuration

```bash
OPENAI_API_KEY=sk-...
AUTOHAR_OPENAI_MODEL=gpt-4o-mini
```

### Role-Specific Overrides

Each role can use different settings:

```bash
AUTOHAR_ORCHESTRATOR_MODEL=gpt-4o
AUTOHAR_MODEL_AGENT_MODEL=gpt-4o-mini
AUTOHAR_AUGMENTATION_AGENT_MODEL=gpt-4o-mini
AUTOHAR_DATA_AGENT_MODEL=gpt-4o-mini
AUTOHAR_JUDGE_MODEL=gpt-4o-mini
```

### Optional Settings

```bash
# Judge thresholds
AUTOHAR_JUDGE_PRIOR_BEST_DELTA=0.015
AUTOHAR_JUDGE_RECENT_DELTA=0.01
AUTOHAR_JUDGE_RELAXED_PRIOR_BEST_DELTA=0.03
AUTOHAR_JUDGE_RELAXED_RECENT_DELTA=0.02

# Sandbox settings
AUTOHAR_SANDBOX_TIMEOUT=5
AUTOHAR_SANDBOX_MODE=thread  # or "subprocess"

# Allow heuristic fallback when LLM is unavailable
AUTOHAR_ALLOW_LLM_FALLBACK=true

# Crash-repair loop settings
AUTOHAR_REPAIR_MAX_RETRIES=3
AUTOHAR_REPAIR_BACKOFF_BASE=2.0
AUTOHAR_REPAIR_FALLBACK_HEURISTIC=true
AUTOHAR_REPAIR_MAX_PATCH_RETRIES=2
```

## Agent Ownership

- `ModelAgent` owns `research/modules/model_module.py`
- `AugmentationAgent` owns `research/modules/augmentation_module.py`
- `DataAgent` owns `research/modules/data_module.py`
- `orchestrator-main` composes those modules into the shared candidate

## Branch Model

Persistent branches:
- `main`
- `agent-model-main`
- `agent-augmentation-main`
- `agent-data-main`
- `orchestrator-main`

Mental model:
- specialists search one axis each
- orchestrator decides what actually wins the round
- the accepted winner becomes the baseline for the next round

## Artifacts

Main outputs in `artifacts/round_N/`:

- `score_history.*`: specialist trial history (can rise or fall)
- `accepted_history.*`: accepted baseline history (should improve)
- `promotion.json`: who won the round and why
- `planner_decision.json`: planner strategy and hints
- `proposal_judgments/*.json`: judge evaluations
- `baseline_sync.json`: baseline propagated to each workspace
- `repair_attempts/*.json`: crash-repair loop attempts and strategies
- `applied_patches/*.json`: file patches applied in each round

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=autohar_researcher
```

## Main Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `agents.py` | Multi-agent orchestration |
| `planner.py` | LLM-based planning |
| `heuristic_planner.py` | Fallback planner |
| `git_agent_bridge.py` | Git communication layer |
| `harness.py` | HAR evaluation |
| `experiment.py` | Experiment configuration |
| `git_runtime.py` | Git worktree management |
| `datasets.py` | UCI HAR dataset |
| `sandbox.py` | Safe code execution |
| `config.py` | Configuration management |
| `reporting.py` | SVG/CSV output |
| `repair_loop.py` | Crash-repair loop with retry logic |
| `file_patcher.py` | Free-form file patching |

## Checklist

- [x] Multi-agent scaffold
- [x] Persistent specialist branches
- [x] UCI HAR evaluation pipeline
- [x] Validation macro-F1 promotion
- [x] Early stopping
- [x] Optional Torch/CUDA path
- [x] Agent-owned executable modules
- [x] Bounded rewrite hooks
- [x] Proposal judge layer
- [x] Resume support
- [x] LM Studio backend
- [x] Heuristic fallback planner
- [x] Configurable thresholds
- [x] Sandboxed code execution
- [x] Test suite
- [x] Crash-repair loop
- [x] Free-form file patching
- [ ] More datasets
- [ ] Compute-aware objectives
