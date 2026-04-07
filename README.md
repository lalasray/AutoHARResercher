# AutoHARResearcher

Multi-agent autonomous research for IMU-based Human Activity Recognition (HAR). Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Multiple AI agents work together, each owning specific code files. They modify code, run experiments, and keep changes that improve the macro-F1 score.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run the multi-agent loop
python -m autohar_researcher.main --never-stop --materialize-git

# Or simple loop
python -m autohar_researcher.autoresearch_loop --max-experiments 100
```

## Architecture: Multiple Agents

Each agent owns specific files and can modify them freely:

```
research/
  train.py              # Orchestrator combines all modules
  modules/
    model_module.py      # ModelAgent owns this
    augmentation_module.py  # AugmentationAgent owns this
    data_module.py       # DataAgent owns this
```

### Agent Responsibilities

| Agent | Owns | Modifies |
|-------|------|----------|
| **ModelAgent** | `model_module.py` | Model architecture, learning rate, optimizer |
| **AugmentationAgent** | `augmentation_module.py` | Data augmentation strategy |
| **DataAgent** | `data_module.py` | Preprocessing, window size |
| **Orchestrator** | `train.py` | Combines everything, runs experiments |

## How It Works

```
1. ModelAgent modifies model_module.py
2. AugmentationAgent modifies augmentation_module.py
3. DataAgent modifies data_module.py
4. Orchestrator combines into train.py
5. Run experiment → get macro-F1 score
6. If improved → keep changes
7. If worse → revert
8. Repeat
```

## What Each Agent Can Edit

### ModelAgent (`model_module.py`)
```python
MODEL_NAME = "deepconvlstm"  # deepconvlstm, tinyhar, cnn, gru, tinierhar, etc.
HIDDEN_DIM = 64
DEPTH = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
```

### AugmentationAgent (`augmentation_module.py`)
```python
AUGMENTATION = "rotation"  # none, rotation, jitter, time_warp, mixup, scaling
STRENGTH = "medium"  # weak, medium, strong
```

### DataAgent (`data_module.py`)
```python
PREPROCESSING = "normalized"  # raw, normalized, normalized+gravity_filter
WINDOW_SIZE = 128  # 64, 96, 128, 192, 256
```

## Running

### Never-Stop Mode (Recommended)
```bash
python -m autohar_researcher.main --never-stop --materialize-git
```
- Runs until Ctrl+C
- Auto-restarts on crash
- Tracks all experiments in results.tsv

### With Time Budget
```bash
python -m autohar_researcher.main --time-budget 300 --never-stop
```
- 5-minute budget per experiment

### Single Rounds
```bash
python -m autohar_researcher.main --rounds 10
```

## Results

All experiments logged to `results.tsv`:
```
commit  val_score  status  description
a1b2c3d  0.8543    keep    improved_model
b2c3d4e  0.8612    keep    better_augmentation
c3d4e5f  0.8598    keep    simplification
d4e5f6g  0.0000    crash   OOM_error
```

### Status
- **keep** - Score improved or stayed same (simplification counts)
- **discard** - Score got worse
- **crash** - Experiment failed (auto-recovered)

## The Metric

**Macro-F1** on UCI HAR validation set:
- 6 classes: walk, walk_upstairs, walk_downstairs, sit, stand, lay
- Higher is better
- All experiments comparable

## Configuration

```bash
# Backend (LLM for planning)
AUTOHAR_LLM_BACKEND=openai|lmstudio|heuristic
OPENAI_API_KEY=sk-...

# LM Studio
AUTOHAR_LLM_BASE_URL=http://127.0.0.1:1234/v1

# Time budget (karpathy/autoresearch style)
AUTOHAR_TIME_BUDGET_SECONDS=300

# Never-stop
AUTOHAR_NEVER_STOP=true
```

## Features

- **Multi-agent** - Each agent owns specific code files
- **Never-stop** - Run indefinitely like karpathy/autoresearch
- **Time budget** - Fixed time per experiment
- **Crash recovery** - Auto-retry with backoff
- **Git-native** - Every change committed automatically
- **Results tracking** - results.tsv with all experiments

## Scripts

```bash
./run_autohar_lmstudio.sh --never-stop
./run_autohar_lmstudio.sh --rounds 10
./run_autohar_openai.sh --never-stop
```

## Testing

```bash
pytest tests/ -v
```

## File Structure

```
research/
  train.py              # Orchestrator combines modules here
  modules/
    model_module.py      # ModelAgent's code
    augmentation_module.py  # AugmentationAgent's code
    data_module.py       # DataAgent's code

agent_guides/            # Agent instructions (used in LLM prompts)
  model_agent.md         # What ModelAgent should do
  augmentation_agent.md  # What AugmentationAgent should do
  data_agent.md          # What DataAgent should do
  orchestrator.md        # What Orchestrator should do

agent.md                 # Top-level agent instructions

autohar_researcher/
  main.py              # Entry point
  agents.py            # Multi-agent orchestration
  planner.py           # LLM planning (loads agent_guides/)
  autoresearch_loop.py # Simple loop (karpathy style)
  harness.py           # READ ONLY - Evaluation
  datasets.py          # READ ONLY - Dataset
  repair_loop.py       # Crash recovery
  time_budget.py       # Time limits
  results_tracker.py   # results.tsv
```

## Agent Guides

Each agent has detailed instructions in `agent_guides/`. These are loaded and included in LLM prompts:

| Guide | Purpose |
|-------|---------|
| `model_agent.md` | ModelAgent - architecture, learning rate, model params |
| `augmentation_agent.md` | AugmentationAgent - data augmentation strategy |
| `data_agent.md` | DataAgent - preprocessing, window size |
| `orchestrator.md` | Orchestrator - global strategy, coordinates agents |
