# AutoHARResearcher Agent Instructions

You are running **multi-agent autonomous research** on IMU-based Human Activity Recognition (HAR).

## Goal

Improve macro-F1 score on UCI HAR validation set. Higher is better.

## The Metric

**Macro-F1** across 6 activity classes:
- walk, walk_upstairs, walk_downstairs, sit, stand, lay

## Architecture

Multiple agents work together, each with specific responsibilities:

| Agent | Owns | Modifies |
|-------|------|----------|
| **ModelAgent** | `model_module.py` | Model architecture, learning rate |
| **AugmentationAgent** | `augmentation_module.py` | Augmentation strategy |
| **DataAgent** | `data_module.py` | Preprocessing, window size |
| **Orchestrator** | `train.py` | Combines everything |

Detailed instructions for each agent are in `agent_guides/`:
- `model_agent.md` - ModelAgent guide
- `augmentation_agent.md` - AugmentationAgent guide
- `data_agent.md` - DataAgent guide
- `orchestrator.md` - Orchestrator guide

## Code Ownership

```
research/
  modules/
    model_module.py         # ModelAgent owns & edits
    augmentation_module.py   # AugmentationAgent owns & edits
    data_module.py          # DataAgent owns & edits
  train.py                  # Orchestrator combines here

agent_guides/               # Detailed agent instructions
  model_agent.md
  augmentation_agent.md
  data_agent.md
  orchestrator.md
```

## READ ONLY (Do Not Modify)

```
autohar_researcher/
  harness.py     # Evaluation harness
  datasets.py    # Dataset loader
```

## How to Run

```bash
# Multi-agent loop (recommended)
python -m autohar_researcher.main --never-stop --materialize-git

# Simple single-script loop
python -m autohar_researcher.autoresearch_loop --max-experiments 100

# Single experiment
python research/train.py
```

## Experiment Flow

1. Each agent reads their guide from `agent_guides/`
2. Agents propose changes to their owned files
3. Orchestrator combines modules into `train.py`
4. Run experiment → get macro-F1 score
5. If improved → keep changes
6. If worse → revert
7. Repeat

## Results

Results logged to `results.tsv`:
```
commit  val_score  status  description
a1b2c3d  0.8543    keep    improved_model
b2c3d4e  0.8612    keep    better_augmentation
c3d4e5f  0.0000    crash   syntax_error
```

## Status

- **keep** - Score improved or stayed same
- **discard** - Score got worse
- **crash** - Experiment failed (auto-recovered)
