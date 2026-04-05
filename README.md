# AutoHARResercher

Multi-agent autoresearch prototype for IMU-based human activity recognition.

## Overview

This project runs a repeated search loop over HAR experiments using:

- `ModelAgent`
- `AugmentationAgent`
- `DataAgent`
- an `Orchestrator`
- a real HAR evaluation harness
- Git branches and worktrees for materialized runs

Each round:

1. plans a round strategy
2. gets one scoped proposal from each specialist
3. evaluates specialist candidates and orchestrator combinations
4. promotes a new baseline only if it beats the current accepted baseline
5. syncs the next round from that accepted baseline

The current search style is best described as:

- hill-climbing style acceptance
- multi-agent guided exploration
- orchestrator combination search
- bounded code evolution inside agent-owned modules

## Current State

- dataset: `UCI HAR`
- promotion metric: validation `macro_f1`
- planner context includes dataset metadata such as activity labels, waist-mounted smartphone setup, sensor modalities, and sampling rate
- specialist agents own separate Python modules
- orchestrator can keep:
  - a single specialist
  - a baseline overlay like `baseline+augmentation`
  - a pairwise combination
  - a full three-way combination

Important: the current planner path still expects a valid `OPENAI_API_KEY`.

## Agent Ownership

- `ModelAgent` owns `research/modules/model_module.py`
- `AugmentationAgent` owns `research/modules/augmentation_module.py`
- `DataAgent` owns `research/modules/data_module.py`
- `orchestrator-main` composes those owned modules into the shared candidate

Specialist evaluation is intended to stay isolated:

- specialists start from a frozen accepted baseline snapshot
- then apply only their own domain mutation
- only orchestrator composition intentionally combines multiple domains

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

## Running

From the repo root:

```bash
./run_autohar.sh
```

Useful commands:

```bash
./run_autohar.sh --rounds 20
./run_autohar.sh --rounds 50
./run_autohar.sh --rounds 10 --resume
AUTOHAR_ROUNDS=50 ./run_autohar.sh
AUTOHAR_MATERIALIZE_GIT=0 ./run_autohar.sh
AUTOHAR_AUTO_PUSH_NON_MAIN=1 ./run_autohar.sh
```

Direct Python entrypoint:

```bash
python3 -m autohar_researcher.main --rounds 20 --materialize-git
python3 -m autohar_researcher.main --rounds 10 --materialize-git --resume
```

Resume behavior:

- `--resume` continues from the next round already present in `artifacts/`
- `--rounds` means additional rounds, not total rounds

## Planner Setup

Create a local env file:

1. copy `.autohar.env.example` to `.autohar.env`
2. add your `OPENAI_API_KEY`

The launcher refuses to run if `.autohar.env` is missing.

## Training Settings

Default training settings live in [experiment.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/experiment.py).

Inside `base_experiment_config()` you can change:

- `epochs`
- `min_epochs`
- `early_stopping_patience`
- `device`
- `metric`

Training optimizes a loss internally, but:

- early stopping uses validation performance
- promotion uses validation `macro_f1`

## Main Files

- [run_autohar.sh](/home/lala/Documents/GitHub/AutoHARResercher/run_autohar.sh)
- [main.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/main.py)
- [agents.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/agents.py)
- [planner.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/planner.py)
- [harness.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/harness.py)
- [experiment.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/experiment.py)
- [git_runtime.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/git_runtime.py)
- [datasets.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/datasets.py)

## Artifacts

Main outputs:

- `score_history.*`: specialist trial history, can rise or fall
- `accepted_history.*`: accepted baseline history, should stay flat or improve
- `promotion.json`: who won the round, whether it was promoted, and why
- `planner_decision.json`: planner strategy, specialist hints, and combination memory
- `proposal_judgments/*.json`: judge view of novelty, plausibility, compatibility, and whether to use the proposal in composition gating
- `baseline_sync.json`: what baseline was propagated into each workspace


## Checklist

- [x] multi-agent scaffold
- [x] Persistent specialist branches and orchestrator branch
- [x] UCI HAR evaluation pipeline
- [x] Validation `macro_f1` promotion metric
- [x] Early stopping
- [x] Optional Torch/CUDA path
- [x] Agent-owned executable module files
- [x] Bounded rewrite hooks for model, augmentation, and data
- [x] Planner artifacts, promotion artifacts, and baseline-sync artifacts
- [x] Per-agent memory artifacts
- [x] Combination-aware orchestrator ordering
- [x] Specialist baseline isolation
- [x] Proposal judge layer
- [x] Report refresh after each round
- [x] Resume support
- [ ] Offline fallback planner when no valid API key is available
- [ ] More datasets
- [ ] Compute-aware objectives such as latency, memory, and size
