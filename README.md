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

Planner backends currently supported:

- `openai` with `OPENAI_API_KEY`
- `lmstudio` through an OpenAI-compatible local endpoint

Each role can now use its own direct LLM connection if you want:

- `orchestrator`
- `model_agent`
- `augmentation_agent`
- `data_agent`
- `judge`

If you do nothing, they inherit the shared planner backend settings. If you want true separation, you can override each role independently.

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

Backend-specific launchers:

```bash
./run_autohar_openai.sh --rounds 20
./run_autohar_lmstudio.sh --rounds 20
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
2. choose one planner backend

OpenAI:

```bash
OPENAI_API_KEY=your_key_here
AUTOHAR_LLM_MODEL=gpt-5-mini
```

LM Studio:

```bash
AUTOHAR_LLM_BACKEND=lmstudio
AUTOHAR_LLM_BASE_URL=http://127.0.0.1:1234/v1
AUTOHAR_LLM_API_KEY=lm-studio
AUTOHAR_LLM_MODEL=your_lmstudio_model_id
```

Role-specific overrides are also supported:

```bash
AUTOHAR_ORCHESTRATOR_BACKEND=lmstudio
AUTOHAR_ORCHESTRATOR_BASE_URL=http://127.0.0.1:1234/v1
AUTOHAR_ORCHESTRATOR_API_KEY=lm-studio
AUTOHAR_ORCHESTRATOR_MODEL=local-orchestrator-model

AUTOHAR_MODEL_AGENT_BACKEND=lmstudio
AUTOHAR_MODEL_AGENT_MODEL=local-model-agent

AUTOHAR_AUGMENTATION_AGENT_BACKEND=lmstudio
AUTOHAR_AUGMENTATION_AGENT_MODEL=local-augmentation-agent

AUTOHAR_DATA_AGENT_BACKEND=lmstudio
AUTOHAR_DATA_AGENT_MODEL=local-data-agent

AUTOHAR_JUDGE_BACKEND=lmstudio
AUTOHAR_JUDGE_MODEL=local-judge-model
```

The launcher still requires `.autohar.env` to exist, but it no longer has to contain an OpenAI key if you use `AUTOHAR_LLM_BACKEND=lmstudio`.

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
- [x] Alternate LM Studio planner backend through an OpenAI-compatible local endpoint
- [ ] Pure offline heuristic fallback planner when no LLM server is available
- [ ] More datasets
- [ ] Compute-aware objectives such as latency, memory, and size
