# AutoHARResercher

Git-first multi-agent autoresearch prototype for IMU-based human activity recognition.

## What It Does

The system runs a repeated search loop with:

- an `Orchestrator`
- three specialist agents:
  - `ModelAgent`
  - `AugmentationAgent`
  - `DataAgent`
- a HAR evaluation harness
- Git branches/worktrees for materialized runs

Each round:

1. the orchestrator plans the round with OpenAI
2. each specialist agent proposes a scoped change
3. the harness evaluates:
   - each single agent
   - pairwise orchestrator combinations
   - the full three-way orchestrator combination
4. the best candidate is promoted only if it beats the current accepted baseline
5. all branches sync from that accepted baseline before the next round

Search style:

- hill-climbing style acceptance: only better candidates replace the accepted baseline
- multi-agent guided exploration: specialist agents still explore within their own domains instead of doing only strict greedy local moves
- orchestrator-level combination search: the system compares single-agent, pairwise, and full-composition candidates before deciding what to keep

## Current Setup

- real OpenAI planner is required
- UCI HAR is the current dataset
- optional Torch/CUDA path with CPU fallback
- early stopping is enabled
- promotion metric is validation `macro_f1`
- specialist agents own separate Python modules
- orchestrator can promote a single agent, a pair, or a full combination

## Agent Ownership

- `ModelAgent` owns `research/modules/model_module.py`
- `AugmentationAgent` owns `research/modules/augmentation_module.py`
- `DataAgent` owns `research/modules/data_module.py`
- `orchestrator-main` composes those modules into the shared experiment state

Specialist evaluation is isolated:

- a specialist branch starts from the frozen accepted `research/experiment.json`
- it does not replay every module in the workspace when building a single-agent baseline
- only the orchestrator intentionally composes multiple modules together

## Branch Model

Persistent branches:

- `main`
- `agent-model-main`
- `agent-augmentation-main`
- `agent-data-main`
- `orchestrator-main`

Mental model:

- specialists improve one axis only
- the orchestrator decides which candidate actually wins the round
- the accepted winner becomes the baseline for all branches in the next round
- when a search axis starts repeating the same accepted pattern too much, the planner now applies explicit exploration pressure inside that agent's own domain

## Running

From the repo root:

```bash
./run_autohar.sh
```

Current launcher defaults:

- `20` rounds
- Git materialization enabled
- auto-push disabled

Useful variants:

```bash
AUTOHAR_ROUNDS=50 ./run_autohar.sh
AUTOHAR_ROUNDS=10 AUTOHAR_MATERIALIZE_GIT=0 ./run_autohar.sh
AUTOHAR_ROUNDS=10 AUTOHAR_AUTO_PUSH_NON_MAIN=1 ./run_autohar.sh
./run_autohar.sh --rounds 20 --materialize-git --auto-push-non-main
./run_autohar.sh --rounds 10 --resume
```

Direct Python entrypoint:

```bash
python3 -m autohar_researcher.main --rounds 20 --materialize-git
python3 -m autohar_researcher.main --rounds 10 --materialize-git --resume
```

## Planner Config

Create a local key file:

1. copy `.autohar.env.example` to `.autohar.env`
2. put your real `OPENAI_API_KEY` there

The launcher refuses to run if `.autohar.env` is missing.

## Main Files

- [run_autohar.sh](/home/lala/Documents/GitHub/AutoHARResercher/run_autohar.sh)
- [autohar_researcher/main.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/main.py)
- [autohar_researcher/agents.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/agents.py)
- [autohar_researcher/planner.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/planner.py)
- [autohar_researcher/harness.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/harness.py)
- [autohar_researcher/experiment.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/experiment.py)
- [autohar_researcher/git_runtime.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/git_runtime.py)

## Artifacts

Important outputs after a run:

- `artifacts/reports/score_history.csv`
- `artifacts/reports/accepted_history.csv`
- `artifacts/round_N/planner_decision.json`
- `artifacts/round_N/baseline_sync.json`
- `artifacts/round_N/promotion.json`
- `artifacts/round_N/results/*.json`

How to read them:

- `score_history.csv`: trial candidate scores, can go up or down
- `accepted_history.csv`: accepted baseline history, should stay flat or rise
- `promotion.json`: which candidate actually won the round
- `planner_decision.json`: what the orchestrator and specialist LLMs decided before evaluation

## Training Settings

Default training settings live in:

- [autohar_researcher/experiment.py](/home/lala/Documents/GitHub/AutoHARResercher/autohar_researcher/experiment.py)

Inside `base_experiment_config()`:

- `epochs`
- `min_epochs`
- `early_stopping_patience`
- `device`
- `metric`

The search objective is validation `macro_f1`. Training still uses a loss internally, but promotion and early stopping are based on validation performance.

## Status

Quick health check from this pass:

- `python3 -m compileall autohar_researcher` passes
- no obvious placeholder stubs in the core package
- launcher script is valid shell

What I did not fully validate in this pass:

- a full long multi-round live run
- full metric/report consistency across old artifact history already on disk

## Next Good Steps

- improve report clarity between trial vs accepted curves
- regenerate fresh reports after the latest isolation fix
- add more datasets later if you want broader HAR evaluation

## Checklist

- [x] Git-first multi-agent scaffold
- [x] Persistent agent branches and orchestrator branch
- [x] Real OpenAI planner
- [x] Separate agent guide files
- [x] Per-agent memory artifacts
- [x] Promotion artifacts and baseline-sync artifacts
- [x] Single-agent, pairwise, and three-way orchestrator evaluation
- [x] Accept-only-if-better baseline promotion
- [x] Specialist baseline isolation from full module replay
- [x] UCI HAR dataset path
- [x] Validation `macro_f1` as the promotion metric
- [x] Early stopping
- [x] Optional Torch/CUDA path
- [x] Checkpoint and experiment artifact writing
- [x] Agent-owned executable module files
- [x] Bounded rewrite hooks for model / augmentation / data modules
- [x] Combination-memory-aware orchestrator ordering
- [x] Freshly regenerate reports after the latest isolation fix
- [x] Make reporting clearer between trial curves and accepted baseline curves
- [ ] Add more datasets
- [ ] Add richer compute-aware objectives like latency / memory / size
