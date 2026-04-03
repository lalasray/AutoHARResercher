# AutoHARResercher

Minimal IMU-based HAR equivalent of Karpathy-style `autoresearch`, starting with a Git-first multi-agent prototype.

## Goal

Build a hierarchical auto-research loop for human activity recognition from IMU data:

- One `Orchestrator` controls the search loop.
- Three parallel agents propose candidate improvements:
  - `ModelAgent`
  - `AugmentationAgent`
  - `DataAgent`
- A `Harness` scores each candidate on a validation objective.
- The orchestrator keeps the best run and repeats for `X` rounds.
- Each agent is modeled as its own Git branch and worktree.

This first version is intentionally simple:

- no LLM calls
- no real IMU training
- no external dependencies
- deterministic dummy validation scores so we can verify the loop quickly
- Git commands are planned and printed, not automatically executed

## Minimal Architecture

```text
Orchestrator
  -> Model agent
  -> Augmentation agent
  -> Data agent
  -> Dummy HAR harness
  -> Git branches / worktrees
```

Role split in this version:

- `ModelAgent`: proposes model-family-centered candidates
- `AugmentationAgent`: proposes augmentation-centered candidates
- `DataAgent`: proposes preprocessing and windowing candidates
- `Orchestrator`: plans each round, launches the 3 agents in parallel, evaluates returned validation scores, and decides which branch to promote
- `Harness`: returns the score for a candidate branch/worktree

The harness turns that configuration into a fake but structured validation score. The orchestrator compares scores across rounds and keeps the best candidate.

## Git-First Mental Model

Use `main` as the stable baseline and give each agent its own branch/worktree:

- `agent/model/main`
- `agent/augmentation/main`
- `agent/data/main`

The orchestrator does not need to "be" a branch. It is better thought of as a controller process that:

- creates or tracks the agent branches
- creates one worktree per agent
- tells each agent when to make a candidate change
- runs the harness on each candidate branch
- selects the best branch to merge back into `main`

## Run

```bash
python3 -m autohar_researcher.main --rounds 4 --show-git-plan
```

## Project Layout

```text
autohar_researcher/
  agents.py
  harness.py
  main.py
  types.py
```

## TODO

### Phase 1: current dummy prototype

- [x] Create hierarchical orchestrator
- [x] Create 3 parallel domain agents
- [x] Add dummy HAR evaluation harness
- [x] Keep best configuration over multiple rounds
- [x] Represent each agent as a branch/worktree in the orchestration plan
- [x] Add CLI entrypoint for quick demo

### Phase 2: make it HAR-realistic

- [ ] Replace dummy harness with a real train/val pipeline for one IMU dataset
- [ ] Add dataset module for windowing, normalization, and label mapping
- [ ] Log metrics such as accuracy, macro-F1, and confusion matrix
- [ ] Save best checkpoint and experiment metadata

### Phase 3: match your architecture sketch

- [ ] Expand architecture module:
  - [ ] model selection
  - [ ] preprocessing
  - [ ] hyperparameter optimization
  - [ ] pretraining hooks
- [ ] Expand data module:
  - [ ] supervised datasets
  - [ ] unsupervised datasets
  - [ ] personalization / PerCom ideas
- [ ] Expand simulation tools:
  - [ ] IMU augmentation
  - [ ] pose-to-IMU
  - [ ] video-to-IMU
  - [ ] text-to-pose / knowledge injection
- [ ] Expand resource optimization:
  - [ ] quantization
  - [ ] ONNX export
  - [ ] latency / energy scoring

### Phase 4: evolve toward `autoresearch`

- [ ] Let agents edit real files in their own worktrees
- [ ] Let the harness launch actual training jobs from each worktree
- [ ] Let the orchestrator read metrics from branch-specific outputs
- [ ] Let the orchestrator merge or reject branches based on validation score
- [ ] Add mutation of the current best experiment rather than static handcrafted proposals
- [ ] Persist all runs in a lightweight experiment registry

## Next Step Suggestion

The cleanest next move is to replace the dummy harness with one real HAR baseline:

- dataset: a tiny IMU dataset or synthetic windows
- model: 1D CNN or CNN-GRU
- metric: validation macro-F1

Once that works, the same orchestrator can optimize real scores instead of fake ones.
