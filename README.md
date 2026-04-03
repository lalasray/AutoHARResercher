# AutoHARResercher

Minimal IMU-based HAR equivalent of Karpathy-style `autoresearch`, starting with a Git-first multi-agent prototype.

## Goal

Build a hierarchical auto-research loop for human activity recognition from IMU data:

- One `Orchestrator` controls the search loop.
- Three parallel agents propose partial improvements:
  - `ModelAgent`
  - `AugmentationAgent`
  - `DataAgent`
- A `Harness` scores each candidate on a validation objective.
- The orchestrator keeps the best run and repeats for `X` rounds.
- Each agent is modeled as its own Git branch and worktree.

This first version is intentionally simple:

- no LLM calls
- no external ML dependencies
- real UCI HAR evaluation with a small dependency-free baseline
- no external dependencies
- Git commands can be either previewed or materialized
- each round writes proposal artifacts so the branch-composition flow is visible

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

- `ModelAgent`: proposes only the model-side part of the next candidate
- `AugmentationAgent`: proposes only the augmentation-side part of the next candidate
- `DataAgent`: proposes only the preprocessing and windowing part of the next candidate
- `Orchestrator`: plans each round, launches the 3 agents in parallel, composes one unified candidate branch from their partial proposals, evaluates that branch, and decides which branch to promote
- `Harness`: trains and evaluates a tiny dependency-free baseline on UCI HAR from the experiment file

The harness now reads the composed experiment file, loads UCI HAR, runs a tiny dependency-free baseline, and returns a real validation macro-F1 score. The orchestrator compares scores across rounds and keeps the best candidate.

## Round Artifacts

Each round now materializes concrete files:

- `artifacts/round_N/agent_proposals/model_agent.json`
- `artifacts/round_N/agent_proposals/augmentation_agent.json`
- `artifacts/round_N/agent_proposals/data_agent.json`
- `artifacts/round_N/orchestrator_candidate.json`

And in the Git materialized flow, each branch now owns a real shared experiment file:

- agent worktrees write `research/experiment.json`
- `ModelAgent` edits only model fields
- `AugmentationAgent` edits only augmentation fields
- `DataAgent` edits only data/preprocessing fields
- `orchestrator-main` commits the composed `research/experiment.json`

This is the bridge between the current dummy prototype and a real Git-based system:

- today: agent branches produce structured proposal files
- now: agent branches can receive real committed experiment files in their worktrees
- next: agent branches produce real code diffs instead of JSON-only files
- later: the orchestrator composes richer commits on its persistent branch

## Git-First Mental Model

Use `main` as the stable baseline and give each agent its own branch/worktree:

- `agent-model-main`
- `agent-augmentation-main`
- `agent-data-main`

The orchestrator does not need to "be" a branch. It is better thought of as a controller process that:

- creates or tracks the agent branches
- creates one worktree per agent
- tells each agent when to make its aspect-specific change
- keeps one persistent orchestrator branch and adds one commit per round
- runs the harness on the latest orchestrator commit
- selects the best branch to merge back into `main`

In the current Git model:

- each agent branch accumulates that agent's own research commits
- the orchestrator branch accumulates one composed commit per round
- before a new agent round starts, each agent branch syncs from the current orchestrator baseline
- `main` stays the stable branch that can later merge the orchestrator branch

## Run

```bash
python3 -m autohar_researcher.main --rounds 4 --show-git-plan
```

To actually create/update the agent worktrees and commit round files:

```bash
python3 -m autohar_researcher.main --rounds 2 --materialize-git --show-git-plan
```

To also push every non-`main` branch after commit:

```bash
python3 -m autohar_researcher.main --rounds 2 --materialize-git --auto-push-non-main --show-git-plan
```

Push policy in this mode:

- pushes `agent-model-main`
- pushes `agent-augmentation-main`
- pushes `agent-data-main`
- pushes `orchestrator-main`
- never pushes `main`

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
- [x] Add synthetic HAR train/validation harness
- [x] Keep best configuration over multiple rounds
- [x] Represent each agent as a branch/worktree in the orchestration plan
- [x] Let the orchestrator compose one candidate from partial agent proposals
- [x] Write per-round proposal artifacts and a composed candidate artifact
- [x] Materialize real branch/worktree files and commits for each round
- [x] Add CLI entrypoint for quick demo

### Phase 2: extend the real-data baseline

- [x] Replace synthetic harness with a real train/val pipeline for one IMU dataset
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
- [ ] Let the orchestrator create richer composed commits by combining selected agent commits
- [ ] Let the orchestrator merge or reject branches based on validation score
- [ ] Add mutation of the current best experiment rather than static handcrafted proposals
- [ ] Persist all runs in a lightweight experiment registry

## Next Step Suggestion

The cleanest next move is to strengthen the current real HAR baseline:

- dataset: UCI HAR
- model: a stronger classifier than the current lightweight linear baseline
- metric: validation macro-F1

Once that is stronger, the same orchestrator can optimize a genuinely meaningful HAR benchmark.
