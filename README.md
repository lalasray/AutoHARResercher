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

- real UCI HAR evaluation
- optional PyTorch acceleration when available
- CUDA auto-detection with CPU fallback
- required OpenAI-based planner for round planning
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
- `AugmentationAgent`: proposes augmentation type and augmentation strength for the next candidate
- `DataAgent`: proposes only the preprocessing and windowing part of the next candidate
- `Orchestrator`: plans each round, launches the 3 agents in parallel, evaluates single-agent and partial-combination candidates, and decides which branch to promote
- `Harness`: trains and evaluates a real UCI HAR baseline from the experiment file, with optional Torch-backed training, early stopping, and richer metrics

The orchestrator now uses a real OpenAI-based planner:

- the planner is required for round planning
- if `OPENAI_API_KEY` is missing, the run fails fast
- if the `openai` Python package is missing, the run fails fast

The harness now reads the experiment file, loads UCI HAR, and returns a real validation macro-F1 score plus richer outputs such as:

- per-class F1
- confusion matrix
- estimated parameter count
- feature dimension
- training report with device, epochs trained, best epoch, and early-stopping state

The current model search space includes stronger HAR-inspired families such as:

- `deepconvlstm`
- `hart`
- `tinierhar`
- plus simpler baselines like `cnn`, `gru`, `tiny_transformer`, and `cnn_gru`

The proposal policy is now adaptive:

- agents mutate around the currently accepted best configuration
- agents avoid replaying identical recent proposals when they still have unseen options
- the orchestrator can keep a stronger partial combination such as `model + data` when `model + augmentation + data` is worse
- the augmentation agent searches both augmentation type and strength

## Round Artifacts

Each round now materializes concrete files:

- `artifacts/round_N/agent_proposals/model_agent.json`
- `artifacts/round_N/agent_proposals/augmentation_agent.json`
- `artifacts/round_N/agent_proposals/data_agent.json`
- `artifacts/round_N/orchestrator_candidate.json`
- `artifacts/round_N/results/model_agent.json`
- `artifacts/round_N/results/augmentation_agent.json`
- `artifacts/round_N/results/data_agent.json`
- `artifacts/round_N/results/orchestrator.json`

And in the Git materialized flow, each branch now owns a real shared experiment file:

- agent worktrees write `research/experiment.json`
- `ModelAgent` edits only model fields
- `AugmentationAgent` edits augmentation name and strength
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

## LLM Planner Config

You can configure the OpenAI planner with a repo-local file:

1. copy `.autohar.env.example` to `.autohar.env`
2. put your real key there

Example:

```bash
cp .autohar.env.example .autohar.env
```

Then edit `.autohar.env`:

```env
OPENAI_API_KEY=your_real_key_here
AUTOHAR_LLM_MODEL=gpt-5-mini
```

Behavior:

- `.autohar.env` is loaded automatically by the planner
- `.autohar.env` is ignored by Git
- if the key is missing, the system raises an error
- if the `openai` package is missing, the system raises an error

Install dependencies before running:

```bash
python3 -m pip install -e .
```

## Project Layout

```text
autohar_researcher/
  agents.py
  datasets.py
  experiment.py
  git_runtime.py
  harness.py
  main.py
  reporting.py
  round_state.py
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
- [x] Log metrics such as accuracy, macro-F1, and confusion matrix
- [x] Add adaptive proposal mutation around the accepted best experiment
- [x] Add augmentation type and strength search
- [x] Add stronger HAR-inspired model choices
- [x] Add optional Torch-backed training with CPU/CUDA device selection
- [x] Add validation-based early stopping
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
- [x] Add mutation of the current best experiment rather than static handcrafted proposals
- [ ] Persist all runs in a lightweight experiment registry

## Training Runtime

The current training config in `research/experiment.json` supports:

- `epochs`
- `min_epochs`
- `early_stopping_patience`
- `device`

Device behavior:

- `device: "auto"` uses CUDA when available, otherwise CPU
- `device: "cuda"` requests CUDA explicitly
- `device: "cpu"` forces CPU

In environments without a visible GPU, the system falls back to CPU cleanly.

## Next Step Suggestion

The cleanest next move is to strengthen the current real HAR baseline:

- dataset: UCI HAR
- model: a stronger sensor-sequence baseline than the current lightweight proxy models
- metric: validation macro-F1

Once that is stronger, the same orchestrator can optimize a genuinely meaningful HAR benchmark.
