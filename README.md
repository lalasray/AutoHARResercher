# AutoHARResearcher

Multi-agent autonomous HAR research. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Quick Start

### Conda + WHAR Datasets

The WHAR-backed dataset runtime is the recommended setup.

```bash
~/miniconda3/bin/conda env create -f environment.yml
~/miniconda3/bin/conda activate autohar-whar
cp .autohar.env.example .autohar.env
```

If you already have an environment:

```bash
pip install -e ".[dev,whar]"
```

Run the test suite:

```bash
pytest tests/ -q
```

Run with the local OpenCode backend:

```bash
./run_autohar_opencode.sh --rounds 10
```

### Minimal Pip Setup

```bash
pip install -e ".[dev]"
cp .autohar.env.example .autohar.env

# Edit .autohar.env with your backend (see below)

# Run the main multi-agent loop
./run_autohar_opencode.sh --never-stop --materialize-git
```

Use `pip install -e ".[dev,whar]"` instead if you want WHAR dataset evaluation.

## Shell Scripts

| Script | Description |
|--------|-------------|
| `./run_mission.sh missions/your_mission.md` | Start directly from a mission markdown file |
| `./run_autohar_opencode.sh` | Use local opencode CLI |
| `./run_autohar_openai.sh` | Use OpenAI API |
| `./run_autoresearch.sh` | Simple single-script loop |

## Flags

### `--never-stop`
Run indefinitely. Stops when Ctrl+C or after `max-rounds-without-improvement` rounds without improvement.

### `--rounds N`
Run exactly N rounds.

### `--materialize-git`
Create agent worktrees and commit changes to git.

### `--auto-push-non-main`
Push agent branches to remote (never pushes main).

### `--resume`
Continue from last completed round in `artifacts/`.

### `--time-budget SECONDS`
Max time per experiment in seconds. If exceeded, experiment is marked as timeout.

### `--mission-file PATH.md`
Load a markdown brief and pass it into the planner every round. Use this when you want the run to follow a persistent research brief instead of only the built-in guides.
The mission file can also include frontmatter to define runtime settings such as backend, rounds, resume, materialize_git, and time_budget.

### `--max-rounds-without-improvement N`
In never-stop mode, stop after N rounds without improvement. Default: 10.

### `--max-experiments N` (autoresearch loop only)
Run exactly N experiments, then stop.

## Environment Variables

Edit `.autohar.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
AUTOHAR_OPENAI_MODEL=gpt-4o-mini

# OpenCode local
AUTOHAR_OPENCODE_BIN=/home/your-user/.opencode/bin/opencode
AUTOHAR_OPENCODE_MODEL=lmstudio/qwen/qwen3-30b-a3b-2507

# Never-stop mode
AUTOHAR_NEVER_STOP=true

# Time budget per experiment
AUTOHAR_TIME_BUDGET_SECONDS=300

# Optional persistent markdown research brief
AUTOHAR_MISSION_FILE=missions/example_mission.md
```

The OpenAI and OpenCode wrapper scripts force their matching backend. `run_mission.sh` does not force a backend; it lets the mission frontmatter choose.

## Architecture

```
research/modules/
  model_module.py          # ModelAgent edits architecture and model hyperparameters
  augmentation_module.py   # AugmentationAgent edits this
  data_module.py          # DataAgent edits this

agent_guides/            # Agent instructions
  model_agent.md
  augmentation_agent.md
  data_agent.md
  orchestrator.md
```

Each agent owns one module file. Orchestrator combines them.

ModelAgent keeps the file name `model_module.py`, but its scope is broad: it can choose from the available architecture list, mutate architecture settings, and tune model-side hyperparameters such as learning rate, hidden size, depth, temporal pooling, spectral bins, interaction features, multiscale features, attention mix, dropout, optimizer, weight decay, loss, and label smoothing. Buildable architecture modules live in `research/modules/architectures/`; each exposes `build_model(input_dim, num_classes, params)`.
It should modify a strong current architecture when tuning is the likely bottleneck, switch architecture after plateau or mismatch, and merge architectures only when two families have complementary strengths worth testing.

Available model architectures:

- `deepconvlstm`
- `deepconvlstmattention`
- `bilstm`
- `bilstmattention`
- `hart`
- `tinierhar`
- `tinyhar`
- `spectra`
- `cnn_gru`
- `cnn`
- `tiny_transformer`
- `gru`
- `custom`

AugmentationAgent can also define a bounded custom augmentation algorithm in its owned module by targeting `custom` and providing:

```python
def custom_augment_window(window, strength, params, rng):
    ...
```

The harness validates the output shape and falls back to the original window if the custom function errors or returns invalid values.

ModelAgent has an analogous bounded `custom` model path for feature construction:

```python
def custom_model_features(window, base_features, params):
    ...
```

The harness validates the returned finite numeric feature list and falls back to the standard feature vector if the custom function fails.

DataAgent has a bounded `custom` preprocessing path:

```python
def custom_prepare_window(window, params):
    ...
```

The harness validates the returned same-shape finite numeric window and falls back to the standard preprocessed window if the custom function fails.

## WHAR Dataset Evaluation

Real datasets are loaded through [`teco-kit/whar-datasets`](https://github.com/teco-kit/whar-datasets). The evaluation catalog is:

- `wisdm`
- `uci_har`
- `utd_mhad`
- `hapt`
- `usc_had`
- `unimib_shar`
- `motion_sense`
- `real_life_har`
- `wisdm_19_phone`
- `wisdm_19_watch`
- `ku_har`
- `hang_time`
- `capture_24`

The DataAgent can train on one or more real datasets, evaluate on a separate held-out dataset, and add class-targeted synthetic rows. The evaluation dataset defines the activity universe; real and synthetic training samples are filtered to classes that overlap the evaluation target classes.

Example data params:

```python
{
    "dataset_policy": "real_synthetic_mix",
    "train_datasets": "wisdm,uci_har,hapt,usc_had",
    "evaluation_dataset": "motion_sense",
    "target_classes": "all",
    "real_ratio": 0.75,
    "per_dataset_real_ratio": {"wisdm": 1.0, "uci_har": 0.5, "hapt": 0.5},
    "per_class_real_ratio": {"walk": 1.0, "sit": 0.75, "stand": 0.75},
    "per_dataset_class_real_ratio": {"usc_had": {"walk": 0.5, "sit": 0.25}},
    "synthetic_ratio": 0.25,
    "synthetic_sources": "synthetic_imu_to_imu,synthetic_text_to_imu,synthetic_video_to_imu",
    "synthetic_source_targets": {
        "synthetic_text_to_imu": "sit,stand",
        "synthetic_video_to_imu": "walk,run",
        "synthetic_imu_to_imu": "all",
    },
    "per_source_synthetic_ratio": {
        "synthetic_text_to_imu": 0.5,
        "synthetic_video_to_imu": 0.25,
        "synthetic_imu_to_imu": 0.2,
    },
}
```

WHAR downloads from original dataset sources. Some datasets may require external credentials or accepting upstream terms before downloading.

## Results

The main multi-agent loop writes round artifacts and reports under `artifacts/`:

```text
artifacts/
  reports/
    score_history.csv       # specialist trial scores
    accepted_history.csv    # promoted baseline history
  round_N/
    planner_decision.json
    agent_proposals/
    proposal_judgments/
    results/
    promotion.json
```

`score_history.csv` is trial history. `accepted_history.csv` is the kept baseline history.
`active_mission.md` is the evolving markdown brief written after each round; it keeps your original mission, preserves mission frontmatter, and adds current best baseline, recent evidence, and tighter guidance for the next rounds.

The simple `run_autoresearch.sh` loop writes `results.tsv`:

```
commit  val_score  status  description
a1b2c3d  0.8543    keep    improved_model
b2c3d4e  0.8612    keep    better_augmentation
```

### Status
- **keep** - Score improved or stayed same
- **discard** - Score worse than baseline
- **crash** - Experiment failed

## Examples

```bash
# Local opencode backend
./run_autohar_opencode.sh --rounds 10

# Run with a markdown mission brief
./run_autohar_opencode.sh --rounds 10 --mission-file missions/example_mission.md

# Let the mission file decide backend and runtime settings
./run_mission.sh missions/example_mission.md

# Never-stop with git
./run_autohar_opencode.sh --never-stop --materialize-git

# With time budget (5 min per experiment)
./run_autohar_opencode.sh --never-stop --time-budget 300

# Simple loop (karpathy style)
./run_autoresearch.sh --max-experiments 100

# Resume previous run
./run_autohar_opencode.sh --resume --rounds 10
```

## Testing

```bash
conda activate autohar-whar
pytest tests/ -q
```
