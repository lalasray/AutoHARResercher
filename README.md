# AutoHARResearcher

Multi-agent autonomous HAR research. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Quick Start

```bash
pip install -e ".[dev]"
cp .autohar.env.example .autohar.env

# Edit .autohar.env with your backend (see below)

# Run
./run_autohar_lmstudio.sh --never-stop --materialize-git
```

## Shell Scripts

| Script | Description |
|--------|-------------|
| `./run_autohar_lmstudio.sh` | Use local LM Studio (no API key) |
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

### `--max-rounds-without-improvement N`
In never-stop mode, stop after N rounds without improvement. Default: 10.

### `--max-experiments N` (autoresearch loop only)
Run exactly N experiments, then stop.

## Environment Variables

Edit `.autohar.env`:

```bash
# Backend: openai | lmstudio | heuristic
AUTOHAR_LLM_BACKEND=lmstudio

# OpenAI
OPENAI_API_KEY=sk-...
AUTOHAR_OPENAI_MODEL=gpt-4o-mini

# LM Studio
AUTOHAR_LLM_BASE_URL=http://127.0.0.1:1234/v1
AUTOHAR_LLM_API_KEY=lm-studio

# Never-stop mode
AUTOHAR_NEVER_STOP=true

# Time budget per experiment
AUTOHAR_TIME_BUDGET_SECONDS=300
```

## Architecture

```
research/modules/
  model_module.py          # ModelAgent edits this
  augmentation_module.py   # AugmentationAgent edits this
  data_module.py          # DataAgent edits this

agent_guides/            # Agent instructions
  model_agent.md
  augmentation_agent.md
  data_agent.md
  orchestrator.md
```

Each agent owns one module file. Orchestrator combines them.

## Results

Results saved to `results.tsv`:

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
# Basic multi-agent loop
./run_autohar_lmstudio.sh --rounds 10

# Never-stop with git
./run_autohar_lmstudio.sh --never-stop --materialize-git

# With time budget (5 min per experiment)
./run_autohar_lmstudio.sh --never-stop --time-budget 300

# Simple loop (karpathy style)
./run_autoresearch.sh --max-experiments 100

# Resume previous run
./run_autohar_lmstudio.sh --resume --rounds 10
```

## Testing

```bash
pytest tests/ -v
```
