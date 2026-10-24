# AutoResearcHAR

Multi-agent search for IMU human activity recognition, based on [AutoResearcHAR](https://doi.org/10.1145/3830727.3834823). PyTorch handles training and GPU augmentation; Codex or OpenCode supplies agent reasoning.

Separate agents optimize data preprocessing and class sampling, model architecture and training, and class-specific augmentation. An orchestrator coordinates proposals; a judge reviews them before evaluation. Agents can change configurations and implementation code. Candidates are promoted only when source-validation macro-F1 improves.

## Setup

Requires Python 3.10+, a CUDA-enabled PyTorch installation, and an authenticated Codex CLI or configured OpenCode provider.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,mcp]'
```

## Data

Create a [manifest](examples/manifest.json) listing `.npy` recordings, subjects, activities, channels, units and sampling rates. Each array must contain one contiguous, single-activity recording with shape `[time, channels]`. Keep raw signals unnormalized and preserve recording boundaries.

The optional WHAR adapter can export recordings:

```bash
pip install -e '.[whar]'
python -m autohar_researcher.whar_export \
  --spec examples/whar-uci-har.json --output data/prepared-uci-har
```

## Run

```bash
autohar search --manifest data/prepared-uci-har/manifest.json \
  --protocol loso --target uci_har \
  --backend codex --model YOUR_MODEL_ID --device cuda \
  --output artifacts/run --final-test
```

- **Fixed-person search (default):** add `--validation-subject ID` to choose the validation person; otherwise the split seed chooses one deterministically. All other people train the candidates.
- **Full LOSO search:** add `--search-mode loso` to run a separate search for every test person, or `--search-mode loso --subject ID` for one test fold.
- **Local LLM:** use `--backend opencode --model LOCAL_PROVIDER/MODEL`.
- **Short run:** add `--seeds 13 --rounds 1 --budget 8`.
- **Parallel fits:** add `--candidate-workers 2` (1–4 supported; requires sufficient GPU memory).
- **Custom start:** use `--baseline examples/baselines/tinyhar.json`, or another pipeline JSON.
- **Fixed pipeline:** add `--method fixed`; no LLM is needed.
- **Inspect splits:** replace `search` with `splits` and omit search options.

UCI-HAR is the default target. Every dataset starts from the same [generic CNN](examples/baseline.json), with 2.56-second windows, 50% overlap, training-fitted standardization and no augmentation. Agents can switch architectures or implement custom models. Available [presets](examples/baselines) include TinyHAR, TinierHAR, DeepConvLSTM, BiLSTM-Attention, HART, MobileHART and SPECTRA; implementation sources and limitations are recorded in [architecture provenance](research/vendor/ARCHITECTURES.json).

Defaults: 100 rounds, 300 candidate fits, seeds 13/14/15/16/17, and CUDA. Use a fresh output directory for each invocation. CPU execution requires explicit `--device cpu`.

## Evaluation

By default, search optimizes on one fixed validation person per seed. With `--final-test`, it freezes the selected pipeline and runs LOSO evaluation across all people, training each fold on everyone except its test person. Epoch counts and learning-rate schedules come from the fixed-person search. This is exploratory LOSO because people in the evaluation also participated in pipeline selection.

With `--search-mode loso`, each test person has an independent search using a distinct validation person. Final fitting combines that fold's training and validation people before testing once. LODO remains available with `--protocol lodo` and a combined manifest.

This LOSO protocol differs from the paper's source split; matching its reported scores is not guaranteed. Data preparation runs on the CPU; model training and built-in augmentation run on the selected PyTorch device.

## Augmentation and synthetic data

Agents can choose [augmentation methods](examples/augmentations), strengths, probabilities and per-class policies. The data agent can also adjust class retention and sampling. Validation and test samples are never augmented.

For synthetic data, pass `--mcp-config PATH`. The [IMUGPT server configuration](examples/imugpt-server.json) connects the included MCP wrapper to an official IMUGPT installation with its upstream environment and checkpoints. IMUTube requires access to the official implementation; contact its original authors.

## Results and checks

Each seed/fold saves `history.jsonl`, `selected.json`, split and initialization records, and round-level proposals and reviews. `--final-test` also saves a checkpoint and `final-test.json`.

```bash
pytest -q
```
