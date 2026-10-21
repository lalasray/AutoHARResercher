# AutoResearcHAR

A reconstruction of the search workflow and evaluation protocol in **AutoResearcHAR: A Multi-Agent Framework for Dataset Adaptive IMU Based Human Activity Recognition**, ISWC 2026, [DOI 10.1145/3830727.3834823](https://doi.org/10.1145/3830727.3834823).

The runtime trains sequence models on `[batch, time, channels]` tensors. Search uses source-validation scores, independent proposal review, and MCP-backed text/video generation.

LOSO uses one held-out test person and one distinct validation person per fold, cycling through all people. This is an explicit protocol adaptation of the paper’s 80/20 source split. Unreported settings use documented reconstruction choices from primary sources, recorded below. This repository does **not** claim that new runs reproduce the paper's reported numbers.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,mcp]'
pytest -q
```

Use Python 3.10+ with NumPy, SciPy, PyTorch and einops. All runtime architectures and training use PyTorch; TensorFlow is not required. Install a CUDA-enabled PyTorch build compatible with your GPU. The optional `whar` extra installs the WHAR adapter dependency. `environment.yml` is an alternative environment specification.

## Prepare recordings

Use a JSON manifest shaped like [examples/manifest.json](examples/manifest.json). Each `.npy` contains one contiguous, single-activity recording as an **unnormalized time × channel** array. Preserve subject/session identities; never concatenate different subjects, activities, or discontinuous recordings. The example illustrates the schema and does not contain a runnable dataset.

Dataset metadata declares the channel order, nominal sampling rate, multiplicative conversion to physical units, ordered label space, and documented raw-label mapping. These values are fixed before search. Optional recording-level `sampling_hz` supports source rates documented per session. Do not derive eligibility or labels from target performance. Unknown labels, repeated recording paths, inconsistent channels, and empty partitions fail explicitly.

For WHAR raw parsers, download/extract the dataset as required by WHAR, review the harmonization specification, then export:

```bash
pip install -e '.[whar]'
python -m autohar_researcher.whar_export \
  --spec examples/whar-uci-har.json --output data/prepared-uci-har
```

The exporter calls WHAR's raw parser and preserves its subject/session metadata; it does not use WHAR's window splits or postprocessing normalization. Parser subject IDs may differ from original dataset IDs. The UCI example uses total acceleration and angular velocity, converting g to m/s². Review the source-column and label names against the installed WHAR version. For LODO, export all three eligible datasets in one specification/manifest, using their documented channel/unit mappings. Complete manifests for the other nine datasets are not bundled.

Inspect the deterministic split without opening signal arrays:

```bash
autohar splits --manifest data/prepared-uci-har/manifest.json \
  --protocol loso --target uci_har --all-subjects
```

For LOSO with N people, each fold holds out one test person and one **different**
validation person; the remaining N−2 people train the candidate. A seeded cycle
assigns the next person as validation (wrapping at the end), so a complete run
uses every person exactly once for testing and once for validation. At least
three people are required. This follows the user's clarified protocol rather
than an 80/20 validation-subject split. Validation and test never share a person.

## Run

Use **OpenCode for local LLMs** and **Codex for remote LLMs**. Codex is the default backend and uses your existing Codex authentication. Set an explicit model ID; no model is silently selected. Export variables from `.autohar.env.example` if desired.

```bash
autohar search --manifest data/prepared-uci-har/manifest.json \
  --protocol loso --target uci_har --all-subjects \
  --device cuda \
  --backend codex --model YOUR_REMOTE_MODEL_ID --output artifacts/uci-har --final-test
```

LOSO cycles through all people by default; `--all-subjects` is an explicit equivalent.
For a short development run, use `--subject ID --seeds 13 --rounds 1 --budget 8`. Training still uses real data; reduce epochs in a separate baseline file for a smoke test. Such a run is not a benchmark result.

For a local model, configure its local provider in OpenCode, then run:

```bash
autohar search --manifest data/prepared-uci-har/manifest.json \
  --protocol loso --target uci_har --subject 0 --seeds 13 \
  --backend opencode --model LOCAL_PROVIDER/MODEL --output artifacts/opencode
```

The OpenCode adapter uses a temporary working directory, a dedicated JSON-only agent, disabled sharing/plugins, and denied agent tools. Configure a local provider (for example LM Studio or Ollama) in OpenCode; the adapter does not infer whether a configured provider URL is local. Temperature/top-p are set explicitly; the provider must honor the 4096-output-token option.

The Codex adapter uses [noninteractive `codex exec`](https://developers.openai.com/codex/noninteractive/) with an explicit remote model, an ephemeral session, a temporary working directory, read-only sandbox, and disabled shell, browsing, apps/plugins and delegation features. It ignores user CLI configuration/rules while retaining Codex authentication. It sends prompts over stdin and reads only the final response file; malformed JSON, timeouts and CLI failures fail the proposal. Install a current Codex CLI supporting the flags shown by `codex exec --help`; use `--codex-bin /path/to/codex` or `--opencode-bin /path/to/opencode` when needed. The adapter does not expose the paper's exact sampling/token controls through Codex: those metadata fields are recorded as null, not claimed as enforced. Backend tests use fixture CLI responses; live model calls have not been verified.

The protected Python harness executes proposed code and generator calls.

LODO uses `--protocol lodo --target uci_har` (or `motion_sense`, `real_life_har`) and a combined manifest. Every source dataset is split by subject independently. Candidate source subsets must retain at least 100 training windows in each of the five fixed shared classes. Validation always includes every fixed source-validation domain, even if a candidate trains on fewer domains.

Defaults are 100 rounds, at most 300 empirical evaluations, and independent search seeds **13, 37, 73, 101, 131**. The split seed stays fixed across searches. Candidate training is sequential; specialist proposal/Judge tasks are concurrent. Use a new output directory for each invocation.

Use `--method single-agent` for the AutoResearch-style controlled comparator:
one generalist, no specialist planning/composition or independent Judge, with the
same harness, baseline, edit limits, generator access, and total fit budget.
Use `--method fixed --baseline PATH` to train/evaluate an explicit fixed baseline without an LLM. The
single-agent runner makes up to three sequential proposals per round; this is
an explicit reconstruction of the unpublished baseline scheduling details.

## Protocol and search

| Paper requirement | Implementation |
| --- | --- |
| Fixed LOSO/LODO protocol | Immutable `Task`; LOSO: one test + one validation person, N−2 training people; LODO: per-source 80/20 subject split |
| Target isolation | Only documented metadata goes to agents; target arrays are first opened in `finalize` |
| Harmonization | Fixed channel order/unit conversion and SciPy 50 Hz polyphase FIR resampling |
| Label space | Predeclared mapping; Macro-F1 includes every fixed class, including absent classes |
| Data Agent | Eligible source selection, class sampling, normalization, 1–5s windows, 0–75% overlap, bounded preprocessing code |
| Model Agent | Sequence architecture and bounded model code, optimizer/loss/training settings |
| Augmentation Agent | Training-only transformations and class-specific MCP synthesis requests |
| Mutation bounds | At most 3 files, 3 changed functions, 8 changed config fields, 120 added/deleted lines per proposal |
| Judge | Integer 1–5 rubrics; novelty ≥2, plausibility ≥3, compatibility ≥4; malformed responses reject |
| Deterministic checks | Syntax, import availability, ownership, configuration, dependencies, class coverage, tensor shape/finiteness |
| Composition | Eligible singletons plus Orchestrator-selected compatible subsets, selected before training |
| Selection | Source-validation Macro-F1; equal source-dataset weighting for LODO; strict improvement only |
| Final evaluation | Freeze pipeline, refit selected source train+validation, recompute normalization, test once |

Source standardization is fitted before synthetic samples are appended. Validation/test receive neither augmentation nor resampling of subjects. Overlapping windows never cross recording boundaries. Training uses fixed source validation for early stopping; there is no random inner window split.

A candidate's custom code lives inside its serialized pipeline, with one owned file per specialist:

- `research/components/data.py`: `preprocess(x)`, a stateless, shape-preserving transformation.
- `research/components/model.py`: `build_model(input_dim, num_classes, params)` returning a PyTorch module.
- `research/components/augmentation.py`: `augment(x, rng)`, used only for training batches.

These virtual files need not exist in the checkout. Configuration changes are partial dictionaries in the specialist's section. The mutation checker compares against the current baseline. Candidate files cannot replace evaluator files. Static checks are not an operating-system security sandbox; use an isolated execution environment for untrusted generated Python. Source/target isolation tests verify the supported interfaces, not arbitrary hostile Python.

## Synthetic data

Install the `mcp` extra and pass `--mcp-config path/to/servers.json`. Each `text` or `video` entry defines a fixed stdio server `command`, `args`, `tool`, optional `env`, and (for video) `allowed_videos`. Server identities and source video eligibility cannot be changed by agents.

The tool request contains `label`, `count`, `sampling_hz=50`, `window_steps`, `channels`, and `seed`; text also gets `prompt`, video gets `videos` and `selection`. The class-specific `ratio` caps added synthetic windows relative to real training windows for that class, additionally bounded by the requested `count`. Responses contain `samples` with shape `[count, window_steps, channels]` in the manifest's physical units/channel order. Missing tools, wrong shapes, invalid labels, or non-finite outputs fail. Real generator wrappers must implement this contract; no generator executable or checkpoint is invented.

The included [IMUGPT MCP server](autohar_researcher/imugpt_mcp.py) runs the official [IMUGPT repository](https://github.com/ZikangLeng/IMUGPT) through a separate upstream Python environment: CLIP → T2M-GPT → VQ-VAE → inverse kinematics/BVH → ideal IMUSim accelerometer **and gyroscope**. The search agent supplies the motion description; the upstream API-key-dependent prompt-generation notebook is not needed. No artificial or zero-filled sensor fallback is used.

Setup:

1. Clone the official IMUGPT repository and check out `4b7a73fa7c8b6fdbd18a389b902c0d935a77fc4f`. Follow its environment and IMUSim installation instructions in a separate environment. Its quantizer requires NVIDIA CUDA. The MCP host uses this project's modern Python environment.
2. In that checkout, run the official `dataset/prepare/download_model.sh` and `dataset/prepare/download_extractor.sh` to obtain the model weights and normalization arrays. Follow upstream asset terms. This worker uses skeleton inverse kinematics, not SMPL mesh rendering; it does not require the optional SMPL body model or GloVe evaluation assets. CLIP downloads its weights on first use.
3. Copy [examples/imugpt-server.json](examples/imugpt-server.json), set absolute repository and upstream Python paths, and review `channel_map`. The example emits Spine1 local-frame accelerometer XYZ in m/s² followed by gyroscope XYZ in rad/s. It is a waist-placement approximation, not an exact match for every dataset. Change the joint, ordering, axis signs and unit scales to match your manifest; the prompt cannot change physical placement. Supported joints are those in upstream `sensor.py`'s `realworld`, `pamap`, and `usc-had` presets. Different channel layouts require different server configurations.
4. Copy [examples/mcp-servers.json](examples/mcp-servers.json), replace its absolute paths, then pass it to search with `--mcp-config`. Install the host dependencies with `pip install -e '.[mcp]'`.

```bash
python -m autohar_researcher.imugpt_mcp --config /absolute/path/to/imugpt-server.json --check
python -m autohar_researcher.imugpt_mcp --config /absolute/path/to/imugpt-server.json
```

`--check` validates checkout revision, asset existence and sensor mapping; it does not load models or verify CUDA/dependency compatibility. The second command serves MCP over stdio. Upstream logs go to stderr. Requests run in temporary directories with a timeout and deterministic request seeds. Each output window comes from a separate sampled motion, cropped at a seeded random offset; short motions are retried within a bounded attempt limit, never looped or padded. The motion model runs at its native 20 Hz and upstream spline-based IMUSim samples the resulting trajectory at 50 Hz. Categorical motion sampling, a default body scale of 1.7/1.55, ideal sensors and no empirical calibration are explicit reconstruction choices. Motion sampling follows the upstream model's categorical option rather than repeating its deterministic greedy output. No IMUGPT 2.0 LLM motion filter/diversity stopping is claimed. Returned provenance records the revision and channel mapping. The official model checkpoints and real CUDA generation have not been validated in this checkout; transport and contract tests use fixtures.

**IMUTube requires access to the official IMUTube implementation. Contact the original authors to obtain access and licensing instructions via the [official IMUTube repository](https://github.com/hyeokhyen/imutube).** No IMUTube replacement is bundled. Configure a video MCP server only after obtaining that access.

[examples/synthetic-policy.json](examples/synthetic-policy.json) contains reconstructed class prompts and mixing settings. The default baseline remains real-only; the augmentation agent can request IMUGPT once the text server is configured.

## Starting baseline

Search selects its starting **model + preprocessing/windowing + augmentation**
combination from a dataset-specific candidate pool. The
[source registry](research/baseline_registry.json) records primary URLs, pinned
code revisions, supported settings and explicit adaptations for all ten datasets.
[Reviewable candidate pools](examples/baseline-pools/) contain complete pipeline
configurations, including each model's training preset.

| Dataset | Window durations considered | Architecture candidates |
|---|---|---|
| WISDM | 4s, 2s | TinierHAR, TinyHAR |
| UCI-HAR | 2.56s | BiLSTM-Attention, TinierHAR |
| UTD-MHAD | 4s, 2s | TinierHAR, DeepConvLSTM |
| HAPT | 4s, 1s | TinierHAR, BiLSTM-Attention |
| USC-HAD | 4s, 2s | TinierHAR, DeepConvLSTM |
| UniMiB-SHAR | 3.02s / 151 samples | TinyHAR, BiLSTM-Attention |
| MotionSense | 4s, 2.56s | TinierHAR, BiLSTM-Attention |
| RealLifeHAR | 5s | BiLSTM-Attention, TinyHAR |
| Hang-Time | 1s | DeepConvLSTM (one 1024-unit recurrent layer), TinyHAR |
| CAPTURE-24 | 5s | TinyHAR, BiLSTM-Attention |

Each LOSO pool has four complete candidates: the first architecture's reference
configuration, a source-training min-max variant, an augmentation variant, and
the second architecture. Most reference candidates use source-training channel
standardization; CAPTURE-24 uses unnormalized physical signals and balanced
sampling. UniMiB-SHAR and CAPTURE-24 use zero overlap; others use 50%.
Augmentation candidates use conservative rotation, except CAPTURE-24, whose
candidate combines time and magnitude warping from its author configuration.
Different methods, strengths and probabilities remain searchable by the agents.

[TinierHAR's dataset configurations](https://github.com/zhaxidele/TinierHAR/blob/f2f1bbd7305689c374fb38cbb7ce853e7cefbb3f/configs/data.yaml)
supply several primary window durations. The short comparison windows are local
hypotheses. [Hang-Time's benchmark](https://ahoelzemann.github.io/hangtime_har/)
supplies its window/recurrent dimensions.
[UniMiB-SHAR](https://doi.org/10.3390/app7101101) supplies the event-window geometry.
[CAPTURE-24](https://github.com/OxWearables/capture24/tree/f861b44f5675cb3e8294cd3d560d7a71a749616f)
uses 10-second source windows; this pool adapts them to the protected 5-second
maximum. [RealLifeHAR's correction](https://doi.org/10.3390/s20164650) documents
20-second source segmentation, also adapted to 5 seconds. These component-based
candidates are not claims of published optimal combinations or equivalent scores.

Every seed/fold evaluates the compatible candidates on **source validation only**.
The highest macro-F1 wins; ties keep the first candidate. Failed fits consume
budget, preflight rejections do not, and the winner is not retrained before agent
search. All initialization fits count toward the same 300-fit maximum. A budget
smaller than the planned pool is rejected before fitting; increase `--budget` or
supply one explicit `--baseline`. Incompatible candidates and failed fits are
recorded; initialization fails if none succeeds.

For LODO, the pool is the deduplicated union of **source-dataset** candidates plus
a fixed MobileHART transfer candidate. Target identity, samples, fitted statistics
and test scores do not determine that pool or its winner. Both search methods use
the same selection procedure and budget accounting. The selected initialization
may differ across folds/seeds because validation is different; this is an explicit
initialization design, not a claim to recover the paper's unpublished starting
pipeline. Full-dataset results require running these experiments.

Pass `--baseline PATH` to use exactly one supplied pipeline. For example,
[examples/baseline.json](examples/baseline.json) is the complete BiLSTM-Attention
configuration, and [examples/baselines](examples/baselines) contains each
architecture's preset. Its convolution widths are 256/128/64/32, with two residual
64-unit bidirectional LSTMs, temporal attention and output batch normalization.
The PyTorch port is checked against pinned author weights; TensorFlow is only an
optional reference-test dependency.

## Architecture fidelity

[Architecture provenance](research/vendor/ARCHITECTURES.json) records sources, revisions, training settings and unresolved differences. [SOURCES.json](research/vendor/SOURCES.json) records SHA-256 hashes for the unchanged author files. Integrity is checked before model construction. Run metadata, selected pipelines and final checkpoints record the implementation status and preset overrides.

| Architecture | Implementation used |
| --- | --- |
| TinyHAR | Pinned author PyTorch code, with transformer/FC/LSTM/temporal-attention settings; only input layout is adapted |
| TinierHAR | Pinned author PyTorch code and model configuration, including residual Conv2d blocks and bidirectional GRU |
| CNN–ResBiLSTM–Attention | PyTorch port of author definitions; explicit task-dimension adaptations |
| HART / MobileHART-XS | Separate PyTorch ports, including the original sensor-wise attention and liteFormer; six ordered acceleration/gyro channels required |
| DeepConvLSTM | Port of the author Lasagne inference graph, restoring peephole LSTMs; paper training dropout, orthogonal matrix initialization and Lasagne RMSProp update rule |
| SPECTRA | Paper reconstruction with STFT, separable convolution, channel attention, BiGRU and attention pooling; unspecified dimensions/reshape/settings are documented assumptions |
| DeepConvLSTM-Attention | Pinned TECO benchmark implementation; explicitly not claimed as the original attention-paper author's code |

TinyHAR retains the released implementation's alternating convolution strides 1/2/1/2. Its paper prose describes a different stride pattern; that difference is recorded rather than silently mixing versions. SPECTRA's paper does not specify all FFT/model dimensions or the reshape between its convolution and sensor-attention stages, so identical parameter counts or scores cannot be claimed. DeepConvLSTM uses a PyTorch implementation of the author architecture; its recurrence is independently checked against the Lasagne equations. Its paper prints learning rate `10e-3`, interpreted literally as 0.01 in the preset; 300 maximum epochs is an explicit assumption because the inference-only release does not supply the training horizon.

Use the complete configurations in [examples/baselines](examples/baselines), for example:

```bash
autohar search --manifest data/prepared-uci-har/manifest.json \
  --protocol loso --target uci_har --method fixed \
  --baseline examples/baselines/tinierhar.json --device cuda \
  --output artifacts/tinierhar --final-test
```

Changing `architecture` resets model settings to that architecture's preset before applying other explicit model changes. This prevents BiLSTM dimensions and Nadam settings leaking into TinyHAR/TinierHAR. These preset selections count as one architecture choice under the mutation rules; all effective settings are serialized. Explicit hyperparameter changes produce variants, recorded as preset overrides. The generic `cnn`, `cnn_gru`, `gru`, `bilstm` and `tiny_transformer` alternatives are experimental models, not claimed reproductions of a named paper.

TinierHAR's validation-loss LR schedule is preserved. Final train+validation refitting replays the LR history observed during source-only selection; it never asks the test set to drive scheduling. The shared harness still selects/early-stops on source Macro-F1. Full result replication also requires the original data preparation, splits, training versions and seeds. Your one-validation-person/one-test-person cycle remains in force, so this is not a claim to recreate the source papers' original evaluation protocols.

## Reconstructed defaults and sources

The user requested plausible literature-based values for details omitted from the paper. [examples/defaults-provenance.json](examples/defaults-provenance.json) records the distinction between paper requirements, externally supported choices, and implementation assumptions.

| Unreported detail | Selected reconstruction and basis |
| --- | --- |
| Initial architecture | Dataset-specific candidate pool; winner selected by source-validation macro-F1 |
| Initial training | Per-architecture presets serialized for every candidate; source-validation checkpoint selection |
| Initial windowing | Dataset-specific durations/overlap in the source registry; protected bounds remain 1–5s and 0–75% overlap |
| Initial regularization | Per-architecture settings and explicit candidate augmentation policy |
| Split membership | Sorted subject IDs, NumPy seeded shuffle with seed 13; LOSO validation is the next person in the full cyclic order; LODO retains ceil(20%) validation subjects per source. Frozen across search seeds |
| Initial normalization/sampling | Standardization, min-max or physical signals; all fitted statistics are source-training-only; candidate sampling policy is explicit |
| Final epoch count | Integer median of best validation epochs recorded for the selected pipeline in that search; ordinarily one observation, because duplicate candidates are rejected |
| Candidate budget | Each initialization/search fit consumes one evaluation; failed fits count and preflight rejections do not; at most 300 total fits |
| Synthetic defaults | Real-only initial pipeline; optional 100 samples per requested class at 0.25 synthetic-to-real ratio cap, ontology prompts, video selection `all`. Counts/ratios are explicit reconstruction choices |

Author-graph equivalence tests cover the architecture ports; full dataset scores have not been reproduced. This project implements the main search workflow; it does not supply all fixed LODO comparison methods, pretrained pools, ablation/statistical analysis scripts, or recreate Tables 2–6. Full-dataset and real LLM/generator benchmark runs remain necessary before reporting results.

## Outputs and development

Each seed/fold directory records `run.json`, `splits.json`, `initialization.json` (candidate evidence, failures, validation scores and selected starting pipeline), per-round proposals/Judge scores/mutation counts/compositions, append-only `history.jsonl`, and `selected.json`. With `--final-test`, it also writes a checkpoint, fitted normalization statistics, and a separate `final-test.json`; target scores are never appended to search history or sent back to agents. An exclusive persistent marker prevents a second final test in the same directory, including after a failed final attempt. Cross-fold test scores are averaged within each seed, then reported with sample standard deviation across seeds.

```bash
pytest -q
python -m compileall -q autohar_researcher research
```

The CLI uses a shared evaluation path for fixed baselines, single-agent search, and multi-agent search.

### Research-derived augmentations

The search agents can select and configure 15 methods from the source-linked
[catalog](research/augmentations.py): jitter, scaling, 3D rotation, magnitude
warp, time warp, permutation, window slice, window warp, time mask, frequency
mask, channel dropout, SPAWNER, random guided warp, Mixup, and temporal CutMix.
[Runnable pipeline overrides](examples/augmentations/) include each method and
a composed policy; pass one through the existing `--baseline` option.

Implementations draw on [Um et al.'s wearable augmentation code](https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data)
and the [Iwana–Uchida time-series augmentation repository](https://github.com/uchidalab/time_series_augmentation),
with pinned upstream revisions in the catalog. Mixup follows
[the original method](https://github.com/facebookresearch/mixup-cifar10).
Temporal CutMix adapts [CutMix](https://github.com/clovaai/CutMix-PyTorch) to
contiguous time spans. Raw time masking and FFT masking are adaptations inspired
by [SpecAugment](https://arxiv.org/abs/1904.08779), not its original spectrogram
implementation. Channel dropout is an input-channel adaptation of dropout.
These are local implementations with explicit defaults, not bit-identical
upstream code or a claim that the AutoResearcHAR paper evaluated every method.

Policies accept `params` and per-window `probability`. Use `kind: compose`
with 1–4 ordered `transforms`, then an optional `mixing` policy. A composition's
top-level probability multiplies each signal step's probability independently;
mixing has its own probability. The `strength` shorthand is supported for jitter,
scaling, and rotation; explicit method parameters override it.

All transforms run on training batches only. SPAWNER and random guided warp
choose distinct donors of the same class within the batch; singleton classes
remain unchanged. Mixup and temporal CutMix produce soft targets consumed by
PyTorch training, including weighted losses. CutMix
targets use the actual replaced time fraction. Rotation applies the same proper
3D rotation to complete XYZ triplets in physical coordinates, then reapplies
the training normalizer. Temporal mappings are shared across channels.

All built-in augmentations use PyTorch on the selected device, including spline
solves, FFT masking, label mixing and DTW cost computation. DTW traceback uses
small CPU control indices and can be expensive. A seeded device generator controls
randomness; PyTorch and NumPy do not produce identical random sequences.
Transform suitability depends on the activity and sensor placement; source
validation selects policies. Initialization compares unaugmented and augmented candidates.

Search defaults to `--device cuda` and fails if CUDA is unavailable; it never silently switches training to CPU. Select `--device cuda:1` for another GPU, or explicitly pass `--device cpu` for CPU debugging. The Python harness also defaults to CUDA; pass `device="cpu"` explicitly for CPU debugging.

PyTorch DataLoader batches the source arrays in pinned host memory for CUDA runs
and transfers batches nonblocking to the selected GPU. Augmentation, forward
passes, loss, gradients and optimizer updates remain on that GPU. File decoding,
fixed resampling, source-only normalization fitting and window construction run
on the host; data loading does not require the entire dataset to fit in VRAM.
Custom augmentation functions receive a device-resident torch.Tensor and
torch.Generator and must return a tensor on the same device.

The three architecture ports are checked against pinned author graphs by
transferring weights and comparing outputs (and HART/MobileHART input gradients).
They preserve Keras population-variance batch normalization, per-head attention
dimensions and released liteFormer reshaping/weight projection. PyTorch RNG and
optimizers differ from TensorFlow; benchmark scores still require full runs.
TensorFlow is only an optional test oracle, not an install or runtime dependency.

Class-specific conventional augmentation is configured with `augmentation.per_class`:
see [the per-class example](examples/augmentations/per-class.json). Keys must
match the task's fixed activity names. Each entry is a complete replacement for
the global conventional policy, with its own method or composition, parameters,
probabilities and optional mixing. Unlisted classes use the global policy;
`{"kind": "none"}` disables all conventional augmentation for that class.
Nested class policies, custom code and synthetic requests inside entries are
rejected; synthetic generation retains its existing separate class-specific
request list.

The policy is selected by each recipient window's original hard label.
Mixup/CutMix donors can come from any class in the training batch, and soft
targets reflect the actual contributions. DTW donors remain same-class.
Policies run on the selected PyTorch device during training only, without
changing validation/test samples or class sampling counts. Probabilities control
how often a training window is augmented, not how many extra windows are added.
Resolved per-class settings and source references are saved in provenance.
