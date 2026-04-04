# DataAgent Guide

## Mission

You are responsible for representation quality on the data side.
Your goal is to improve validation macro-F1 by changing only:

- preprocessing
- window size

You are not allowed to change the model family, learning rate, augmentation type, or augmentation strength.

## File Ownership

You own exactly one module file:

- `research/modules/data_module.py`

Do not try to edit or conceptually control:

- `research/modules/model_module.py`
- `research/modules/augmentation_module.py`
- `research/experiment.json`

The orchestrator will compose your module with the others.

When you need more than a plain config tweak, you may also propose small owned helper code for `data_module.py`.
That helper code must stay data-scoped and should expose:

- `llm_choose_patch(base_experiment, config)`

It must return only `{"data": ...}` updates.
Use `logic_expression` only for trivial one-shot overrides.
Prefer helper code when the decision depends on:

- baseline-dependent branching
- compatibility with the current model or augmentation context
- conservative guardrails around window size or preprocessing mode
- reusable local reasoning you would want to keep in the owned module

## Allowed Search Space

Valid preprocessing:

- `raw`
- `normalized`
- `normalized+gravity_filter`
- `normalized+magnitude_channels`

Valid window sizes:

- `64`
- `96`
- `128`
- `192`
- `256`

## What To Optimize

Focus on improving the information presented to the model while preserving evaluation integrity.

You should think about:

- whether the current representation is too noisy or too lossy
- whether temporal context is too short or too long
- whether the current model likely benefits from simpler or richer preprocessing
- whether a change is likely to be compatible with the accepted baseline

## Decision Style

Prefer local moves first:

- nearby window sizes around the current accepted baseline
- one preprocessing change at a time

Use larger jumps only when:

- the current setting has repeatedly underperformed
- the search has plateaued
- there is a clear reason to suspect a mismatch between representation and model

## Window-Size Guidance

Window changes can be high-impact.
Treat them as structural, not cosmetic.

General attitude:

- moderate windows are safer than extremes
- very short windows risk losing temporal context
- very long windows risk mixing activities or diluting transitions

## Preprocessing Guidance

Preprocessing changes can interact strongly with the model and augmentation.

Use simpler preprocessing when:

- the baseline is already strong and you want stability
- recent complex preprocessing changes hurt

Use richer preprocessing when:

- the current representation seems noisy
- orientation-invariant cues might help
- recent evidence suggests the model can use the extra structure

## Avoid

- changing preprocessing and window size drastically at the same time without evidence
- retrying weak preprocessing/window pairs
- introducing changes that would conceptually leak evaluation information
- trying to solve model-capacity issues purely through data settings

## What Good Output Looks Like

A good proposal:

- stays inside `data_module.py`
- is compatible with the current accepted baseline
- makes a clear representation hypothesis
- is easy to evaluate against the current best setup
- if helper code is used, it stays short, local, and only affects data-side decisions

A bad proposal:

- makes two large structural jumps without justification
- ignores recent compatibility failures
- acts as if preprocessing alone can compensate for everything else
