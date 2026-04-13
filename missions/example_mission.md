---
backend: auto
rounds: 20
resume: false
materialize_git: true
auto_push_non_main: false
never_stop: false
---

# AutoHAR Mission Brief

## Goal

Improve validation macro-F1 on UCI HAR while keeping the search interpretable and stable.

## Priorities

- Prefer real gains over noisy one-off jumps.
- Encourage specialist diversity before forcing combinations.
- Let the orchestrator reuse historically strong specialist slices when current round proposals are weak.
- Use bounded custom code only when it is likely to express a meaningful new idea.

## Search Guidance

- ModelAgent:
  Focus on strong representation changes, not only tiny learning-rate tweaks.
- AugmentationAgent:
  Explore both standard augmentations and bounded custom augmentations when the baseline is plateauing.
- DataAgent:
  Explore preprocessing and windowing changes that fit a single waist-mounted smartphone IMU setup.

## Acceptance Guidance

- Keep the current accepted baseline unless a candidate is measurably better.
- If multiple candidates are close, prefer the simpler and more interpretable one.
