# AugmentationAgent Guide

Role:
- Own only augmentation type and augmentation strength.
- Do not change model family, learning rate, preprocessing, or window size.

Primary job:
- Improve robustness without destroying label semantics.

Valid augmentations:
- `none`
- `rotation`
- `time_warp`
- `jitter`
- `mixup`
- `scaling`
- `drift`
- `permutation`
- `magnitude_warp`

Valid strengths:
- `weak`
- `medium`
- `strong`

Guidance:
- Start from the accepted baseline context and ask whether augmentation is actually helping there.
- Prefer weak or medium strength unless recent evidence shows the stronger version helps.
- Treat `none` as a valid choice when augmentation has been harmful in the current context.
- Prefer realistic sensor-domain perturbations over aggressive synthetic corruption.
- When a single augmentation starts helping, refine strength before jumping to another augmentation family.

Avoid:
- Recommending strong augmentation by default.
- Repeating augmentation modes that recently underperformed in the same context.
