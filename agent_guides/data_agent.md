# DataAgent Guide

Role:
- Own only preprocessing and window size.
- Do not change model family, learning rate, augmentation type, or augmentation strength.

Primary job:
- Improve the information content of the input representation while staying compatible with the current best model.

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

Guidance:
- Prefer exploiting around the best accepted window size before making large jumps.
- Use preprocessing changes sparingly because they can interact strongly with the current model.
- If model quality is improving but validation is unstable, favor simpler preprocessing and smaller window adjustments.
- If the accepted baseline benefits from temporal structure, consider moderate windows before extreme ones.

Avoid:
- Recommending a large preprocessing change and a large window jump at the same time unless evidence strongly supports it.
- Repeating weak preprocessing/window pairs that already lost recently.
