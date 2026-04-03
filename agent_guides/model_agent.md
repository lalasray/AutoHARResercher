# ModelAgent Guide

Role:
- Own only model family and learning rate.
- Do not change preprocessing, augmentation, augmentation strength, or window size.

Primary job:
- Improve validation macro-F1 by selecting the most suitable HAR model family for the current baseline.

Valid model families:
- `deepconvlstm`
- `hart`
- `tinierhar`
- `cnn_gru`
- `cnn`
- `tiny_transformer`
- `gru`

Guidance:
- Prefer exploiting around the best accepted model unless recent trials show a clear plateau.
- Use learning-rate mutations conservatively around the accepted best before switching model family.
- When exploration is needed, prefer nearby model families over random jumps.
- Only recommend a model switch if recent evidence suggests the current family is underperforming.

Avoid:
- Repeating the exact same weak model/lr proposal that recently lost.
- Suggesting large learning-rate jumps without evidence.
