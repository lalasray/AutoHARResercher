# Model Agent Module
#
# This file is managed by the ModelAgent. Keep the file name stable.
# Edit this to change architecture or tune model-side hyperparameters.

AVAILABLE_ARCHITECTURES = [
    "deepconvlstm",
    "deepconvlstmattention",
    "bilstm",
    "bilstmattention",
    "hart",
    "tinierhar",
    "tinyhar",
    "spectra",
    "cnn_gru",
    "cnn",
    "tiny_transformer",
    "gru",
    "custom",
]

MODEL_NAME = "deepconvlstm"

# Architecture hyperparameters
HIDDEN_DIM = 64
DEPTH = 2
ATTENTION_MIX = 0.9
DROPOUT = 0.0
BIDIRECTIONAL = False

# Optimizer and loss
OPTIMIZER = "adam"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
LOSS = "cross_entropy"
LABEL_SMOOTHING = 0.0

MUTATION_SPACE = {
    "architecture": AVAILABLE_ARCHITECTURES,
    "layers": ["add", "remove", "widen", "narrow", "change_pooling"],
    "hyperparameters": [
        "hidden_dim",
        "depth",
        "attention_mix",
        "dropout",
        "bidirectional",
        "learning_rate",
        "weight_decay",
    ],
    "optimizers": ["adam", "adamw", "sgd"],
    "losses": ["cross_entropy", "weighted_cross_entropy"],
    "merge_strategies": [
        "temporal_plus_spectral",
        "attention_plus_compact",
        "recurrent_plus_convolutional",
    ],
}

DECISION_POLICY = {
    "modify": (
        "Use when the current architecture is competitive and the likely bottleneck "
        "is capacity, pooling, regularization, optimizer, loss, or learning rate."
    ),
    "switch": (
        "Use when the current architecture has plateaued or the context suggests "
        "a different temporal, spectral, convolutional, or attention bias."
    ),
    "merge": (
        "Use only when two architectures have complementary strengths. Record "
        "parents in parent_architectures and describe the merge in merge_strategy."
    ),
}
