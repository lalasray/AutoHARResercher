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
}
