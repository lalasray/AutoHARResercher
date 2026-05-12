"""BiLSTM with attention architecture spec."""

ARCHITECTURE_NAME = "bilstmattention"
DEFAULT_PARAMS = {
    "hidden_dim": 72,
    "depth": 2,
    "temporal_pool": "attention",
    "bidirectional": True,
    "attention_mix": 0.7,
    "dropout": 0.1,
}
MUTABLE_PARTS = [
    "hidden_dim",
    "depth",
    "bidirectional",
    "attention_mix",
    "dropout",
    "layers",
]
