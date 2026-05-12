"""CNN-GRU architecture spec."""

ARCHITECTURE_NAME = "cnn_gru"
DEFAULT_PARAMS = {
    "hidden_dim": 56,
    "depth": 2,
    "temporal_pool": "last",
}
MUTABLE_PARTS = ["hidden_dim", "depth", "temporal_pool", "layers"]
