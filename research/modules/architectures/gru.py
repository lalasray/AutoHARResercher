"""GRU architecture spec."""

ARCHITECTURE_NAME = "gru"
DEFAULT_PARAMS = {
    "hidden_dim": 40,
    "depth": 1,
    "temporal_pool": "last",
}
MUTABLE_PARTS = ["hidden_dim", "depth", "dropout", "layers"]
