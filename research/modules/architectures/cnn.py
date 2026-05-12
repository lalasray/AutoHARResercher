"""CNN architecture spec."""

ARCHITECTURE_NAME = "cnn"
DEFAULT_PARAMS = {
    "hidden_dim": 32,
    "depth": 1,
    "temporal_pool": "avg",
    "use_interactions": False,
}
MUTABLE_PARTS = ["hidden_dim", "depth", "layers"]
