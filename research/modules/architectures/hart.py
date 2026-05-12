"""HART architecture spec."""

ARCHITECTURE_NAME = "hart"
DEFAULT_PARAMS = {
    "hidden_dim": 72,
    "depth": 2,
    "temporal_pool": "concat",
    "use_interactions": True,
    "use_multiscale": True,
}
MUTABLE_PARTS = ["hidden_dim", "depth", "temporal_pool", "layers"]
