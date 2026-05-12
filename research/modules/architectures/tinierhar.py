"""TinierHAR architecture spec."""

ARCHITECTURE_NAME = "tinierhar"
DEFAULT_PARAMS = {
    "hidden_dim": 32,
    "depth": 1,
    "temporal_pool": "avg",
    "use_multiscale": False,
}
MUTABLE_PARTS = ["hidden_dim", "depth", "temporal_pool", "layers"]
