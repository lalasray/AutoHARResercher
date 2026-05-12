"""Custom bounded model-side representation spec."""

ARCHITECTURE_NAME = "custom"
DEFAULT_PARAMS = {
    "hidden_dim": 64,
    "depth": 2,
    "temporal_pool": "avg",
    "custom_code": "",
}
MUTABLE_PARTS = ["custom_code", "hidden_dim", "depth", "layers"]
