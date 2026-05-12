"""Tiny transformer architecture spec."""

ARCHITECTURE_NAME = "tiny_transformer"
DEFAULT_PARAMS = {
    "hidden_dim": 48,
    "depth": 2,
    "temporal_pool": "concat",
}
MUTABLE_PARTS = ["hidden_dim", "depth", "temporal_pool", "attention_mix", "layers"]
