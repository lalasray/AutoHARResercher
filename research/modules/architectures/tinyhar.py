"""TinyHAR architecture spec."""

ARCHITECTURE_NAME = "tinyhar"
DEFAULT_PARAMS = {
    "hidden_dim": 40,
    "depth": 1,
    "temporal_pool": "avg",
    "spectral_bins": 4,
}
MUTABLE_PARTS = ["hidden_dim", "depth", "spectral_bins", "layers"]
