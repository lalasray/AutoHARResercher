"""Spectra architecture spec."""

ARCHITECTURE_NAME = "spectra"
DEFAULT_PARAMS = {
    "hidden_dim": 56,
    "depth": 2,
    "temporal_pool": "concat",
    "spectral_bins": 10,
    "use_interactions": False,
}
MUTABLE_PARTS = ["hidden_dim", "depth", "spectral_bins", "temporal_pool", "layers"]
