"""DeepConvLSTM architecture spec."""

ARCHITECTURE_NAME = "deepconvlstm"
DEFAULT_PARAMS = {
    "hidden_dim": 64,
    "depth": 2,
    "temporal_pool": "avg",
    "spectral_bins": 6,
    "use_interactions": True,
    "use_multiscale": True,
}
MUTABLE_PARTS = ["hidden_dim", "depth", "temporal_pool", "layers"]
