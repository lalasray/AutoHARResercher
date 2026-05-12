"""DeepConvLSTM with attention architecture spec."""

ARCHITECTURE_NAME = "deepconvlstmattention"
DEFAULT_PARAMS = {
    "hidden_dim": 80,
    "depth": 2,
    "temporal_pool": "attention",
    "spectral_bins": 8,
    "attention_mix": 0.8,
    "use_interactions": True,
    "use_multiscale": True,
}
MUTABLE_PARTS = ["hidden_dim", "depth", "attention_mix", "temporal_pool", "layers"]
