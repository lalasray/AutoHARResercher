"""BiLSTM architecture spec."""

ARCHITECTURE_NAME = "bilstm"
DEFAULT_PARAMS = {
    "hidden_dim": 64,
    "depth": 2,
    "temporal_pool": "last",
    "bidirectional": True,
    "dropout": 0.1,
}
MUTABLE_PARTS = ["hidden_dim", "depth", "bidirectional", "dropout", "layers"]
