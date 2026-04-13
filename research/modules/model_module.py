# Model Agent Module
#
# This file is managed by the ModelAgent.
# Edit this to change model architecture.

# Model candidates: deepconvlstm, deepconvlstmattention, hart,
#                  tinierhar, tinyhar, spectra, cnn_gru, cnn, tiny_transformer, gru

MODEL_NAME = "deepconvlstm"

# Architecture parameters
HIDDEN_DIM = 64
DEPTH = 2
ATTENTION_MIX = 0.9

# Optimizer
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
