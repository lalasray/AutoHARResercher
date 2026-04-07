# Data Agent Module
#
# This file is managed by the DataAgent.
# Edit this to change data preprocessing.

# Preprocessing candidates: raw, normalized, normalized+gravity_filter, normalized+magnitude_channels

PREPROCESSING = "normalized"
WINDOW_SIZE = 128

# Data parameters
LOWPASS_CUTOFF = 5.0
GRAVITY_FILTER_ORDER = 4
