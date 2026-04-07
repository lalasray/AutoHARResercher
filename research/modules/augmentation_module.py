# Augmentation Agent Module
#
# This file is managed by the AugmentationAgent.
# Edit this to change augmentation strategy.

# Augmentation candidates: none, rotation, time_warp, jitter,
#                         mixup, scaling, drift, permutation, magnitude_warp

AUGMENTATION = "none"
STRENGTH = "medium"

# Augmentation parameters
ROTATION_ANGLE = 15
JITTER_SIGMA = 0.05
SCALE_FACTOR = 0.1
