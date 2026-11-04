"""Local MotionAndStatistics variant with early/late slope statistics."""
from .motion_statistics import MotionAndStatistics

ARCHITECTURE_NAME = 'motion_slope'


class MotionSlope(MotionAndStatistics):
    def __init__(self, channels, classes, params):
        super().__init__(channels, classes, params, slopes=True, default_stride=1)


def build_model(input_dim, num_classes, params):
    return MotionSlope(int(params['input_channels']), num_classes, params)
