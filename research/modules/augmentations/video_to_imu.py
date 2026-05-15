"""Video-to-IMU placeholder.

Until a real generator is integrated, this delegates to a bounded IMU-to-IMU
proxy while preserving video-conditioning metadata in params.
"""

from __future__ import annotations

from research.modules.augmentations.imu_to_imu import (
    augment_window as _proxy_augment_window,
)

NAME = "video_to_imu"


def augment_window(window, strength, params, rng):
    """Generate a video-conditioned proxy sample with the shared IMU transform."""
    return _proxy_augment_window(window, strength, params, rng)
