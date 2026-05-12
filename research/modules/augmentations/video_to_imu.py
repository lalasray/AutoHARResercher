"""Video-to-IMU placeholder.

Until a real generator is integrated, this delegates to a bounded IMU-to-IMU
proxy while preserving video-conditioning metadata in params.
"""

from __future__ import annotations

from research.modules.augmentations.imu_to_imu import augment_window

NAME = "video_to_imu"
