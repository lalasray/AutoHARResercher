# Augmentation Agent Module
#
# This file is managed by the AugmentationAgent.
# Edit this to change augmentation strategy, generation placeholders, and variation controls.

# Augmentation candidates: none, rotation, time_warp, jitter,
#                         mixup, scaling, drift, permutation, magnitude_warp,
#                         channel_dropout, sensor_bias, frequency_mask,
#                         imu_to_imu, text_to_imu, video_to_imu, custom

AVAILABLE_AUGMENTATIONS = [
    "none",
    "rotation",
    "time_warp",
    "jitter",
    "mixup",
    "scaling",
    "drift",
    "permutation",
    "magnitude_warp",
    "channel_dropout",
    "sensor_bias",
    "frequency_mask",
    "imu_to_imu",
    "text_to_imu",
    "video_to_imu",
    "custom",
]

AUGMENTATION = "none"
STRENGTH = "medium"

# Augmentation parameters
ROTATION_ANGLE = 15
JITTER_SIGMA = 0.05
SCALE_FACTOR = 0.1

# Generative and variation controls. These are placeholders until real generators
# such as T2M-GPT or video-to-IMU models are integrated.
VARIATION_COUNT = 1
VARIATION_AMOUNT = 0.1
VARIATION_KIND = "noise+warp+bias"
NOISE_INJECTION = "sensor_jitter"
SENTENCE_INPUT = "person performs the labeled activity with natural phone motion"
PROMPT_STYLE = "activity-conditioned"
TEXT_TO_IMU_GENERATOR = "t2mgpt_placeholder"
VIDEO_CONDITION = "activity video motion summary placeholder"
VIDEO_TO_IMU_GENERATOR = "video_to_imu_placeholder"
INVENTED_METHOD = ""

# Synthetic data controls: how much, from what source, and for which class.
SYNTHETIC_SOURCE = "same_window"  # same_window, imu_to_imu, text_to_imu, video_to_imu, custom
SYNTHETIC_CLASS = "same_as_source"  # same_as_source, minority_classes, or explicit class label
SYNTHETIC_RATIO = 0.2
SYNTHETIC_COUNT = 1
CLASS_BALANCE_POLICY = "preserve"  # preserve, boost_minority, class_targeted

MUTATION_SPACE = {
    "families": AVAILABLE_AUGMENTATIONS,
    "strengths": ["weak", "medium", "strong"],
    "variation_controls": [
        "variation_count",
        "variation_amount",
        "variation_kind",
        "noise_injection",
    ],
    "generative_placeholders": ["text_to_imu", "video_to_imu"],
    "prompt_controls": ["sentence_input", "prompt_style", "video_condition"],
    "synthetic_data_controls": [
        "synthetic_source",
        "synthetic_class",
        "synthetic_ratio",
        "synthetic_count",
        "class_balance_policy",
    ],
    "imu_to_imu_methods": [
        "jitter",
        "scaling",
        "rotation",
        "time_warp",
        "permutation",
        "drift",
        "magnitude_warp",
        "channel_dropout",
        "sensor_bias",
        "frequency_mask",
        "custom",
    ],
}
