# Data Agent Module
#
# This file is managed by the DataAgent.
# Edit this to change data sources, source mix, windowing, and preprocessing.

# Preprocessing candidates: raw, normalized, normalized+gravity_filter,
# normalized+magnitude_channels, custom

AVAILABLE_PREPROCESSING = [
    "raw",
    "normalized",
    "normalized+gravity_filter",
    "normalized+magnitude_channels",
    "custom",
]

AVAILABLE_DATA_SOURCES = [
    "wisdm",
    "uci_har",
    "utd_mhad",
    "hapt",
    "usc_had",
    "unimib_shar",
    "motion_sense",
    "real_life_har",
    "wisdm_19_phone",
    "wisdm_19_watch",
    "ku_har",
    "hang_time",
    "capture_24",
    "synthetic_imu_to_imu",
    "synthetic_text_to_imu",
    "synthetic_video_to_imu",
]

READILY_DOWNLOADABLE_DATASETS = {
    name: {
        "status": "whar",
        "source": "teco-kit/whar-datasets",
        "modalities": ["configured_by_whar"],
        "download": "managed by WHAR dataset config",
    }
    for name in AVAILABLE_DATA_SOURCES
    if not name.startswith("synthetic_")
}

PREPROCESSING = "normalized"
WINDOW_SIZE = 128
STRIDE = 64

# Data parameters
LOWPASS_CUTOFF = 5.0
GRAVITY_FILTER_ORDER = 4

# Dataset/source-mix controls. The runtime uses WHAR Datasets for real datasets.
DATASET_POLICY = "single_dataset"  # single_dataset, multi_dataset, cross_dataset, real_synthetic_mix
PRIMARY_DATASET = "uci_har"
EVALUATION_DATASET = ""  # blank means evaluate on PRIMARY_DATASET
SECONDARY_DATASETS = []
DATASET_SOURCES = ["uci_har"]
REAL_RATIO = 1.0
SYNTHETIC_RATIO = 0.0
SYNTHETIC_SOURCES = []  # synthetic_imu_to_imu, synthetic_text_to_imu, synthetic_video_to_imu
SYNTHETIC_TARGET_CLASSES = "all"
SYNTHETIC_SOURCE_TARGETS = {}
PER_SOURCE_SYNTHETIC_RATIO = {}
CLASS_SAMPLING_POLICY = "preserve"  # preserve, boost_minority, class_targeted
TARGET_CLASSES = "all"
PER_CLASS_REAL_RATIO = {}
PER_CLASS_SYNTHETIC_RATIO = {}

MUTATION_SPACE = {
    "data_sources": AVAILABLE_DATA_SOURCES,
    "readily_downloadable_real_datasets": [
        "wisdm",
        "uci_har",
        "utd_mhad",
        "hapt",
        "usc_had",
        "unimib_shar",
        "motion_sense",
        "real_life_har",
        "wisdm_19_phone",
        "wisdm_19_watch",
        "ku_har",
        "hang_time",
        "capture_24",
    ],
    "dataset_policies": ["single_dataset", "multi_dataset", "cross_dataset", "real_synthetic_mix"],
    "preprocessing": AVAILABLE_PREPROCESSING,
    "windowing": {
        "window_size": [64, 96, 128, 192, 256],
        "stride": [16, 32, 48, 64, 96, 128],
    },
    "source_mix_controls": [
        "primary_dataset",
        "evaluation_dataset",
        "secondary_datasets",
        "dataset_sources",
        "real_ratio",
        "synthetic_ratio",
        "synthetic_sources",
        "synthetic_target_classes",
        "synthetic_source_targets",
        "per_source_synthetic_ratio",
    ],
    "class_mix_controls": [
        "class_sampling_policy",
        "target_classes",
        "per_class_real_ratio",
        "per_class_synthetic_ratio",
    ],
}

DECISION_POLICY = {
    "choose_existing_preprocessing": (
        "Use an existing preprocessing mode when the representation goal maps "
        "cleanly to raw, normalization, gravity filtering, or magnitude channels."
    ),
    "mutate_preprocessing": (
        "Mutate preprocessing when the mode is plausible but channel layout, "
        "normalization, stride, resampling, gravity scale, or magnitude settings "
        "need tuning."
    ),
    "swap_dataset_policy": (
        "Switch dataset policy when the hypothesis is about one dataset versus "
        "multiple real datasets versus real/synthetic mixtures."
    ),
    "mutate_source_mix": (
        "Mutate source mix when the likely issue is amount: real_ratio, "
        "synthetic_ratio, which synthetic source, target class, or per-class ratio."
    ),
    "create_new_preprocessing": (
        "Use custom only when no listed preprocessing expresses the idea. The "
        "custom transform must be bounded, same-shape, finite, and label-safe."
    ),
}
