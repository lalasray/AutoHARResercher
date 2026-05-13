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
    "uci_har",
    "pamap2",
    "mhealth",
    "opportunity",
    "motionsense",
    "synthetic_imu_to_imu",
    "synthetic_text_to_imu",
    "synthetic_video_to_imu",
]

READILY_DOWNLOADABLE_DATASETS = {
    "uci_har": {
        "status": "implemented",
        "source": "UCI",
        "modalities": ["accelerometer", "gyroscope"],
        "download": "https://cdn.uci-ics-mlr-prod.aws.uci.edu/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip",
    },
    "pamap2": {
        "status": "implemented",
        "source": "UCI",
        "modalities": ["multi-IMU", "heart_rate"],
        "download": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
    },
    "mhealth": {
        "status": "implemented",
        "source": "UCI",
        "modalities": ["multi-IMU", "ECG"],
        "download": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    },
    "opportunity": {
        "status": "implemented",
        "source": "UCI",
        "modalities": ["body_IMU", "object_sensors", "ambient_sensors"],
        "download": "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
    },
    "motionsense": {
        "status": "implemented",
        "source": "GitHub",
        "modalities": ["smartphone_accelerometer", "smartphone_gyroscope"],
        "download": "https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip",
    },
}

PREPROCESSING = "normalized"
WINDOW_SIZE = 128
STRIDE = 64

# Data parameters
LOWPASS_CUTOFF = 5.0
GRAVITY_FILTER_ORDER = 4

# Dataset/source-mix controls. The current runtime has loaders for the five
# common public IMU datasets listed above.
DATASET_POLICY = "single_dataset"  # single_dataset, multi_dataset, real_synthetic_mix
PRIMARY_DATASET = "uci_har"
SECONDARY_DATASETS = []
DATASET_SOURCES = ["uci_har"]
REAL_RATIO = 1.0
SYNTHETIC_RATIO = 0.0
SYNTHETIC_SOURCES = []  # synthetic_imu_to_imu, synthetic_text_to_imu, synthetic_video_to_imu
CLASS_SAMPLING_POLICY = "preserve"  # preserve, boost_minority, class_targeted
TARGET_CLASSES = "all"
PER_CLASS_REAL_RATIO = {}
PER_CLASS_SYNTHETIC_RATIO = {}

MUTATION_SPACE = {
    "data_sources": AVAILABLE_DATA_SOURCES,
    "readily_downloadable_real_datasets": [
        "uci_har",
        "pamap2",
        "mhealth",
        "opportunity",
        "motionsense",
    ],
    "dataset_policies": ["single_dataset", "multi_dataset", "real_synthetic_mix"],
    "preprocessing": AVAILABLE_PREPROCESSING,
    "windowing": {
        "window_size": [64, 96, 128, 192, 256],
        "stride": [16, 32, 48, 64, 96, 128],
    },
    "source_mix_controls": [
        "primary_dataset",
        "secondary_datasets",
        "dataset_sources",
        "real_ratio",
        "synthetic_ratio",
        "synthetic_sources",
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
