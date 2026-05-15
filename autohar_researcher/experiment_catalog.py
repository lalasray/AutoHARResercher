from __future__ import annotations

MODEL_ARCHITECTURES: tuple[str, ...] = (
    "deepconvlstm",
    "deepconvlstmattention",
    "bilstm",
    "bilstmattention",
    "hart",
    "tinierhar",
    "tinyhar",
    "spectra",
    "cnn_gru",
    "cnn",
    "tiny_transformer",
    "gru",
    "custom",
)

MODEL_HYPERPARAMETERS: tuple[str, ...] = (
    "hidden_dim",
    "depth",
    "temporal_pool",
    "spectral_bins",
    "use_interactions",
    "use_multiscale",
    "attention_mix",
    "dropout",
    "bidirectional",
    "optimizer",
    "weight_decay",
    "loss",
    "label_smoothing",
    "parent_architectures",
    "merge_strategy",
    "learning_rate",
)

MODEL_ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    "deepconvlstm": "Balanced convolutional temporal baseline.",
    "deepconvlstmattention": "DeepConvLSTM variant with attention pooling.",
    "bilstm": "Bidirectional LSTM temporal baseline.",
    "bilstmattention": "Bidirectional LSTM variant with attention pooling.",
    "hart": "HAR transformer-style architecture anchor.",
    "tinierhar": "Very small low-capacity HAR model.",
    "tinyhar": "Compact HAR model with modest spectral features.",
    "spectra": "Spectral-feature-heavy architecture.",
    "cnn_gru": "Convolutional feature extractor with recurrent pooling.",
    "cnn": "Small convolutional baseline.",
    "tiny_transformer": "Small transformer-like temporal model.",
    "gru": "Recurrent GRU temporal baseline.",
    "custom": "Bounded custom model-side feature representation.",
}

AVAILABLE_AUGMENTATIONS: tuple[str, ...] = (
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
)

AVAILABLE_PREPROCESSING: tuple[str, ...] = (
    "raw",
    "normalized",
    "normalized+gravity_filter",
    "normalized+magnitude_channels",
    "custom",
)

REAL_DATASETS: tuple[str, ...] = (
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
)


def available_model_architectures() -> list[str]:
    return list(MODEL_ARCHITECTURES)


def model_architecture_catalog() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "description": MODEL_ARCHITECTURE_DESCRIPTIONS[name],
            "default_params": default_model_params(name),
        }
        for name in MODEL_ARCHITECTURES
    ]


def available_augmentations() -> list[str]:
    return list(AVAILABLE_AUGMENTATIONS)


def available_preprocessing_modes() -> list[str]:
    return list(AVAILABLE_PREPROCESSING)


def default_model_params(name: str) -> dict[str, float | int | str | bool]:
    params: dict[str, float | int | str | bool] = {
        "hidden_dim": 64,
        "depth": 2,
        "temporal_pool": "avg",
        "spectral_bins": 6,
        "use_interactions": True,
        "use_multiscale": True,
        "attention_mix": 0.5,
        "dropout": 0.0,
        "bidirectional": False,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "loss": "cross_entropy",
        "label_smoothing": 0.0,
        "parent_architectures": "",
        "merge_strategy": "",
    }
    if name == "cnn":
        params.update({"hidden_dim": 32, "depth": 1, "use_interactions": False})
    elif name == "gru":
        params.update({"hidden_dim": 40, "depth": 1, "temporal_pool": "last"})
    elif name == "bilstm":
        params.update(
            {
                "hidden_dim": 64,
                "depth": 2,
                "temporal_pool": "last",
                "bidirectional": True,
                "dropout": 0.1,
            }
        )
    elif name == "bilstmattention":
        params.update(
            {
                "hidden_dim": 72,
                "depth": 2,
                "temporal_pool": "attention",
                "bidirectional": True,
                "attention_mix": 0.7,
                "dropout": 0.1,
            }
        )
    elif name == "tiny_transformer":
        params.update({"hidden_dim": 48, "depth": 2, "temporal_pool": "concat"})
    elif name == "hart":
        params.update({"hidden_dim": 72, "depth": 2, "temporal_pool": "concat"})
    elif name == "deepconvlstmattention":
        params.update(
            {
                "hidden_dim": 80,
                "depth": 2,
                "temporal_pool": "attention",
                "spectral_bins": 8,
                "attention_mix": 0.8,
            }
        )
    elif name == "tinierhar":
        params.update(
            {
                "hidden_dim": 32,
                "depth": 1,
                "temporal_pool": "avg",
                "use_multiscale": False,
            }
        )
    elif name == "tinyhar":
        params.update(
            {"hidden_dim": 40, "depth": 1, "temporal_pool": "avg", "spectral_bins": 4}
        )
    elif name == "spectra":
        params.update(
            {
                "hidden_dim": 56,
                "depth": 2,
                "temporal_pool": "concat",
                "spectral_bins": 10,
                "use_interactions": False,
            }
        )
    elif name == "cnn_gru":
        params.update({"hidden_dim": 56, "depth": 2, "temporal_pool": "last"})
    elif name == "custom":
        params.update(
            {
                "hidden_dim": 64,
                "depth": 2,
                "temporal_pool": "avg",
                "custom_code": "",
            }
        )
    return params


def default_augmentation_params(name: str, strength: str) -> dict[str, float | int | str]:
    strength_key = strength if strength in {"weak", "medium", "strong"} else "medium"
    strength_scale = {
        "weak": 0.5,
        "medium": 1.0,
        "strong": 1.5,
    }[strength_key]
    if name == "rotation":
        return {"angle_deg": round(10.0 * strength_scale, 3)}
    if name == "jitter":
        return {"jitter_std": round(0.02 * strength_scale, 5)}
    if name == "scaling":
        return {"scale_span": round(0.10 * strength_scale, 5)}
    if name == "time_warp":
        return {"compression_ratio": round(max(0.7, 1.0 - (0.08 * strength_scale)), 5)}
    if name == "permutation":
        return {"segment_count": int(min(6, max(3, round(2 + (2 * strength_scale)))))}
    if name == "drift":
        return {"drift_max": round(0.015 * strength_scale, 5)}
    if name == "magnitude_warp":
        return {"warp_span": round(0.15 * strength_scale, 5)}
    if name == "mixup":
        return {"blend": round(min(0.45, 0.15 + (0.15 * strength_scale)), 5)}
    if name == "channel_dropout":
        return {"drop_probability": round(min(0.35, 0.08 * strength_scale), 5)}
    if name == "sensor_bias":
        return {"bias_radius": round(0.02 * strength_scale, 5)}
    if name == "frequency_mask":
        return {"mask_ratio": round(min(0.4, 0.12 * strength_scale), 5)}
    if name == "imu_to_imu":
        return {
            "variation_count": int(max(1, round(strength_scale))),
            "variation_amount": round(0.12 * strength_scale, 5),
            "variation_kind": "noise+warp+bias",
            "noise_injection": "gaussian",
            "synthetic_source": "imu_to_imu",
            "synthetic_class": "same_as_source",
            "synthetic_ratio": round(0.25 * strength_scale, 5),
            "synthetic_count": int(max(1, round(strength_scale))),
            "method": "placeholder_composed_imu_transform",
            "invented_method": "",
        }
    if name == "text_to_imu":
        return {
            "variation_count": int(max(1, round(strength_scale))),
            "variation_amount": round(0.10 * strength_scale, 5),
            "sentence_input": "person performs the labeled activity with natural phone motion",
            "prompt_style": "activity-conditioned",
            "generator": "t2mgpt_placeholder",
            "noise_injection": "sensor_jitter",
            "synthetic_source": "text_to_imu",
            "synthetic_class": "same_as_prompt_activity",
            "synthetic_ratio": round(0.20 * strength_scale, 5),
            "synthetic_count": int(max(1, round(strength_scale))),
            "placeholder": "returns bounded IMU-to-IMU proxy until T2M-GPT is integrated",
        }
    if name == "video_to_imu":
        return {
            "variation_count": int(max(1, round(strength_scale))),
            "variation_amount": round(0.10 * strength_scale, 5),
            "video_condition": "activity video motion summary placeholder",
            "generator": "video_to_imu_placeholder",
            "noise_injection": "camera_motion_to_sensor_jitter",
            "synthetic_source": "video_to_imu",
            "synthetic_class": "same_as_video_activity",
            "synthetic_ratio": round(0.20 * strength_scale, 5),
            "synthetic_count": int(max(1, round(strength_scale))),
            "placeholder": "returns bounded IMU-to-IMU proxy until video-to-IMU is integrated",
        }
    if name == "custom":
        return {
            "custom_code": "",
            "variation_count": int(max(1, round(strength_scale))),
            "variation_amount": round(0.10 * strength_scale, 5),
            "synthetic_source": "custom",
            "synthetic_class": "same_as_source",
            "synthetic_ratio": round(0.20 * strength_scale, 5),
            "synthetic_count": int(max(1, round(strength_scale))),
            "invented_method": "",
        }
    return {}


def default_data_params(
    preprocessing: str, window_size: int
) -> dict[str, float | int | str | bool]:
    params: dict[str, float | int | str | bool] = {
        "channel_layout": "total_acc+gyro",
        "resample_mode": "nearest",
        "normalize": preprocessing
        in {"normalized", "normalized+gravity_filter", "normalized+magnitude_channels"},
        "gravity_scale": 1.0,
        "append_magnitude": preprocessing == "normalized+magnitude_channels",
        "stride": max(16, window_size // 2),
        "dataset_policy": "single_dataset",
        "train_datasets": "uci_har",
        "dataset_sources": "uci_har",
        "evaluation_dataset": "",
        "available_real_datasets": ",".join(REAL_DATASETS),
        "real_ratio": 1.0,
        "per_dataset_real_ratio": "",
        "synthetic_ratio": 0.0,
        "synthetic_sources": "",
        "synthetic_target_classes": "all",
        "synthetic_source_targets": "",
        "per_source_synthetic_ratio": "",
        "class_sampling_policy": "preserve",
        "target_classes": "all",
        "per_class_real_ratio": "",
        "per_dataset_class_real_ratio": "",
        "per_class_synthetic_ratio": "",
        "dataset_mix_notes": "real datasets are loaded through teco-kit/whar-datasets",
    }
    if preprocessing == "normalized+gravity_filter":
        params["channel_layout"] = "body_acc+gyro"
    if preprocessing == "custom":
        params["custom_code"] = ""
    if window_size >= 192:
        params["resample_mode"] = "linear"
    return params
