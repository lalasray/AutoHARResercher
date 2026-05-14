from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol


WHAR_DATASET_IDS = [
    "uci_har",
    "wisdm",
    "pamap2",
    "motion_sense",
    "opportunity",
    "mhealth",
    "dsads",
    "ku_har",
    "daphnet",
    "har_sense",
    "hapt",
    "w_har",
    "usc_had",
    "hugadb",
    "wisdm_19_phone",
    "wisdm_19_watch",
    "hang_time",
    "hhar",
    "capture_24",
    "real_world",
    "real_life_har",
    "sad",
    "unimib_shar",
    "uma_fall",
    "real_disp",
    "harth",
    "falldet",
    "har70",
    "gotov",
    "utd_mhad",
    "up_fall",
    "bmhad",
    "uca_ehar",
    "wear",
    "skoda",
    "actrectut_gestures",
    "actrectut_walking",
]

WHAR_EVALUATION_DATASET_IDS = [
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
]


def readily_downloadable_imu_datasets() -> dict[str, dict[str, object]]:
    return {
        dataset_id: {
            "name": dataset_id,
            "loader_status": "whar",
            "source": "teco-kit/whar-datasets",
            "download_url": "managed by WHAR dataset config",
            "notes": (
                "Loaded through the WHAR Datasets library. The library downloads "
                "from original sources and requires accepting each dataset's terms."
            ),
        }
        for dataset_id in WHAR_EVALUATION_DATASET_IDS
    }


class WindowedHARDataset(Protocol):
    def ensure_downloaded(self) -> None: ...

    def load_split(self, split: str) -> list[tuple[list[list[float]], str]]: ...

    def metadata(self) -> dict[str, object]: ...


def create_har_dataset(name: str, root_dir: Path | None = None) -> WindowedHARDataset:
    normalized = _normalize_whar_dataset_id(name)
    return WHARDataset(dataset_id=normalized, root_dir=root_dir)


class WHARDataset:
    """Windowed HAR dataset adapter backed only by teco-kit/whar-datasets."""

    TRAIN_LIMIT_PER_LABEL = 240
    TEST_LIMIT_PER_LABEL = 120

    def __init__(self, dataset_id: str = "uci_har", root_dir: Path | None = None) -> None:
        self.dataset_id = _normalize_whar_dataset_id(dataset_id)
        self.root_dir = root_dir or (Path.cwd() / "data" / "whar")
        self._cfg = None
        self._pre_pipeline = None
        self._session_df = None
        self._window_df = None
        self._split = None
        self._loader = None
        self._samples = None

    def ensure_downloaded(self) -> None:
        self._ensure_loader()

    def load_split(self, split: str) -> list[tuple[list[list[float]], str]]:
        self._ensure_loader()
        if split == "train":
            indices = list(self._split.train_indices)
            limit = self.TRAIN_LIMIT_PER_LABEL
        elif split in {"test", "val", "validation"}:
            indices = (
                list(self._split.test_indices)
                if split == "test"
                else list(self._split.val_indices)
            )
            limit = self.TEST_LIMIT_PER_LABEL
        else:
            raise ValueError(f"Unsupported WHAR split {split!r}")

        samples: list[tuple[list[list[float]], str]] = []
        counts: dict[str, int] = {}
        for index in indices:
            activity_label, _, sample = self._loader.get_item(index)
            label_name = self._label_name(activity_label)
            if counts.get(label_name, 0) >= limit:
                continue
            window = self._coerce_window(sample[0])
            if not window:
                continue
            counts[label_name] = counts.get(label_name, 0) + 1
            samples.append((window, label_name))
        return samples

    def metadata(self) -> dict[str, object]:
        try:
            cfg = self._load_cfg()
        except RuntimeError:
            cfg = None
        sampling_freq = getattr(cfg, "sampling_freq", None) if cfg is not None else None
        window_time = float(getattr(cfg, "window_time", 0.0)) if cfg is not None else 0.0
        resampling_freq = (
            getattr(cfg, "resampling_freq", None) if cfg is not None else None
        )
        activity_labels = (
            [str(label) for label in getattr(cfg, "available_activities", [])]
            if cfg is not None
            else []
        )
        return {
            "dataset_name": self.dataset_id,
            "task": "human_activity_recognition",
            "backend": "whar_datasets",
            "source_repository": "https://github.com/teco-kit/whar-datasets",
            "sensor_count": "configured_by_whar",
            "sensor_modalities": (
                list(getattr(cfg, "available_channels", [])) if cfg is not None else []
            ),
            "sensor_placement": "configured_by_whar",
            "sampling_rate_hz": sampling_freq,
            "window_length_steps": int(
                window_time * float(resampling_freq or sampling_freq or 0)
            ),
            "activity_labels": activity_labels,
            "notes": (
                "WHAR-only dataset backend. Install with "
                'pip install "whar-datasets @ git+https://github.com/teco-kit/whar-datasets.git".'
            ),
        }

    def _ensure_loader(self) -> None:
        if self._loader is not None:
            return

        whar = self._import_whar()
        cfg = self._load_cfg()
        pre_pipeline = whar.PreProcessingPipeline(cfg)
        _, session_df, window_df = pre_pipeline.run()

        splits = whar.LOSOSplitter(cfg).get_splits(session_df, window_df)
        if not splits:
            raise RuntimeError(f"WHAR dataset {self.dataset_id!r} did not produce splits.")
        split = splits[0]

        post_pipeline = whar.PostProcessingPipeline(
            cfg,
            pre_pipeline,
            window_df,
            split.train_indices,
        )
        samples = post_pipeline.run()
        loader = whar.Loader(session_df, window_df, post_pipeline.samples_dir, samples)

        self._pre_pipeline = pre_pipeline
        self._session_df = session_df
        self._window_df = window_df
        self._split = split
        self._samples = samples
        self._loader = loader

    def _load_cfg(self):
        if self._cfg is None:
            whar = self._import_whar()
            dataset_enum = self._dataset_enum(whar.WHARDatasetID)
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._cfg = whar.get_dataset_cfg(dataset_enum, datasets_dir=str(self.root_dir))
        return self._cfg

    def _dataset_enum(self, enum_cls):
        for member in enum_cls:
            if str(member.value).lower() == self.dataset_id:
                return member
        supported = ", ".join(WHAR_DATASET_IDS)
        raise ValueError(
            f"Unsupported WHAR dataset {self.dataset_id!r}. Supported datasets: {supported}"
        )

    def _label_name(self, activity_label: int) -> str:
        labels = list(getattr(self._load_cfg(), "available_activities", []))
        if 0 <= int(activity_label) < len(labels):
            return str(labels[int(activity_label)])
        return f"activity_{activity_label}"

    def _coerce_window(self, sample: object) -> list[list[float]]:
        rows = sample.tolist() if hasattr(sample, "tolist") else sample
        if not isinstance(rows, list) or not rows:
            return []
        if rows and all(isinstance(value, (int, float)) for value in rows):
            rows = [rows]
        if not all(isinstance(row, list) for row in rows):
            return []

        matrix = [
            [float(value) if math.isfinite(float(value)) else 0.0 for value in row]
            for row in rows
            if row
        ]
        if not matrix:
            return []

        # WHAR samples are commonly channel x time; the harness expects time x channel.
        if len(matrix) < len(matrix[0]):
            matrix = [list(row) for row in zip(*matrix)]
        return matrix

    def _import_whar(self):
        try:
            import whar_datasets
        except ImportError as exc:  # pragma: no cover - dependency guidance path
            raise RuntimeError(
                "WHAR dataset support requires teco-kit/whar-datasets. Install it with "
                'pip install "whar-datasets @ git+https://github.com/teco-kit/whar-datasets.git".'
            ) from exc
        return whar_datasets


def _normalize_whar_dataset_id(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized.startswith("whar:"):
        normalized = normalized.split(":", 1)[1].replace("-", "_")
    elif normalized.startswith("whar_"):
        normalized = normalized.removeprefix("whar_")
    aliases = {
        "ucihar": "uci_har",
        "utdmhad": "utd_mhad",
        "unimibshar": "unimib_shar",
        "motionsense": "motion_sense",
        "reallifehar": "real_life_har",
        "wisdm19phone": "wisdm_19_phone",
        "wisdm19watch": "wisdm_19_watch",
        "kuhar": "ku_har",
        "hangtime": "hang_time",
        "capture24": "capture_24",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in WHAR_DATASET_IDS:
        supported = ", ".join(WHAR_DATASET_IDS)
        raise ValueError(f"Unsupported WHAR dataset {name!r}. Supported datasets: {supported}")
    return normalized
