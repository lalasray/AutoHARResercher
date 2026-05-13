from __future__ import annotations

import csv
import math
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path
from typing import Protocol


READILY_DOWNLOADABLE_IMU_DATASETS: dict[str, dict[str, object]] = {
    "uci_har": {
        "name": "UCI HAR",
        "download_url": "https://cdn.uci-ics-mlr-prod.aws.uci.edu/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip",
        "loader_status": "implemented",
        "modalities": ["smartphone accelerometer", "smartphone gyroscope"],
        "activities": ["walk", "walk_upstairs", "walk_downstairs", "sit", "stand", "lay"],
        "notes": "Current default runtime dataset; 30 subjects, 50 Hz, 128-step windows.",
    },
    "pamap2": {
        "name": "PAMAP2 Physical Activity Monitoring",
        "download_url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "loader_status": "implemented",
        "modalities": ["IMU hand", "IMU chest", "IMU ankle", "heart rate"],
        "activities": "18 activity IDs",
        "notes": "UCI-hosted multi-IMU dataset.",
    },
    "mhealth": {
        "name": "MHEALTH",
        "download_url": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
        "loader_status": "implemented",
        "modalities": ["chest IMU", "wrist IMU", "ankle IMU", "ECG"],
        "activities": "12 activities",
        "notes": "UCI-hosted multimodal body-motion dataset.",
    },
    "opportunity": {
        "name": "OPPORTUNITY Activity Recognition",
        "download_url": "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip",
        "loader_status": "implemented",
        "modalities": ["body-worn IMUs", "object sensors", "ambient sensors"],
        "activities": "daily living gestures and locomotion labels",
        "notes": "UCI-hosted multi-sensor dataset; large download.",
    },
    "motionsense": {
        "name": "MotionSense",
        "download_url": "https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip",
        "loader_status": "implemented",
        "modalities": ["smartphone accelerometer", "smartphone gyroscope"],
        "activities": ["downstairs", "upstairs", "walking", "jogging", "sitting", "standing"],
        "notes": "GitHub-hosted smartphone IMU dataset.",
    },
}


def readily_downloadable_imu_datasets() -> dict[str, dict[str, object]]:
    return {key: dict(value) for key, value in READILY_DOWNLOADABLE_IMU_DATASETS.items()}


class WindowedHARDataset(Protocol):
    def ensure_downloaded(self) -> None: ...

    def load_split(self, split: str) -> list[tuple[list[list[float]], str]]: ...

    def metadata(self) -> dict[str, object]: ...


def create_har_dataset(name: str, root_dir: Path | None = None) -> WindowedHARDataset:
    normalized = name.lower().replace("-", "_")
    if normalized in {"dummy_imu_har", "uci_har"}:
        return UCIHARDataset(root_dir=root_dir)
    if normalized == "pamap2":
        return PAMAP2Dataset(root_dir=root_dir)
    if normalized == "mhealth":
        return MHEALTHDataset(root_dir=root_dir)
    if normalized == "opportunity":
        return OpportunityDataset(root_dir=root_dir)
    if normalized == "motionsense":
        return MotionSenseDataset(root_dir=root_dir)
    raise ValueError(f"Unsupported dataset {name!r}")


class DownloadableWindowDataset:
    DOWNLOAD_URL = ""
    DATASET_NAME = ""
    WINDOW_LENGTH = 128
    STRIDE = 64
    TRAIN_LIMIT_PER_LABEL = 240
    TEST_LIMIT_PER_LABEL = 120

    def __init__(self, root_dir: Path | None = None) -> None:
        safe_name = self.DATASET_NAME or self.__class__.__name__.lower()
        self.root_dir = root_dir or (Path.cwd() / "data" / safe_name)
        self.zip_path = self.root_dir / f"{safe_name}.zip"

    def ensure_downloaded(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.zip_path.exists():
            context = ssl.create_default_context()
            with urllib.request.urlopen(self.DOWNLOAD_URL, context=context) as response:
                self.zip_path.write_bytes(response.read())
        marker = self.root_dir / ".extracted"
        if marker.exists():
            return
        with zipfile.ZipFile(self.zip_path) as archive:
            archive.extractall(self.root_dir)
        marker.write_text("ok\n", encoding="utf-8")

    def _window_stream(
        self,
        rows: list[tuple[list[float], str]],
        *,
        limit_per_label: int,
    ) -> list[tuple[list[list[float]], str]]:
        grouped_counts: dict[str, int] = {}
        windows: list[tuple[list[list[float]], str]] = []
        idx = 0
        while idx + self.WINDOW_LENGTH <= len(rows):
            chunk = rows[idx : idx + self.WINDOW_LENGTH]
            labels = [label for _, label in chunk]
            label = max(set(labels), key=labels.count)
            if label and labels.count(label) >= max(1, int(self.WINDOW_LENGTH * 0.6)):
                count = grouped_counts.get(label, 0)
                if count < limit_per_label:
                    grouped_counts[label] = count + 1
                    windows.append(([values for values, _ in chunk], label))
            idx += self.STRIDE
        return windows

    def _clean(self, values: list[float], previous: list[float] | None = None) -> list[float]:
        previous = previous or [0.0] * len(values)
        clean = []
        for idx, value in enumerate(values):
            if not math.isfinite(value):
                clean.append(previous[idx] if idx < len(previous) else 0.0)
            else:
                clean.append(float(value))
        return clean

    def _float_token(self, token: str) -> float:
        try:
            return float(token)
        except ValueError:
            return float("nan")


class UCIHARDataset:
    DOWNLOAD_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
    EXTRACTED_DIRNAME = "UCI HAR Dataset"
    LABELS = {
        1: "walk",
        2: "walk_upstairs",
        3: "walk_downstairs",
        4: "sit",
        5: "stand",
        6: "lay",
    }
    TRAIN_LIMIT_PER_LABEL = 240
    TEST_LIMIT_PER_LABEL = 120
    SAMPLING_RATE_HZ = 50
    WINDOW_LENGTH = 128
    SENSOR_COUNT = 1
    SENSOR_MODALITIES = ["accelerometer", "gyroscope"]
    SENSOR_PLACEMENT = "smartphone at waist"
    CHANNEL_GROUPS = {
        "total_acc": ["x", "y", "z"],
        "body_acc": ["x", "y", "z"],
        "body_gyro": ["x", "y", "z"],
    }
    NOTES = (
        "Single smartphone-based IMU HAR benchmark with waist-mounted device, "
        "50 Hz sampling, and 128-step windows."
    )

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or (Path.cwd() / "data" / "uci_har")
        self.zip_path = self.root_dir / "UCI_HAR_Dataset.zip"
        self.extracted_dir = self.root_dir / self.EXTRACTED_DIRNAME

    def ensure_downloaded(self) -> None:
        if self.extracted_dir.exists():
            return
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.zip_path.exists():
            context = ssl.create_default_context()
            with urllib.request.urlopen(self.DOWNLOAD_URL, context=context) as response:
                self.zip_path.write_bytes(response.read())
        with zipfile.ZipFile(self.zip_path) as archive:
            archive.extractall(self.root_dir)
        inner_zip = self.root_dir / f"{self.EXTRACTED_DIRNAME}.zip"
        if inner_zip.exists() and not self.extracted_dir.exists():
            with zipfile.ZipFile(inner_zip) as archive:
                archive.extractall(self.root_dir)

    def load_split(self, split: str) -> list[tuple[list[list[float]], str]]:
        split_dir = self.extracted_dir / split
        labels = self._read_vector(split_dir / f"y_{split}.txt", cast=int)
        total_acc = self._read_signal_block(split_dir / "Inertial Signals", split, prefix="total_acc")
        body_acc = self._read_signal_block(split_dir / "Inertial Signals", split, prefix="body_acc")
        body_gyro = self._read_signal_block(split_dir / "Inertial Signals", split, prefix="body_gyro")

        grouped_counts = {label: 0 for label in self.LABELS.values()}
        limit = self.TRAIN_LIMIT_PER_LABEL if split == "train" else self.TEST_LIMIT_PER_LABEL
        samples: list[tuple[list[list[float]], str]] = []
        for sample_index, label_id in enumerate(labels):
            label_name = self.LABELS[label_id]
            if grouped_counts[label_name] >= limit:
                continue
            grouped_counts[label_name] += 1
            window = []
            for timestep in range(len(total_acc[0][0])):
                window.append(
                    [
                        total_acc[0][sample_index][timestep],
                        total_acc[1][sample_index][timestep],
                        total_acc[2][sample_index][timestep],
                        body_acc[0][sample_index][timestep],
                        body_acc[1][sample_index][timestep],
                        body_acc[2][sample_index][timestep],
                        body_gyro[0][sample_index][timestep],
                        body_gyro[1][sample_index][timestep],
                        body_gyro[2][sample_index][timestep],
                    ]
                )
            samples.append((window, label_name))
        return samples

    def _read_signal_block(
        self,
        signal_dir: Path,
        split: str,
        prefix: str,
    ) -> list[list[list[float]]]:
        return [
            self._read_matrix(signal_dir / f"{prefix}_{axis}_{split}.txt")
            for axis in ("x", "y", "z")
        ]

    def _read_matrix(self, path: Path) -> list[list[float]]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            values = [float(token) for token in line.strip().split()]
            rows.append(values)
        return rows

    def _read_vector(self, path: Path, cast: type[int] | type[float]) -> list[int] | list[float]:
        return [cast(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def metadata(self) -> dict[str, object]:
        return {
            "dataset_name": "uci_har",
            "task": "human_activity_recognition",
            "sensor_count": self.SENSOR_COUNT,
            "sensor_modalities": list(self.SENSOR_MODALITIES),
            "sensor_placement": self.SENSOR_PLACEMENT,
            "sampling_rate_hz": self.SAMPLING_RATE_HZ,
            "window_length_steps": self.WINDOW_LENGTH,
            "channel_groups": dict(self.CHANNEL_GROUPS),
            "activity_labels": list(self.LABELS.values()),
            "notes": self.NOTES,
        }


class MHEALTHDataset(DownloadableWindowDataset):
    DATASET_NAME = "mhealth"
    DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
    LABELS = {
        1: "standing_still",
        2: "sitting_relaxing",
        3: "lying_down",
        4: "walking",
        5: "climbing_stairs",
        6: "waist_bends",
        7: "front_elevation_arms",
        8: "knees_bending",
        9: "cycling",
        10: "jogging",
        11: "running",
        12: "jump_front_back",
    }
    SAMPLING_RATE_HZ = 50

    def load_split(self, split: str) -> list[tuple[list[list[float]], str]]:
        files = sorted(self.root_dir.rglob("mHealth_subject*.log"))
        if not files:
            files = sorted(self.root_dir.rglob("*.log"))
        selected = [
            path
            for path in files
            if (self._subject_id(path) <= 7 if split == "train" else self._subject_id(path) > 7)
        ]
        rows: list[tuple[list[float], str]] = []
        previous: list[float] | None = None
        for path in selected:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                tokens = line.split()
                if len(tokens) < 24:
                    continue
                label_id = int(float(tokens[-1]))
                if label_id == 0 or label_id not in self.LABELS:
                    continue
                values = [self._float_token(token) for token in tokens]
                channels = values[0:3] + values[5:8] + values[8:11]
                previous = self._clean(channels, previous)
                rows.append((previous, self.LABELS[label_id]))
        limit = self.TRAIN_LIMIT_PER_LABEL if split == "train" else self.TEST_LIMIT_PER_LABEL
        return self._window_stream(rows, limit_per_label=limit)

    def _subject_id(self, path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits or 0)

    def metadata(self) -> dict[str, object]:
        return {
            "dataset_name": "mhealth",
            "task": "human_activity_recognition",
            "sensor_count": 3,
            "sensor_modalities": ["accelerometer", "gyroscope", "magnetometer", "ecg"],
            "sensor_placement": "chest, wrist, ankle",
            "sampling_rate_hz": self.SAMPLING_RATE_HZ,
            "window_length_steps": self.WINDOW_LENGTH,
            "activity_labels": list(self.LABELS.values()),
            "notes": "MHEALTH loader maps raw logs to 9 IMU channels.",
        }


class PAMAP2Dataset(DownloadableWindowDataset):
    DATASET_NAME = "pamap2"
    DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
    LABELS = {
        1: "lying",
        2: "sitting",
        3: "standing",
        4: "walking",
        5: "running",
        6: "cycling",
        7: "nordic_walking",
        9: "watching_tv",
        10: "computer_work",
        11: "car_driving",
        12: "ascending_stairs",
        13: "descending_stairs",
        16: "vacuum_cleaning",
        17: "ironing",
        18: "folding_laundry",
        19: "house_cleaning",
        20: "playing_soccer",
        24: "rope_jumping",
    }
    SAMPLING_RATE_HZ = 100

    def load_split(self, split: str) -> list[tuple[list[list[float]], str]]:
        files = sorted(self.root_dir.rglob("subject*.dat"))
        selected = [
            path
            for path in files
            if (self._subject_id(path) <= 7 if split == "train" else self._subject_id(path) > 7)
        ]
        rows: list[tuple[list[float], str]] = []
        previous: list[float] | None = None
        for path in selected:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                tokens = line.split()
                if len(tokens) < 54:
                    continue
                activity_id = int(float(tokens[1]))
                if activity_id == 0 or activity_id not in self.LABELS:
                    continue
                values = [self._float_token(token) for token in tokens]
                channels = values[7:13] + values[24:27]
                previous = self._clean(channels, previous)
                rows.append((previous, self.LABELS[activity_id]))
        limit = self.TRAIN_LIMIT_PER_LABEL if split == "train" else self.TEST_LIMIT_PER_LABEL
        return self._window_stream(rows, limit_per_label=limit)

    def _subject_id(self, path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits or 0)

    def metadata(self) -> dict[str, object]:
        return {
            "dataset_name": "pamap2",
            "task": "human_activity_recognition",
            "sensor_count": 3,
            "sensor_modalities": ["accelerometer", "gyroscope", "magnetometer", "heart_rate"],
            "sensor_placement": "hand, chest, ankle",
            "sampling_rate_hz": self.SAMPLING_RATE_HZ,
            "window_length_steps": self.WINDOW_LENGTH,
            "activity_labels": list(self.LABELS.values()),
            "notes": "PAMAP2 loader maps protocol files to 9 IMU channels.",
        }


class MotionSenseDataset(DownloadableWindowDataset):
    DATASET_NAME = "motionsense"
    DOWNLOAD_URL = "https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip"
    LABEL_MAP = {
        "dws": "walk_downstairs",
        "ups": "walk_upstairs",
        "wlk": "walk",
        "jog": "jog",
        "sit": "sit",
        "std": "stand",
    }
    SAMPLING_RATE_HZ = 50

    def load_split(self, split: str) -> list[tuple[list[list[float]], str]]:
        files = sorted(self.root_dir.rglob("*.csv"))
        rows_by_label: list[tuple[list[tuple[list[float], str]], int]] = []
        for path in files:
            label = self._label_from_path(path)
            subject = self._subject_from_path(path)
            if not label or subject is None:
                continue
            if (split == "train" and subject > 18) or (split != "train" and subject <= 18):
                continue
            rows = self._read_motion_csv(path, label)
            if rows:
                rows_by_label.append((rows, subject))
        samples: list[tuple[list[list[float]], str]] = []
        limit = self.TRAIN_LIMIT_PER_LABEL if split == "train" else self.TEST_LIMIT_PER_LABEL
        counts: dict[str, int] = {}
        for rows, _ in rows_by_label:
            for window, label in self._window_stream(rows, limit_per_label=10_000):
                if counts.get(label, 0) >= limit:
                    continue
                counts[label] = counts.get(label, 0) + 1
                samples.append((window, label))
        return samples

    def _read_motion_csv(self, path: Path, label: str) -> list[tuple[list[float], str]]:
        rows: list[tuple[list[float], str]] = []
        previous: list[float] | None = None
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                keys = {
                    key.strip(): value
                    for key, value in row.items()
                    if key is not None
                }
                channels = [
                    self._float_token(keys.get("userAcceleration.x", "nan")),
                    self._float_token(keys.get("userAcceleration.y", "nan")),
                    self._float_token(keys.get("userAcceleration.z", "nan")),
                    self._float_token(keys.get("rotationRate.x", "nan")),
                    self._float_token(keys.get("rotationRate.y", "nan")),
                    self._float_token(keys.get("rotationRate.z", "nan")),
                    self._float_token(keys.get("gravity.x", "nan")),
                    self._float_token(keys.get("gravity.y", "nan")),
                    self._float_token(keys.get("gravity.z", "nan")),
                ]
                previous = self._clean(channels, previous)
                rows.append((previous, label))
        return rows

    def _label_from_path(self, path: Path) -> str | None:
        for part in path.parts:
            prefix = part.split("_", 1)[0]
            if prefix in self.LABEL_MAP:
                return self.LABEL_MAP[prefix]
        return None

    def _subject_from_path(self, path: Path) -> int | None:
        stem = path.stem
        if stem.startswith("sub_"):
            try:
                return int(stem.split("_", 1)[1])
            except ValueError:
                return None
        return None

    def metadata(self) -> dict[str, object]:
        return {
            "dataset_name": "motionsense",
            "task": "human_activity_recognition",
            "sensor_count": 1,
            "sensor_modalities": ["accelerometer", "gyroscope", "attitude/gravity"],
            "sensor_placement": "smartphone",
            "sampling_rate_hz": self.SAMPLING_RATE_HZ,
            "window_length_steps": self.WINDOW_LENGTH,
            "activity_labels": list(self.LABEL_MAP.values()),
            "notes": "MotionSense loader maps CSV motion files to 9 channels.",
        }


class OpportunityDataset(DownloadableWindowDataset):
    DATASET_NAME = "opportunity"
    DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"
    SAMPLING_RATE_HZ = 30
    LABELS = {
        1: "stand",
        2: "walk",
        4: "sit",
        5: "lie",
    }

    def load_split(self, split: str) -> list[tuple[list[list[float]], str]]:
        files = sorted(self.root_dir.rglob("*.dat"))
        selected = [
            path
            for path in files
            if (self._subject_id(path) <= 3 if split == "train" else self._subject_id(path) > 3)
        ]
        rows: list[tuple[list[float], str]] = []
        previous: list[float] | None = None
        for path in selected:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                tokens = line.split()
                if len(tokens) < 20:
                    continue
                values = [self._float_token(token) for token in tokens]
                label_id = self._label_from_values(values)
                if label_id is None:
                    continue
                feature_values = [value for value in values[1:-7] if math.isfinite(value)]
                if len(feature_values) < 9:
                    continue
                channels = feature_values[:9]
                previous = self._clean(channels, previous)
                rows.append((previous, self.LABELS.get(label_id, f"label_{label_id}")))
        limit = self.TRAIN_LIMIT_PER_LABEL if split == "train" else self.TEST_LIMIT_PER_LABEL
        return self._window_stream(rows, limit_per_label=limit)

    def _label_from_values(self, values: list[float]) -> int | None:
        for value in reversed(values[-12:]):
            if math.isfinite(value) and int(value) in self.LABELS:
                return int(value)
        return None

    def _subject_id(self, path: Path) -> int:
        match = re.search(r"S(\d+)", "_".join(path.parts), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def metadata(self) -> dict[str, object]:
        return {
            "dataset_name": "opportunity",
            "task": "human_activity_recognition",
            "sensor_count": "multiple",
            "sensor_modalities": ["body_worn_imu", "object_sensors", "ambient_sensors"],
            "sensor_placement": "body and environment",
            "sampling_rate_hz": self.SAMPLING_RATE_HZ,
            "window_length_steps": self.WINDOW_LENGTH,
            "activity_labels": list(self.LABELS.values()),
            "notes": "OPPORTUNITY loader extracts finite sensor channels and locomotion labels.",
        }
