from __future__ import annotations

import ssl
import urllib.request
import zipfile
from pathlib import Path


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
