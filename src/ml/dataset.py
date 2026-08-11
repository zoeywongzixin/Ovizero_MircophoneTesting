from __future__ import annotations

import csv
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FeatureConfig, decode_audio_file, iter_audio_windows, log_mel_spectrogram


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
DEFAULT_HARD_NEGATIVE_CATEGORIES = (
    "wind",
    "engine",
    "car_horn",
    "chirping_birds",
    "crow",
    "pouring_water",
    "rain",
    "sea_waves",
    "water_drops",
    "insects",
    "crickets",
)


@dataclass(frozen=True)
class AudioSample:
    path: Path
    label: int
    source: str
    group: str


def build_sample_manifest(
    input_dir: str | Path = "input",
    esc50_dir: str | Path = "output/datasets/esc50",
) -> list[AudioSample]:
    input_path = Path(input_dir)
    esc_path = Path(esc50_dir)
    samples: list[AudioSample] = []
    samples.extend(_positive_samples(input_path))
    samples.extend(_local_negative_samples(input_path))
    samples.extend(_esc50_negative_samples(esc_path))
    samples.sort(key=lambda sample: (sample.label, str(sample.path)))
    return samples


def split_by_group(
    samples: Iterable[AudioSample],
    validation_fraction: float = 0.2,
    seed: int = 7,
) -> tuple[list[AudioSample], list[AudioSample]]:
    grouped: dict[str, list[AudioSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.group, []).append(sample)

    group_names = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    validation_group_count = max(1, int(round(len(group_names) * validation_fraction)))
    validation_groups = set(group_names[:validation_group_count])

    train: list[AudioSample] = []
    validation: list[AudioSample] = []
    for group_name, group_samples in grouped.items():
        if group_name in validation_groups:
            validation.extend(group_samples)
        else:
            train.extend(group_samples)
    return train, validation


class WindowedFeatureDataset(Dataset):
    def __init__(
        self,
        samples: Iterable[AudioSample],
        feature_config: FeatureConfig | None = None,
        max_windows_per_class: int | None = None,
        hard_negative_categories: Iterable[str] = DEFAULT_HARD_NEGATIVE_CATEGORIES,
        hard_negative_multiplier: int = 3,
        seed: int = 7,
    ) -> None:
        self.feature_config = feature_config or FeatureConfig()
        self.hard_negative_categories = tuple(hard_negative_categories)
        self.hard_negative_multiplier = hard_negative_multiplier
        self.items = self._build_items(list(samples), max_windows_per_class, seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, label = self.items[index]
        x = torch.from_numpy(features).unsqueeze(0)
        y = torch.tensor([label], dtype=torch.float32)
        return x, y

    def _build_items(
        self,
        samples: list[AudioSample],
        max_windows_per_class: int | None,
        seed: int,
    ) -> list[tuple[np.ndarray, float]]:
        by_label: dict[int, list[tuple[np.ndarray, float]]] = {0: [], 1: []}
        shuffled_samples = list(samples)
        random.Random(seed).shuffle(shuffled_samples)
        for sample in shuffled_samples:
            if max_windows_per_class is not None and len(by_label[sample.label]) >= max_windows_per_class:
                continue
            audio = decode_audio_file(sample.path, self.feature_config)
            for window in iter_audio_windows(audio, self.feature_config, drop_silent=sample.label == 0):
                features = log_mel_spectrogram(window, self.feature_config)
                repeats = negative_sample_multiplier(
                    sample,
                    self.hard_negative_categories,
                    self.hard_negative_multiplier,
                )
                for _ in range(repeats):
                    if max_windows_per_class is not None and len(by_label[sample.label]) >= max_windows_per_class:
                        break
                    by_label[sample.label].append((features, float(sample.label)))
                if max_windows_per_class is not None and len(by_label[sample.label]) >= max_windows_per_class:
                    break

        items = by_label[0] + by_label[1]
        random.Random(seed).shuffle(items)
        return items


def negative_sample_multiplier(
    sample: AudioSample,
    hard_negative_categories: Iterable[str] = DEFAULT_HARD_NEGATIVE_CATEGORIES,
    hard_negative_multiplier: int = 3,
) -> int:
    if sample.label != 0:
        return 1
    multiplier = max(1, int(hard_negative_multiplier))
    if sample.source == "local_negative":
        return multiplier
    category = _category_from_source(sample.source)
    if category in set(hard_negative_categories):
        return multiplier
    return 1


def ensure_esc50_dataset(target_dir: str | Path = "output/datasets/esc50") -> Path:
    target = Path(target_dir)
    audio_dir = _find_esc50_audio_dir(target)
    if audio_dir is not None:
        return audio_dir.parents[0]

    target.mkdir(parents=True, exist_ok=True)
    archive_path = target / "esc50-master.zip"
    urllib.request.urlretrieve(ESC50_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target)
    audio_dir = _find_esc50_audio_dir(target)
    if audio_dir is None:
        raise RuntimeError(f"ESC-50 download did not contain an audio directory under {target}")
    return audio_dir.parents[0]


def read_esc50_categories(esc50_dir: str | Path) -> dict[str, str]:
    root = Path(esc50_dir)
    metadata_files = list(root.rglob("esc50.csv"))
    if not metadata_files:
        return {}
    categories = {}
    with metadata_files[0].open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            categories[row["filename"]] = row["category"]
    return categories


def _positive_samples(input_dir: Path) -> list[AudioSample]:
    samples = []
    if not input_dir.exists():
        return samples
    for path in input_dir.rglob("*"):
        if not _is_audio(path):
            continue
        is_cleaned = "CleanedData_SnippedToIsolateMosquitoSounds" in path.parts
        is_root_example = path.parent == input_dir and path.suffix.lower() in {".mp3", ".wav", ".m4a"}
        if not (is_cleaned or is_root_example):
            continue
        group = "mosquito:" + _source_group_from_stem(path.stem)
        samples.append(AudioSample(path=path, label=1, source="mosquito", group=group))
    return samples


def _local_negative_samples(input_dir: Path) -> list[AudioSample]:
    negative_dir = input_dir / "negative"
    samples = []
    if not negative_dir.exists():
        return samples
    for path in negative_dir.rglob("*"):
        if not _is_audio(path):
            continue
        group = "local_negative:" + _source_group_from_stem(path.stem)
        samples.append(AudioSample(path=path, label=0, source="local_negative", group=group))
    return samples


def _esc50_negative_samples(esc50_dir: Path) -> list[AudioSample]:
    audio_dir = _find_esc50_audio_dir(esc50_dir)
    if audio_dir is None:
        return []
    categories = read_esc50_categories(esc50_dir)
    samples = []
    for path in sorted(audio_dir.iterdir()):
        if not _is_audio(path):
            continue
        category = categories.get(path.name, "unknown")
        group = f"esc50:{path.stem}"
        samples.append(AudioSample(path=path, label=0, source=f"esc50:{category}", group=group))
    return samples


def _find_esc50_audio_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    for candidate in [path / "audio", path / "ESC-50-master" / "audio"]:
        if candidate.exists():
            return candidate
    matches = [candidate for candidate in path.rglob("audio") if candidate.is_dir()]
    return matches[0] if matches else None


def _source_group_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0].split(" ", 1)[0]


def _category_from_source(source: str) -> str:
    if ":" not in source:
        return source
    return source.split(":", 1)[1]


def _is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
