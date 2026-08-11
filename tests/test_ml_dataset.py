import numpy as np
from scipy.io import wavfile

from src.ml.dataset import (
    AudioSample,
    WindowedFeatureDataset,
    build_sample_manifest,
    negative_sample_multiplier,
    split_by_group,
)
from src.ml.features import FeatureConfig


def write_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, 44100, np.zeros(44100, dtype=np.float32))


def test_build_sample_manifest_labels_cleaned_mosquito_and_esc50_negative(tmp_path):
    input_dir = tmp_path / "input"
    positive_path = (
        input_dir
        / "doi"
        / "Aedes+sierrensis"
        / "Aedes sierrensis"
        / "CleanedData_SnippedToIsolateMosquitoSounds"
        / "mosquito.wav"
    )
    root_mp3 = input_dir / "mosquito-squeaks.mp3"
    esc_audio = tmp_path / "esc50" / "ESC-50-master" / "audio" / "1-100032-A-0.wav"
    write_wav(positive_path)
    write_wav(root_mp3)
    write_wav(esc_audio)

    samples = build_sample_manifest(input_dir=input_dir, esc50_dir=tmp_path / "esc50")

    labels = {sample.path.name: sample.label for sample in samples}
    assert labels["mosquito.wav"] == 1
    assert labels["mosquito-squeaks.mp3"] == 1
    assert labels["1-100032-A-0.wav"] == 0


def test_build_sample_manifest_labels_input_negative_directory_as_negative(tmp_path):
    input_dir = tmp_path / "input"
    positive_path = input_dir / "mosquito-squeaks.mp3"
    wind_path = input_dir / "negative" / "wind.wav"
    car_path = input_dir / "negative" / "traffic" / "car.wav"
    write_wav(positive_path)
    write_wav(wind_path)
    write_wav(car_path)

    samples = build_sample_manifest(input_dir=input_dir, esc50_dir=tmp_path / "missing-esc50")

    by_name = {sample.path.name: sample for sample in samples}
    assert by_name["mosquito-squeaks.mp3"].label == 1
    assert by_name["wind.wav"].label == 0
    assert by_name["wind.wav"].source == "local_negative"
    assert by_name["car.wav"].label == 0
    assert by_name["car.wav"].source == "local_negative"


def test_negative_sample_multiplier_prioritizes_hard_esc50_and_local_negatives(tmp_path):
    wind = AudioSample(tmp_path / "wind.wav", label=0, source="esc50:wind", group="wind")
    dog = AudioSample(tmp_path / "dog.wav", label=0, source="esc50:dog", group="dog")
    local = AudioSample(tmp_path / "fan.wav", label=0, source="local_negative", group="fan")
    mosquito = AudioSample(tmp_path / "mosquito.wav", label=1, source="mosquito", group="m")

    assert negative_sample_multiplier(wind, ("wind",), 4) == 4
    assert negative_sample_multiplier(local, ("wind",), 4) == 4
    assert negative_sample_multiplier(dog, ("wind",), 4) == 1
    assert negative_sample_multiplier(mosquito, ("wind",), 4) == 1


def test_split_by_group_keeps_source_groups_separate(tmp_path):
    input_dir = tmp_path / "input"
    esc_dir = tmp_path / "esc50" / "ESC-50-master" / "audio"
    for index in range(4):
        write_wav(
            input_dir
            / "doi"
            / "Aedes+sierrensis"
            / "Aedes sierrensis"
            / "CleanedData_SnippedToIsolateMosquitoSounds"
            / f"source{index}_clip.wav"
        )
        write_wav(esc_dir / f"1-1000{index}-A-{index}.wav")

    samples = build_sample_manifest(input_dir=input_dir, esc50_dir=tmp_path / "esc50")
    train, validation = split_by_group(samples, validation_fraction=0.25, seed=123)

    train_groups = {sample.group for sample in train}
    validation_groups = {sample.group for sample in validation}
    assert train
    assert validation
    assert train_groups.isdisjoint(validation_groups)


def test_windowed_feature_dataset_respects_max_windows_per_class(tmp_path):
    positive = tmp_path / "positive.wav"
    negative = tmp_path / "negative.wav"
    write_wav(positive)
    write_wav(negative)
    samples = [
        AudioSample(path=positive, label=1, source="mosquito", group="p1"),
        AudioSample(path=negative, label=0, source="esc50:test", group="n1"),
    ]

    dataset = WindowedFeatureDataset(
        samples,
        FeatureConfig(n_mels=16, hop_seconds=0.1),
        max_windows_per_class=1,
    )

    assert len(dataset) == 2


def test_windowed_feature_dataset_applies_hard_negative_multiplier_before_window_cap(tmp_path):
    positive = tmp_path / "positive.wav"
    wind = tmp_path / "wind.wav"
    dog = tmp_path / "dog.wav"
    write_wav(positive)
    write_wav(wind)
    write_wav(dog)
    samples = [
        AudioSample(path=positive, label=1, source="mosquito", group="p1"),
        AudioSample(path=wind, label=0, source="esc50:wind", group="wind"),
        AudioSample(path=dog, label=0, source="esc50:dog", group="dog"),
    ]

    dataset = WindowedFeatureDataset(
        samples,
        FeatureConfig(n_mels=16, hop_seconds=0.1),
        max_windows_per_class=3,
        hard_negative_categories=("wind",),
        hard_negative_multiplier=3,
    )

    negative_count = sum(1 for _, label in dataset.items if label == 0.0)
    assert negative_count == 3
