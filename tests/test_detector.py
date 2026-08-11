import numpy as np
import pytest

from src.detector import DetectionConfig, MosquitoDetector


def tone(freq_hz, duration_s=1.0, sample_rate=44100, amplitude=0.45):
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def mosquito_tone(f0_hz, duration_s=1.0, sample_rate=44100):
    signal = tone(f0_hz, duration_s, sample_rate, 0.42)
    signal += tone(2.0 * f0_hz, duration_s, sample_rate, 0.22)
    signal += tone(3.0 * f0_hz, duration_s, sample_rate, 0.14)
    return signal.astype(np.float32)


def detector():
    config = DetectionConfig(sample_rate=44100, window_seconds=1.0)
    return MosquitoDetector(config)


def test_detects_366_hz_mosquito_harmonic_family():
    result = detector().analyze(mosquito_tone(366.0))

    assert result.is_mosquito
    assert result.confidence >= 0.70
    assert result.candidate_f0_hz == pytest.approx(366.0, abs=8.0)
    assert result.harmonic_db >= -16.0


def test_detects_702_hz_mosquito_harmonic_family():
    result = detector().analyze(mosquito_tone(702.0))

    assert result.is_mosquito
    assert result.confidence >= 0.70
    assert result.candidate_f0_hz == pytest.approx(702.0, abs=8.0)
    assert result.harmonic_db >= -16.0


def test_rejects_white_noise():
    rng = np.random.default_rng(42)
    audio = rng.normal(0.0, 0.18, 44100).astype(np.float32)

    result = detector().analyze(audio)

    assert not result.is_mosquito
    assert result.confidence < 0.55


def test_rejects_low_frequency_fan_like_tone():
    audio = tone(120.0, amplitude=0.45).astype(np.float32)

    result = detector().analyze(audio)

    assert not result.is_mosquito
    assert result.candidate_f0_hz == 0.0 or result.confidence < 0.55


def test_rejects_pure_high_frequency_buzz():
    audio = tone(2400.0, amplitude=0.45).astype(np.float32)

    result = detector().analyze(audio)

    assert not result.is_mosquito
    assert result.low_band_energy_share < 0.12


def test_calibration_suppresses_matching_background_signal():
    audio = mosquito_tone(366.0)
    mosquito_detector = detector()
    mosquito_detector.calibrate(audio)

    result = mosquito_detector.analyze(audio)

    assert not result.is_mosquito
    assert result.main_db < 6.0
