from __future__ import annotations

import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EPSILON = 1e-10


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 44100
    window_seconds: float = 1.0
    hop_seconds: float = 0.25
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    f_min: float = 150.0
    f_max: float = 6000.0

    @property
    def window_size(self) -> int:
        return int(round(self.sample_rate * self.window_seconds))

    @property
    def window_hop_size(self) -> int:
        return int(round(self.sample_rate * self.hop_seconds))

    @property
    def frames_per_window(self) -> int:
        return math.ceil(max(0, self.window_size - self.n_fft) / self.hop_length) + 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})


def decode_audio_file(path: str | Path, config: FeatureConfig | None = None) -> np.ndarray:
    cfg = config or FeatureConfig()
    audio_path = Path(path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(cfg.sample_rate),
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH; install ffmpeg to decode audio") from exc
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {audio_path}: {error}") from exc

    audio = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"decoded audio is empty: {audio_path}")
    return np.nan_to_num(audio, copy=False)


def iter_audio_windows(
    audio: Iterable[float] | np.ndarray,
    config: FeatureConfig | None = None,
    drop_silent: bool = False,
) -> list[np.ndarray]:
    cfg = config or FeatureConfig()
    samples = _prepare_audio(audio)
    if samples.size <= cfg.window_size:
        return [_fit_window(samples, cfg.window_size)]

    windows = []
    for start in range(0, samples.size - cfg.window_size + 1, cfg.window_hop_size):
        window = samples[start : start + cfg.window_size]
        if drop_silent and _rms_dbfs(window) < -60.0:
            continue
        windows.append(window.astype(np.float32, copy=False))

    return windows or [_fit_window(samples[-cfg.window_size :], cfg.window_size)]


def log_mel_spectrogram(
    audio: Iterable[float] | np.ndarray,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    cfg = config or FeatureConfig()
    samples = _fit_window(_prepare_audio(audio), cfg.window_size)
    frames = _frame_audio(samples, cfg.n_fft, cfg.hop_length, cfg.frames_per_window)
    window = np.hanning(cfg.n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=cfg.n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    mel_basis = mel_filterbank(cfg)
    mel_power = mel_basis @ power.T
    log_mel = np.log(np.maximum(mel_power, EPSILON))
    log_mel = _standardize(log_mel)
    return log_mel.astype(np.float32, copy=False)


def mel_filterbank(config: FeatureConfig | None = None) -> np.ndarray:
    cfg = config or FeatureConfig()
    fft_freqs = np.fft.rfftfreq(cfg.n_fft, d=1.0 / cfg.sample_rate)
    mel_min = _hz_to_mel(cfg.f_min)
    mel_max = _hz_to_mel(cfg.f_max)
    mel_points = np.linspace(mel_min, mel_max, cfg.n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    basis = np.zeros((cfg.n_mels, fft_freqs.size), dtype=np.float32)
    for mel_index in range(cfg.n_mels):
        left = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        right = hz_points[mel_index + 2]
        up = (fft_freqs - left) / max(center - left, EPSILON)
        down = (right - fft_freqs) / max(right - center, EPSILON)
        basis[mel_index] = np.maximum(0.0, np.minimum(up, down))
        normalizer = np.sum(basis[mel_index])
        if normalizer > 0:
            basis[mel_index] /= normalizer
    return basis


def _prepare_audio(audio: Iterable[float] | np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    samples = np.nan_to_num(samples, copy=False)
    samples = np.clip(samples, -1.0, 1.0)
    if samples.size:
        samples = samples - float(np.mean(samples))
    return samples.astype(np.float32, copy=False)


def _fit_window(samples: np.ndarray, size: int) -> np.ndarray:
    if samples.size == size:
        return samples.astype(np.float32, copy=False)
    if samples.size < size:
        return np.pad(samples, (0, size - samples.size)).astype(np.float32, copy=False)
    return samples[:size].astype(np.float32, copy=False)


def _frame_audio(samples: np.ndarray, n_fft: int, hop_length: int, frame_count: int) -> np.ndarray:
    needed = (frame_count - 1) * hop_length + n_fft
    if samples.size < needed:
        samples = np.pad(samples, (0, needed - samples.size))
    starts = np.arange(frame_count)[:, None] * hop_length
    offsets = np.arange(n_fft)[None, :]
    return samples[starts + offsets].astype(np.float32, copy=False)


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = float(np.mean(features))
    std = float(np.std(features))
    if std < 1e-6:
        return np.zeros_like(features, dtype=np.float32)
    return ((features - mean) / std).astype(np.float32, copy=False)


def _rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return 20.0 * math.log10(rms + EPSILON)


def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)
