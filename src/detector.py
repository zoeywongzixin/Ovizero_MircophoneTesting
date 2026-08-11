from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class DetectionConfig:
    sample_rate: int = 44100
    f0_min: float = 300.0
    f0_max: float = 850.0
    window_seconds: float = 1.0
    analysis_min_hz: float = 100.0
    analysis_max_hz: float = 5000.0
    display_max_hz: float = 3500.0
    low_band_min_hz: float = 150.0
    low_band_max_hz: float = 1200.0
    low_band_min_share: float = 0.12
    min_main_db: float = 10.0
    min_harmonic_db: float = -18.0
    harmonic_numbers: tuple[int, ...] = (2, 3, 4)
    harmonic_tolerance_hz: float = 18.0
    harmonic_tolerance_ratio: float = 0.025
    max_flatness: float = 0.45
    detect_confidence: float = 0.62

    @property
    def window_size(self) -> int:
        return max(256, int(round(self.sample_rate * self.window_seconds)))


@dataclass(frozen=True)
class DetectionResult:
    is_mosquito: bool
    confidence: float
    candidate_f0_hz: float
    main_db: float
    harmonic_db: float
    flatness: float
    low_band_energy_share: float
    rms_dbfs: float
    frequency_hz: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spectrum_db: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    harmonic_frequencies_hz: tuple[float, ...] = ()


class MosquitoDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()
        self._background_db: np.ndarray | None = None
        self._background_freqs: np.ndarray | None = None

    @property
    def has_calibration(self) -> bool:
        return self._background_db is not None

    def calibrate(self, audio: Iterable[float] | np.ndarray) -> None:
        spectrum = self._spectrum(audio)
        self._background_freqs = spectrum.frequency_hz
        self._background_db = spectrum.spectrum_db

    def clear_calibration(self) -> None:
        self._background_db = None
        self._background_freqs = None

    def analyze(self, audio: Iterable[float] | np.ndarray) -> DetectionResult:
        spectrum = self._spectrum(audio)
        cfg = self.config

        search_mask = (
            (spectrum.frequency_hz >= cfg.f0_min)
            & (spectrum.frequency_hz <= cfg.f0_max)
        )
        if not np.any(search_mask):
            return self._empty_result(spectrum)

        search_indices = np.where(search_mask)[0]
        candidate_index = self._find_candidate_index(spectrum, search_indices)
        candidate_f0 = float(spectrum.frequency_hz[candidate_index])
        raw_peak_db = float(spectrum.spectrum_db[candidate_index])
        main_db = self._relative_db_at(spectrum, candidate_index)

        harmonic_dbs: list[float] = []
        harmonic_freqs: list[float] = []
        for number in cfg.harmonic_numbers:
            harmonic_freq = candidate_f0 * number
            if harmonic_freq > cfg.analysis_max_hz:
                continue
            harmonic_index = self._peak_near(spectrum.frequency_hz, spectrum.spectrum_db, harmonic_freq)
            if harmonic_index is None:
                continue
            harmonic_relative = self._relative_db_at(spectrum, harmonic_index) - main_db
            harmonic_dbs.append(float(harmonic_relative))
            harmonic_freqs.append(float(spectrum.frequency_hz[harmonic_index]))

        harmonic_db = max(harmonic_dbs) if harmonic_dbs else -120.0
        low_share = self._band_energy_share(
            spectrum.frequency_hz,
            spectrum.power,
            cfg.low_band_min_hz,
            cfg.low_band_max_hz,
            cfg.analysis_min_hz,
            cfg.analysis_max_hz,
        )
        flatness = self._spectral_flatness(
            spectrum.frequency_hz,
            spectrum.power,
            cfg.analysis_min_hz,
            cfg.analysis_max_hz,
        )

        confidence = self._confidence(main_db, harmonic_db, low_share, flatness)
        is_mosquito = bool(confidence >= cfg.detect_confidence)
        if raw_peak_db < -95.0:
            is_mosquito = False
            confidence = min(confidence, 0.2)

        return DetectionResult(
            is_mosquito=is_mosquito,
            confidence=confidence,
            candidate_f0_hz=candidate_f0 if confidence > 0.05 else 0.0,
            main_db=main_db,
            harmonic_db=harmonic_db,
            flatness=flatness,
            low_band_energy_share=low_share,
            rms_dbfs=spectrum.rms_dbfs,
            frequency_hz=spectrum.frequency_hz,
            spectrum_db=spectrum.spectrum_db,
            harmonic_frequencies_hz=tuple(harmonic_freqs),
        )

    def _confidence(
        self,
        main_db: float,
        harmonic_db: float,
        low_share: float,
        flatness: float,
    ) -> float:
        cfg = self.config
        main_score = _clamp((main_db - 4.0) / (cfg.min_main_db - 4.0))
        harmonic_score = _clamp((harmonic_db - (-30.0)) / (cfg.min_harmonic_db - (-30.0)))
        low_score = _clamp(low_share / cfg.low_band_min_share)
        flatness_score = _clamp((cfg.max_flatness - flatness) / cfg.max_flatness)
        score = (
            0.38 * main_score
            + 0.31 * harmonic_score
            + 0.18 * low_score
            + 0.13 * flatness_score
        )
        if main_db < cfg.min_main_db:
            score *= 0.72
        if harmonic_db < cfg.min_harmonic_db:
            score *= 0.58
        if low_share < cfg.low_band_min_share:
            score *= 0.45
        if flatness > cfg.max_flatness:
            score *= 0.55
        return _clamp(score)

    def _find_candidate_index(self, spectrum: "_Spectrum", indices: np.ndarray) -> int:
        primary_index = int(indices[int(np.argmax(spectrum.spectrum_db[indices]))])
        primary_hz = float(spectrum.frequency_hz[primary_index])
        primary_db = float(spectrum.spectrum_db[primary_index])

        for divisor in (2.0, 3.0):
            subharmonic_hz = primary_hz / divisor
            if not self.config.f0_min <= subharmonic_hz <= self.config.f0_max:
                continue
            subharmonic_index = self._peak_near(
                spectrum.frequency_hz,
                spectrum.spectrum_db,
                subharmonic_hz,
            )
            if subharmonic_index is None:
                continue
            subharmonic_db = float(spectrum.spectrum_db[subharmonic_index])
            if subharmonic_db >= primary_db - 14.0:
                return subharmonic_index

        return primary_index

    def _relative_db_at(self, spectrum: "_Spectrum", index: int) -> float:
        if self._background_db is None:
            noise_floor = float(np.percentile(spectrum.spectrum_db, 35))
            return float(spectrum.spectrum_db[index] - noise_floor)
        background_index = index
        if self._background_freqs is not None and len(self._background_freqs) == len(self._background_db):
            background_index = int(
                np.argmin(np.abs(self._background_freqs - spectrum.frequency_hz[index]))
            )
        return float(spectrum.spectrum_db[index] - self._background_db[background_index])

    def _peak_near(
        self,
        freqs: np.ndarray,
        spectrum_db: np.ndarray,
        target_hz: float,
    ) -> int | None:
        tolerance = max(
            self.config.harmonic_tolerance_hz,
            target_hz * self.config.harmonic_tolerance_ratio,
        )
        mask = (freqs >= target_hz - tolerance) & (freqs <= target_hz + tolerance)
        if not np.any(mask):
            return None
        indices = np.where(mask)[0]
        return int(indices[int(np.argmax(spectrum_db[indices]))])

    def _spectrum(self, audio: Iterable[float] | np.ndarray) -> "_Spectrum":
        cfg = self.config
        samples = np.asarray(audio, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            samples = np.zeros(cfg.window_size, dtype=np.float64)
        if samples.size < cfg.window_size:
            samples = np.pad(samples, (cfg.window_size - samples.size, 0))
        elif samples.size > cfg.window_size:
            samples = samples[-cfg.window_size :]

        samples = np.nan_to_num(samples, copy=False)
        samples = np.clip(samples, -1.0, 1.0)
        samples = samples - float(np.mean(samples))
        rms = float(np.sqrt(np.mean(samples * samples)))
        rms_dbfs = 20.0 * np.log10(rms + EPSILON)

        window = np.hanning(samples.size)
        windowed = samples * window
        magnitude = np.abs(np.fft.rfft(windowed)) / (np.sum(window) / 2.0 + EPSILON)
        power = magnitude * magnitude
        spectrum_db = 20.0 * np.log10(magnitude + EPSILON)
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / cfg.sample_rate)
        return _Spectrum(freqs, spectrum_db, power, rms_dbfs)

    def _empty_result(self, spectrum: "_Spectrum") -> DetectionResult:
        return DetectionResult(
            is_mosquito=False,
            confidence=0.0,
            candidate_f0_hz=0.0,
            main_db=0.0,
            harmonic_db=-120.0,
            flatness=1.0,
            low_band_energy_share=0.0,
            rms_dbfs=spectrum.rms_dbfs,
            frequency_hz=spectrum.frequency_hz,
            spectrum_db=spectrum.spectrum_db,
        )

    @staticmethod
    def _band_energy_share(
        freqs: np.ndarray,
        power: np.ndarray,
        numerator_min_hz: float,
        numerator_max_hz: float,
        denominator_min_hz: float,
        denominator_max_hz: float,
    ) -> float:
        num_mask = (freqs >= numerator_min_hz) & (freqs <= numerator_max_hz)
        den_mask = (freqs >= denominator_min_hz) & (freqs <= denominator_max_hz)
        numerator = float(np.sum(power[num_mask]))
        denominator = float(np.sum(power[den_mask]) + EPSILON)
        return numerator / denominator

    @staticmethod
    def _spectral_flatness(
        freqs: np.ndarray,
        power: np.ndarray,
        min_hz: float,
        max_hz: float,
    ) -> float:
        mask = (freqs >= min_hz) & (freqs <= max_hz)
        band_power = power[mask] + EPSILON
        return float(np.exp(np.mean(np.log(band_power))) / (np.mean(band_power) + EPSILON))


@dataclass(frozen=True)
class _Spectrum:
    frequency_hz: np.ndarray
    spectrum_db: np.ndarray
    power: np.ndarray
    rms_dbfs: float


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return float(max(minimum, min(maximum, value)))
