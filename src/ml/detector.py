from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..detector import DetectionConfig, DetectionResult, MosquitoDetector
from .features import FeatureConfig, log_mel_spectrogram
from .model import create_model, predict_probability


DEFAULT_MODEL_PATH = Path("output/models/mosquito_panns.pt")
LEGACY_CRNN_MODEL_PATH = Path("output/models/mosquito_crnn.pt")
DEFAULT_PANNS_RUNTIME_THRESHOLD_FLOOR = 0.65


class AIMosquitoDetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: torch.device | str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        artifact = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.feature_config = FeatureConfig.from_dict(artifact["feature_config"])
        self.artifact_threshold = float(artifact["threshold"])
        self.metrics = artifact.get("metrics", {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_type = artifact.get("model_type", "crnn")
        self.threshold = self._runtime_threshold(self.artifact_threshold, self.model_type)
        self.model = create_model(self.model_type, artifact["model_config"])
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self._fft_detector = MosquitoDetector(
            DetectionConfig(
                sample_rate=self.feature_config.sample_rate,
                window_seconds=self.feature_config.window_seconds,
            )
        )

    def _runtime_threshold(self, artifact_threshold: float, model_type: str) -> float:
        if _is_panns_model(model_type):
            return max(artifact_threshold, DEFAULT_PANNS_RUNTIME_THRESHOLD_FLOOR)
        return artifact_threshold

    @property
    def has_calibration(self) -> bool:
        return self._fft_detector.has_calibration

    def calibrate(self, audio: Iterable[float] | np.ndarray) -> None:
        self._fft_detector.calibrate(audio)

    def clear_calibration(self) -> None:
        self._fft_detector.clear_calibration()

    def analyze(self, audio: Iterable[float] | np.ndarray) -> DetectionResult:
        fft_result = self._fft_detector.analyze(audio)
        features = log_mel_spectrogram(audio, self.feature_config)
        tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(self.device)
        probability = float(predict_probability(self.model, tensor).detach().cpu()[0])
        return DetectionResult(
            is_mosquito=probability >= self.threshold,
            confidence=probability,
            candidate_f0_hz=fft_result.candidate_f0_hz,
            main_db=fft_result.main_db,
            harmonic_db=fft_result.harmonic_db,
            flatness=fft_result.flatness,
            low_band_energy_share=fft_result.low_band_energy_share,
            rms_dbfs=fft_result.rms_dbfs,
            frequency_hz=fft_result.frequency_hz,
            spectrum_db=fft_result.spectrum_db,
            harmonic_frequencies_hz=fft_result.harmonic_frequencies_hz,
        )


def load_ai_detector_if_available(
    model_path: str | Path | None = None,
) -> AIMosquitoDetector | None:
    paths = [Path(model_path)] if model_path is not None else [
        DEFAULT_MODEL_PATH,
        LEGACY_CRNN_MODEL_PATH,
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            return AIMosquitoDetector(path)
        except Exception:
            continue
    return None


def _is_panns_model(model_type: str) -> bool:
    return model_type.startswith("panns_")
