import numpy as np
import torch

from src.detector import DetectionResult
from src.ml.detector import AIMosquitoDetector, DEFAULT_PANNS_RUNTIME_THRESHOLD_FLOOR
from src.ml.features import FeatureConfig
from src.ml.model import MosquitoCRNN, PANNsMosquitoClassifier


def test_ai_detector_loads_artifact_and_returns_detection_result(tmp_path):
    feature_config = FeatureConfig(n_mels=40)
    model = MosquitoCRNN(n_mels=40, lstm_hidden_size=16)
    artifact_path = tmp_path / "mosquito_crnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "feature_config": feature_config.to_dict(),
            "threshold": 0.0,
            "metrics": {"precision": 1.0},
        },
        artifact_path,
    )

    detector = AIMosquitoDetector(artifact_path)
    result = detector.analyze(np.zeros(feature_config.window_size, dtype=np.float32))

    assert isinstance(result, DetectionResult)
    assert result.is_mosquito
    assert 0.0 <= result.confidence <= 1.0
    assert result.frequency_hz.size > 0
    assert result.spectrum_db.size > 0


def test_ai_detector_loads_panns_artifact_with_same_runtime_interface(tmp_path):
    feature_config = FeatureConfig(n_mels=64)
    model = PANNsMosquitoClassifier(backbone="cnn10", n_mels=64)
    artifact_path = tmp_path / "mosquito_panns.pt"
    torch.save(
        {
            "model_type": "panns_cnn10",
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "feature_config": feature_config.to_dict(),
            "threshold": 0.0,
            "metrics": {"recall": 1.0},
        },
        artifact_path,
    )

    detector = AIMosquitoDetector(artifact_path)
    result = detector.analyze(np.zeros(feature_config.window_size, dtype=np.float32))

    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.confidence <= 1.0


def test_ai_detector_raises_low_panns_threshold_to_reduce_false_positives(tmp_path):
    feature_config = FeatureConfig(n_mels=64)
    model = PANNsMosquitoClassifier(backbone="cnn10", n_mels=64)
    artifact_path = tmp_path / "mosquito_panns.pt"
    torch.save(
        {
            "model_type": "panns_cnn10",
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "feature_config": feature_config.to_dict(),
            "threshold": 0.09,
            "metrics": {"precision": 0.52, "recall": 0.96},
        },
        artifact_path,
    )

    detector = AIMosquitoDetector(artifact_path)

    assert detector.artifact_threshold == 0.09
    assert detector.threshold == DEFAULT_PANNS_RUNTIME_THRESHOLD_FLOOR
