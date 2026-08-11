import numpy as np

from src.detector import MosquitoDetector
from src.detector import DetectionResult
from src.ui import create_default_detector, format_metric_lines, format_status_text


def make_result():
    return DetectionResult(
        is_mosquito=True,
        confidence=0.82,
        candidate_f0_hz=702.4,
        main_db=18.2,
        harmonic_db=-7.5,
        flatness=0.02,
        low_band_energy_share=0.58,
        rms_dbfs=-21.3,
        frequency_hz=np.array([100.0, 702.0]),
        spectrum_db=np.array([-80.0, -20.0]),
        harmonic_frequencies_hz=(1404.0, 2106.0),
    )


def test_format_status_text_uses_smoothed_state():
    assert format_status_text(True) == "Mosquito detected"
    assert format_status_text(False) == "No mosquito detected"


def test_format_metric_lines_includes_key_detection_values():
    lines = format_metric_lines(make_result(), calibrated=True)

    assert "Confidence: 82%" in lines
    assert "Main peak: 702.4 Hz" in lines
    assert "Calibration: calibrated" in lines


def test_create_default_detector_falls_back_when_ai_model_is_missing(tmp_path):
    detector, message = create_default_detector(tmp_path / "missing.pt")

    assert isinstance(detector, MosquitoDetector)
    assert "AI model not found" in message
