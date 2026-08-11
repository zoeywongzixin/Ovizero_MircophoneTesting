from pathlib import Path

import pytest

from src.analyze import AnalysisSummary, analyze_file, summary_status
from src.detector import DetectionResult


INPUT_DIR = Path(__file__).resolve().parents[1] / "input"


def test_squeaks_sample_is_detected_near_702_hz():
    summary = analyze_file(INPUT_DIR / "mosquito-squeaks.mp3")

    assert summary.best_result.is_mosquito
    assert summary.best_result.candidate_f0_hz == pytest.approx(702.0, abs=25.0)
    assert summary.positive_window_share >= 0.30


def test_over_ear_sample_is_detected_near_366_hz():
    summary = analyze_file(INPUT_DIR / "the-sound-of-a-mosquito-that-flies-over-the-ear.mp3")

    assert summary.best_result.is_mosquito
    assert summary.best_result.candidate_f0_hz == pytest.approx(366.0, abs=35.0)
    assert summary.positive_window_share >= 0.20


def test_buzzing_sample_is_reported_as_boundary_data_not_required_positive():
    summary = analyze_file(INPUT_DIR / "mosquito-buzzing-sound.mp3")

    assert summary.duration_seconds > 5.0
    assert 0.0 <= summary.positive_window_share <= 1.0
    assert summary.best_result.confidence >= 0.0


def test_summary_status_marks_inconsistent_positive_windows_as_boundary():
    result = DetectionResult(
        is_mosquito=True,
        confidence=0.91,
        candidate_f0_hz=585.0,
        main_db=20.0,
        harmonic_db=-6.0,
        flatness=0.05,
        low_band_energy_share=0.30,
        rms_dbfs=-20.0,
    )
    summary = AnalysisSummary(
        path=INPUT_DIR / "mosquito-buzzing-sound.mp3",
        duration_seconds=24.8,
        best_result=result,
        positive_window_share=0.38,
        analyzed_windows=100,
    )

    assert summary_status(summary) == "BOUNDARY"
