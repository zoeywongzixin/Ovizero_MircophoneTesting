from src.state import DetectionStateSmoother


def test_state_requires_sustained_high_confidence_before_detection():
    smoother = DetectionStateSmoother(detect_after_seconds=0.8, clear_after_seconds=1.2)

    assert not smoother.update(confidence=0.80, is_positive=True, timestamp=0.0)
    assert not smoother.update(confidence=0.82, is_positive=True, timestamp=0.4)
    assert smoother.update(confidence=0.83, is_positive=True, timestamp=0.8)


def test_state_requires_sustained_low_confidence_before_clearing():
    smoother = DetectionStateSmoother(detect_after_seconds=0.8, clear_after_seconds=1.2)
    smoother.update(confidence=0.80, is_positive=True, timestamp=0.0)
    smoother.update(confidence=0.81, is_positive=True, timestamp=0.8)

    assert smoother.is_detected
    assert smoother.update(confidence=0.20, is_positive=False, timestamp=1.0)
    assert smoother.update(confidence=0.20, is_positive=False, timestamp=1.8)
    assert not smoother.update(confidence=0.20, is_positive=False, timestamp=2.2)


def test_state_recovers_when_signal_returns_before_clear_timeout():
    smoother = DetectionStateSmoother(detect_after_seconds=0.8, clear_after_seconds=1.2)
    smoother.update(confidence=0.80, is_positive=True, timestamp=0.0)
    smoother.update(confidence=0.81, is_positive=True, timestamp=0.8)
    smoother.update(confidence=0.20, is_positive=False, timestamp=1.0)

    assert smoother.update(confidence=0.82, is_positive=True, timestamp=1.4)
