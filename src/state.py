from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class DetectionStateSmoother:
    detect_after_seconds: float = 0.8
    clear_after_seconds: float = 1.2

    def __post_init__(self) -> None:
        self.is_detected = False
        self._positive_since: float | None = None
        self._clear_since: float | None = None

    def update(
        self,
        confidence: float,
        is_positive: bool,
        timestamp: float | None = None,
    ) -> bool:
        now = time.monotonic() if timestamp is None else timestamp
        positive = bool(is_positive and confidence > 0.0)

        if positive:
            if self._positive_since is None:
                self._positive_since = now
            self._clear_since = None
            if now - self._positive_since >= self.detect_after_seconds:
                self.is_detected = True
            return self.is_detected

        self._positive_since = None
        if self.is_detected:
            if self._clear_since is None:
                self._clear_since = now
            if now - self._clear_since >= self.clear_after_seconds:
                self.is_detected = False
                self._clear_since = None
        else:
            self._clear_since = None
        return self.is_detected

    def reset(self) -> None:
        self.is_detected = False
        self._positive_since = None
        self._clear_since = None
