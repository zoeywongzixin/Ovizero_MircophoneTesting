from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .detector import DetectionConfig

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - exercised only when dependency is absent.
    sd = None


class AudioInputError(RuntimeError):
    """Raised when microphone input cannot be opened or read."""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int


class RollingAudioBuffer:
    def __init__(self, sample_rate: int, seconds: float) -> None:
        self.sample_rate = sample_rate
        self.capacity = max(1, int(round(sample_rate * seconds)))
        self._data = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()

    def append(self, samples: Iterable[float] | np.ndarray) -> None:
        chunk = np.asarray(samples, dtype=np.float32)
        if chunk.ndim > 1:
            chunk = chunk[:, 0]
        chunk = np.nan_to_num(chunk.reshape(-1), copy=False)
        if chunk.size == 0:
            return
        with self._lock:
            combined = np.concatenate((self._data, chunk))
            if combined.size > self.capacity:
                combined = combined[-self.capacity :]
            self._data = combined.astype(np.float32, copy=False)

    def latest(self, seconds: float) -> np.ndarray:
        size = max(1, int(round(self.sample_rate * seconds)))
        with self._lock:
            data = self._data[-size:].copy()
        if data.size < size:
            data = np.pad(data, (size - data.size, 0))
        return data.astype(np.float32, copy=False)

    def clear(self) -> None:
        with self._lock:
            self._data = np.zeros(0, dtype=np.float32)


class MicrophoneStream:
    def __init__(
        self,
        config: DetectionConfig,
        device_index: int | None = None,
        buffer_seconds: float = 8.0,
    ) -> None:
        self.config = config
        self.device_index = device_index
        self.buffer = RollingAudioBuffer(config.sample_rate, buffer_seconds)
        self.last_status = ""
        self._stream = None

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        sounddevice = _require_sounddevice()
        blocksize = max(256, int(round(self.config.sample_rate * 0.05)))
        try:
            self._stream = sounddevice.InputStream(
                samplerate=self.config.sample_rate,
                device=self.device_index,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioInputError(f"Could not open microphone input: {exc}") from exc

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:
            raise AudioInputError(f"Could not stop microphone input cleanly: {exc}") from exc

    def latest_window(self, seconds: float | None = None) -> np.ndarray:
        return self.buffer.latest(seconds or self.config.window_seconds)

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.last_status = str(status)
        self.buffer.append(indata)


def list_input_devices(
    query_devices: Callable[[], Iterable[dict]] | None = None,
) -> list[AudioDevice]:
    if query_devices is None:
        query_devices = _require_sounddevice().query_devices

    devices: list[AudioDevice] = []
    for index, raw in enumerate(query_devices()):
        max_input_channels = int(raw.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        devices.append(
            AudioDevice(
                index=index,
                name=str(raw.get("name", f"Input {index}")),
                max_input_channels=max_input_channels,
                default_sample_rate=int(round(float(raw.get("default_samplerate", 0)))),
            )
        )
    return devices


def _require_sounddevice():
    if sd is None:
        raise AudioInputError(
            "sounddevice is not installed. Run `python -m pip install -r requirements.txt`."
        )
    return sd
