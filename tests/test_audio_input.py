import numpy as np

from src.audio_input import RollingAudioBuffer, list_input_devices


def test_rolling_buffer_returns_latest_window_with_padding():
    buffer = RollingAudioBuffer(sample_rate=10, seconds=1.0)

    buffer.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    latest = buffer.latest(1.0)
    assert latest.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_rolling_buffer_keeps_only_capacity():
    buffer = RollingAudioBuffer(sample_rate=5, seconds=1.0)

    buffer.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    buffer.append(np.array([4.0, 5.0, 6.0], dtype=np.float32))

    latest = buffer.latest(1.0)
    assert latest.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_list_input_devices_filters_output_only_devices():
    raw_devices = [
        {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 44100},
        {"name": "USB Mic", "max_input_channels": 2, "default_samplerate": 48000},
    ]

    devices = list_input_devices(lambda: raw_devices)

    assert len(devices) == 1
    assert devices[0].index == 1
    assert devices[0].name == "USB Mic"
    assert devices[0].default_sample_rate == 48000
