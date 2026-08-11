import subprocess

import numpy as np
from scipy.io import wavfile

from src.ml.features import FeatureConfig, decode_audio_file, log_mel_spectrogram


def write_tone(path, sample_rate=44100, seconds=1.0, freq_hz=440.0):
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    audio = 0.35 * np.sin(2.0 * np.pi * freq_hz * t)
    wavfile.write(path, sample_rate, audio.astype(np.float32))


def transcode(src, dst):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), str(dst)],
        check=True,
    )


def test_log_mel_spectrogram_has_fixed_shape_for_short_and_long_audio():
    config = FeatureConfig(n_mels=40, sample_rate=44100, window_seconds=1.0)
    short_audio = np.zeros(int(config.sample_rate * 0.25), dtype=np.float32)
    long_audio = np.zeros(int(config.sample_rate * 2.0), dtype=np.float32)

    short_features = log_mel_spectrogram(short_audio, config)
    long_features = log_mel_spectrogram(long_audio, config)

    assert short_features.shape == (40, config.frames_per_window)
    assert long_features.shape == (40, config.frames_per_window)
    assert short_features.dtype == np.float32


def test_decode_audio_file_supports_wav_mp3_and_m4a(tmp_path):
    wav_path = tmp_path / "tone.wav"
    mp3_path = tmp_path / "tone.mp3"
    m4a_path = tmp_path / "tone.m4a"
    write_tone(wav_path)
    transcode(wav_path, mp3_path)
    transcode(wav_path, m4a_path)

    config = FeatureConfig(sample_rate=44100)
    for path in (wav_path, mp3_path, m4a_path):
        audio = decode_audio_file(path, config)
        features = log_mel_spectrogram(audio, config)
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        assert features.shape == (config.n_mels, config.frames_per_window)
