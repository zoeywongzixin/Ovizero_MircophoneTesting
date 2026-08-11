from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import DetectionConfig, DetectionResult, MosquitoDetector


@dataclass(frozen=True)
class AnalysisSummary:
    path: Path
    duration_seconds: float
    best_result: DetectionResult
    positive_window_share: float
    analyzed_windows: int


def decode_audio(path: str | Path, sample_rate: int = 44100) -> np.ndarray:
    audio_path = Path(path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH; install ffmpeg to analyze files") from exc
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode {audio_path}: {error}") from exc

    audio = np.frombuffer(completed.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError(f"decoded audio is empty: {audio_path}")
    return np.nan_to_num(audio, copy=False)


def analyze_file(
    path: str | Path,
    config: DetectionConfig | None = None,
    hop_seconds: float = 0.2,
) -> AnalysisSummary:
    cfg = config or DetectionConfig()
    audio = decode_audio(path, cfg.sample_rate)
    detector = MosquitoDetector(cfg)
    windows = list(_active_windows(audio, cfg, hop_seconds))
    if not windows:
        windows = [audio]

    results = [detector.analyze(window) for window in windows]
    best = max(results, key=lambda result: result.confidence)
    positive_share = sum(1 for result in results if result.is_mosquito) / len(results)
    return AnalysisSummary(
        path=Path(path),
        duration_seconds=float(audio.size / cfg.sample_rate),
        best_result=best,
        positive_window_share=float(positive_share),
        analyzed_windows=len(results),
    )


def analyze_directory(input_dir: str | Path = "input") -> list[AnalysisSummary]:
    paths = sorted(Path(input_dir).glob("*.mp3"))
    return [analyze_file(path) for path in paths]


def summary_status(summary: AnalysisSummary) -> str:
    if summary.best_result.is_mosquito and summary.positive_window_share >= 0.50:
        return "MOSQUITO"
    if summary.best_result.is_mosquito:
        return "BOUNDARY"
    return "clear"


def _active_windows(
    audio: np.ndarray,
    config: DetectionConfig,
    hop_seconds: float,
) -> list[np.ndarray]:
    window_size = config.window_size
    hop_size = max(1, int(round(config.sample_rate * hop_seconds)))
    if audio.size <= window_size:
        return [audio]

    windows: list[np.ndarray] = []
    rms_values: list[float] = []
    for start in range(0, audio.size - window_size + 1, hop_size):
        window = audio[start : start + window_size]
        rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
        windows.append(window)
        rms_values.append(rms)

    rms_db = 20.0 * np.log10(np.asarray(rms_values) + 1e-12)
    threshold = max(float(np.max(rms_db) - 35.0), -55.0)
    return [window for window, db in zip(windows, rms_db) if db >= threshold]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mosquito audio samples with FFT rules.")
    parser.add_argument("paths", nargs="*", type=Path, help="MP3/WAV files to analyze")
    parser.add_argument("--input-dir", type=Path, default=Path("input"), help="Directory used when no paths are passed")
    args = parser.parse_args()

    paths = args.paths or sorted(args.input_dir.glob("*.mp3"))
    for path in paths:
        summary = analyze_file(path)
        result = summary.best_result
        status = summary_status(summary)
        print(
            f"{path.name}: {status} "
            f"confidence={result.confidence:.2f} "
            f"f0={result.candidate_f0_hz:.1f}Hz "
            f"main={result.main_db:.1f}dB "
            f"harmonic={result.harmonic_db:.1f}dB "
            f"low_share={result.low_band_energy_share:.2f} "
            f"positive_windows={summary.positive_window_share:.0%}"
        )


if __name__ == "__main__":
    main()
