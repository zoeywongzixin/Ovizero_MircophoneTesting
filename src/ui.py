from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .audio_input import AudioInputError, MicrophoneStream, list_input_devices
from .detector import DetectionConfig, DetectionResult, MosquitoDetector
from .ml.detector import DEFAULT_MODEL_PATH, LEGACY_CRNN_MODEL_PATH, load_ai_detector_if_available
from .state import DetectionStateSmoother


REFRESH_MS = 200
CALIBRATION_SECONDS = 3.0


def format_status_text(is_detected: bool) -> str:
    return "Mosquito detected" if is_detected else "No mosquito detected"


def format_metric_lines(result: DetectionResult | None, calibrated: bool) -> str:
    calibration = "calibrated" if calibrated else "not calibrated"
    if result is None:
        return (
            "Confidence: 0%\n"
            "Main peak: -- Hz\n"
            "Harmonic: -- dB\n"
            "Low-band share: --\n"
            "Volume: -- dBFS\n"
            f"Calibration: {calibration}"
        )
    return (
        f"Confidence: {result.confidence:.0%}\n"
        f"Main peak: {result.candidate_f0_hz:.1f} Hz\n"
        f"Main strength: {result.main_db:.1f} dB\n"
        f"Harmonic: {result.harmonic_db:.1f} dB\n"
        f"Low-band share: {result.low_band_energy_share:.0%}\n"
        f"Flatness: {result.flatness:.3f}\n"
        f"Volume: {result.rms_dbfs:.1f} dBFS\n"
        f"Calibration: {calibration}"
    )


def create_default_detector(model_path: str | Path | None = None):
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    ai_detector = load_ai_detector_if_available(model_path)
    if ai_detector is not None:
        return ai_detector, "AI model loaded"
    if path.exists():
        return MosquitoDetector(), "AI model could not be loaded; using FFT fallback"
    if model_path is None and LEGACY_CRNN_MODEL_PATH.exists():
        return MosquitoDetector(), "AI model could not be loaded; using FFT fallback"
    return (
        MosquitoDetector(),
        "AI model not found; using FFT fallback. Train with `python -m src.ml.train`.",
    )


class MosquitoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.config = DetectionConfig()
        self.detector, self.detector_message = create_default_detector()
        self.smoother = DetectionStateSmoother()
        self.stream: MicrophoneStream | None = None
        self.last_result: DetectionResult | None = None
        self.calibration_deadline: float | None = None
        self.devices = []

        self.status_var = tk.StringVar(value=format_status_text(False))
        self.metrics_var = tk.StringVar(value=format_metric_lines(None, False))
        self.message_var = tk.StringVar(value=self.detector_message)
        self.device_var = tk.StringVar()

        self._build_layout()
        self._load_devices()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(REFRESH_MS, self._tick)

    def _build_layout(self) -> None:
        self.root.title("Mosquito Sound Detector")
        self.root.minsize(900, 620)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Microphone").pack(side=tk.LEFT)
        self.device_combo = ttk.Combobox(
            controls,
            textvariable=self.device_var,
            width=42,
            state="readonly",
        )
        self.device_combo.pack(side=tk.LEFT, padx=(8, 12))

        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=4)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=4)
        self.calibrate_button = ttk.Button(
            controls,
            text="Calibrate background",
            command=self.start_calibration,
            state=tk.DISABLED,
        )
        self.calibrate_button.pack(side=tk.LEFT, padx=4)

        status_row = ttk.Frame(main)
        status_row.pack(fill=tk.X, pady=(18, 10))

        self.status_label = ttk.Label(
            status_row,
            textvariable=self.status_var,
            font=("Microsoft YaHei UI", 30, "bold"),
            anchor=tk.CENTER,
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        metrics = ttk.Label(
            status_row,
            textvariable=self.metrics_var,
            font=("Consolas", 11),
            justify=tk.LEFT,
            anchor=tk.W,
        )
        metrics.pack(side=tk.LEFT, padx=(20, 0))

        self.figure = Figure(figsize=(8, 4.4), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title("Live Spectrum")
        self.axis.set_xlabel("Frequency (Hz)")
        self.axis.set_ylabel("Level (dB)")
        self.axis.set_xlim(100, self.config.display_max_hz)
        self.axis.set_ylim(-120, 0)
        (self.spectrum_line,) = self.axis.plot([], [], color="#1f77b4", linewidth=1.4)
        self.peak_line = self.axis.axvline(0, color="#d62728", linewidth=1.2, visible=False)
        self.harmonic_lines = [
            self.axis.axvline(0, color="#ff7f0e", linewidth=1.0, alpha=0.75, visible=False)
            for _ in self.config.harmonic_numbers
        ]
        self.axis.grid(True, color="#d9d9d9", linewidth=0.8)

        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(6, 8))

        ttk.Label(main, textvariable=self.message_var, anchor=tk.W).pack(fill=tk.X)

    def _load_devices(self) -> None:
        try:
            self.devices = list_input_devices()
        except AudioInputError as exc:
            self.devices = []
            self.message_var.set(str(exc))
            return

        values = [
            f"{device.index}: {device.name} ({device.default_sample_rate} Hz)"
            for device in self.devices
        ]
        self.device_combo["values"] = values
        if values:
            self.device_combo.current(0)
        else:
            self.message_var.set("No microphone input devices found")

    def start(self) -> None:
        device_index = self._selected_device_index()
        if device_index is None and self.devices:
            self.message_var.set("Select a microphone first")
            return

        self.stop()
        self.stream = MicrophoneStream(self.config, device_index=device_index)
        try:
            self.stream.start()
        except AudioInputError as exc:
            self.stream = None
            self.message_var.set(str(exc))
            self._set_running_controls(False)
            return

        self.detector.clear_calibration()
        self.smoother.reset()
        self.status_var.set(format_status_text(False))
        self.metrics_var.set(format_metric_lines(None, False))
        self.message_var.set(f"Listening. {self.detector_message}")
        self._set_running_controls(True)

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
            except AudioInputError as exc:
                self.message_var.set(str(exc))
            self.stream = None
        self.calibration_deadline = None
        self._set_running_controls(False)

    def start_calibration(self) -> None:
        if self.stream is None:
            self.message_var.set("Start listening before calibrating the background")
            return
        self.calibration_deadline = time.monotonic() + CALIBRATION_SECONDS
        self.message_var.set("Calibrating. Keep mosquito sounds out of the room")
        self.calibrate_button.configure(state=tk.DISABLED)

    def close(self) -> None:
        self.stop()
        self.root.destroy()

    def _tick(self) -> None:
        if self.stream is not None:
            self._analyze_latest_audio()
        self.root.after(REFRESH_MS, self._tick)

    def _analyze_latest_audio(self) -> None:
        assert self.stream is not None
        if self.calibration_deadline is not None and time.monotonic() >= self.calibration_deadline:
            calibration_audio = self.stream.buffer.latest(CALIBRATION_SECONDS)
            self.detector.calibrate(calibration_audio)
            self.smoother.reset()
            self.calibration_deadline = None
            self.calibrate_button.configure(state=tk.NORMAL)
            self.message_var.set("Background calibration complete")

        audio = self.stream.latest_window()
        result = self.detector.analyze(audio)
        self.last_result = result
        smoothed_detected = self.smoother.update(result.confidence, result.is_mosquito)
        self.status_var.set(format_status_text(smoothed_detected))
        self.metrics_var.set(format_metric_lines(result, self.detector.has_calibration))
        if self.stream.last_status:
            self.message_var.set(f"Audio status: {self.stream.last_status}")
        self._update_plot(result)

    def _update_plot(self, result: DetectionResult) -> None:
        mask = (
            (result.frequency_hz >= self.config.analysis_min_hz)
            & (result.frequency_hz <= self.config.display_max_hz)
        )
        self.spectrum_line.set_data(result.frequency_hz[mask], result.spectrum_db[mask])
        if np.any(mask):
            visible_db = result.spectrum_db[mask]
            top = float(np.nanmax(visible_db))
            bottom = max(-130.0, min(-90.0, top - 85.0))
            self.axis.set_ylim(bottom, max(-20.0, top + 10.0))

        if result.candidate_f0_hz > 0:
            self.peak_line.set_xdata([result.candidate_f0_hz, result.candidate_f0_hz])
            self.peak_line.set_visible(True)
        else:
            self.peak_line.set_visible(False)

        for line, frequency in zip(self.harmonic_lines, result.harmonic_frequencies_hz):
            line.set_xdata([frequency, frequency])
            line.set_visible(frequency <= self.config.display_max_hz)
        for line in self.harmonic_lines[len(result.harmonic_frequencies_hz) :]:
            line.set_visible(False)
        self.canvas.draw_idle()

    def _set_running_controls(self, running: bool) -> None:
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.calibrate_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _selected_device_index(self) -> int | None:
        selection = self.device_var.get()
        if not selection:
            return None
        try:
            return int(selection.split(":", 1)[0])
        except ValueError:
            return None


def run_app() -> None:
    root = tk.Tk()
    MosquitoApp(root)
    root.mainloop()
