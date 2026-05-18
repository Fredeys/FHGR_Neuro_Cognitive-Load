"""
EEG filter pipeline for IDUN/Guardian single-channel EEG.

This script performs the actual signal preparation after CSV/timestamp
validation:

1. Center signal by removing DC offset
2. Visualize raw vs. centered signal
3. Apply zero-phase Butterworth bandpass filter, 1-40 Hz
4. Apply 50 Hz notch filter after bandpass filtering
5. Store raw, centered, bandpassed and final filtered signals
6. Visualize filtered signal and before/after comparisons
7. Compute Welch PSD before and after filtering
8. Summarize filtering quality indicators

No classification or deep-learning steps are included.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

from src.pipeline_config import CONFIG


RAW_COL = "ch1"
CENTERED_COL = "ch1_centered"
BANDPASSED_COL = "ch1_bandpassed"
FILTERED_COL = "ch1_filtered"
PREPROCESSING_CONFIG = CONFIG["preprocessing"]
BANDPASS_LOW_HZ = float(PREPROCESSING_CONFIG["bandpass_low_hz"])
BANDPASS_HIGH_HZ = float(PREPROCESSING_CONFIG["bandpass_high_hz"])
BANDPASS_ORDER = int(PREPROCESSING_CONFIG["bandpass_order"])
NOTCH_FREQ_HZ = float(PREPROCESSING_CONFIG["notch_freq_hz"])
NOTCH_QUALITY_FACTOR = float(PREPROCESSING_CONFIG["notch_quality_factor"])
PSD_CONFIG = CONFIG["psd"]
PLOT_DPI = int(CONFIG["plots"]["dpi"])


def _load_step_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Konnte Modul nicht laden: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_first_step = _load_step_module("02_first_step_pipeline.py", "step02_first_step_pipeline")
load_eeg_csv = _first_step.load_eeg_csv
estimate_sampling_rate = _first_step.estimate_sampling_rate


def center_signal(
    df: pd.DataFrame,
    signal_col: str = RAW_COL,
    output_col: str = CENTERED_COL,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove DC offset from the EEG signal.

    EEG analyses focus mainly on fluctuations around a local baseline.
    A DC offset shifts the full signal vertically and can distort filtering,
    PSD estimates and amplitude-based quality metrics. Therefore, the mean is
    removed before frequency filtering.
    """
    df = df.copy()
    df["ch1_raw"] = df[signal_col].astype(float)

    dc_offset = float(df[signal_col].mean())
    df[output_col] = df[signal_col] - dc_offset

    report = {
        "dc_offset_removed": dc_offset,
        "raw_mean": float(df[signal_col].mean()),
        "centered_mean": float(df[output_col].mean()),
    }
    return df, report


def plot_raw_vs_centered(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot raw signal against centered signal."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df["ch1_raw"], linewidth=0.6, alpha=0.65, label="Raw signal")
    ax.plot(df["datetime"], df[CENTERED_COL], linewidth=0.7, alpha=0.85, label="Centered signal")
    _finish_axes(ax, "Raw Signal vs. Centered Signal", "Time", "Amplitude")
    _save_figure(fig, output_path)
    return fig


def plot_centered_zoom(
    df: pd.DataFrame,
    seconds: float = 20.0,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot raw vs. centered signal for the first 10-30 seconds."""
    if seconds < 10 or seconds > 30:
        warnings.warn("Für den Zoom werden 10-30 Sekunden empfohlen.")

    zoom_df = df[df["time_seconds"] <= seconds]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(zoom_df["datetime"], zoom_df["ch1_raw"], linewidth=0.8, label="Raw signal")
    ax.plot(zoom_df["datetime"], zoom_df[CENTERED_COL], linewidth=0.8, label="Centered signal")
    _finish_axes(ax, f"Raw vs. Centered Signal, First {seconds:g} Seconds", "Time", "Amplitude")
    _save_figure(fig, output_path)
    return fig


def plot_mean_comparison(
    report: dict,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot mean before and after centering."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Raw mean", "Centered mean"], [report["raw_mean"], report["centered_mean"]], color=["#4C78A8", "#F58518"])
    _finish_axes(ax, "Mean Comparison Before and After DC Offset Removal", "Signal", "Mean amplitude", legend=False)
    _save_figure(fig, output_path)
    return fig


def apply_bandpass_filter(
    df: pd.DataFrame,
    fs: float,
    input_col: str = CENTERED_COL,
    output_col: str = BANDPASSED_COL,
    lowcut: float = BANDPASS_LOW_HZ,
    highcut: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> pd.DataFrame:
    """
    Apply a zero-phase Butterworth bandpass filter.

    sosfiltfilt is used instead of lfilter to avoid phase shifts.
    The 1-40 Hz range keeps common EEG rhythms while reducing slow drift and
    high-frequency noise.
    """
    nyquist = fs / 2.0
    if lowcut <= 0:
        raise ValueError("lowcut muss > 0 Hz sein.")
    if highcut >= nyquist:
        raise ValueError(f"highcut={highcut} Hz muss unter Nyquist={nyquist:.2f} Hz liegen.")

    df = df.copy()
    sos = butter(N=order, Wn=[lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    df[output_col] = sosfiltfilt(sos, df[input_col].to_numpy(dtype=float))
    return df


def apply_notch_filter(
    df: pd.DataFrame,
    fs: float,
    input_col: str = BANDPASSED_COL,
    output_col: str = FILTERED_COL,
    notch_freq: float = NOTCH_FREQ_HZ,
    quality_factor: float = NOTCH_QUALITY_FACTOR,
) -> pd.DataFrame:
    """
    Apply a 50 Hz notch filter after bandpass filtering.

    The notch targets mains interference. filtfilt is used to keep the output
    zero-phase, matching the no-phase-shift requirement of EEG preprocessing.
    """
    nyquist = fs / 2.0
    df = df.copy()

    if notch_freq >= nyquist:
        warnings.warn(
            f"Notch-Frequenz {notch_freq} Hz liegt oberhalb Nyquist={nyquist:.2f} Hz. "
            "Notch-Filter wird übersprungen."
        )
        df[output_col] = df[input_col]
        return df

    b, a = iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
    df[output_col] = filtfilt(b, a, df[input_col].to_numpy(dtype=float))
    return df


def plot_filtered_signal(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot final filtered EEG signal."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df[FILTERED_COL], linewidth=0.7, label="Filtered signal")
    _finish_axes(ax, "Filtered EEG Signal, 1-40 Hz Bandpass + 50 Hz Notch", "Time", "Amplitude")
    _save_figure(fig, output_path)
    return fig


def plot_raw_vs_filtered(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot raw signal against final filtered signal."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df["ch1_raw"], linewidth=0.55, alpha=0.45, label="Raw signal")
    ax.plot(df["datetime"], df[FILTERED_COL], linewidth=0.75, alpha=0.9, label="Filtered signal")
    _finish_axes(ax, "Raw Signal vs. Filtered Signal", "Time", "Amplitude")
    _save_figure(fig, output_path)
    return fig


def plot_filtered_zoom(
    df: pd.DataFrame,
    seconds: float = 20.0,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot filtered signal zoomed to first 10-30 seconds."""
    if seconds < 10 or seconds > 30:
        warnings.warn("Für den Zoom werden 10-30 Sekunden empfohlen.")

    zoom_df = df[df["time_seconds"] <= seconds]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(zoom_df["datetime"], zoom_df[FILTERED_COL], linewidth=0.8, label="Filtered signal")
    _finish_axes(ax, f"Filtered Signal, First {seconds:g} Seconds", "Time", "Amplitude")
    _save_figure(fig, output_path)
    return fig


def plot_filter_stages(
    df: pd.DataFrame,
    seconds: float = 20.0,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Compare centered, bandpassed and final filtered signal in a zoomed view."""
    zoom_df = df[df["time_seconds"] <= seconds]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(zoom_df["datetime"], zoom_df[CENTERED_COL], linewidth=0.7, alpha=0.55, label="Centered")
    ax.plot(zoom_df["datetime"], zoom_df[BANDPASSED_COL], linewidth=0.7, alpha=0.75, label="Bandpassed")
    ax.plot(zoom_df["datetime"], zoom_df[FILTERED_COL], linewidth=0.8, alpha=0.95, label="Bandpass + Notch")
    _finish_axes(ax, f"Comparison Before and After Filtering, First {seconds:g} Seconds", "Time", "Amplitude")
    _save_figure(fig, output_path)
    return fig


def compute_psd(
    signal: np.ndarray | pd.Series,
    fs: float,
    nperseg_max: int = int(PSD_CONFIG["full_signal_nperseg_max"]),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Welch PSD is preferred over a raw FFT for inspection because it produces a
    more stable spectral estimate by averaging over segments.
    """
    signal = np.asarray(signal, dtype=float)
    nperseg = min(nperseg_max, len(signal))
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def compare_psd(
    df: pd.DataFrame,
    fs: float,
    before_col: str = CENTERED_COL,
    after_col: str = FILTERED_COL,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, dict]:
    """Compute and plot PSD before and after filtering."""
    freqs_before, psd_before = compute_psd(df[before_col], fs)
    freqs_after, psd_after = compute_psd(df[after_col], fs)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(freqs_before, psd_before, linewidth=0.8, alpha=0.7, label="Before filtering")
    ax.semilogy(freqs_after, psd_after, linewidth=0.9, alpha=0.9, label="After filtering")
    ax.set_xlim(0, min(80, fs / 2.0))
    _finish_axes(ax, "Welch PSD Before vs. After Filtering", "Frequency [Hz]", "PSD")
    _save_figure(fig, output_path)

    report = evaluate_filter_quality(freqs_before, psd_before, freqs_after, psd_after)
    return fig, report


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def evaluate_filter_quality(
    freqs_before: np.ndarray,
    psd_before: np.ndarray,
    freqs_after: np.ndarray,
    psd_after: np.ndarray,
) -> dict:
    """
    Summarize basic quality indicators after filtering.

    These checks are descriptive. They should be interpreted together with the
    time-domain plots and knowledge about the recording context.
    """
    drift_before = _band_power(freqs_before, psd_before, 0.0, 1.0)
    drift_after = _band_power(freqs_after, psd_after, 0.0, 1.0)

    total_low = float(PSD_CONFIG["total_power_low_hz"])
    total_high = float(PSD_CONFIG["total_power_high_hz"])
    eeg_before = _band_power(freqs_before, psd_before, total_low, total_high)
    eeg_after = _band_power(freqs_after, psd_after, total_low, total_high)

    notch_freq = NOTCH_FREQ_HZ
    notch_before = _band_power(freqs_before, psd_before, notch_freq - 1.0, notch_freq + 1.0)
    notch_after = _band_power(freqs_after, psd_after, notch_freq - 1.0, notch_freq + 1.0)

    high_freq_before = _band_power(freqs_before, psd_before, total_high, min(2.0 * total_high, freqs_before[-1]))
    high_freq_after = _band_power(freqs_after, psd_after, total_high, min(2.0 * total_high, freqs_after[-1]))

    report = {
        "drift_power_before_0_1hz": drift_before,
        "drift_power_after_0_1hz": drift_after,
        "drift_reduction_ratio": drift_after / (drift_before + 1e-12),
        "line_noise_power_before_49_51hz": notch_before,
        "line_noise_power_after_49_51hz": notch_after,
        "line_noise_reduction_ratio": notch_after / (notch_before + 1e-12),
        "eeg_band_power_before_1_40hz": eeg_before,
        "eeg_band_power_after_1_40hz": eeg_after,
        "eeg_band_retention_ratio": eeg_after / (eeg_before + 1e-12),
        "high_frequency_power_before_40_80hz": high_freq_before,
        "high_frequency_power_after_40_80hz": high_freq_after,
        "high_frequency_reduction_ratio": high_freq_after / (high_freq_before + 1e-12),
    }

    comments = []
    if report["line_noise_reduction_ratio"] < 0.5:
        comments.append("50-Hz-Netzbrummen wurde deutlich reduziert.")
    else:
        comments.append("50-Hz-Netzbrummen wurde nicht stark reduziert oder war vorher gering.")

    if report["drift_reduction_ratio"] < 0.5:
        comments.append("Langsame Drift unter 1 Hz wurde deutlich reduziert.")
    else:
        comments.append("Driftreduktion ist gering; Rohsignal und PSD visuell prüfen.")

    if 0.2 <= report["eeg_band_retention_ratio"] <= 1.5:
        comments.append("Der physiologisch relevante 1-40-Hz-Bereich wirkt grob plausibel erhalten.")
    else:
        comments.append("Auffällige Veränderung im 1-40-Hz-Bereich; Filterparameter und Datenqualität prüfen.")

    if report["high_frequency_reduction_ratio"] < 0.5:
        comments.append("Hochfrequente Aktivität oberhalb 40 Hz wurde reduziert.")

    report["comments"] = comments
    return report


def _finish_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, legend: bool = True) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    ax.grid(True, alpha=0.3)


def _save_figure(fig: plt.Figure, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")


def run_filter_pipeline(
    csv_path: str | Path,
    output_dir: str | Path = "outputs/filter_pipeline",
    detail_seconds: float = 20.0,
    show_plots: bool = True,
) -> tuple[pd.DataFrame, dict, dict[str, plt.Figure]]:
    """
    Run the full filter preparation pipeline directly from a CSV file.

    The CSV is first loaded and timestamp-validated using the first-step
    pipeline functions, then the actual signal preparation is performed.
    """
    output_dir = Path(output_dir)

    df, timestamp_col, validation_report = load_eeg_csv(csv_path)
    df, sampling_report = estimate_sampling_rate(df, timestamp_col)
    fs = sampling_report["estimated_fs_median_hz"]

    figures = {}

    df, centering_report = center_signal(df)
    figures["raw_vs_centered"] = plot_raw_vs_centered(df, output_dir / "01_raw_vs_centered.png")
    figures["centered_zoom"] = plot_centered_zoom(df, detail_seconds, output_dir / "02_centered_zoom.png")
    figures["mean_comparison"] = plot_mean_comparison(centering_report, output_dir / "03_mean_comparison.png")

    df = apply_bandpass_filter(df, fs)
    df = apply_notch_filter(df, fs)

    figures["raw_vs_filtered"] = plot_raw_vs_filtered(df, output_dir / "04_raw_vs_filtered.png")
    figures["filtered_signal"] = plot_filtered_signal(df, output_dir / "05_filtered_signal.png")
    figures["filtered_zoom"] = plot_filtered_zoom(df, detail_seconds, output_dir / "06_filtered_zoom.png")
    figures["filter_stages"] = plot_filter_stages(df, detail_seconds, output_dir / "07_filter_stages.png")

    psd_fig, quality_report = compare_psd(df, fs, output_path=output_dir / "08_psd_before_after.png")
    figures["psd_before_after"] = psd_fig

    if show_plots:
        for fig in figures.values():
            fig.show()

    summary = {
        "validation": validation_report,
        "sampling": sampling_report,
        "centering": centering_report,
        "filtering": {
            "bandpass": "Butterworth 1-40 Hz, order 4, sosfiltfilt zero-phase",
            "notch": "50 Hz iirnotch, Q=30, applied after bandpass with filtfilt",
        },
        "quality": quality_report,
    }

    print("\n=== Filterpipeline Zusammenfassung ===")
    print(f"Geschätzte Samplingrate: {fs:.2f} Hz")
    print(f"Entfernter DC-Offset: {centering_report['dc_offset_removed']:.6f}")
    for comment in quality_report["comments"]:
        print(f"- {comment}")

    return df, summary, figures


if __name__ == "__main__":
    # Example:
    # python 03_filter_pipeline.py
    CSV_PATH = "data/raw/eeg_Work_PC_Morning.csv"
    run_filter_pipeline(CSV_PATH, show_plots=True)
