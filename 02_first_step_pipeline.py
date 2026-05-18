"""
First-step EEG preprocessing pipeline for IDUN/Guardian CSV files.

Scope:
1. Load CSV
2. Validate data types and required columns
3. Analyze timestamps and estimate sampling rate
4. Inspect raw EEG signal
5. Create first Matplotlib visualizations
6. Summarize validation and signal-quality findings

Expected CSV columns:
- 'timestamps' or 'timestamp'
- 'ch1'

No filtering, feature extraction, or deep-learning steps are included here.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline_config import CONFIG


EEG_CHANNEL = "ch1"
TIMESTAMP_CANDIDATES = ("timestamps", "timestamp")
EXPECTED_FS = float(CONFIG["sampling"]["expected_fs_hz"])
EPS = 1e-12


def validate_columns(df: pd.DataFrame) -> str:
    """
    Validate required EEG columns and return the detected timestamp column.

    Returns
    -------
    timestamp_col:
        Either 'timestamps' or 'timestamp'.
    """
    timestamp_col = next((col for col in TIMESTAMP_CANDIDATES if col in df.columns), None)

    if timestamp_col is None:
        raise ValueError(
            "Keine gültige Zeitspalte gefunden. Erwartet wird entweder "
            "'timestamps' oder 'timestamp'."
        )

    if EEG_CHANNEL not in df.columns:
        raise ValueError(
            f"EEG-Spalte '{EEG_CHANNEL}' fehlt. Vorhandene Spalten: {list(df.columns)}"
        )

    return timestamp_col


def load_eeg_csv(csv_path: str | Path) -> tuple[pd.DataFrame, str, dict]:
    """
    Load and validate an IDUN/Guardian EEG CSV.

    Processing order:
    - Load CSV with pandas
    - Detect timestamp column
    - Validate 'ch1'
    - Convert timestamp and ch1 to numeric
    - Remove invalid/missing values
    - Remove duplicate timestamps
    - Sort by timestamp
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)
    timestamp_col = validate_columns(df)

    n_initial = len(df)
    df = df.copy()

    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
    df[EEG_CHANNEL] = pd.to_numeric(df[EEG_CHANNEL], errors="coerce")

    n_invalid_or_missing = int(df[[timestamp_col, EEG_CHANNEL]].isna().any(axis=1).sum())
    df = df.dropna(subset=[timestamp_col, EEG_CHANNEL])

    n_before_duplicates = len(df)
    df = df.drop_duplicates(subset=[timestamp_col])
    n_duplicates = n_before_duplicates - len(df)

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            "Nach Bereinigung sind weniger als zwei valide Samples vorhanden. "
            "Samplingrate und Zeitdifferenzen können nicht berechnet werden."
        )

    validation_report = {
        "csv_path": str(csv_path),
        "timestamp_column": timestamp_col,
        "rows_initial": int(n_initial),
        "invalid_or_missing_rows_removed": n_invalid_or_missing,
        "duplicate_timestamps_removed": int(n_duplicates),
        "rows_final": int(len(df)),
    }

    return df, timestamp_col, validation_report


def estimate_sampling_rate(
    df: pd.DataFrame,
    timestamp_col: str,
    expected_fs: float = EXPECTED_FS,
    tolerance: float = float(CONFIG["sampling"]["fs_tolerance_fraction"]),
) -> tuple[pd.DataFrame, dict]:
    """
    Analyze timestamps and estimate sampling rate.

    The median sample interval is used as the primary estimator because it is
    more robust to occasional gaps than the mean interval.
    """
    df = df.copy()
    timestamps = df[timestamp_col].to_numpy(dtype=float)
    median_timestamp = float(np.nanmedian(timestamps))

    if median_timestamp > 1e11:
        timestamp_unit = "ms"
        df["timestamp_seconds"] = df[timestamp_col] / 1000.0
    else:
        timestamp_unit = "s"
        df["timestamp_seconds"] = df[timestamp_col]

    df["datetime"] = pd.to_datetime(df["timestamp_seconds"], unit="s", errors="coerce")
    n_invalid_datetime = int(df["datetime"].isna().sum())
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Nach datetime-Konvertierung sind zu wenige Samples vorhanden.")

    df["time_seconds"] = df["timestamp_seconds"] - df["timestamp_seconds"].iloc[0]
    df["sample_interval"] = df["timestamp_seconds"].diff()

    intervals = df["sample_interval"].dropna().to_numpy(dtype=float)
    intervals = intervals[intervals > 0]

    if len(intervals) == 0:
        raise ValueError("Keine positiven Zeitabstände zwischen Samples gefunden.")

    mean_dt = float(np.mean(intervals))
    median_dt = float(np.median(intervals))
    std_dt = float(np.std(intervals))
    fs_mean = float(1.0 / mean_dt)
    fs_median = float(1.0 / median_dt)

    expected_dt = 1.0 / expected_fs
    large_gap_threshold = max(
        float(CONFIG["sampling"]["large_gap_min_seconds"]),
        float(CONFIG["sampling"]["large_gap_expected_dt_factor"]) * expected_dt,
    )
    large_gaps = intervals[intervals > large_gap_threshold]

    irregular_threshold = 0.20 * expected_dt
    irregular_intervals = intervals[np.abs(intervals - median_dt) > irregular_threshold]

    fs_lower = expected_fs * (1.0 - tolerance)
    fs_upper = expected_fs * (1.0 + tolerance)
    fs_plausible = fs_lower <= fs_median <= fs_upper

    warning_messages = []
    if not fs_plausible:
        warning_messages.append(
            f"Geschätzte Samplingrate {fs_median:.2f} Hz weicht deutlich von "
            f"{expected_fs:.2f} Hz ab."
        )
    if len(large_gaps) > 0:
        warning_messages.append(
            f"{len(large_gaps)} grössere Zeitlücken erkannt; "
            f"maximale Lücke: {np.max(large_gaps):.3f} s."
        )
    if len(irregular_intervals) > 0:
        warning_messages.append(
            f"{len(irregular_intervals)} unregelmässige Samplingintervalle erkannt."
        )
    if n_invalid_datetime > 0:
        warning_messages.append(
            f"{n_invalid_datetime} Zeitstempel konnten nicht in datetime konvertiert werden."
        )

    sampling_report = {
        "timestamp_unit": timestamp_unit,
        "start_time": df["datetime"].iloc[0],
        "end_time": df["datetime"].iloc[-1],
        "duration_seconds": float(df["time_seconds"].iloc[-1]),
        "mean_sample_interval_seconds": mean_dt,
        "median_sample_interval_seconds": median_dt,
        "std_sample_interval_seconds": std_dt,
        "estimated_fs_mean_hz": fs_mean,
        "estimated_fs_median_hz": fs_median,
        "fs_plausible_around_250hz": bool(fs_plausible),
        "n_large_gaps": int(len(large_gaps)),
        "max_gap_seconds": float(np.max(large_gaps)) if len(large_gaps) else 0.0,
        "n_irregular_intervals": int(len(irregular_intervals)),
        "warnings": warning_messages,
    }

    return df, sampling_report


def inspect_signal(df: pd.DataFrame, fs: float, signal_col: str = EEG_CHANNEL) -> dict:
    """
    Compute raw-signal statistics and basic signal-quality indicators.

    This function marks suspicious properties but does not modify the signal.
    """
    signal = df[signal_col].to_numpy(dtype=float)
    n_samples = len(signal)
    duration = float(df["time_seconds"].iloc[-1] - df["time_seconds"].iloc[0])

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    min_value = float(np.min(signal))
    max_value = float(np.max(signal))
    peak_to_peak = float(np.ptp(signal))
    max_abs = float(np.max(np.abs(signal)))

    diffs = np.diff(signal)
    flat_threshold = max(EPS, 1e-9 * max(1.0, max_abs))
    min_flat_len = int(round(0.5 * fs))

    flat_regions = 0
    current_run = 0
    for is_flat in np.abs(diffs) <= flat_threshold:
        if is_flat:
            current_run += 1
        else:
            if current_run >= min_flat_len:
                flat_regions += 1
            current_run = 0
    if current_run >= min_flat_len:
        flat_regions += 1

    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median)))
    robust_sigma = 1.4826 * mad + EPS
    extreme_peak_mask = np.abs(signal - median) > 8.0 * robust_sigma

    min_count = int(np.sum(signal == min_value))
    max_count = int(np.sum(signal == max_value))
    clipping_fraction = float((min_count + max_count) / n_samples)

    abs_signal = np.abs(signal)
    abs_median = float(np.median(abs_signal))
    abs_mad = float(np.median(np.abs(abs_signal - abs_median)))
    high_amplitude_threshold = abs_median + 8.0 * 1.4826 * abs_mad
    high_amplitude_samples = int(np.sum(abs_signal > high_amplitude_threshold))

    suspicious_sections = []
    if flat_regions > 0:
        suspicious_sections.append("konstante oder nahezu konstante Bereiche")
    if int(np.sum(extreme_peak_mask)) > 0:
        suspicious_sections.append("extreme Peaks")
    if clipping_fraction > 0.001:
        suspicious_sections.append("mögliches Clipping")
    if high_amplitude_samples > 0:
        suspicious_sections.append("ungewöhnlich hohe Amplituden")

    return {
        "mean": mean,
        "std": std,
        "min": min_value,
        "max": max_value,
        "peak_to_peak": peak_to_peak,
        "n_samples": int(n_samples),
        "duration_seconds": duration,
        "max_abs": max_abs,
        "flat_regions": int(flat_regions),
        "extreme_peak_samples": int(np.sum(extreme_peak_mask)),
        "clipping_fraction": clipping_fraction,
        "high_amplitude_samples": high_amplitude_samples,
        "suspicious_sections": suspicious_sections,
    }


def plot_raw_signal(df: pd.DataFrame, signal_col: str = EEG_CHANNEL) -> plt.Figure:
    """Create a Matplotlib line plot of the complete raw EEG signal."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df[signal_col], linewidth=0.7)
    _finish_axes(ax, "Komplettes Rohsignal", "Zeit", "Amplitude")
    return fig


def plot_raw_detail(
    df: pd.DataFrame,
    seconds: float = 20.0,
    signal_col: str = EEG_CHANNEL,
) -> plt.Figure:
    """Create a detail plot of the first 10-30 seconds."""
    if seconds < 10 or seconds > 30:
        warnings.warn("Für den Detailplot werden wissenschaftlich sinnvoll 10-30 Sekunden empfohlen.")

    detail_df = df[df["time_seconds"] <= seconds]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(detail_df["datetime"], detail_df[signal_col], linewidth=0.9)
    _finish_axes(ax, f"Rohsignal Detailplot: erste {seconds:g} Sekunden", "Zeit", "Amplitude")
    return fig


def plot_signal_histogram(df: pd.DataFrame, signal_col: str = EEG_CHANNEL) -> plt.Figure:
    """Create a histogram of the raw signal distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[signal_col].dropna(), bins=100, color="#4C78A8", alpha=0.85)
    _finish_axes(ax, "Verteilung der EEG-Amplituden", "Amplitude", "Anzahl")
    return fig


def plot_sample_interval_distribution(df: pd.DataFrame) -> plt.Figure:
    """Create a histogram of inter-sample intervals."""
    intervals = df["sample_interval"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(intervals, bins=100, color="#72B7B2", alpha=0.85)
    _finish_axes(ax, "Verteilung der Zeitabstände zwischen Samples", "Sample-Abstand [s]", "Anzahl")
    return fig


def _finish_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def summarize_results(
    validation_report: dict,
    sampling_report: dict,
    signal_report: dict,
) -> dict:
    """Combine validation, sampling and signal-quality results."""
    potential_problems = []
    potential_problems.extend(sampling_report.get("warnings", []))

    if signal_report["flat_regions"] > 0:
        potential_problems.append("Konstante Bereiche im Rohsignal gefunden.")
    if signal_report["extreme_peak_samples"] > 0:
        potential_problems.append("Extreme Peaks im Rohsignal gefunden.")
    if signal_report["clipping_fraction"] > 0.001:
        potential_problems.append("Mögliches Clipping erkannt.")
    if signal_report["high_amplitude_samples"] > 0:
        potential_problems.append("Ungewöhnlich hohe Amplituden erkannt.")

    return {
        "validation": validation_report,
        "sampling": sampling_report,
        "raw_signal": signal_report,
        "estimated_sampling_rate_hz": sampling_report["estimated_fs_median_hz"],
        "potential_problems": potential_problems,
    }


def run_first_step_pipeline(
    csv_path: str | Path,
    detail_seconds: float = 20.0,
    show_plots: bool = True,
) -> tuple[pd.DataFrame, dict, dict[str, plt.Figure]]:
    """
    Run the first part of the EEG processing pipeline.

    Returns
    -------
    df:
        Cleaned dataframe with datetime, relative time and sample intervals.
    summary:
        Summary of validation, sampling and signal-quality checks.
    figures:
        Matplotlib figures for raw signal, detail plot, histogram and intervals.
    """
    df, timestamp_col, validation_report = load_eeg_csv(csv_path)
    df, sampling_report = estimate_sampling_rate(df, timestamp_col)

    fs = sampling_report["estimated_fs_median_hz"]
    signal_report = inspect_signal(df, fs)

    figures = {
        "raw_signal": plot_raw_signal(df),
        "raw_detail": plot_raw_detail(df, seconds=detail_seconds),
        "signal_histogram": plot_signal_histogram(df),
        "sample_intervals": plot_sample_interval_distribution(df),
    }

    if show_plots:
        for fig in figures.values():
            fig.show()

    summary = summarize_results(validation_report, sampling_report, signal_report)

    print("\n=== Zusammenfassung ===")
    print(f"CSV: {validation_report['csv_path']}")
    print(f"Zeitspalte: {validation_report['timestamp_column']}")
    print(f"Valide Samples: {validation_report['rows_final']}")
    print(f"Geschätzte Samplingrate: {summary['estimated_sampling_rate_hz']:.2f} Hz")
    print(f"Dauer: {signal_report['duration_seconds']:.2f} s")
    print(f"Mittelwert: {signal_report['mean']:.6f}")
    print(f"Standardabweichung: {signal_report['std']:.6f}")
    print(f"Min/Max: {signal_report['min']:.6f} / {signal_report['max']:.6f}")
    print(f"Peak-to-Peak: {signal_report['peak_to_peak']:.6f}")

    if summary["potential_problems"]:
        print("\nPotenzielle Probleme:")
        for problem in summary["potential_problems"]:
            print(f"- {problem}")
    else:
        print("\nKeine auffälligen Probleme in den Basisprüfungen erkannt.")

    return df, summary, figures


if __name__ == "__main__":
    # Beispiel:
    # df_clean, summary, figures = run_first_step_pipeline("../eeg_Work_PC_Morning.csv")
    CSV_PATH = "data/raw/eeg_Work_PC_Morning.csv"
    run_first_step_pipeline(CSV_PATH, detail_seconds=20.0, show_plots=True)
