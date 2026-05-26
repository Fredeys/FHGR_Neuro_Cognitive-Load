"""
Artifact detection and segmentation pipeline for filtered IDUN/Guardian EEG.

Assumptions:
- CSV validation and sampling-rate estimation have already been established.
- The EEG signal is centered, bandpass-filtered and notch-filtered before
  artifact analysis.

Scope:
1. Create overlapping windows
2. Detect artifacts window-wise using robust Median/MAD thresholds
3. Create artifact overview table
4. Plot artifact diagnostics with Matplotlib
5. Compute Welch PSD for clean windows
6. Plot window PSDs and clean/artifact comparisons
7. Summarize segmentation and artifact quality

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
from scipy.signal import welch

from src.pipeline_config import CONFIG


WINDOW_SECONDS = float(CONFIG["windowing"]["window_seconds"])
OVERLAP = float(CONFIG["windowing"]["overlap"])
PLOT_DPI = int(CONFIG["plots"]["dpi"])
EPS = 1e-12


def _load_step_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Konnte Modul nicht laden: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_first_step = _load_step_module("02_first_step_pipeline.py", "step02_first_step_pipeline")
_filter_step = _load_step_module("03_filter_pipeline.py", "step03_filter_pipeline")

estimate_sampling_rate = _first_step.estimate_sampling_rate
load_eeg_csv = _first_step.load_eeg_csv
FILTERED_COL = _filter_step.FILTERED_COL
apply_bandpass_filter = _filter_step.apply_bandpass_filter
apply_notch_filter = _filter_step.apply_notch_filter
center_signal = _filter_step.center_signal


def create_windows(
    df: pd.DataFrame,
    fs: float,
    signal_col: str = FILTERED_COL,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = OVERLAP,
) -> list[dict]:
    """
    Segment filtered EEG into overlapping windows.

    Each returned window contains:
    - window_id
    - start_sample
    - end_sample
    - start_time
    - end_time
    - signal_segment
    """
    if signal_col not in df.columns:
        raise ValueError(f"Signalspalte '{signal_col}' fehlt.")
    if not 0 <= overlap < 1:
        raise ValueError("overlap muss im Bereich [0, 1) liegen.")

    window_size = int(round(window_seconds * fs))
    step_size = int(round(window_size * (1.0 - overlap)))

    if window_size < 2:
        raise ValueError("Fensterlänge ist zu kurz.")
    if step_size < 1:
        raise ValueError("Schrittweite ist zu klein. Overlap reduzieren.")
    if len(df) < window_size:
        raise ValueError("Aufnahme ist kürzer als ein vollständiges Fenster.")

    windows = []
    for window_id, start_sample in enumerate(range(0, len(df) - window_size + 1, step_size)):
        end_sample = start_sample + window_size
        segment = df[signal_col].iloc[start_sample:end_sample].to_numpy(dtype=float)

        windows.append(
            {
                "window_id": int(window_id),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_time": df["datetime"].iloc[start_sample],
                "end_time": df["datetime"].iloc[end_sample - 1],
                "signal_segment": segment,
            }
        )

    return windows


def _mad(values: np.ndarray) -> float:
    """Median Absolute Deviation."""
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - median)))


def _robust_threshold(values: np.ndarray, factor: float = float(CONFIG["artifact_detection"]["mad_factor"])) -> float:
    """
    Robust upper threshold: median + factor * 1.4826 * MAD.

    The factor 1.4826 scales MAD to a standard-deviation-like quantity under
    normally distributed data, while remaining robust to extreme windows.
    """
    values = np.asarray(values, dtype=float)
    return float(np.nanmedian(values) + factor * 1.4826 * _mad(values))


def _configured_threshold(config_key: str, values: np.ndarray, threshold_factor: float) -> float:
    configured = CONFIG["artifact_detection"].get(config_key)
    if configured is not None:
        return float(configured)
    return _robust_threshold(values, threshold_factor)


def compute_window_metrics(window: dict) -> dict:
    """
    Compute artifact metrics for one EEG window.

    Metrics:
    - peak-to-peak amplitude
    - standard deviation
    - variance
    - absolute maximum amplitude
    - mean signal energy
    - sample-to-sample gradient metrics
    """
    segment = np.asarray(window["signal_segment"], dtype=float)
    abs_gradients = np.abs(np.diff(segment))
    max_abs_gradient = float(np.max(abs_gradients)) if abs_gradients.size else 0.0
    mean_abs_gradient = float(np.mean(abs_gradients)) if abs_gradients.size else 0.0
    gradient_p95 = float(np.percentile(abs_gradients, 95)) if abs_gradients.size else 0.0

    return {
        "window_id": window["window_id"],
        "start_sample": window["start_sample"],
        "end_sample": window["end_sample"],
        "start_time": window["start_time"],
        "end_time": window["end_time"],
        "p2p": float(np.ptp(segment)),
        "std": float(np.std(segment)),
        "variance": float(np.var(segment)),
        "max_abs": float(np.max(np.abs(segment))),
        "energy": float(np.mean(segment**2)),
        "max_abs_gradient": max_abs_gradient,
        "mean_abs_gradient": mean_abs_gradient,
        "gradient_p95": gradient_p95,
    }


def detect_artifacts(
    windows: list[dict],
    threshold_factor: float = float(CONFIG["artifact_detection"]["mad_factor"]),
) -> tuple[pd.DataFrame, dict]:
    """
    Detect artifacts using robust thresholds over window metrics.

    Artifacts are only marked as artifact=True. They are not deleted.
    """
    if len(windows) == 0:
        raise ValueError("Keine Fenster für Artefakterkennung vorhanden.")

    artifact_df = pd.DataFrame([compute_window_metrics(window) for window in windows])

    thresholds = {
        "p2p_threshold": _configured_threshold("peak_to_peak_threshold", artifact_df["p2p"].to_numpy(), threshold_factor),
        "std_threshold": _configured_threshold("std_threshold", artifact_df["std"].to_numpy(), threshold_factor),
        "variance_threshold": _configured_threshold("variance_threshold", artifact_df["variance"].to_numpy(), threshold_factor),
        "max_abs_threshold": _configured_threshold("absolute_amplitude_threshold", artifact_df["max_abs"].to_numpy(), threshold_factor),
        "energy_threshold": _configured_threshold("energy_threshold", artifact_df["energy"].to_numpy(), threshold_factor),
        "gradient_threshold": _configured_threshold("gradient_threshold", artifact_df["max_abs_gradient"].to_numpy(), threshold_factor),
    }

    artifact_df["artifact"] = (
        (artifact_df["p2p"] > thresholds["p2p_threshold"])
        | (artifact_df["std"] > thresholds["std_threshold"])
        | (artifact_df["variance"] > thresholds["variance_threshold"])
        | (artifact_df["max_abs"] > thresholds["max_abs_threshold"])
        | (artifact_df["energy"] > thresholds["energy_threshold"])
        | (artifact_df["max_abs_gradient"] > thresholds["gradient_threshold"])
    )

    report = {
        **thresholds,
        "n_windows": int(len(artifact_df)),
        "n_clean_windows": int((~artifact_df["artifact"]).sum()),
        "n_artifact_windows": int(artifact_df["artifact"].sum()),
        "artifact_fraction": float(artifact_df["artifact"].mean()),
    }

    return artifact_df, report


def plot_artifacts(
    df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, plt.Figure]:
    """
    Create Matplotlib artifact visualizations:
    - artifact windows over time
    - histograms of artifact metrics
    - clean vs artifact comparison
    - signal plot with marked artifact windows
    """
    output_dir = Path(output_dir) if output_dir is not None else None
    figures = {}

    plot_df = artifact_df.copy()
    colors = np.where(plot_df["artifact"], "#E45756", "#4C78A8")

    fig_over_time, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(plot_df["start_time"], plot_df["p2p"], c=colors, s=18, alpha=0.85)
    _finish_axes(ax, "Artifact Windows Over Time", "Window start", "Peak-to-peak amplitude", legend=False)
    figures["artifact_windows_over_time"] = fig_over_time
    _save_figure(fig_over_time, _path(output_dir, "01_artifact_windows_over_time.png"))

    metric_cols = ["p2p", "std", "max_abs", "energy"]
    for metric in metric_cols:
        fig_hist, ax = plt.subplots(figsize=(10, 5))
        ax.hist(plot_df.loc[~plot_df["artifact"], metric].dropna(), bins=80, alpha=0.7, label="Clean", color="#4C78A8")
        ax.hist(plot_df.loc[plot_df["artifact"], metric].dropna(), bins=80, alpha=0.7, label="Artifact", color="#E45756")
        _finish_axes(ax, f"Artifact Metric Distribution: {metric}", metric, "Count")
        figures[f"hist_{metric}"] = fig_hist
        _save_figure(fig_hist, _path(output_dir, f"02_hist_{metric}.png"))

    fig_box, ax = plt.subplots(figsize=(12, 6))
    data = []
    labels = []
    for metric in metric_cols:
        data.extend([plot_df.loc[~plot_df["artifact"], metric].dropna(), plot_df.loc[plot_df["artifact"], metric].dropna()])
        labels.extend([f"Clean {metric}", f"Artifact {metric}"])
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    _finish_axes(ax, "Clean vs. Artifact Windows: Metric Comparison", "Metric", "Metric value", legend=False)
    figures["clean_vs_artifact_metrics"] = fig_box
    _save_figure(fig_box, _path(output_dir, "03_clean_vs_artifact_metrics.png"))

    fig_signal, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df[FILTERED_COL], linewidth=0.7, color="#4C78A8", label="Filtered EEG")

    artifact_windows = artifact_df[artifact_df["artifact"]]
    for _, row in artifact_windows.iterrows():
        ax.axvspan(row["start_time"], row["end_time"], color="#E45756", alpha=0.18)
    _finish_axes(ax, "Filtered Signal with Marked Artifact Windows", "Time", "Amplitude")
    figures["signal_with_artifacts"] = fig_signal
    _save_figure(fig_signal, _path(output_dir, "04_signal_with_artifacts.png"))

    return figures


def compute_window_psd(
    windows: list[dict],
    artifact_df: pd.DataFrame,
    fs: float,
    include_artifacts: bool = False,
    nperseg_max: int = int(CONFIG["psd"]["nperseg_max"]),
) -> pd.DataFrame:
    """
    Compute Welch PSD for windows.

    By default, PSD is computed only for clean windows. Artifact windows can be
    included for diagnostic comparisons by setting include_artifacts=True.
    """
    artifact_lookup = artifact_df.set_index("window_id")["artifact"].to_dict()
    rows = []

    for window in windows:
        window_id = window["window_id"]
        is_artifact = bool(artifact_lookup[window_id])

        if is_artifact and not include_artifacts:
            continue

        segment = np.asarray(window["signal_segment"], dtype=float)
        nperseg = min(nperseg_max, len(segment))
        freqs, psd = welch(
            segment,
            fs=fs,
            nperseg=nperseg,
            detrend="constant",
            scaling="density",
        )

        rows.append(
            {
                "window_id": int(window_id),
                "start_time": window["start_time"],
                "end_time": window["end_time"],
                "artifact": is_artifact,
                "freqs": freqs,
                "psd": psd,
            }
        )

    return pd.DataFrame(rows)


def plot_window_psd(
    psd_df: pd.DataFrame,
    artifact_psd_df: pd.DataFrame | None = None,
    max_clean_windows: int = 5,
    output_dir: str | Path | None = None,
) -> dict[str, plt.Figure]:
    """
    Create PSD visualizations:
    - PSD of a single clean window
    - comparison of multiple clean windows
    - comparison of one artifact window vs one clean window, if available
    """
    if psd_df.empty:
        raise ValueError("Keine sauberen Fenster für PSD-Visualisierung vorhanden.")

    output_dir = Path(output_dir) if output_dir is not None else None
    figures = {}

    first_clean = psd_df.iloc[0]
    fig_single, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(first_clean["freqs"], first_clean["psd"], linewidth=0.9, label=f"Clean window {int(first_clean['window_id'])}")
    ax.set_xlim(0, 45)
    _finish_axes(ax, "Welch PSD of One Clean Window", "Frequency [Hz]", "PSD")
    figures["single_clean_window_psd"] = fig_single
    _save_figure(fig_single, _path(output_dir, "05_single_clean_window_psd.png"))

    fig_multi, ax = plt.subplots(figsize=(12, 5))
    for _, row in psd_df.head(max_clean_windows).iterrows():
        ax.semilogy(row["freqs"], row["psd"], linewidth=0.8, alpha=0.75, label=f"Window {int(row['window_id'])}")
    ax.set_xlim(0, 45)
    _finish_axes(ax, f"Welch PSD Comparison of {min(max_clean_windows, len(psd_df))} Clean Windows", "Frequency [Hz]", "PSD")
    figures["multiple_clean_window_psd"] = fig_multi
    _save_figure(fig_multi, _path(output_dir, "06_multiple_clean_window_psd.png"))

    if artifact_psd_df is not None and not artifact_psd_df.empty:
        artifact_only = artifact_psd_df[artifact_psd_df["artifact"]]
        clean_only = artifact_psd_df[~artifact_psd_df["artifact"]]

        if not artifact_only.empty and not clean_only.empty:
            clean_row = clean_only.iloc[0]
            artifact_row = artifact_only.iloc[0]

            fig_compare, ax = plt.subplots(figsize=(12, 5))
            ax.semilogy(clean_row["freqs"], clean_row["psd"], linewidth=0.9, label=f"Clean window {int(clean_row['window_id'])}")
            ax.semilogy(artifact_row["freqs"], artifact_row["psd"], linewidth=0.9, label=f"Artifact window {int(artifact_row['window_id'])}")
            ax.set_xlim(0, 45)
            _finish_axes(ax, "Welch PSD: Artifact Window vs. Clean Window", "Frequency [Hz]", "PSD")
            figures["artifact_vs_clean_psd"] = fig_compare
            _save_figure(fig_compare, _path(output_dir, "07_artifact_vs_clean_psd.png"))

    return figures


def quality_check(artifact_df: pd.DataFrame, fs: float) -> dict:
    """Analyze whether artifact detection and windowing look plausible."""
    n_windows = len(artifact_df)
    n_artifacts = int(artifact_df["artifact"].sum())
    artifact_fraction = float(n_artifacts / max(n_windows, 1))

    problematic_runs = []
    current_start = None
    previous_id = None
    for _, row in artifact_df[artifact_df["artifact"]].iterrows():
        window_id = int(row["window_id"])
        if current_start is None:
            current_start = window_id
        elif previous_id is not None and window_id != previous_id + 1:
            problematic_runs.append((current_start, previous_id))
            current_start = window_id
        previous_id = window_id

    if current_start is not None:
        problematic_runs.append((current_start, previous_id))

    comments = []
    comments.append(f"{n_artifacts} von {n_windows} Fenstern enthalten Artefakte ({artifact_fraction:.1%}).")

    if artifact_fraction == 0:
        comments.append("Keine Artefaktfenster erkannt; Schwellenwerte und Rohsignal visuell prüfen.")
    elif artifact_fraction > 0.30:
        comments.append("Hoher Artefaktanteil; Aufnahmequalität oder Schwellenwerte kritisch prüfen.")
    else:
        comments.append("Artefaktanteil wirkt grundsätzlich plausibel, muss aber visuell validiert werden.")

    if problematic_runs:
        longest_run = max(problematic_runs, key=lambda run: run[1] - run[0])
        comments.append(
            f"Problematische zusammenhängende Abschnitte vorhanden; längster Artefaktlauf: "
            f"Fenster {longest_run[0]} bis {longest_run[1]}."
        )
    else:
        comments.append("Keine zusammenhängenden Artefaktabschnitte erkannt.")

    comments.append(f"Fensterung nutzt geschätzte Samplingrate {fs:.2f} Hz.")

    return {
        "n_windows": int(n_windows),
        "n_artifact_windows": n_artifacts,
        "n_clean_windows": int(n_windows - n_artifacts),
        "artifact_fraction": artifact_fraction,
        "artifact_runs": problematic_runs,
        "comments": comments,
    }


def prepare_filtered_signal_from_csv(csv_path: str | Path) -> tuple[pd.DataFrame, float, dict]:
    """
    Convenience loader for this standalone script.

    It reproduces the established preprocessing sequence:
    CSV load -> timestamp validation -> centering -> bandpass -> notch.
    """
    df, timestamp_col, validation_report = load_eeg_csv(csv_path)
    df, sampling_report = estimate_sampling_rate(df, timestamp_col)
    fs = sampling_report["estimated_fs_median_hz"]

    df, centering_report = center_signal(df)
    df = apply_bandpass_filter(df, fs)
    df = apply_notch_filter(df, fs)

    summary = {
        "validation": validation_report,
        "sampling": sampling_report,
        "centering": centering_report,
    }
    return df, fs, summary


def run_artifact_segmentation_pipeline(
    csv_path: str | Path,
    output_dir: str | Path = "outputs/artifact_segmentation",
    show_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, plt.Figure]]:
    """
    Run windowing, artifact detection and clean-window PSD preparation.

    Returns
    -------
    df:
        Signal dataframe including ch1_filtered.
    artifact_df:
        Window-wise artifact overview.
    clean_psd_df:
        Welch PSD table for clean windows only.
    summary:
        Reports and quality comments.
    figures:
        Matplotlib figures.
    """
    output_dir = Path(output_dir)
    df, fs, preparation_summary = prepare_filtered_signal_from_csv(csv_path)

    windows = create_windows(df, fs, signal_col=FILTERED_COL)
    artifact_df, artifact_report = detect_artifacts(windows)

    artifact_figures = plot_artifacts(df, artifact_df, output_dir=output_dir)

    clean_psd_df = compute_window_psd(windows, artifact_df, fs, include_artifacts=False)
    diagnostic_psd_df = compute_window_psd(windows, artifact_df, fs, include_artifacts=True)
    psd_figures = plot_window_psd(clean_psd_df, diagnostic_psd_df, output_dir=output_dir)

    qc_report = quality_check(artifact_df, fs)

    figures = {**artifact_figures, **psd_figures}
    if show_plots:
        for fig in figures.values():
            fig.show()

    summary = {
        **preparation_summary,
        "artifact_detection": artifact_report,
        "quality_check": qc_report,
        "method_notes": [
            "Artefakte wurden markiert und nicht gelöscht.",
            "Welch-PSD wurde für saubere Fenster berechnet.",
            "Artefaktfenster bleiben für Diagnostik und spätere Ausschlussregeln erhalten.",
            "Keine Klassifikation und keine Deep-Learning-Modelle wurden verwendet.",
        ],
    }

    artifact_df.to_csv(output_dir / "artifact_overview.csv", index=False)
    clean_psd_export = clean_psd_df.drop(columns=["freqs", "psd"]).copy()
    clean_psd_export.to_csv(output_dir / "clean_window_psd_index.csv", index=False)

    print("\n=== Artefakt- und Segmentierungs-Zusammenfassung ===")
    for comment in qc_report["comments"]:
        print(f"- {comment}")
    print(f"Saubere PSD-Fenster: {len(clean_psd_df)}")
    print(f"Artefaktübersicht: {output_dir / 'artifact_overview.csv'}")

    return df, artifact_df, clean_psd_df, summary, figures


def _path(output_dir: Path | None, filename: str) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _finish_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str, legend: bool = True) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    ax.grid(True, alpha=0.3)


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path is None:
        return
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")


if __name__ == "__main__":
    CSV_PATH = "data/raw/eeg_Work_PC_Morning.csv"
    run_artifact_segmentation_pipeline(CSV_PATH, show_plots=True)
