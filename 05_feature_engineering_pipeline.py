"""
EEG feature-engineering pipeline for IDUN/Guardian single-channel EEG.

Previous pipeline steps:
- CSV and timestamps validated
- Sampling rate estimated
- Signal centered
- Bandpass + notch filtering applied
- Artifact windows detected
- Welch PSD available per window

Scope of this file:
1. Define EEG frequency bands
2. Compute absolute bandpower from Welch PSD
3. Compute total power from 1-40 Hz
4. Compute relative bandpower
5. Compute EEG band ratios
6. Compute additional signal features
7. Prepare optional baseline normalization
8. Create final feature table
9. Create Matplotlib visualizations
10. Export features, artifacts and cleaned signal

No classification or deep-learning steps are implemented.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import welch

from src.pipeline_config import CONFIG, frequency_bands


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
_artifact_step = _load_step_module("04_artifact_segmentation_pipeline.py", "step04_artifact_segmentation_pipeline")

estimate_sampling_rate = _first_step.estimate_sampling_rate
load_eeg_csv = _first_step.load_eeg_csv
FILTERED_COL = _filter_step.FILTERED_COL
apply_bandpass_filter = _filter_step.apply_bandpass_filter
apply_notch_filter = _filter_step.apply_notch_filter
center_signal = _filter_step.center_signal
create_windows = _artifact_step.create_windows
detect_artifacts = _artifact_step.detect_artifacts

# Gamma is included for completeness, but should be interpreted cautiously in
# single-channel/in-ear EEG because 30-40 Hz can be strongly contaminated by EMG.
FREQUENCY_BANDS = frequency_bands()
PSD_CONFIG = CONFIG["psd"]
PLOT_DPI = int(CONFIG["plots"]["dpi"])


def compute_bandpower(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """
    Compute absolute bandpower by numerical integration of Welch PSD.

    trapezoid integration is numerically stable and appropriate for PSD samples
    on an approximately regular frequency grid returned by scipy.signal.welch.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)
    low, high = band
    mask = (freqs >= low) & (freqs < high)

    if np.sum(mask) < 2:
        return 0.0

    return float(trapezoid(psd[mask], freqs[mask]))


def compute_relative_bandpower(absolute_bandpowers: dict[str, float], total_power: float) -> dict[str, float]:
    """Compute relative bandpower for all available absolute bandpower values."""
    return {
        band.replace("absolute_", "relative_"): power / (total_power + EPS)
        for band, power in absolute_bandpowers.items()
    }


def compute_ratios(absolute_bandpowers: dict[str, float]) -> dict[str, float]:
    """Compute common EEG band ratios with epsilon protection."""
    theta = absolute_bandpowers["absolute_theta"]
    alpha = absolute_bandpowers["absolute_alpha"]
    beta = absolute_bandpowers["absolute_beta"]

    return {
        "theta_alpha_ratio": theta / (alpha + EPS),
        "alpha_theta_ratio": alpha / (theta + EPS),
        "beta_alpha_ratio": beta / (alpha + EPS),
    }


def compute_spectral_entropy(psd: np.ndarray) -> float:
    """
    Compute normalized spectral entropy from PSD.

    Higher entropy indicates a more broadly distributed spectrum; lower entropy
    indicates power concentrated in fewer frequencies.
    """
    psd = np.asarray(psd, dtype=float)
    psd_sum = float(np.sum(psd))

    if psd_sum <= EPS:
        return 0.0

    probability = psd / psd_sum
    probability = probability[probability > 0]
    entropy = -np.sum(probability * np.log2(probability))
    normalizer = np.log2(len(probability)) if len(probability) > 1 else 1.0
    return float(entropy / normalizer)


def compute_window_psd(
    signal_segment: np.ndarray,
    fs: float,
    nperseg_max: int = int(PSD_CONFIG["nperseg_max"]),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD for one EEG window."""
    signal_segment = np.asarray(signal_segment, dtype=float)
    nperseg = min(nperseg_max, len(signal_segment))
    freqs, psd = welch(
        signal_segment,
        fs=fs,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def create_feature_dataframe(
    windows: list[dict],
    artifact_df: pd.DataFrame,
    fs: float,
    include_artifacts: bool = False,
) -> pd.DataFrame:
    """
    Create final window-wise feature dataframe.

    By default, feature values are computed only for artifact=False windows.
    Artifact windows remain represented in the final table, but their feature
    values are NaN so they cannot silently enter downstream analyses.
    """
    artifact_lookup = artifact_df.set_index("window_id")["artifact"].to_dict()
    rows = []

    for window in windows:
        window_id = int(window["window_id"])
        is_artifact = bool(artifact_lookup[window_id])

        row = {
            "window_id": window_id,
            "start_time": window["start_time"],
            "end_time": window["end_time"],
            "start_sample": int(window["start_sample"]),
            "end_sample": int(window["end_sample"]),
            "artifact": is_artifact,
        }

        if is_artifact and not include_artifacts:
            rows.append(row)
            continue

        signal = np.asarray(window["signal_segment"], dtype=float)
        freqs, psd = compute_window_psd(signal, fs)

        absolute_bandpowers = {
            f"absolute_{band_name}": compute_bandpower(freqs, psd, band_range)
            for band_name, band_range in FREQUENCY_BANDS.items()
        }

        total_power = compute_bandpower(
            freqs,
            psd,
            (float(PSD_CONFIG["total_power_low_hz"]), float(PSD_CONFIG["total_power_high_hz"])),
        )
        relative_bandpowers = compute_relative_bandpower(absolute_bandpowers, total_power)
        ratios = compute_ratios(absolute_bandpowers)

        row.update(
            {
                **absolute_bandpowers,
                "total_power_1_40hz": total_power,
                **relative_bandpowers,
                **ratios,
                "signal_mean": float(np.mean(signal)),
                "signal_std": float(np.std(signal)),
                "signal_variance": float(np.var(signal)),
                "peak_to_peak": float(np.ptp(signal)),
                "max_abs": float(np.max(np.abs(signal))),
                "spectral_entropy": compute_spectral_entropy(psd),
            }
        )
        rows.append(row)

    features_df = pd.DataFrame(rows)
    return features_df


def baseline_normalize(
    features_df: pd.DataFrame,
    baseline_window_ids: list[int] | None = None,
    baseline_start: str | pd.Timestamp | None = None,
    baseline_end: str | pd.Timestamp | None = None,
    exclude_artifacts: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Create z-normalized feature columns from baseline windows.

    If no baseline is provided, the function returns the dataframe unchanged
    and reports that individual baseline normalization is recommended for
    cognitive-overload work.
    """
    features_df = features_df.copy()

    if baseline_window_ids is None and baseline_start is None and baseline_end is None:
        return features_df, {
            "applied": False,
            "message": (
                "Keine Baseline angegeben. Für Overload-Analysen wird eine "
                "individuelle Baseline pro Person/Session empfohlen."
            ),
        }

    if baseline_window_ids is not None:
        baseline_mask = features_df["window_id"].isin(baseline_window_ids)
    else:
        baseline_mask = pd.Series(True, index=features_df.index)
        if baseline_start is not None:
            baseline_mask &= features_df["start_time"] >= pd.Timestamp(baseline_start)
        if baseline_end is not None:
            baseline_mask &= features_df["end_time"] <= pd.Timestamp(baseline_end)

    if exclude_artifacts:
        baseline_mask &= ~features_df["artifact"]

    baseline_df = features_df[baseline_mask].copy()
    if baseline_df.empty:
        raise ValueError("Keine gültigen artifact=False Baseline-Fenster gefunden.")

    metadata_cols = {"window_id", "start_time", "end_time", "start_sample", "end_sample", "artifact"}
    feature_cols = [
        col
        for col in features_df.columns
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(features_df[col])
    ]

    baseline_stats = {}
    for col in feature_cols:
        mean = float(baseline_df[col].mean(skipna=True))
        std = float(baseline_df[col].std(skipna=True))

        if not np.isfinite(mean) or not np.isfinite(std):
            continue

        features_df[f"{col}_z"] = (features_df[col] - mean) / (std + EPS)
        baseline_stats[col] = {"mean": mean, "std": std}

    report = {
        "applied": True,
        "baseline_windows": int(len(baseline_df)),
        "normalized_features": int(len(baseline_stats)),
        "exclude_artifacts": bool(exclude_artifacts),
        "baseline_stats": baseline_stats,
    }
    return features_df, report


def plot_relative_bandpower(features_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    """Plot relative bandpower over time for clean windows."""
    clean_df = features_df[~features_df["artifact"]].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    for band in FREQUENCY_BANDS:
        col = f"relative_{band}"
        if col in clean_df.columns:
            ax.plot(clean_df["start_time"], clean_df[col], linewidth=0.9, marker=".", markersize=3, label=col)
    _finish_axes(ax, "Relative Bandpower Over Time", "Window start", "Relative bandpower")
    _save_figure(fig, output_path)
    return fig


def plot_theta_alpha_ratio(features_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    """Plot theta/alpha ratio over the session."""
    clean_df = features_df[~features_df["artifact"]].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(clean_df["start_time"], clean_df["theta_alpha_ratio"], linewidth=0.9, marker=".", markersize=3)
    _finish_axes(ax, "Theta/Alpha Ratio Over Time", "Window start", "Theta / Alpha", legend=False)
    _save_figure(fig, output_path)
    return fig


def plot_feature_correlation_heatmap(features_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    """Plot feature correlations for clean-window numeric features."""
    clean_df = features_df[~features_df["artifact"]].copy()
    numeric = clean_df.select_dtypes(include=[np.number]).drop(
        columns=["window_id", "start_sample", "end_sample"],
        errors="ignore",
    )
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Feature Correlation Heatmap")
    _save_figure(fig, output_path)
    return fig


def plot_feature_boxplots(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Create boxplots for selected clean-window features."""
    clean_df = features_df[~features_df["artifact"]].copy()
    if feature_cols is None:
        feature_cols = [
            "relative_delta",
            "relative_theta",
            "relative_alpha",
            "relative_beta",
            "theta_alpha_ratio",
            "beta_alpha_ratio",
            "spectral_entropy",
        ]

    available = [col for col in feature_cols if col in clean_df.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([clean_df[col].dropna() for col in available], labels=available, showfliers=True)
    ax.set_xticklabels(available, rotation=35, ha="right")
    _finish_axes(ax, "Boxplots of Selected Clean-Window Features", "Feature", "Value", legend=False)
    _save_figure(fig, output_path)
    return fig


def plot_feature_timecourse(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot important feature trajectories over the recording session."""
    clean_df = features_df[~features_df["artifact"]].copy()
    if feature_cols is None:
        feature_cols = [
            "relative_theta",
            "relative_alpha",
            "relative_beta",
            "theta_alpha_ratio",
            "spectral_entropy",
        ]

    fig, ax = plt.subplots(figsize=(14, 5))
    for col in feature_cols:
        if col in clean_df.columns:
            ax.plot(clean_df["start_time"], clean_df[col], linewidth=0.9, marker=".", markersize=3, label=col)
    _finish_axes(ax, "Important Feature Trajectories Over Session", "Window start", "Feature value")
    _save_figure(fig, output_path)
    return fig


def create_feature_plots(features_df: pd.DataFrame, output_dir: str | Path | None = None) -> dict[str, plt.Figure]:
    """Create all requested Matplotlib feature-engineering visualizations."""
    output_dir = Path(output_dir) if output_dir is not None else None
    figures = {
        "relative_bandpower": plot_relative_bandpower(features_df, _path(output_dir, "01_relative_bandpower.png")),
        "theta_alpha_ratio": plot_theta_alpha_ratio(features_df, _path(output_dir, "02_theta_alpha_ratio.png")),
        "feature_correlation_heatmap": plot_feature_correlation_heatmap(
            features_df,
            _path(output_dir, "03_feature_correlation_heatmap.png"),
        ),
        "feature_boxplots": plot_feature_boxplots(features_df, output_path=_path(output_dir, "04_feature_boxplots.png")),
        "feature_timecourse": plot_feature_timecourse(features_df, output_path=_path(output_dir, "05_feature_timecourse.png")),
    }
    return figures


def export_features(
    signal_df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_dir: str | Path = "outputs/feature_engineering",
    prefix: str = "eeg",
    summary: dict | None = None,
) -> dict[str, Path]:
    """Export final feature table, artifact overview and cleaned signal as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_path = output_dir / f"{prefix}_cleaned_signal.csv"
    artifact_path = output_dir / f"{prefix}_artifact_overview.csv"
    features_path = output_dir / f"{prefix}_features.csv"
    summary_path = output_dir / f"{prefix}_feature_summary.json"

    signal_df.to_csv(signal_path, index=False)
    artifact_df.to_csv(artifact_path, index=False)
    features_df.to_csv(features_path, index=False)

    exported = {
        "cleaned_signal_csv": signal_path,
        "artifact_overview_csv": artifact_path,
        "features_csv": features_path,
    }

    if summary is not None:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        exported["summary_json"] = summary_path

    return exported


def prepare_filtered_signal_and_artifacts(csv_path: str | Path) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, float, dict]:
    """
    Convenience preparation for standalone execution.

    Reuses the established scientific order:
    load CSV -> validate timestamps -> center -> bandpass -> notch -> windowing
    -> artifact marking.
    """
    df, timestamp_col, validation_report = load_eeg_csv(csv_path)
    df, sampling_report = estimate_sampling_rate(df, timestamp_col)
    fs = sampling_report["estimated_fs_median_hz"]

    df, centering_report = center_signal(df)
    df = apply_bandpass_filter(df, fs)
    df = apply_notch_filter(df, fs)

    windows = create_windows(df, fs, signal_col=FILTERED_COL)
    artifact_df, artifact_report = detect_artifacts(windows)

    preparation_summary = {
        "validation": validation_report,
        "sampling": sampling_report,
        "centering": centering_report,
        "artifact_detection": artifact_report,
    }
    return df, windows, artifact_df, fs, preparation_summary


def run_feature_engineering_pipeline(
    csv_path: str | Path,
    output_dir: str | Path = "outputs/feature_engineering",
    prefix: str | None = None,
    baseline_window_ids: list[int] | None = None,
    baseline_start: str | pd.Timestamp | None = None,
    baseline_end: str | pd.Timestamp | None = None,
    include_artifact_features: bool = False,
    show_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict[str, plt.Figure]]:
    """
    Run complete EEG feature-engineering pipeline.

    Returns
    -------
    signal_df:
        Cleaned and filtered signal dataframe.
    artifact_df:
        Artifact overview dataframe.
    features_df:
        Final window-wise feature table.
    summary:
        Processing and baseline reports.
    figures:
        Matplotlib feature figures.
    """
    csv_path = Path(csv_path)
    prefix = prefix or csv_path.stem
    output_dir = Path(output_dir)

    signal_df, windows, artifact_df, fs, preparation_summary = prepare_filtered_signal_and_artifacts(csv_path)

    features_df = create_feature_dataframe(
        windows=windows,
        artifact_df=artifact_df,
        fs=fs,
        include_artifacts=include_artifact_features,
    )

    features_df, baseline_report = baseline_normalize(
        features_df,
        baseline_window_ids=baseline_window_ids,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        exclude_artifacts=True,
    )

    figures = create_feature_plots(features_df, output_dir / "plots")
    if show_plots:
        for fig in figures.values():
            fig.show()

    summary = {
        **preparation_summary,
        "feature_engineering": {
            "frequency_bands": FREQUENCY_BANDS,
            "gamma_note": "Gamma 30-40 Hz ist enthalten, aber bei Single-Channel/In-Ear EEG vorsichtig zu interpretieren.",
            "include_artifact_features": bool(include_artifact_features),
            "n_feature_rows": int(len(features_df)),
            "n_clean_feature_rows": int((~features_df["artifact"]).sum()),
            "n_artifact_rows": int(features_df["artifact"].sum()),
        },
        "baseline_normalization": baseline_report,
        "method_notes": [
            "Absolute Bandpower wurde durch Integration der Welch-PSD berechnet.",
            "Relative Bandpower wurde gegen total_power_1_40hz normalisiert.",
            "Ratios nutzen ein kleines Epsilon gegen Division durch 0.",
            "Artifact=True Fenster werden standardmässig nicht für Features ausgewertet.",
            "Keine Klassifikation wurde implementiert.",
        ],
    }

    exported = export_features(
        signal_df=signal_df,
        artifact_df=artifact_df,
        features_df=features_df,
        output_dir=output_dir,
        prefix=prefix,
        summary=summary,
    )
    summary["exported"] = exported

    print("\n=== Feature-Engineering-Zusammenfassung ===")
    print(f"Geschätzte Samplingrate: {fs:.2f} Hz")
    print(f"Feature-Zeilen: {len(features_df)}")
    print(f"Saubere Fenster: {(~features_df['artifact']).sum()}")
    print(f"Artefaktfenster: {features_df['artifact'].sum()}")
    print("Gamma 30-40 Hz wurde berechnet, sollte aber bei In-Ear/Single-Channel EEG vorsichtig interpretiert werden.")
    if not baseline_report["applied"]:
        warnings.warn(baseline_report["message"])
        print(f"Hinweis: {baseline_report['message']}")
    for name, path in exported.items():
        print(f"{name}: {path}")

    return signal_df, artifact_df, features_df, summary, figures


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


def _save_figure(fig: plt.Figure, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


if __name__ == "__main__":
    CSV_PATH = "data/raw/eeg_Work_PC_Morning.csv"
    run_feature_engineering_pipeline(CSV_PATH, show_plots=True)
