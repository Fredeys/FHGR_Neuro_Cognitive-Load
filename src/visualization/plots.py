from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.extraction import FREQUENCY_BANDS
from src.features.psd import compute_psd
from src.pipeline_config import CONFIG


PLOT_DPI = int(CONFIG["plots"]["dpi"])


def _save(fig: plt.Figure, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")


def _close_or_return(fig: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    _save(fig, output_path)
    return fig


def plot_raw_signal(df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df["ch1"], linewidth=0.7)
    ax.set_title("Raw EEG Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_detail_signal(df: pd.DataFrame, seconds: float = 20.0, output_path: str | Path | None = None) -> plt.Figure:
    detail = df[df["time_seconds"] <= seconds]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(detail["datetime"], detail["ch1"], linewidth=0.9)
    ax.set_title(f"Raw EEG Detail: First {seconds:g} Seconds")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_centered_comparison(df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df["ch1"], linewidth=0.6, alpha=0.55, label="Raw")
    ax.plot(df["datetime"], df["ch1_centered"], linewidth=0.7, alpha=0.85, label="Centered")
    ax.set_title("Raw vs. Centered EEG")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_filtered_signal(df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df["ch1_filtered"], linewidth=0.7, color="#F58518")
    ax.set_title("Filtered EEG Signal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_before_after_filter(df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["datetime"], df["ch1_centered"], linewidth=0.55, alpha=0.45, label="Centered")
    ax.plot(df["datetime"], df["ch1_filtered"], linewidth=0.75, alpha=0.9, label="Filtered")
    ax.set_title("Before vs. After Filtering")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_signal_histogram(df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["ch1"].dropna(), bins=100, color="#4C78A8", alpha=0.85)
    ax.set_title("Raw Signal Distribution")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_sample_intervals(df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["sample_interval"].dropna(), bins=100, color="#72B7B2", alpha=0.85)
    ax.set_title("Distribution of Sample Intervals")
    ax.set_xlabel("Sample interval [s]")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_artifact_overview(artifact_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = np.where(artifact_df["artifact"], "#E45756", "#4C78A8")
    ax.scatter(artifact_df["start_time"], artifact_df["p2p"], c=colors, s=18, alpha=0.85)
    ax.set_title("Artifact Overview")
    ax.set_xlabel("Window start")
    ax.set_ylabel("Peak-to-peak amplitude")
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_example_psd(
    df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    fs: float,
    output_path: str | Path | None = None,
) -> plt.Figure | None:
    clean = artifact_df[~artifact_df["artifact"]]
    if clean.empty:
        return None

    win = clean.iloc[0]
    signal = df["ch1_filtered"].iloc[int(win["start_sample"]):int(win["end_sample"])].to_numpy(dtype=float)
    freqs, psd = compute_psd(signal, fs)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(freqs, psd, linewidth=0.9)
    ax.set_xlim(0, 45)
    ax.set_title(f"Example Welch PSD of Clean Window {int(win['window_id'])}")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_relative_bandpower(features_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    for band in FREQUENCY_BANDS:
        col = f"relative_{band}"
        if col in features_df:
            ax.plot(features_df["start_time"], features_df[col], linewidth=0.9, marker=".", markersize=3, label=col)
    ax.set_title("Relative Bandpower Over Time")
    ax.set_xlabel("Window start")
    ax.set_ylabel("Relative power")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_theta_alpha_ratio(features_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(features_df["start_time"], features_df["theta_alpha_ratio"], linewidth=0.9, marker=".", markersize=3)
    ax.set_title("Theta/Alpha Ratio Over Time")
    ax.set_xlabel("Window start")
    ax.set_ylabel("Theta / Alpha")
    ax.grid(True, alpha=0.3)
    return _close_or_return(fig, output_path)


def plot_feature_correlation_heatmap(features_df: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    numeric = features_df.select_dtypes(include=[np.number]).drop(
        columns=["window_id", "start_sample", "end_sample"],
        errors="ignore",
    )
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Feature Correlation Heatmap")
    return _close_or_return(fig, output_path)
