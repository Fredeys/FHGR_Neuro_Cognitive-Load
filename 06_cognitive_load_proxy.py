"""
Cognitive Load Proxy Score extension for an existing EEG feature table.

Input assumption:
features_df already exists and contains at least:
- artifact
- relative_theta
- relative_alpha
- beta_alpha_ratio
- theta_alpha_ratio
- spectral_entropy
- start_time
- end_time

Important:
This is not a supervised overload classifier. It is an EEG-based deviation
index relative to an individual/session baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline_config import CONFIG, load_features
from src.scoring.state_machine import apply_cognitive_load_state_machine


EPS = 1e-12

LOAD_FEATURES = load_features()
BASELINE_CONFIG = CONFIG["baseline"]
PROXY_CONFIG = CONFIG["cognitive_load_proxy"]
PLOT_DPI = int(CONFIG["plots"]["dpi"])

ZSCORE_COLUMNS = {
    "theta_alpha_ratio": "theta_alpha_ratio_z",
    "beta_alpha_ratio": "beta_alpha_ratio_z",
    "relative_theta": "relative_theta_z",
    "relative_alpha": "relative_alpha_z",
    "spectral_entropy": "spectral_entropy_z",
}


def define_baseline(
    features_df: pd.DataFrame,
    baseline_minutes: float = float(BASELINE_CONFIG["baseline_minutes"]),
    fallback_clean_windows: int = int(BASELINE_CONFIG["fallback_clean_windows"]),
) -> tuple[pd.DataFrame, dict]:
    """
    Define baseline windows.

    Default:
    - first 3-5 minutes of recording
    - artifact == False only

    If start_time is unavailable or unusable, the first clean windows are used.
    """
    required = ["artifact", *LOAD_FEATURES]
    missing = [col for col in required if col not in features_df.columns]
    if missing:
        raise ValueError(f"features_df fehlt erforderliche Spalten: {missing}")

    df = features_df.copy()
    clean_df = df[df["artifact"] == False].copy()

    if clean_df.empty:
        raise ValueError("Keine artifact=False Fenster für Baseline vorhanden.")

    baseline_df = pd.DataFrame()
    used_method = "first_clean_windows"

    if "start_time" in clean_df.columns:
        clean_df["start_time"] = pd.to_datetime(clean_df["start_time"], errors="coerce")
        valid_time_df = clean_df.dropna(subset=["start_time"]).copy()

        if not valid_time_df.empty:
            baseline_start = valid_time_df["start_time"].min()
            baseline_end = baseline_start + pd.Timedelta(minutes=baseline_minutes)
            baseline_df = valid_time_df[
                (valid_time_df["start_time"] >= baseline_start)
                & (valid_time_df["start_time"] <= baseline_end)
            ].copy()
            used_method = "time_based"

    if baseline_df.empty:
        baseline_df = clean_df.head(fallback_clean_windows).copy()
        used_method = "first_clean_windows"

    if baseline_df.empty:
        raise ValueError("Baseline konnte nicht definiert werden.")

    report = {
        "baseline_method": used_method,
        "baseline_minutes": float(baseline_minutes),
        "fallback_clean_windows": int(fallback_clean_windows),
        "baseline_windows": int(len(baseline_df)),
        "baseline_start": baseline_df["start_time"].min() if "start_time" in baseline_df.columns else None,
        "baseline_end": baseline_df["end_time"].max() if "end_time" in baseline_df.columns else None,
    }

    return baseline_df, report


def compute_baseline_zscores(
    features_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    epsilon: float = EPS,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute baseline means/stds and add z-scored feature columns.

    Formula:
    feature_z = (feature - baseline_mean) / baseline_std
    """
    df = features_df.copy()
    baseline_stats = {}

    for feature in LOAD_FEATURES:
        z_col = ZSCORE_COLUMNS[feature]
        baseline_mean = float(baseline_df[feature].mean(skipna=True))
        baseline_std = float(baseline_df[feature].std(skipna=True))

        if not np.isfinite(baseline_std) or baseline_std == 0:
            baseline_std = epsilon

        df[z_col] = (df[feature] - baseline_mean) / (baseline_std + epsilon)
        baseline_stats[feature] = {
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
        }

    return df, baseline_stats


def compute_cognitive_load_proxy_score(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute EEG-based Cognitive Load Proxy Score.

    This is a baseline-relative deviation index, not a true overload classifier.
    """
    df = features_df.copy()

    required_z = list(ZSCORE_COLUMNS.values())
    missing = [col for col in required_z if col not in df.columns]
    if missing:
        raise ValueError(f"Z-Score-Spalten fehlen: {missing}")

    df["Cognitive_Load_Proxy_Score"] = (
        sum(float(weight) * df[z_col] for z_col, weight in PROXY_CONFIG["weights"].items())
    )

    df.loc[df["artifact"] == True, "Cognitive_Load_Proxy_Score"] = np.nan

    return df


def smooth_cognitive_load_score(
    features_df: pd.DataFrame,
    rolling_windows: int = int(PROXY_CONFIG["rolling_windows"]),
) -> pd.DataFrame:
    """
    Smooth Cognitive Load Proxy Score with rolling mean over 3-6 windows.
    """
    if rolling_windows < 3 or rolling_windows > 6:
        raise ValueError("rolling_windows sollte zwischen 3 und 6 liegen.")

    df = features_df.copy()
    df["Cognitive_Load_Proxy_Score_Smoothed"] = (
        df["Cognitive_Load_Proxy_Score"]
        .rolling(window=rolling_windows, min_periods=1, center=True)
        .mean()
    )

    return df


def assign_cognitive_load_states(
    features_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    score_col: str = "Cognitive_Load_Proxy_Score",
) -> tuple[pd.DataFrame, dict]:
    """
    Assign adaptive states based on baseline score distribution.

    normal:
        Score <= baseline_mean + 1 * baseline_std
    elevated:
        Score > baseline_mean + 1 * baseline_std
    strong:
        Score > baseline_mean + 2 * baseline_std
    """
    df = features_df.copy()

    baseline_scores = df.loc[baseline_df.index, score_col].dropna()
    if baseline_scores.empty:
        raise ValueError("Keine gültigen Baseline-Scores für Zustandsableitung vorhanden.")

    baseline_score_mean = float(baseline_scores.mean())
    baseline_score_std = float(baseline_scores.std())

    if not np.isfinite(baseline_score_std) or baseline_score_std == 0:
        baseline_score_std = EPS

    return apply_cognitive_load_state_machine(
        df,
        baseline_score_mean=baseline_score_mean,
        baseline_score_std=baseline_score_std,
        score_col=score_col,
        artifact_col="artifact",
    )


def plot_cognitive_load_score(
    features_df: pd.DataFrame,
    thresholds: dict,
    output_dir: str | Path | None = None,
) -> dict[str, plt.Figure]:
    """
    Create Matplotlib visualizations:
    - Cognitive Load Proxy Score over time
    - smoothed score over time
    - threshold lines
    - colored state markers
    - theta/alpha ratio over time
    - relative alpha and theta over time
    """
    output_dir = Path(output_dir) if output_dir is not None else None
    df = features_df.copy()

    x_col = "start_time" if "start_time" in df.columns else "window_id"

    state_colors = {
        "normal": "#4C78A8",
        "elevated": "#F58518",
        "strong": "#E45756",
        "artifact": "#9D9D9D",
    }

    figures = {}

    fig_score, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[x_col], df["Cognitive_Load_Proxy_Score"], linewidth=0.9, label="Proxy Score", color="#4C78A8")
    ax.plot(
        df[x_col],
        df["Cognitive_Load_Proxy_Score_Smoothed"],
        linewidth=1.5,
        label="Smoothed Proxy Score",
        color="#F58518",
    )
    for state, color in state_colors.items():
        state_df = df[df["cognitive_load_state"] == state]
        if state_df.empty:
            continue
        ax.scatter(state_df[x_col], state_df["Cognitive_Load_Proxy_Score"], s=18, color=color, label=state)
    ax.axhline(thresholds["elevated_threshold"], linestyle="--", color="#F58518", label="elevated")
    ax.axhline(thresholds["strongly_elevated_threshold"], linestyle="--", color="#E45756", label="strong")
    ax.set_title("EEG Cognitive Load Proxy Score Over Time")
    ax.set_xlabel("Time" if x_col == "start_time" else "Window")
    ax.set_ylabel("Baseline-relative proxy score [z-weighted]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    figures["cognitive_load_proxy_score"] = fig_score
    _save_figure(fig_score, _path(output_dir, "01_cognitive_load_proxy_score.png"))

    fig_ta, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[x_col], df["theta_alpha_ratio"], linewidth=0.9, marker=".", markersize=3)
    ax.set_title("Theta/Alpha Ratio Over Time")
    ax.set_xlabel("Time" if x_col == "start_time" else "Window")
    ax.set_ylabel("Theta / Alpha")
    ax.grid(True, alpha=0.3)
    figures["theta_alpha_ratio"] = fig_ta
    _save_figure(fig_ta, _path(output_dir, "02_theta_alpha_ratio.png"))

    fig_theta_alpha, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[x_col], df["relative_theta"], linewidth=0.9, marker=".", markersize=3, label="Relative Theta")
    ax.plot(df[x_col], df["relative_alpha"], linewidth=0.9, marker=".", markersize=3, label="Relative Alpha")
    ax.set_title("Relative Theta and Relative Alpha Over Time")
    ax.set_xlabel("Time" if x_col == "start_time" else "Window")
    ax.set_ylabel("Relative bandpower")
    ax.legend()
    ax.grid(True, alpha=0.3)
    figures["relative_theta_alpha"] = fig_theta_alpha
    _save_figure(fig_theta_alpha, _path(output_dir, "03_relative_theta_alpha.png"))

    return figures


def export_score_results(
    features_df: pd.DataFrame,
    output_dir: str | Path = "outputs/cognitive_load_proxy",
    prefix: str = "eeg",
) -> dict[str, Path]:
    """
    Export updated feature table and compact score table.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / f"{prefix}_features_with_cognitive_load_proxy.csv"
    score_path = output_dir / f"{prefix}_cognitive_load_proxy_scores.csv"

    features_df.to_csv(features_path, index=False)

    score_cols = [
        "window_id",
        "start_time",
        "end_time",
        "Cognitive_Load_Proxy_Score",
        "Cognitive_Load_Proxy_Score_Smoothed",
        "cognitive_load_state",
    ]
    available_score_cols = [col for col in score_cols if col in features_df.columns]
    features_df[available_score_cols].to_csv(score_path, index=False)

    return {
        "updated_features_csv": features_path,
        "score_table_csv": score_path,
    }


def add_cognitive_load_proxy_score(
    features_df: pd.DataFrame,
    baseline_minutes: float = float(BASELINE_CONFIG["baseline_minutes"]),
    fallback_clean_windows: int = int(BASELINE_CONFIG["fallback_clean_windows"]),
    rolling_windows: int = int(PROXY_CONFIG["rolling_windows"]),
    output_dir: str | Path | None = None,
    prefix: str = "eeg",
    make_plots: bool = True,
    export: bool = True,
) -> tuple[pd.DataFrame, dict, dict[str, plt.Figure]]:
    """
    Main extension function for an existing features_df.

    Returns:
    - updated features_df
    - report with baseline statistics and thresholds
    - Matplotlib figures
    """
    baseline_df, baseline_report = define_baseline(
        features_df,
        baseline_minutes=baseline_minutes,
        fallback_clean_windows=fallback_clean_windows,
    )

    updated_df, baseline_stats = compute_baseline_zscores(features_df, baseline_df)
    updated_df = compute_cognitive_load_proxy_score(updated_df)
    updated_df = smooth_cognitive_load_score(updated_df, rolling_windows=rolling_windows)

    # Re-select baseline rows from updated_df to include newly computed score.
    baseline_df_for_score = updated_df.loc[baseline_df.index]
    updated_df, thresholds = assign_cognitive_load_states(updated_df, baseline_df_for_score)

    figures = {}
    if make_plots:
        figures = plot_cognitive_load_score(updated_df, thresholds, output_dir=output_dir)

    exported = {}
    if export and output_dir is not None:
        exported = export_score_results(updated_df, output_dir=output_dir, prefix=prefix)

    report = {
        "interpretation": (
            "Der Cognitive Load Proxy Score ist kein echter Overload-Klassifikator, "
            "sondern ein EEG-basierter Abweichungsindex relativ zur individuellen Baseline."
        ),
        "baseline": baseline_report,
        "baseline_feature_stats": baseline_stats,
        "score_thresholds": thresholds,
        "rolling_windows": int(rolling_windows),
        "exports": exported,
    }

    return updated_df, report, figures


def _path(output_dir: Path | None, filename: str) -> Path | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
