from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.psd import compute_bandpower, compute_psd, compute_spectral_entropy
from src.pipeline_config import CONFIG, frequency_bands


EPS = 1e-12

FREQUENCY_BANDS = frequency_bands()


def extract_features(
    df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    fs: float,
    signal_col: str = "ch1_filtered",
) -> pd.DataFrame:
    """
    Extract window-wise features from filtered EEG.

    Artifact windows remain in the table and should be excluded or modelled
    consciously downstream.
    """
    rows = []

    for _, win in artifact_df.iterrows():
        start_sample = int(win["start_sample"])
        end_sample = int(win["end_sample"])
        signal = df[signal_col].iloc[start_sample:end_sample].to_numpy(dtype=float)

        freqs, psd = compute_psd(signal, fs)
        bandpowers = {
            f"{name}_power": compute_bandpower(freqs, psd, band)
            for name, band in FREQUENCY_BANDS.items()
        }
        total_power = compute_bandpower(
            freqs,
            psd,
            (float(CONFIG["psd"]["total_power_low_hz"]), float(CONFIG["psd"]["total_power_high_hz"])),
        )

        relative = {
            f"relative_{name}": bandpowers[f"{name}_power"] / (total_power + EPS)
            for name in FREQUENCY_BANDS
        }

        theta = bandpowers["theta_power"]
        alpha = bandpowers["alpha_power"]
        beta = bandpowers["beta_power"]

        rows.append(
            {
                "window_id": int(win["window_id"]),
                "start_time": win["start_time"],
                "end_time": win["end_time"],
                "start_sample": start_sample,
                "end_sample": end_sample,
                "artifact": bool(win["artifact"]),
                "total_power_1_40": total_power,
                **bandpowers,
                **relative,
                "theta_alpha_ratio": theta / (alpha + EPS),
                "beta_alpha_ratio": beta / (alpha + EPS),
                "alpha_theta_ratio": alpha / (theta + EPS),
                "signal_mean": float(np.mean(signal)),
                "signal_std": float(np.std(signal)),
                "signal_variance": float(np.var(signal)),
                "peak_to_peak": float(np.ptp(signal)),
                "max_abs": float(np.max(np.abs(signal))),
                "spectral_entropy": compute_spectral_entropy(psd),
            }
        )

    return pd.DataFrame(rows)


def baseline_normalize(
    features_df: pd.DataFrame,
    baseline_start: pd.Timestamp | None = None,
    baseline_end: pd.Timestamp | None = None,
    baseline_window_ids: list[int] | None = None,
    exclude_artifacts: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Create z-normalized features based on clean baseline windows."""
    features_df = features_df.copy()

    if baseline_start is None and baseline_end is None and baseline_window_ids is None:
        return features_df, {
            "applied": False,
            "message": (
                "Keine Baseline angegeben. Für Cognitive-Overload-Erkennung ist "
                "eine individuelle Baseline pro Person/Session empfohlen."
            ),
        }

    if baseline_window_ids is not None:
        baseline_mask = features_df["window_id"].isin(baseline_window_ids)
    else:
        baseline_mask = pd.Series(True, index=features_df.index)
        if baseline_start is not None:
            baseline_mask &= features_df["start_time"] >= baseline_start
        if baseline_end is not None:
            baseline_mask &= features_df["end_time"] <= baseline_end

    if exclude_artifacts:
        baseline_mask &= ~features_df["artifact"]

    baseline_df = features_df[baseline_mask]
    if baseline_df.empty:
        raise ValueError("Keine sauberen Baseline-Fenster gefunden.")

    metadata_cols = {"window_id", "start_time", "end_time", "start_sample", "end_sample", "artifact"}
    feature_cols = [
        col for col in features_df.columns
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(features_df[col])
    ]

    for col in feature_cols:
        mean = baseline_df[col].mean()
        std = baseline_df[col].std()
        features_df[f"{col}_z"] = (features_df[col] - mean) / (std + EPS)

    return features_df, {
        "applied": True,
        "baseline_windows": int(len(baseline_df)),
        "normalized_features": int(len(feature_cols)),
    }
