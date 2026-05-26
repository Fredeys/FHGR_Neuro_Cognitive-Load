from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.pipeline_config import CONFIG


RESAMPLING_CONFIG = CONFIG.get("resampling", {})


def resampling_enabled() -> bool:
    return bool(RESAMPLING_CONFIG.get("enabled", False))


def maybe_resample_signal(
    df: pd.DataFrame,
    fs: float,
    signal_col: str = "ch1_filtered",
    target_fs_hz: float | None = RESAMPLING_CONFIG.get("target_fs_hz"),
) -> tuple[pd.DataFrame, float, dict]:
    """
    Optionally downsample a filtered signal with anti-aliasing.

    This is intentionally disabled by default. If downsampling is enabled, a
    lowpass filter is applied before resampling so spectral content above the new
    Nyquist frequency cannot alias into the retained EEG band.
    """
    if not resampling_enabled():
        return df, fs, {"applied": False, "reason": "resampling.enabled=false"}

    if target_fs_hz is None:
        raise ValueError("resampling.target_fs_hz muss gesetzt sein, wenn Resampling aktiviert ist.")

    target_fs = float(target_fs_hz)
    if target_fs <= 0:
        raise ValueError("target_fs_hz muss > 0 sein.")
    if signal_col not in df.columns:
        raise ValueError(f"Signalspalte '{signal_col}' fehlt.")
    if target_fs >= fs:
        warnings.warn("target_fs_hz ist >= aktueller Samplingrate; Resampling wird uebersprungen.")
        return df, fs, {"applied": False, "reason": "target_fs_hz>=fs", "target_fs_hz": target_fs}

    working = df.copy()
    signal = working[signal_col].to_numpy(dtype=float)
    aa_config = RESAMPLING_CONFIG.get("anti_aliasing", {})
    if bool(aa_config.get("enabled", True)):
        lowpass_hz = aa_config.get("lowpass_hz")
        cutoff = float(lowpass_hz) if lowpass_hz is not None else 0.45 * target_fs
        cutoff = min(cutoff, 0.95 * target_fs / 2.0)
        signal = _anti_alias_lowpass(signal, fs=fs, cutoff_hz=cutoff, order=int(aa_config.get("order", 4)))
    else:
        cutoff = None

    up, down = _resample_ratio(fs, target_fs)
    resampled = resample_poly(signal, up, down)
    resampled_len = len(resampled)
    resampled_df = working.iloc[:resampled_len].copy().reset_index(drop=True)
    resampled_df[signal_col] = resampled
    resampled_df["time_seconds"] = np.arange(resampled_len, dtype=float) / target_fs
    start_timestamp = float(working["timestamp_seconds"].iloc[0])
    resampled_df["timestamp_seconds"] = start_timestamp + resampled_df["time_seconds"]
    resampled_df["datetime"] = pd.to_datetime(resampled_df["timestamp_seconds"], unit="s", errors="coerce")
    resampled_df["sample_interval"] = resampled_df["timestamp_seconds"].diff()

    return resampled_df, target_fs, {
        "applied": True,
        "source_fs_hz": float(fs),
        "target_fs_hz": target_fs,
        "anti_aliasing_lowpass_hz": cutoff,
        "up": int(up),
        "down": int(down),
    }


def _anti_alias_lowpass(signal: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    nyquist = fs / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"Anti-Aliasing-Cutoff {cutoff_hz} Hz muss zwischen 0 und Nyquist {nyquist:.2f} Hz liegen.")
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def _resample_ratio(fs: float, target_fs: float) -> tuple[int, int]:
    ratio = target_fs / fs
    denominator = 1000
    numerator = int(round(ratio * denominator))
    gcd = int(np.gcd(numerator, denominator))
    return numerator // gcd, denominator // gcd
