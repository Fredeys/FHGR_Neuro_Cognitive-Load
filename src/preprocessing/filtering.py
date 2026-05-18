import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from src.pipeline_config import CONFIG


PREPROCESSING_CONFIG = CONFIG["preprocessing"]

def center_signal(df: pd.DataFrame, input_col: str = "ch1", output_col: str = "ch1_centered") -> tuple[pd.DataFrame, dict]:
    """Remove DC offset from EEG signal."""
    df = df.copy()
    dc_offset = float(df[input_col].mean())
    df[output_col] = df[input_col] - dc_offset
    return df, {"dc_offset_removed": dc_offset}


def apply_bandpass_filter(
    df: pd.DataFrame,
    fs: float,
    input_col: str = "ch1_centered",
    output_col: str = "ch1_bandpassed",
    lowcut: float = float(PREPROCESSING_CONFIG["bandpass_low_hz"]),
    highcut: float = float(PREPROCESSING_CONFIG["bandpass_high_hz"]),
    order: int = int(PREPROCESSING_CONFIG["bandpass_order"]),
) -> pd.DataFrame:
    """Apply zero-phase Butterworth bandpass filter using second-order sections."""
    nyquist = fs / 2.0
    if lowcut <= 0:
        raise ValueError("lowcut muss > 0 Hz sein.")
    if highcut >= nyquist:
        raise ValueError(f"highcut={highcut} Hz muss unter Nyquist={nyquist:.2f} Hz liegen.")

    df = df.copy()
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    df[output_col] = sosfiltfilt(sos, df[input_col].to_numpy(dtype=float))
    return df


def apply_notch_filter(
    df: pd.DataFrame,
    fs: float,
    input_col: str = "ch1_bandpassed",
    output_col: str = "ch1_filtered",
    notch_freq: float = float(PREPROCESSING_CONFIG["notch_freq_hz"]),
    quality_factor: float = float(PREPROCESSING_CONFIG["notch_quality_factor"]),
) -> pd.DataFrame:
    """Apply zero-phase 50 Hz notch filter against line noise."""
    df = df.copy()
    nyquist = fs / 2.0

    if notch_freq >= nyquist:
        warnings.warn(
            f"Notch-Frequenz {notch_freq} Hz liegt oberhalb Nyquist {nyquist:.2f} Hz. Notch wird übersprungen."
        )
        df[output_col] = df[input_col]
        return df

    b, a = iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
    df[output_col] = filtfilt(b, a, df[input_col].to_numpy(dtype=float))
    return df


def inspect_signal(df: pd.DataFrame, fs: float, signal_col: str = "ch1") -> dict:
    """Compute descriptive statistics and basic raw-signal quality checks."""
    signal = df[signal_col].to_numpy(dtype=float)
    n_samples = len(signal)
    duration = float(df["time_seconds"].iloc[-1] - df["time_seconds"].iloc[0])
    max_abs = float(np.max(np.abs(signal)))

    diffs = np.diff(signal)
    flat_threshold = max(1e-12, 1e-9 * max(1.0, max_abs))
    min_flat_len = int(round(0.5 * fs))

    flat_regions = 0
    run_length = 0
    for is_flat in np.abs(diffs) <= flat_threshold:
        if is_flat:
            run_length += 1
        else:
            if run_length >= min_flat_len:
                flat_regions += 1
            run_length = 0
    if run_length >= min_flat_len:
        flat_regions += 1

    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median)))
    robust_sigma = 1.4826 * mad + 1e-12
    extreme_peaks = np.abs(signal - median) > 8.0 * robust_sigma

    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    clipping_fraction = float((np.sum(signal == min_val) + np.sum(signal == max_val)) / n_samples)

    abs_signal = np.abs(signal)
    abs_median = float(np.median(abs_signal))
    abs_mad = float(np.median(np.abs(abs_signal - abs_median)))
    high_amp_threshold = abs_median + 8.0 * 1.4826 * abs_mad

    return {
        "n_samples": int(n_samples),
        "duration_seconds": duration,
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": min_val,
        "max": max_val,
        "peak_to_peak": float(np.ptp(signal)),
        "max_abs": max_abs,
        "flat_regions": int(flat_regions),
        "extreme_peak_samples": int(np.sum(extreme_peaks)),
        "clipping_fraction": clipping_fraction,
        "high_amplitude_samples": int(np.sum(abs_signal > high_amp_threshold)),
    }
