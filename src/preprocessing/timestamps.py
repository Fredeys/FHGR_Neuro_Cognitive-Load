import numpy as np
import pandas as pd

from src.pipeline_config import CONFIG

EXPECTED_FS = float(CONFIG["sampling"]["expected_fs_hz"])


def validate_timestamps(df: pd.DataFrame, timestamp_col: str) -> tuple[pd.DataFrame, dict]:
    """Detect Unix timestamp unit, convert to datetime and add timing columns."""
    df = df.copy()
    median_timestamp = float(np.nanmedian(df[timestamp_col].to_numpy(dtype=float)))

    if median_timestamp > 1e11:
        unit = "ms"
        df["timestamp_seconds"] = df[timestamp_col] / 1000.0
    else:
        unit = "s"
        df["timestamp_seconds"] = df[timestamp_col]

    df["datetime"] = pd.to_datetime(df["timestamp_seconds"], unit="s", errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    df["time_seconds"] = df["timestamp_seconds"] - df["timestamp_seconds"].iloc[0]
    df["sample_interval"] = df["timestamp_seconds"].diff()

    report = {
        "timestamp_unit": unit,
        "start_time": df["datetime"].iloc[0],
        "end_time": df["datetime"].iloc[-1],
        "duration_seconds": float(df["time_seconds"].iloc[-1]),
    }

    return df, report


def estimate_sampling_rate(
    df: pd.DataFrame,
    expected_fs: float = EXPECTED_FS,
    tolerance: float = float(CONFIG["sampling"]["fs_tolerance_fraction"]),
) -> dict:
    """
    Estimate sampling rate from timestamp intervals.

    Median interval is used as primary estimate because it is robust to gaps.
    """
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

    lower = expected_fs * (1.0 - tolerance)
    upper = expected_fs * (1.0 + tolerance)
    plausible_250hz = lower <= fs_median <= upper

    warnings = []
    if not plausible_250hz:
        warnings.append(
            f"Samplingrate {fs_median:.2f} Hz weicht deutlich von erwarteten {expected_fs:.2f} Hz ab."
        )
    if len(large_gaps) > 0:
        warnings.append(f"{len(large_gaps)} grosse Zeitlücken erkannt; maximale Lücke: {np.max(large_gaps):.3f} s.")
    if len(irregular_intervals) > 0:
        warnings.append(f"{len(irregular_intervals)} unregelmässige Samplingintervalle erkannt.")

    return {
        "mean_dt": mean_dt,
        "median_dt": median_dt,
        "std_dt": std_dt,
        "fs_mean": fs_mean,
        "fs_median": fs_median,
        "fs_plausible_around_250hz": bool(plausible_250hz),
        "n_large_gaps": int(len(large_gaps)),
        "max_gap_seconds": float(np.max(large_gaps)) if len(large_gaps) else 0.0,
        "n_irregular_intervals": int(len(irregular_intervals)),
        "warnings": warnings,
    }
