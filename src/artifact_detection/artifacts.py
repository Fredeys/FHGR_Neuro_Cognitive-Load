import numpy as np
import pandas as pd

from src.pipeline_config import CONFIG


ARTIFACT_CONFIG = CONFIG["artifact_detection"]


def _mad(values: np.ndarray) -> float:
    median = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - median)))


def _robust_threshold(values: np.ndarray, factor: float = float(ARTIFACT_CONFIG["mad_factor"])) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmedian(values) + factor * 1.4826 * _mad(values))


def _configured_threshold(metric: str, values: np.ndarray, threshold_factor: float) -> float:
    configured = ARTIFACT_CONFIG.get(metric)
    if configured is not None:
        return float(configured)
    return _robust_threshold(values, threshold_factor)


def _artifact_metrics(signal: np.ndarray) -> dict:
    signal = np.asarray(signal, dtype=float)
    gradients = np.diff(signal)
    abs_gradients = np.abs(gradients)
    if abs_gradients.size == 0:
        max_abs_gradient = 0.0
        mean_abs_gradient = 0.0
        gradient_p95 = 0.0
    else:
        max_abs_gradient = float(np.max(abs_gradients))
        mean_abs_gradient = float(np.mean(abs_gradients))
        gradient_p95 = float(np.percentile(abs_gradients, 95))

    return {
        "p2p": float(np.ptp(signal)),
        "std": float(np.std(signal)),
        "variance": float(np.var(signal)),
        "max_abs": float(np.max(np.abs(signal))),
        "energy": float(np.mean(signal**2)),
        "max_abs_gradient": max_abs_gradient,
        "mean_abs_gradient": mean_abs_gradient,
        "gradient_p95": gradient_p95,
    }


def detect_artifacts(
    df: pd.DataFrame,
    windows: list[dict],
    signal_col: str = "ch1_filtered",
    threshold_factor: float = float(ARTIFACT_CONFIG["mad_factor"]),
) -> tuple[pd.DataFrame, dict]:
    """
    Mark artifact windows using robust thresholds.

    Criteria:
    - peak-to-peak amplitude
    - standard deviation
    - variance
    - maximum absolute amplitude
    - mean signal energy
    - sample-to-sample gradients
    """
    rows = []
    for win in windows:
        signal = df[signal_col].iloc[win["start_sample"]:win["end_sample"]].to_numpy(dtype=float)
        rows.append({**win, **_artifact_metrics(signal)})

    artifact_df = pd.DataFrame(rows)
    thresholds = {
        "p2p_threshold": _configured_threshold(
            "peak_to_peak_threshold", artifact_df["p2p"].to_numpy(), threshold_factor
        ),
        "std_threshold": _configured_threshold(
            "std_threshold", artifact_df["std"].to_numpy(), threshold_factor
        ),
        "variance_threshold": _configured_threshold(
            "variance_threshold", artifact_df["variance"].to_numpy(), threshold_factor
        ),
        "max_abs_threshold": _configured_threshold(
            "absolute_amplitude_threshold", artifact_df["max_abs"].to_numpy(), threshold_factor
        ),
        "energy_threshold": _configured_threshold(
            "energy_threshold", artifact_df["energy"].to_numpy(), threshold_factor
        ),
        "gradient_threshold": _configured_threshold(
            "gradient_threshold", artifact_df["max_abs_gradient"].to_numpy(), threshold_factor
        ),
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
        "artifact_windows": int(artifact_df["artifact"].sum()),
        "total_windows": int(len(artifact_df)),
        "artifact_fraction": float(artifact_df["artifact"].mean()),
    }
    return artifact_df, report
