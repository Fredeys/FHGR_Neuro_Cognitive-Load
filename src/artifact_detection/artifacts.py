import numpy as np
import pandas as pd

from src.pipeline_config import CONFIG


def _mad(values: np.ndarray) -> float:
    median = np.nanmedian(values)
    return float(np.nanmedian(np.abs(values - median)))


def _robust_threshold(values: np.ndarray, factor: float = float(CONFIG["artifact_detection"]["mad_factor"])) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmedian(values) + factor * 1.4826 * _mad(values))


def detect_artifacts(
    df: pd.DataFrame,
    windows: list[dict],
    signal_col: str = "ch1_filtered",
    threshold_factor: float = float(CONFIG["artifact_detection"]["mad_factor"]),
) -> tuple[pd.DataFrame, dict]:
    """
    Mark artifact windows using robust thresholds.

    Criteria:
    - peak-to-peak amplitude
    - standard deviation
    - maximum absolute amplitude
    - mean signal energy
    """
    rows = []
    for win in windows:
        signal = df[signal_col].iloc[win["start_sample"]:win["end_sample"]].to_numpy(dtype=float)
        rows.append(
            {
                **win,
                "p2p": float(np.ptp(signal)),
                "std": float(np.std(signal)),
                "max_abs": float(np.max(np.abs(signal))),
                "energy": float(np.mean(signal**2)),
            }
        )

    artifact_df = pd.DataFrame(rows)
    thresholds = {
        "p2p_threshold": _robust_threshold(artifact_df["p2p"].to_numpy(), threshold_factor),
        "std_threshold": _robust_threshold(artifact_df["std"].to_numpy(), threshold_factor),
        "max_abs_threshold": _robust_threshold(artifact_df["max_abs"].to_numpy(), threshold_factor),
        "energy_threshold": _robust_threshold(artifact_df["energy"].to_numpy(), threshold_factor),
    }

    artifact_df["artifact"] = (
        (artifact_df["p2p"] > thresholds["p2p_threshold"])
        | (artifact_df["std"] > thresholds["std_threshold"])
        | (artifact_df["max_abs"] > thresholds["max_abs_threshold"])
        | (artifact_df["energy"] > thresholds["energy_threshold"])
    )

    report = {
        **thresholds,
        "artifact_windows": int(artifact_df["artifact"].sum()),
        "total_windows": int(len(artifact_df)),
        "artifact_fraction": float(artifact_df["artifact"].mean()),
    }
    return artifact_df, report
