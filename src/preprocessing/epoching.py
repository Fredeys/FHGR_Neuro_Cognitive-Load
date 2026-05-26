from __future__ import annotations

import pandas as pd

from src.pipeline_config import CONFIG


EPOCHING_CONFIG = CONFIG.get("epoching", {})


def create_epochs(
    df: pd.DataFrame,
    fs: float,
    signal_col: str = "ch1_filtered",
    epoch_seconds: float = float(EPOCHING_CONFIG.get("epoch_seconds", 1.0)),
    overlap: float = float(EPOCHING_CONFIG.get("overlap", 0.0)),
) -> list[dict]:
    """
    Create optional short QC epochs without replacing feature windows.

    Epochs are intended for quality control and finer artifact inspection. The
    10-second feature windows remain the default basis for PSD, bandpower and
    cognitive-load scoring.
    """
    if signal_col not in df.columns:
        raise ValueError(f"Signalspalte '{signal_col}' fehlt.")
    if epoch_seconds <= 0:
        raise ValueError("epoch_seconds muss > 0 sein.")
    if not 0 <= overlap < 1:
        raise ValueError("overlap muss im Bereich [0, 1) liegen.")

    epoch_size = int(round(epoch_seconds * fs))
    step_size = int(round(epoch_size * (1.0 - overlap)))
    if epoch_size <= 1:
        raise ValueError("Epoch-Laenge ist zu kurz.")
    if step_size < 1:
        raise ValueError("Schrittweite ist zu klein. Overlap reduzieren.")

    epochs = []
    for epoch_id, start_sample in enumerate(range(0, len(df) - epoch_size + 1, step_size)):
        end_sample = start_sample + epoch_size
        epochs.append(
            {
                "epoch_id": int(epoch_id),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_time": df["datetime"].iloc[start_sample],
                "end_time": df["datetime"].iloc[end_sample - 1],
                "signal_segment": df[signal_col].iloc[start_sample:end_sample].to_numpy(dtype=float),
            }
        )

    return epochs


def epoching_enabled() -> bool:
    return bool(EPOCHING_CONFIG.get("enabled", False))
