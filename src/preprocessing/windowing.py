import pandas as pd

from src.pipeline_config import CONFIG


def create_windows(
    df: pd.DataFrame,
    fs: float,
    window_seconds: float = float(CONFIG["windowing"]["window_seconds"]),
    overlap: float = float(CONFIG["windowing"]["overlap"]),
) -> list[dict]:
    """Create overlapping windows using the estimated sampling rate."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap muss im Bereich [0, 1) liegen.")

    window_size = int(round(window_seconds * fs))
    step_size = int(round(window_size * (1.0 - overlap)))

    if window_size <= 1:
        raise ValueError("Fenstergrösse ist zu klein.")
    if step_size < 1:
        raise ValueError("Schrittgrösse ist zu klein.")

    windows = []
    for window_id, start_sample in enumerate(range(0, len(df) - window_size + 1, step_size)):
        end_sample = start_sample + window_size
        windows.append(
            {
                "window_id": window_id,
                "start_time": df["datetime"].iloc[start_sample],
                "end_time": df["datetime"].iloc[end_sample - 1],
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
            }
        )

    if not windows:
        raise ValueError("Keine vollständigen Fenster erzeugt. Aufnahme ist evtl. kürzer als die Fensterlänge.")

    return windows
