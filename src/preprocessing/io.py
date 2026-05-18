from __future__ import annotations

from pathlib import Path

import pandas as pd


EEG_CHANNEL = "ch1"
TIMESTAMP_CANDIDATES = ("timestamps", "timestamp")


def validate_columns(df: pd.DataFrame) -> str:
    """Validate required EEG columns and return detected timestamp column."""
    timestamp_col = next((col for col in TIMESTAMP_CANDIDATES if col in df.columns), None)

    if timestamp_col is None:
        raise ValueError("Keine Zeitspalte gefunden. Erwartet: 'timestamps' oder 'timestamp'.")

    if EEG_CHANNEL not in df.columns:
        raise ValueError(f"EEG-Spalte '{EEG_CHANNEL}' fehlt. Vorhandene Spalten: {list(df.columns)}")

    return timestamp_col


def load_eeg_csv(csv_path: str | Path) -> tuple[pd.DataFrame, str, dict]:
    """
    Load, validate and clean an IDUN/Guardian EEG CSV.

    Cleaning steps:
    - detect timestamp column
    - convert timestamp and ch1 to numeric
    - remove invalid/missing rows
    - remove duplicate timestamps
    - sort by timestamp
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)
    timestamp_col = validate_columns(df)

    n_initial = len(df)
    df = df.copy()
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
    df[EEG_CHANNEL] = pd.to_numeric(df[EEG_CHANNEL], errors="coerce")

    n_after_numeric = df[[timestamp_col, EEG_CHANNEL]].notna().all(axis=1).sum()
    df = df.dropna(subset=[timestamp_col, EEG_CHANNEL])

    n_before_duplicates = len(df)
    df = df.drop_duplicates(subset=[timestamp_col])
    n_duplicates = n_before_duplicates - len(df)

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Nach Bereinigung sind zu wenige valide Samples vorhanden.")

    report = {
        "csv_path": str(csv_path),
        "timestamp_column": timestamp_col,
        "rows_initial": int(n_initial),
        "rows_after_numeric_validation": int(n_after_numeric),
        "rows_removed_missing_or_invalid": int(n_initial - n_after_numeric),
        "duplicate_timestamps_removed": int(n_duplicates),
        "rows_final": int(len(df)),
    }

    return df, timestamp_col, report
