"""
Strictly ordered EEG pipeline for IDUN/Guardian single-channel EEG.

IMPORTANT:
The processing order follows the requested scientific sequence exactly:

1. Rohdaten einlesen
2. Spalten validieren: timestamp/timestamps, ch1
3. Datentypen bereinigen
4. fehlende Werte / Duplikate entfernen
5. Zeitstempel sortieren
6. Samplingrate prüfen
7. Rohsignal visualisieren
8. Signal zentrieren / DC-Offset entfernen
9. Bandpass: 1-40 Hz
10. Notch: 50 Hz
11. gefiltertes Signal visualisieren
12. Fensterbildung: 10 Sekunden, 50 % Overlap
13. Artefakt-Metriken pro Fenster berechnen
14. Artefaktfenster markieren
15. PSD mit Welch nur für gültige Fenster berechnen
16. Bandpower berechnen
17. relative Bandpower berechnen
18. Ratios berechnen
19. optionale Baseline-Normalisierung
20. Cognitive Load Proxy Score berechnen
21. Score glätten
22. Feature-/Score-Tabelle erzeugen
23. Export

Methodological notes:
- Welch PSD is used instead of raw FFT.
- Relative bandpower is emphasized.
- Artifact windows are marked, not deleted.
- PSD and feature extraction are performed only for valid artifact=False windows.
- Cognitive Load Proxy Score is interpreted only as baseline deviation.
- No supervised overload classifier is implemented.
- Raw CSV files are loaded from data/raw/ by default.
- Plots are created with Matplotlib PNG output only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch
from scipy.stats import kurtosis, pearsonr, skew, spearmanr

from src.pipeline_config import CONFIG, frequency_bands, load_features


EPS = 1e-12
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
SUMMARY_DIR = PROJECT_ROOT / "outputs"

SAMPLING_CONFIG = CONFIG["sampling"]
PREPROCESSING_CONFIG = CONFIG["preprocessing"]
WINDOWING_CONFIG = CONFIG["windowing"]
ARTIFACT_CONFIG = CONFIG["artifact_detection"]
PSD_CONFIG = CONFIG["psd"]
BASELINE_CONFIG = CONFIG["baseline"]
PROXY_CONFIG = CONFIG["cognitive_load_proxy"]
EXTERNAL_SCORE_CONFIG = CONFIG["external_score"]
PLOT_CONFIG = CONFIG["plots"]

EXPECTED_FS = float(SAMPLING_CONFIG["expected_fs_hz"])
FREQUENCY_BANDS = frequency_bands()
LOAD_FEATURES = load_features()
USE_EXTERNAL_SCORE = bool(EXTERNAL_SCORE_CONFIG["use_external_score"])
EXTERNAL_SCORE_COLUMN = str(EXTERNAL_SCORE_CONFIG["column"])
EXTERNAL_SCORE_TYPE = str(EXTERNAL_SCORE_CONFIG["type"])
HIGH_WORKLOAD_THRESHOLD = float(EXTERNAL_SCORE_CONFIG["high_workload_threshold"])
BANDPASS_LOW_HZ = float(PREPROCESSING_CONFIG["bandpass_low_hz"])
BANDPASS_HIGH_HZ = float(PREPROCESSING_CONFIG["bandpass_high_hz"])
BANDPASS_ORDER = int(PREPROCESSING_CONFIG["bandpass_order"])
NOTCH_FREQ_HZ = float(PREPROCESSING_CONFIG["notch_freq_hz"])
NOTCH_QUALITY_FACTOR = float(PREPROCESSING_CONFIG["notch_quality_factor"])
WINDOW_SECONDS = float(WINDOWING_CONFIG["window_seconds"])
WINDOW_OVERLAP = float(WINDOWING_CONFIG["overlap"])
BASELINE_MINUTES = float(BASELINE_CONFIG["baseline_minutes"])
ROLLING_WINDOWS = int(PROXY_CONFIG["rolling_windows"])
TOTAL_POWER_BAND = (float(PSD_CONFIG["total_power_low_hz"]), float(PSD_CONFIG["total_power_high_hz"]))
PSD_NPERSEG_MAX = int(PSD_CONFIG["nperseg_max"])
PSD_FULL_SIGNAL_NPERSEG_MAX = int(PSD_CONFIG["full_signal_nperseg_max"])
DETAIL_SECONDS = float(PLOT_CONFIG["detail_seconds"])
PLOT_DPI = int(PLOT_CONFIG["dpi"])


def _resolve_raw_csv_path(csv_path: str | Path | None) -> Path:
    """Resolve input CSV path, using data/raw/ as the default source directory."""
    if csv_path is None:
        csv_files = sorted(RAW_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"Keine CSV-Datei in {RAW_DIR} gefunden. Lege Rohdaten dort ab "
                "oder übergib einen konkreten CSV-Pfad."
            )
        return csv_files[0]

    path = Path(csv_path)
    if path.exists():
        return path

    raw_path = RAW_DIR / path
    if raw_path.exists():
        return raw_path

    raise FileNotFoundError(
        f"CSV-Datei nicht gefunden: {csv_path}. Standard-Suchordner ist {RAW_DIR}."
    )


def run_ordered_eeg_pipeline(
    csv_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
    baseline_minutes: float = BASELINE_MINUTES,
    rolling_windows: int = ROLLING_WINDOWS,
    show_plots: bool = False,
    use_external_score: bool = USE_EXTERNAL_SCORE,
    external_score_column: str = EXTERNAL_SCORE_COLUMN,
    external_score_type: str = EXTERNAL_SCORE_TYPE,
    high_workload_threshold: float = HIGH_WORKLOAD_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run the EEG pipeline in the exact requested order."""
    csv_path = _resolve_raw_csv_path(csv_path)
    output_base = Path(output_dir) if output_dir is not None else PROJECT_ROOT
    prefix = prefix or csv_path.stem
    processed_dir = output_base / "data" / "processed"
    features_dir = output_base / "data" / "features"
    artifacts_dir = output_base / "data" / "artifacts"
    plots_dir = output_base / "outputs" / "plots"
    summary_dir = output_base / "outputs"
    reports_dir = output_base / "outputs" / "reports"

    summary: dict = {
        "method_notes": [],
        "pipeline_config": CONFIG,
        "external_score_config": {
            "USE_EXTERNAL_SCORE": bool(use_external_score),
            "EXTERNAL_SCORE_COLUMN": external_score_column,
            "EXTERNAL_SCORE_TYPE": external_score_type,
            "HIGH_WORKLOAD_THRESHOLD": float(high_workload_threshold),
        },
    }

    # 1. Rohdaten einlesen
    print("\n1. Rohdaten einlesen")
    raw_df = pd.read_csv(csv_path)
    summary["step_1_csv_path"] = str(csv_path)
    summary["step_1_raw_rows"] = int(len(raw_df))
    print(f"CSV-Datei geladen: {csv_path}")
    print(f"Anzahl Zeilen / Spalten: {raw_df.shape[0]} / {raw_df.shape[1]}")

    # 2. Spalten validieren: timestamp/timestamps, ch1
    print("\n2. Spalten validieren")
    timestamp_col = _validate_columns(raw_df)
    summary["step_2_timestamp_column"] = timestamp_col
    print(f"Erkannte Zeitspalte: {timestamp_col}")
    print("Spalte ch1 gefunden.")

    # 3. Datentypen bereinigen
    print("\n3. Datentypen bereinigen")
    df = raw_df.copy()
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
    df["ch1"] = pd.to_numeric(df["ch1"], errors="coerce")
    external_score_available = bool(use_external_score and external_score_column in df.columns)
    if external_score_available:
        df[external_score_column] = pd.to_numeric(df[external_score_column], errors="coerce")
        print(f"Externer Workload-Score erkannt: {external_score_column}")
    elif use_external_score:
        print("Hinweis: Kein externer Workload-Score vorhanden.")
    missing_before_cleaning = int(df[[timestamp_col, "ch1"]].isna().sum().sum())
    print(f"Fehlende/ungültige Werte vor Bereinigung: {missing_before_cleaning}")

    # 4. fehlende Werte / Duplikate entfernen
    print("\n4. Fehlende Werte / Duplikate entfernen")
    rows_before_cleaning = len(df)
    df = df.dropna(subset=[timestamp_col, "ch1"])
    rows_after_missing = len(df)
    df = df.drop_duplicates(subset=[timestamp_col])
    missing_after_cleaning = int(df[[timestamp_col, "ch1"]].isna().sum().sum())
    summary["step_4_removed_missing"] = int(rows_before_cleaning - rows_after_missing)
    summary["step_4_removed_duplicate_timestamps"] = int(rows_after_missing - len(df))
    print(f"Fehlende Werte nach Bereinigung: {missing_after_cleaning}")
    print(f"Anzahl Duplikate entfernt: {summary['step_4_removed_duplicate_timestamps']}")
    print(f"Verbleibende Zeilen: {len(df)}")

    if len(df) < 2:
        raise ValueError("Nach Bereinigung sind weniger als zwei valide Samples vorhanden.")

    # 5. Zeitstempel sortieren
    print("\n5. Zeitstempel sortieren")
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    print("Zeitstempel sortiert.")

    # 6. Samplingrate prüfen
    print("\n6. Samplingrate prüfen")
    df, sampling_report = _check_sampling_rate(df, timestamp_col)
    fs = sampling_report["estimated_fs_median_hz"]
    summary["step_6_sampling"] = sampling_report
    for message in sampling_report["warnings"]:
        warnings.warn(message)
    duration_seconds = float(df["time_seconds"].iloc[-1])
    ch1_stats = {
        "mean": float(df["ch1"].mean()),
        "std": float(df["ch1"].std()),
        "min": float(df["ch1"].min()),
        "max": float(df["ch1"].max()),
    }
    print(f"Geschätzte Samplingrate: {fs:.2f} Hz")
    print(f"Gesamtdauer der Aufnahme: {duration_seconds:.2f} s")
    print(
        "Mittelwert / Standardabweichung / Min / Max von ch1: "
        f"{ch1_stats['mean']:.6f} / {ch1_stats['std']:.6f} / "
        f"{ch1_stats['min']:.6f} / {ch1_stats['max']:.6f}"
    )

    # 7. Rohsignal visualisieren
    print("\n7. Rohsignal visualisieren")
    _plot_signal(df, "ch1", "Rohsignal", "Zeit", "Amplitude", plots_dir / "raw_signal.png")
    _plot_signal_zoom(df, "ch1", "Rohsignal Zoom", plots_dir / "raw_signal_zoom.png", seconds=DETAIL_SECONDS)
    _plot_distribution(df["ch1"], "Signalverteilung ch1", "Amplitude", plots_dir / "signal_distribution.png")

    # 8. Signal zentrieren / DC-Offset entfernen
    print("\n8. Signal zentrieren / DC-Offset entfernen")
    df["ch1_raw"] = df["ch1"]
    dc_offset = float(df["ch1_raw"].mean())
    df["ch1_centered"] = df["ch1_raw"] - dc_offset
    summary["step_8_dc_offset_removed"] = dc_offset
    print(f"DC-Offset entfernt: {dc_offset:.6f}")

    # 9. Bandpass: 1-40 Hz
    print(f"\n9. Bandpass {BANDPASS_LOW_HZ:g}-{BANDPASS_HIGH_HZ:g} Hz")
    sos = butter(N=BANDPASS_ORDER, Wn=[BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ], btype="bandpass", fs=fs, output="sos")
    df["ch1_bandpassed"] = sosfiltfilt(sos, df["ch1_centered"].to_numpy(dtype=float))
    print("Bandpassfilter erfolgreich angewendet.")

    # 10. Notch: 50 Hz
    print(f"\n10. Notch {NOTCH_FREQ_HZ:g} Hz")
    if NOTCH_FREQ_HZ >= fs / 2.0:
        warnings.warn(f"{NOTCH_FREQ_HZ:g}-Hz-Notch liegt oberhalb Nyquist und wird übersprungen.")
        df["ch1_filtered"] = df["ch1_bandpassed"]
        print(f"Notch-Filter übersprungen, da {NOTCH_FREQ_HZ:g} Hz oberhalb Nyquist liegt.")
    else:
        b, a = iirnotch(w0=NOTCH_FREQ_HZ, Q=NOTCH_QUALITY_FACTOR, fs=fs)
        df["ch1_filtered"] = filtfilt(b, a, df["ch1_bandpassed"].to_numpy(dtype=float))
        print("Notch-Filter erfolgreich angewendet.")
    print("Filter erfolgreich angewendet.")

    # 11. gefiltertes Signal visualisieren
    print("\n11. Gefiltertes Signal visualisieren")
    _plot_signal(df, "ch1_filtered", "Gefiltertes Signal", "Zeit", "Amplitude", plots_dir / "filtered_signal.png")
    _plot_raw_vs_filtered(df, plots_dir / "raw_vs_filtered.png")
    _plot_psd_before_after_filtering(df, fs, plots_dir / "psd_before_after_filtering.png")

    # 12. Fensterbildung: 10 Sekunden, 50 % Overlap
    print("\n12. Fensterbildung")
    windows = _create_windows(df, fs, window_seconds=WINDOW_SECONDS, overlap=WINDOW_OVERLAP)
    summary["step_12_n_windows"] = int(len(windows))
    print(f"Anzahl erzeugter Fenster: {len(windows)}")

    # 13. Artefakt-Metriken pro Fenster berechnen
    print("\n13. Artefakt-Metriken pro Fenster berechnen")
    artifact_df = pd.DataFrame([_compute_artifact_metrics(win) for win in windows])
    print(f"Artefakt-Metriken berechnet für {len(artifact_df)} Fenster.")

    # 14. Artefaktfenster markieren
    print("\n14. Artefaktfenster markieren")
    artifact_thresholds = _artifact_thresholds(artifact_df)
    artifact_df["artifact"] = (
        (artifact_df["p2p"] > artifact_thresholds["p2p_threshold"])
        | (artifact_df["std"] > artifact_thresholds["std_threshold"])
        | (artifact_df["max_abs"] > artifact_thresholds["max_abs_threshold"])
        | (artifact_df["energy"] > artifact_thresholds["energy_threshold"])
    )
    summary["step_14_artifacts"] = {
        **artifact_thresholds,
        "artifact_windows": int(artifact_df["artifact"].sum()),
        "clean_windows": int((~artifact_df["artifact"]).sum()),
    }
    n_artifacts = int(artifact_df["artifact"].sum())
    n_valid = int((~artifact_df["artifact"]).sum())
    artifact_fraction = 100.0 * n_artifacts / max(len(artifact_df), 1)
    print(f"Anzahl Artefaktfenster: {n_artifacts}")
    print(f"Anteil Artefaktfenster in %: {artifact_fraction:.2f}")
    print(f"Anzahl gültiger Fenster: {n_valid}")
    _plot_artifact_windows(artifact_df, plots_dir / "artifact_windows.png")

    # 15. PSD mit Welch nur für gültige Fenster berechnen
    print("\n15. PSD mit Welch nur für gültige Fenster berechnen")
    psd_rows = []
    artifact_lookup = artifact_df.set_index("window_id")["artifact"].to_dict()
    for win in windows:
        if artifact_lookup[win["window_id"]]:
            continue
        freqs, psd = welch(
            win["signal_segment"],
            fs=fs,
            nperseg=min(PSD_NPERSEG_MAX, len(win["signal_segment"])),
            detrend="constant",
            scaling="density",
        )
        psd_rows.append({**_window_metadata(win), "freqs": freqs, "psd": psd})
    psd_df = pd.DataFrame(psd_rows)
    print(f"Anzahl berechneter PSDs: {len(psd_df)}")

    # 16. Bandpower berechnen
    print("\n16. Bandpower berechnen")
    feature_rows = []
    for _, row in psd_df.iterrows():
        absolute = {
            f"absolute_{name}": _bandpower(row["freqs"], row["psd"], band)
            for name, band in FREQUENCY_BANDS.items()
        }
        total_power = _bandpower(row["freqs"], row["psd"], TOTAL_POWER_BAND)

        # 17. relative Bandpower berechnen
        relative = {
            f"relative_{name}": absolute[f"absolute_{name}"] / (total_power + EPS)
            for name in FREQUENCY_BANDS
        }

        # 18. Ratios berechnen
        ratios = {
            "theta_alpha_ratio": absolute["absolute_theta"] / (absolute["absolute_alpha"] + EPS),
            "alpha_theta_ratio": absolute["absolute_alpha"] / (absolute["absolute_theta"] + EPS),
            "beta_alpha_ratio": absolute["absolute_beta"] / (absolute["absolute_alpha"] + EPS),
        }

        win = windows[int(row["window_id"])]
        signal = win["signal_segment"]
        feature_rows.append(
            {
                "window_id": int(row["window_id"]),
                "start_sample": int(row["start_sample"]),
                "end_sample": int(row["end_sample"]),
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "artifact": False,
                **absolute,
                "total_power_1_40hz": total_power,
                **relative,
                **ratios,
                "signal_mean": float(np.mean(signal)),
                "signal_std": float(np.std(signal)),
                "signal_variance": float(np.var(signal)),
                "peak_to_peak": float(np.ptp(signal)),
                "max_abs": float(np.max(np.abs(signal))),
                "spectral_entropy": _spectral_entropy(row["psd"]),
            }
        )

    clean_features_df = pd.DataFrame(feature_rows)
    print(f"Absolute Bandpower berechnet für {len(clean_features_df)} gültige Fenster.")
    print("\n17. Relative Bandpower berechnen")
    print("Relative Bandpower berechnet.")
    print("\n18. Ratios berechnen")
    print("Theta/Alpha, Alpha/Theta und Beta/Alpha Ratios berechnet.")

    # 19. optionale Baseline-Normalisierung
    print("\n19. Optionale Baseline-Normalisierung")
    clean_features_df, baseline_report = _baseline_normalize(clean_features_df, baseline_minutes)
    summary["step_19_baseline"] = baseline_report
    print(
        "Baseline-Zeitraum: "
        f"{baseline_report['baseline_start']} bis {baseline_report['baseline_end']} "
        f"({baseline_report['baseline_windows']} Fenster)"
    )

    # 20. Cognitive Load Proxy Score berechnen
    print("\n20. Cognitive Load Proxy Score berechnen")
    clean_features_df["Cognitive_Load_Proxy_Score"] = _compute_proxy_score(clean_features_df)
    clean_features_df, score_thresholds = _assign_load_states(clean_features_df)
    summary["step_20_score_thresholds"] = score_thresholds
    print("Cognitive Load Score berechnet.")

    # 21. Score glätten
    print("\n21. Score glätten")
    if not 3 <= rolling_windows <= 6:
        raise ValueError("rolling_windows muss zwischen 3 und 6 liegen.")
    clean_features_df["Cognitive_Load_Proxy_Score_Smoothed"] = (
        clean_features_df["Cognitive_Load_Proxy_Score"]
        .rolling(window=rolling_windows, min_periods=1, center=True)
        .mean()
    )
    print(f"Cognitive Load Score geglättet mit Rolling Mean über {rolling_windows} Fenster.")

    # 22. Feature-/Score-Tabelle erzeugen
    print("\n22. Feature-/Score-Tabelle erzeugen")
    artifact_feature_rows = artifact_df[artifact_df["artifact"]][
        ["window_id", "start_sample", "end_sample", "start_time", "end_time", "artifact"]
    ].copy()
    artifact_feature_rows["cognitive_load_state"] = "artifact"
    features_df = pd.concat([clean_features_df, artifact_feature_rows], ignore_index=True)
    features_df = features_df.sort_values("window_id").reset_index(drop=True)

    features_df, deviation_report = _add_eeg_deviation_index(features_df)

    if external_score_available:
        print("\n22b. Externen Workload-Score pro Fenster zuordnen und validieren")
        features_df, external_report = _attach_and_validate_external_score(
            signal_df=df,
            features_df=features_df,
            windows=windows,
            score_thresholds=score_thresholds,
            external_score_column=external_score_column,
            external_score_type=external_score_type,
            high_workload_threshold=high_workload_threshold,
            plots_dir=plots_dir,
        )
        summary["external_score_validation"] = external_report
        print(f"Externe Score-Fenster mit validem Wert: {external_report['valid_external_score_windows']}")
        print(f"Pearson r: {external_report['continuous_metrics']['pearson_r']}")
        print(f"Spearman rho: {external_report['continuous_metrics']['spearman_rho']}")
    else:
        summary["external_score_validation"] = {
            "available": False,
            "message": "Kein externer Workload-Score vorhanden. Pipeline wurde ohne externe Validierung ausgefuehrt.",
        }

    score_df = features_df[
        [col for col in [
            "window_id",
            "start_time",
            "end_time",
            "artifact",
            "Cognitive_Load_Proxy_Score",
            "Cognitive_Load_Proxy_Score_Smoothed",
            "cognitive_load_state",
            external_score_column,
            "high_workload",
            "predicted_high_workload_proxy",
        ] if col in features_df.columns]
    ].copy()
    print(f"Anzahl erzeugter Feature-Zeilen: {len(features_df)}")
    _plot_bandpower_over_time(clean_features_df, plots_dir / "bandpower_over_time.png")
    _plot_cognitive_load_score(clean_features_df, plots_dir / "cognitive_load_score.png")
    _plot_smoothed_cognitive_load_score(clean_features_df, plots_dir / "smoothed_cognitive_load_score.png")

    print("\nErweiterte wissenschaftliche Analyse")
    extended_analysis = _compute_extended_analysis(
        signal_df=df,
        artifact_df=artifact_df,
        features_df=features_df,
        clean_features_df=clean_features_df,
        deviation_report=deviation_report,
    )
    _create_extended_analysis_plots(features_df, plots_dir)
    report_paths = _export_extended_reports(extended_analysis, reports_dir)
    for path in report_paths.values():
        print("Gespeichert:", path)

    # 23. Export
    print("\n23. Export")
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    signal_path = processed_dir / f"{prefix}_cleaned_filtered_signal.csv"
    artifact_path = artifacts_dir / f"{prefix}_artifact_windows.csv"
    features_path = features_dir / f"{prefix}_features_scores.csv"
    score_path = features_dir / f"{prefix}_score_table.csv"
    summary_path = summary_dir / f"{prefix}_summary.json"

    df.to_csv(signal_path, index=False)
    print("Gespeichert:", signal_path)
    artifact_df.to_csv(artifact_path, index=False)
    print("Gespeichert:", artifact_path)
    features_df.to_csv(features_path, index=False)
    print("Gespeichert:", features_path)
    score_df.to_csv(score_path, index=False)
    print("Gespeichert:", score_path)

    summary["exports"] = {
        "cleaned_filtered_signal": str(signal_path),
        "artifact_windows": str(artifact_path),
        "features_scores": str(features_path),
        "score_table": str(score_path),
        "extended_analysis_report_txt": str(report_paths["txt"]),
        "extended_analysis_report_json": str(report_paths["json"]),
    }
    summary["extended_analysis"] = extended_analysis
    summary["method_notes"] = [
        "Welch PSD statt roher FFT.",
        "Relative Bandpower wird bevorzugt interpretiert.",
        "Artefakte werden markiert, nicht sofort gelöscht.",
        "PSD und Feature-Extraktion erfolgen ausschliesslich auf artifact=False Fenstern.",
        "Cognitive Load Proxy Score ist eine Baseline-Abweichung, kein klassischer Klassifikator.",
        "NASA-TLX/externe Workload-Scores werden nie als EEG-Merkmal verwendet, sondern nur als externe Validierung.",
        "Ohne externen Referenzscore wird kein Cognitive Overload behauptet.",
        "Visualisierungen werden ausschliesslich mit Matplotlib erzeugt.",
    ]
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Gespeichert:", summary_path)
    print("\nPipeline abgeschlossen.")

    return df, artifact_df, features_df, summary


def _validate_columns(df: pd.DataFrame) -> str:
    timestamp_col = "timestamps" if "timestamps" in df.columns else "timestamp" if "timestamp" in df.columns else None
    if timestamp_col is None:
        raise ValueError("Erwartete Zeitspalte fehlt: 'timestamp' oder 'timestamps'.")
    if "ch1" not in df.columns:
        raise ValueError("Erwartete EEG-Spalte fehlt: 'ch1'.")
    return timestamp_col


def _check_sampling_rate(df: pd.DataFrame, timestamp_col: str) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    median_ts = float(np.nanmedian(df[timestamp_col]))
    unit = "ms" if median_ts > 1e11 else "s"
    df["timestamp_seconds"] = df[timestamp_col] / 1000.0 if unit == "ms" else df[timestamp_col]
    df["datetime"] = pd.to_datetime(df["timestamp_seconds"], unit="s", errors="coerce")
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    df["time_seconds"] = df["timestamp_seconds"] - df["timestamp_seconds"].iloc[0]
    df["sample_interval"] = df["timestamp_seconds"].diff()

    intervals = df["sample_interval"].dropna().to_numpy(dtype=float)
    intervals = intervals[intervals > 0]
    median_dt = float(np.median(intervals))
    mean_dt = float(np.mean(intervals))
    fs = float(1.0 / median_dt)

    warnings_list = []
    tolerance = float(SAMPLING_CONFIG["fs_tolerance_fraction"])
    if not (EXPECTED_FS * (1.0 - tolerance) <= fs <= EXPECTED_FS * (1.0 + tolerance)):
        warnings_list.append(f"Samplingrate {fs:.2f} Hz weicht von ca. 250 Hz ab.")

    large_gap_min = float(SAMPLING_CONFIG["large_gap_min_seconds"])
    large_gap_factor = float(SAMPLING_CONFIG["large_gap_expected_dt_factor"])
    large_gaps = intervals[intervals > max(large_gap_min, large_gap_factor / EXPECTED_FS)]
    if len(large_gaps):
        warnings_list.append(f"{len(large_gaps)} grössere Zeitlücken erkannt.")

    return df, {
        "timestamp_unit": unit,
        "mean_dt": mean_dt,
        "median_dt": median_dt,
        "estimated_fs_median_hz": fs,
        "estimated_fs_mean_hz": float(1.0 / mean_dt),
        "warnings": warnings_list,
    }


def _plot_signal(
    df: pd.DataFrame,
    col: str,
    title: str,
    x_label: str,
    y_label: str,
    path: Path,
) -> None:
    """Create and save a Matplotlib line plot for a signal column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(df["datetime"], df[col], linewidth=0.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_signal_zoom(
    df: pd.DataFrame,
    col: str,
    title: str,
    path: Path,
    seconds: float = 20.0,
) -> None:
    """Create and save a zoomed signal plot for the first seconds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    zoom_df = df[df["time_seconds"] <= seconds]
    plt.figure(figsize=(14, 5))
    plt.plot(zoom_df["datetime"], zoom_df[col], linewidth=0.9)
    plt.title(title)
    plt.xlabel("Zeit")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_distribution(values: pd.Series, title: str, x_label: str, path: Path) -> None:
    """Create and save a histogram for signal distribution."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(values.dropna(), bins=100, color="#4C78A8", alpha=0.85)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Anzahl")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_raw_vs_filtered(df: pd.DataFrame, path: Path) -> None:
    """Create and save raw vs filtered signal comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(df["datetime"], df["ch1_raw"], linewidth=0.6, alpha=0.45, label="Rohsignal")
    plt.plot(df["datetime"], df["ch1_filtered"], linewidth=0.8, alpha=0.90, label="Gefiltert")
    plt.title("Rohsignal vs. gefiltertes Signal")
    plt.xlabel("Zeit")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_psd_before_after_filtering(df: pd.DataFrame, fs: float, path: Path) -> None:
    """Create and save Welch PSD before and after filtering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    freqs_before, psd_before = welch(
        df["ch1_centered"].to_numpy(dtype=float),
        fs=fs,
        nperseg=min(PSD_FULL_SIGNAL_NPERSEG_MAX, len(df)),
        detrend="constant",
        scaling="density",
    )
    freqs_after, psd_after = welch(
        df["ch1_filtered"].to_numpy(dtype=float),
        fs=fs,
        nperseg=min(PSD_FULL_SIGNAL_NPERSEG_MAX, len(df)),
        detrend="constant",
        scaling="density",
    )
    plt.figure(figsize=(12, 5))
    plt.semilogy(freqs_before, psd_before, linewidth=0.8, alpha=0.65, label="Vor Filterung")
    plt.semilogy(freqs_after, psd_after, linewidth=0.9, alpha=0.90, label="Nach Filterung")
    plt.xlim(0, min(80, fs / 2.0))
    plt.title("PSD vor/nach Filterung")
    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("PSD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_artifact_windows(artifact_df: pd.DataFrame, path: Path) -> None:
    """Create and save artifact windows over time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.where(artifact_df["artifact"], "#E45756", "#4C78A8")
    plt.figure(figsize=(14, 5))
    plt.scatter(artifact_df["start_time"], artifact_df["p2p"], c=colors, s=18)
    plt.title("Artefaktfenster über Zeit")
    plt.xlabel("Zeit")
    plt.ylabel("Peak-to-Peak-Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_bandpower_over_time(features_df: pd.DataFrame, path: Path) -> None:
    """Create and save relative bandpower trajectories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    for col in ["relative_delta", "relative_theta", "relative_alpha", "relative_beta", "relative_gamma"]:
        if col in features_df.columns:
            plt.plot(features_df["start_time"], features_df[col], linewidth=0.9, label=col)
    plt.title("Relative Bandpower über Zeit")
    plt.xlabel("Zeit")
    plt.ylabel("Relative Bandpower")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_cognitive_load_score(features_df: pd.DataFrame, path: Path) -> None:
    """Create and save Cognitive Load Proxy Score trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(features_df["start_time"], features_df["Cognitive_Load_Proxy_Score"], linewidth=0.9)
    plt.title("Cognitive Load Proxy Score über Zeit")
    plt.xlabel("Zeit")
    plt.ylabel("Proxy Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_smoothed_cognitive_load_score(features_df: pd.DataFrame, path: Path) -> None:
    """Create and save smoothed Cognitive Load Proxy Score trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(
        features_df["start_time"],
        features_df["Cognitive_Load_Proxy_Score_Smoothed"],
        linewidth=1.2,
        color="#F58518",
    )
    plt.title("Geglätteter Cognitive Load Proxy Score über Zeit")
    plt.xlabel("Zeit")
    plt.ylabel("Geglätteter Proxy Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _show_plot() -> None:
    """Call plt.show() while keeping non-interactive Matplotlib runs quiet."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")
        plt.show()


def _compute_proxy_score(features_df: pd.DataFrame) -> pd.Series:
    """Compute the configured baseline-relative Cognitive Load Proxy Score."""
    score = pd.Series(0.0, index=features_df.index, dtype=float)
    for z_col, weight in PROXY_CONFIG["weights"].items():
        if z_col not in features_df.columns:
            raise ValueError(f"Z-Score-Spalte fehlt fuer Proxy Score: {z_col}")
        score = score + float(weight) * features_df[z_col]
    return score


def _attach_and_validate_external_score(
    signal_df: pd.DataFrame,
    features_df: pd.DataFrame,
    windows: list[dict],
    score_thresholds: dict,
    external_score_column: str,
    external_score_type: str,
    high_workload_threshold: float,
    plots_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Attach an external workload reference to windows and validate the EEG proxy.

    The external score is never used as an EEG feature. It is aggregated per
    window and used only as ground-truth/reference information after the EEG
    proxy score already exists.
    """
    df = features_df.copy()
    window_scores = []
    for win in windows:
        start_sample = int(win["start_sample"])
        end_sample = int(win["end_sample"])
        values = signal_df[external_score_column].iloc[start_sample:end_sample]
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            score = np.nan
        elif external_score_type == "binary":
            score = float(values.mode().iloc[0]) if not values.mode().empty else float(values.median())
        else:
            score = float(values.median())
        window_scores.append({"window_id": int(win["window_id"]), external_score_column: score})

    score_map = pd.DataFrame(window_scores)
    df = df.drop(columns=[external_score_column], errors="ignore")
    df = df.merge(score_map, on="window_id", how="left")

    df["high_workload"] = df[external_score_column] >= float(high_workload_threshold)
    elevated_threshold = float(score_thresholds["elevated_threshold"])
    df["predicted_high_workload_proxy"] = df["Cognitive_Load_Proxy_Score"] >= elevated_threshold
    df.loc[df["artifact"] == True, "predicted_high_workload_proxy"] = np.nan

    valid = df[
        (df["artifact"] == False)
        & df[external_score_column].notna()
        & df["Cognitive_Load_Proxy_Score"].notna()
    ].copy()

    continuous_metrics = _external_continuous_metrics(
        valid["Cognitive_Load_Proxy_Score"],
        valid[external_score_column],
    )
    binary_metrics = _external_binary_metrics(
        valid["predicted_high_workload_proxy"],
        valid["high_workload"],
        valid["Cognitive_Load_Proxy_Score"],
    )

    _plot_external_score_scatter(
        valid,
        external_score_column,
        plots_dir / "external_score_vs_eeg_proxy.png",
    )
    _plot_external_confusion_matrix(
        binary_metrics["confusion_matrix"],
        plots_dir / "external_score_confusion_matrix.png",
    )

    return df, {
        "available": True,
        "external_score_column": external_score_column,
        "external_score_type": external_score_type,
        "high_workload_threshold": float(high_workload_threshold),
        "valid_external_score_windows": int(valid[external_score_column].notna().sum()),
        "validation_windows": int(len(valid)),
        "continuous_metrics": continuous_metrics,
        "binary_metrics": binary_metrics,
        "method_note": (
            "Der externe Score wurde nicht als EEG-Merkmal verwendet, sondern nur "
            "als Referenz/Ground Truth zur Validierung des Cognitive_Load_Proxy_Score."
        ),
    }


def _external_continuous_metrics(proxy: pd.Series, external: pd.Series) -> dict:
    """Compute correlation and scale-aware error metrics for an external score."""
    pair = pd.DataFrame({"proxy": proxy, "external": external}).dropna()
    if len(pair) < 3:
        return {
            "n": int(len(pair)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "mae_proxy_scaled_to_external": np.nan,
            "rmse_proxy_scaled_to_external": np.nan,
            "r2_proxy_scaled_to_external": np.nan,
        }

    if pair["proxy"].nunique() < 2 or pair["external"].nunique() < 2:
        pearson_r_value, pearson_p_value = np.nan, np.nan
        spearman_r_value, spearman_p_value = np.nan, np.nan
    else:
        pearson_r_value, pearson_p_value = pearsonr(pair["proxy"], pair["external"])
        spearman_r_value, spearman_p_value = spearmanr(pair["proxy"], pair["external"])

    proxy_scaled = _minmax_scale_to_reference(pair["proxy"], pair["external"])
    residual = proxy_scaled - pair["external"]
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    ss_res = float(np.sum(np.square(pair["external"] - proxy_scaled)))
    ss_tot = float(np.sum(np.square(pair["external"] - pair["external"].mean())))
    r2 = float(1.0 - ss_res / (ss_tot + EPS)) if ss_tot > 0 else np.nan

    return {
        "n": int(len(pair)),
        "pearson_r": float(pearson_r_value) if np.isfinite(pearson_r_value) else np.nan,
        "pearson_p": float(pearson_p_value) if np.isfinite(pearson_p_value) else np.nan,
        "spearman_rho": float(spearman_r_value) if np.isfinite(spearman_r_value) else np.nan,
        "spearman_p": float(spearman_p_value) if np.isfinite(spearman_p_value) else np.nan,
        "mae_proxy_scaled_to_external": mae,
        "rmse_proxy_scaled_to_external": rmse,
        "r2_proxy_scaled_to_external": r2,
        "error_metric_note": (
            "MAE/RMSE/R2 nutzen den min-max skalierten Proxy im Wertebereich des externen Scores, "
            "weil der EEG-Proxy ein baseline-relativer Index ist."
        ),
    }


def _external_binary_metrics(
    predicted: pd.Series,
    actual: pd.Series,
    proxy_score: pd.Series,
) -> dict:
    """Compute binary validation metrics for high workload labels."""
    pair = pd.DataFrame({"pred": predicted, "actual": actual, "score": proxy_score}).dropna()
    if pair.empty:
        return _empty_binary_metrics()

    y_true = pair["actual"].astype(bool).to_numpy()
    y_pred = pair["pred"].astype(bool).to_numpy()

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    roc_auc = _binary_roc_auc(y_true, pair["score"].to_numpy(dtype=float))

    return {
        "n": int(len(pair)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "roc_auc_note": "ROC-AUC wird nur berechnet, wenn beide Klassen vorhanden sind.",
    }


def _empty_binary_metrics() -> dict:
    return {
        "n": 0,
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": np.nan,
        "roc_auc": np.nan,
    }


def _binary_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC from ranks without adding a sklearn dependency."""
    y_true = np.asarray(y_true, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if np.sum(y_true) == 0 or np.sum(~y_true) == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    n_pos = float(np.sum(y_true))
    n_neg = float(np.sum(~y_true))
    rank_sum_pos = float(np.sum(ranks[y_true]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg + EPS)
    return float(auc)


def _minmax_scale_to_reference(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Scale values to the min/max range of a reference series."""
    v_min = float(values.min())
    v_max = float(values.max())
    r_min = float(reference.min())
    r_max = float(reference.max())
    if not np.isfinite(v_max - v_min) or abs(v_max - v_min) <= EPS:
        return pd.Series(np.full(len(values), reference.mean()), index=values.index)
    return r_min + (values - v_min) * (r_max - r_min) / (v_max - v_min + EPS)


def _plot_external_score_scatter(features_df: pd.DataFrame, score_col: str, path: Path) -> None:
    """Plot EEG proxy score against the external workload reference."""
    if features_df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    colors = np.where(features_df["high_workload"], "#E45756", "#4C78A8")
    plt.scatter(
        features_df[score_col],
        features_df["Cognitive_Load_Proxy_Score"],
        c=colors,
        alpha=0.75,
        edgecolor="none",
    )
    plt.title("EEG Proxy Score vs. externer Workload-Score")
    plt.xlabel(score_col)
    plt.ylabel("Cognitive_Load_Proxy_Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_external_confusion_matrix(confusion: dict, path: Path) -> None:
    """Plot a compact confusion matrix for external high-workload labels."""
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix: Proxy vs. externer Score")
    plt.xticks([0, 1], ["Proxy low", "Proxy high"])
    plt.yticks([0, 1], ["Extern low", "Extern high"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    plt.colorbar(label="Fenster")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _add_eeg_deviation_index(features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add score_z, rolling stability metrics and EEG Deviation Index."""
    df = features_df.copy()
    clean_mask = df["artifact"] == False
    if "baseline_window" in df.columns:
        baseline_col = df["baseline_window"].fillna(False).astype(bool)
    else:
        baseline_col = pd.Series(False, index=df.index)
    baseline_mask = clean_mask & baseline_col
    if not baseline_mask.any():
        baseline_mask = clean_mask

    score_mean = float(df.loc[baseline_mask, "Cognitive_Load_Proxy_Score"].mean())
    score_std = float(df.loc[baseline_mask, "Cognitive_Load_Proxy_Score"].std())
    if not np.isfinite(score_std) or score_std == 0:
        score_std = EPS

    df["score_z"] = np.nan
    df.loc[clean_mask, "score_z"] = (
        df.loc[clean_mask, "Cognitive_Load_Proxy_Score"] - score_mean
    ) / (score_std + EPS)

    df["rolling_std_score"] = (
        df["Cognitive_Load_Proxy_Score"].rolling(window=int(PROXY_CONFIG["rolling_std_windows"]), min_periods=2, center=True).std()
    )
    df["rolling_std_theta_alpha_ratio"] = (
        df["theta_alpha_ratio"].rolling(window=int(PROXY_CONFIG["rolling_std_windows"]), min_periods=2, center=True).std()
    )
    df["rolling_std_relative_theta"] = (
        df["relative_theta"].rolling(window=int(PROXY_CONFIG["rolling_std_windows"]), min_periods=2, center=True).std()
    )

    df["EEG_Deviation_Index"] = (
        0.25 * df["theta_alpha_ratio_z"].abs()
        + 0.20 * df["relative_theta_z"].abs()
        + 0.20 * df["relative_alpha_z"].abs()
        + 0.20 * df["spectral_entropy_z"].abs()
        + 0.15 * df["score_z"].abs()
    )

    baseline_index = df.loc[baseline_mask, "EEG_Deviation_Index"].dropna()
    if baseline_index.empty:
        index_mean = 0.0
        index_std = EPS
    else:
        index_mean = float(baseline_index.mean())
        index_std = float(baseline_index.std())
        if not np.isfinite(index_std) or index_std == 0:
            index_std = EPS

    moderate_threshold = index_mean + index_std
    high_threshold = index_mean + 2.0 * index_std
    very_high_threshold = index_mean + 3.0 * index_std

    df["EEG_Deviation_Index_Level"] = "niedrig"
    df.loc[df["EEG_Deviation_Index"] > moderate_threshold, "EEG_Deviation_Index_Level"] = "moderat"
    df.loc[df["EEG_Deviation_Index"] > high_threshold, "EEG_Deviation_Index_Level"] = "hoch"
    df.loc[df["EEG_Deviation_Index"] > very_high_threshold, "EEG_Deviation_Index_Level"] = "stark auffällig"
    df.loc[df["artifact"] == True, "EEG_Deviation_Index_Level"] = "artifact"

    return df, {
        "score_z_baseline_mean": score_mean,
        "score_z_baseline_std": score_std,
        "index_baseline_mean": index_mean,
        "index_baseline_std": index_std,
        "moderate_threshold": moderate_threshold,
        "high_threshold": high_threshold,
        "very_high_threshold": very_high_threshold,
    }


def _compute_extended_analysis(
    signal_df: pd.DataFrame,
    artifact_df: pd.DataFrame,
    features_df: pd.DataFrame,
    clean_features_df: pd.DataFrame,
    deviation_report: dict,
) -> dict:
    """Compute extended scientific-but-non-diagnostic EEG analysis metrics."""
    clean_df = features_df[features_df["artifact"] == False].copy()
    total_windows = len(artifact_df)
    artifact_windows = int(artifact_df["artifact"].sum())
    valid_windows = int(total_windows - artifact_windows)
    artifact_percent = 100.0 * artifact_windows / max(total_windows, 1)
    valid_percent = 100.0 * valid_windows / max(total_windows, 1)

    noise_proxy = signal_df["ch1_centered"] - signal_df["ch1_filtered"]
    noise_std = float(noise_proxy.std())
    signal_std = float(signal_df["ch1_filtered"].std())
    snr_proxy = signal_std / (noise_std + EPS)
    stability_denominator = float(clean_df["signal_mean"].std()) if "signal_mean" in clean_df else np.nan
    stability_score = 1.0 / (stability_denominator + EPS) if np.isfinite(stability_denominator) else np.nan

    signal_quality = {
        "signal_to_noise_proxy": snr_proxy,
        "signal_std": signal_std,
        "noise_std": noise_std,
        "valid_windows_percent": valid_percent,
        "artifact_windows_percent": artifact_percent,
        "average_peak_to_peak_amplitude": float(artifact_df["p2p"].mean()),
        "average_signal_energy": float(artifact_df["energy"].mean()),
        "stability_score": stability_score,
        "quality_rating": _rate_signal_quality(artifact_percent, snr_proxy, stability_score),
    }

    feature_stat_cols = [
        "relative_theta",
        "relative_alpha",
        "relative_beta",
        "theta_alpha_ratio",
        "beta_alpha_ratio",
    ]
    feature_statistics = {col: _descriptive_stats(clean_df[col]) for col in feature_stat_cols if col in clean_df}

    dynamics_cols = [
        "relative_theta",
        "relative_alpha",
        "theta_alpha_ratio",
        "Cognitive_Load_Proxy_Score",
    ]
    feature_dynamics = {}
    for col in dynamics_cols:
        if col not in clean_df:
            continue
        delta = clean_df[col].diff().dropna()
        feature_dynamics[col] = {
            "mean_change": float(delta.mean()),
            "mean_absolute_change": float(delta.abs().mean()),
            "max_change": float(delta.abs().max()),
            "std_change": float(delta.std()),
        }

    score = clean_df["Cognitive_Load_Proxy_Score"].dropna()
    cognitive_load_analysis = {
        "mean_score": float(score.mean()),
        "median_score": float(score.median()),
        "std_score": float(score.std()),
        "min_score": float(score.min()),
        "max_score": float(score.max()),
        "p90_score": float(score.quantile(0.90)),
        "iqr_score": float(score.quantile(0.75) - score.quantile(0.25)),
        "skewness": float(skew(score, nan_policy="omit")) if len(score) > 2 else np.nan,
        "kurtosis": float(kurtosis(score, nan_policy="omit")) if len(score) > 3 else np.nan,
        "score_variability": float(score.std() / (abs(score.mean()) + EPS)),
    }

    high_load_phases = _high_load_phase_metrics(clean_df)

    corr_cols = [
        "relative_theta",
        "relative_alpha",
        "relative_beta",
        "theta_alpha_ratio",
        "beta_alpha_ratio",
        "spectral_entropy",
        "Cognitive_Load_Proxy_Score",
    ]
    corr_df = clean_df[[col for col in corr_cols if col in clean_df]].dropna()
    pearson_corr = corr_df.corr(method="pearson")
    spearman_corr = corr_df.corr(method="spearman")
    correlation_analysis = {
        "pearson": pearson_corr.to_dict(),
        "spearman": spearman_corr.to_dict(),
        "top_pearson_correlations": _top_correlations(pearson_corr),
        "top_spearman_correlations": _top_correlations(spearman_corr),
    }

    feature_importance = _feature_score_correlations(clean_df, corr_cols)
    baseline_deviation = _baseline_deviation_metrics(clean_df)

    temporal_stability = {
        "rolling_window_count": 5,
        "mean_rolling_std_score": float(clean_df["rolling_std_score"].mean()),
        "max_rolling_std_score": float(clean_df["rolling_std_score"].max()),
        "mean_rolling_std_theta_alpha_ratio": float(clean_df["rolling_std_theta_alpha_ratio"].mean()),
        "mean_rolling_std_relative_theta": float(clean_df["rolling_std_relative_theta"].mean()),
    }

    eeg_deviation = {
        **deviation_report,
        "mean_index": float(clean_df["EEG_Deviation_Index"].mean()),
        "median_index": float(clean_df["EEG_Deviation_Index"].median()),
        "max_index": float(clean_df["EEG_Deviation_Index"].max()),
        "level_distribution_percent": (
            clean_df["EEG_Deviation_Index_Level"].value_counts(normalize=True).mul(100).to_dict()
        ),
    }

    automatic_interpretation = _automatic_interpretation(
        signal_quality,
        cognitive_load_analysis,
        high_load_phases,
        feature_dynamics,
        eeg_deviation,
    )

    return {
        "SIGNAL QUALITY": signal_quality,
        "FEATURE STATISTICS": feature_statistics,
        "FEATURE DYNAMICS": feature_dynamics,
        "COGNITIVE LOAD ANALYSIS": cognitive_load_analysis,
        "HIGH LOAD PHASES": high_load_phases,
        "CORRELATION ANALYSIS": correlation_analysis,
        "FEATURE IMPORTANCE FOR SCORE": feature_importance,
        "BASELINE DEVIATION": baseline_deviation,
        "TEMPORAL STABILITY": temporal_stability,
        "EEG DEVIATION INDEX": eeg_deviation,
        "AUTOMATIC INTERPRETATION": automatic_interpretation,
        "IMPORTANT LIMITATIONS": [
            "Keine medizinische Diagnose.",
            "Cognitive Overload wird ohne Labels oder Fragebögen nicht bewiesen.",
            "Alle Aussagen sind baseline-relative EEG-Abweichungen.",
        ],
    }


def _rate_signal_quality(artifact_percent: float, snr_proxy: float, stability_score: float) -> str:
    """Heuristic non-diagnostic signal-quality rating."""
    if artifact_percent <= 5 and snr_proxy >= 2.0:
        return "sehr gut"
    if artifact_percent <= 15 and snr_proxy >= 1.2:
        return "gut"
    if artifact_percent <= 30 and snr_proxy >= 0.7:
        return "brauchbar"
    return "kritisch"


def _descriptive_stats(series: pd.Series) -> dict:
    clean = series.dropna()
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
    }


def _high_load_phase_metrics(clean_df: pd.DataFrame) -> dict:
    """Detect contiguous strongly elevated phases."""
    phases = []
    current_rows = []
    sorted_df = clean_df.sort_values("window_id")

    for _, row in sorted_df.iterrows():
        is_high = row.get("cognitive_load_state") == "stark erhöht"
        if is_high:
            current_rows.append(row)
        elif current_rows:
            phases.append(pd.DataFrame(current_rows))
            current_rows = []
    if current_rows:
        phases.append(pd.DataFrame(current_rows))

    phase_summaries = []
    total_high_duration = 0.0
    for idx, phase_df in enumerate(phases):
        start = phase_df["start_time"].iloc[0]
        end = phase_df["end_time"].iloc[-1]
        duration = float((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds())
        total_high_duration += max(duration, 0.0)
        phase_summaries.append(
            {
                "phase_id": idx,
                "start_time": start,
                "end_time": end,
                "duration_seconds": duration,
                "mean_score": float(phase_df["Cognitive_Load_Proxy_Score"].mean()),
                "max_score": float(phase_df["Cognitive_Load_Proxy_Score"].max()),
                "n_windows": int(len(phase_df)),
            }
        )

    if phase_summaries:
        durations = [phase["duration_seconds"] for phase in phase_summaries]
        mean_scores = [phase["mean_score"] for phase in phase_summaries]
        max_scores = [phase["max_score"] for phase in phase_summaries]
    else:
        durations = [0.0]
        mean_scores = [np.nan]
        max_scores = [np.nan]

    recording_duration = float(
        (pd.Timestamp(clean_df["end_time"].max()) - pd.Timestamp(clean_df["start_time"].min())).total_seconds()
    )
    return {
        "n_high_load_phases": int(len(phase_summaries)),
        "average_duration_seconds": float(np.nanmean(durations)),
        "max_duration_seconds": float(np.nanmax(durations)),
        "average_score_during_high_load": float(np.nanmean(mean_scores)),
        "max_score_during_high_load": float(np.nanmax(max_scores)),
        "recording_fraction_high_load_percent": float(100.0 * total_high_duration / (recording_duration + EPS)),
        "phases": phase_summaries,
    }


def _top_correlations(corr: pd.DataFrame, n: int = 10) -> list[dict]:
    """Return strongest absolute off-diagonal correlations."""
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            value = corr.loc[a, b]
            if pd.isna(value):
                continue
            rows.append({"feature_a": a, "feature_b": b, "correlation": float(value)})
    return sorted(rows, key=lambda item: abs(item["correlation"]), reverse=True)[:n]


def _feature_score_correlations(clean_df: pd.DataFrame, candidate_cols: list[str]) -> list[dict]:
    """Rank feature-score associations."""
    rows = []
    score_col = "Cognitive_Load_Proxy_Score"
    for col in candidate_cols:
        if col == score_col or col not in clean_df:
            continue
        pair = clean_df[[col, score_col]].dropna()
        if len(pair) < 3:
            continue
        rows.append(
            {
                "feature": col,
                "pearson_correlation_with_score": float(pair[col].corr(pair[score_col], method="pearson")),
                "spearman_correlation_with_score": float(pair[col].corr(pair[score_col], method="spearman")),
            }
        )
    return sorted(rows, key=lambda item: abs(item["pearson_correlation_with_score"]), reverse=True)


def _baseline_deviation_metrics(clean_df: pd.DataFrame) -> dict:
    """Compute absolute baseline deviations for z-scored features."""
    z_cols = [
        "theta_alpha_ratio_z",
        "beta_alpha_ratio_z",
        "relative_theta_z",
        "relative_alpha_z",
        "spectral_entropy_z",
        "score_z",
    ]
    results = {}
    for col in z_cols:
        if col not in clean_df:
            continue
        abs_dev = clean_df[col].abs()
        if abs_dev.dropna().empty:
            continue
        max_idx = abs_dev.idxmax()
        results[col.replace("_z", "")] = {
            "mean_deviation": float(abs_dev.mean()),
            "max_deviation": float(abs_dev.max()),
            "window_with_max_deviation": int(clean_df.loc[max_idx, "window_id"]),
            "time_with_max_deviation": clean_df.loc[max_idx, "start_time"],
        }
    return results


def _automatic_interpretation(
    signal_quality: dict,
    cognitive_load: dict,
    high_load: dict,
    dynamics: dict,
    eeg_deviation: dict,
) -> str:
    """Generate cautious baseline-relative interpretation text."""
    dynamic_ranking = sorted(
        dynamics.items(),
        key=lambda item: abs(item[1].get("mean_absolute_change", 0.0)),
        reverse=True,
    )
    strongest = ", ".join([item[0] for item in dynamic_ranking[:2]]) if dynamic_ranking else "keine eindeutigen Features"
    level_distribution = eeg_deviation.get("level_distribution_percent", {})
    dominant_level = max(level_distribution, key=level_distribution.get) if level_distribution else "nicht bestimmbar"
    return (
        f"Die EEG-Aufnahme zeigt einen Artefaktanteil von "
        f"{signal_quality['artifact_windows_percent']:.2f} %. "
        f"Die mittlere baseline-relative Aktivierung lag bei "
        f"{cognitive_load['mean_score']:.3f}. "
        f"Es wurden {high_load['n_high_load_phases']} zusammenhängende stark erhöhte Phasen erkannt. "
        f"Die stärksten zeitlichen Veränderungen zeigten {strongest}. "
        f"Der EEG Deviation Index liegt überwiegend im Bereich '{dominant_level}'. "
        "Diese Aussagen beschreiben nur baseline-relative EEG-Abweichungen und beweisen keinen Cognitive Overload."
    )


def _create_extended_analysis_plots(features_df: pd.DataFrame, plots_dir: Path) -> None:
    """Create additional Matplotlib plots for the extended report."""
    clean_df = features_df[features_df["artifact"] == False].copy()
    _plot_score_histogram(clean_df, plots_dir / "cognitive_load_score_histogram.png")
    _plot_feature_boxplots(clean_df, plots_dir / "important_feature_boxplots.png")
    _plot_correlation_heatmap(clean_df, plots_dir / "correlation_heatmap.png")
    _plot_score_distribution(clean_df, plots_dir / "score_distribution.png")
    _plot_rolling_std(clean_df, plots_dir / "rolling_standard_deviation.png")
    _plot_feature_dynamics(clean_df, plots_dir / "feature_dynamics.png")
    _plot_eeg_deviation_index(clean_df, plots_dir / "eeg_deviation_index.png")


def _plot_score_histogram(features_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(features_df["Cognitive_Load_Proxy_Score"].dropna(), bins=40, color="#4C78A8", alpha=0.85)
    plt.title("Histogramm des Cognitive Load Proxy Scores")
    plt.xlabel("Cognitive Load Proxy Score")
    plt.ylabel("Anzahl Fenster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_feature_boxplots(features_df: pd.DataFrame, path: Path) -> None:
    cols = [
        "relative_theta",
        "relative_alpha",
        "relative_beta",
        "theta_alpha_ratio",
        "beta_alpha_ratio",
        "spectral_entropy",
        "Cognitive_Load_Proxy_Score",
    ]
    available = [col for col in cols if col in features_df]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 6))
    plt.boxplot([features_df[col].dropna() for col in available], labels=available, showfliers=True)
    plt.title("Boxplots wichtiger EEG-Features")
    plt.xlabel("Feature")
    plt.ylabel("Wert")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_correlation_heatmap(features_df: pd.DataFrame, path: Path) -> None:
    cols = [
        "relative_theta",
        "relative_alpha",
        "relative_beta",
        "theta_alpha_ratio",
        "beta_alpha_ratio",
        "spectral_entropy",
        "Cognitive_Load_Proxy_Score",
    ]
    corr = features_df[[col for col in cols if col in features_df]].corr(method="pearson")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Korrelationsmatrix wichtiger Features")
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_score_distribution(features_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    values = features_df["Cognitive_Load_Proxy_Score"].dropna()
    plt.hist(values, bins=40, density=True, alpha=0.75, color="#72B7B2")
    plt.axvline(values.mean(), color="#E45756", linestyle="--", label="Mittelwert")
    plt.axvline(values.median(), color="#F58518", linestyle="--", label="Median")
    plt.title("Score-Verteilung")
    plt.xlabel("Cognitive Load Proxy Score")
    plt.ylabel("Dichte")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_rolling_std(features_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    for col in ["rolling_std_score", "rolling_std_theta_alpha_ratio", "rolling_std_relative_theta"]:
        if col in features_df:
            plt.plot(features_df["start_time"], features_df[col], linewidth=0.9, label=col)
    plt.title("Rolling Standardabweichung")
    plt.xlabel("Zeit")
    plt.ylabel("Rolling Std")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_feature_dynamics(features_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    for col in ["relative_theta", "relative_alpha", "theta_alpha_ratio", "Cognitive_Load_Proxy_Score"]:
        if col in features_df:
            plt.plot(features_df["start_time"], features_df[col].diff(), linewidth=0.9, label=f"delta_{col}")
    plt.title("Feature-Dynamik zwischen Fenstern")
    plt.xlabel("Zeit")
    plt.ylabel("Änderung zum vorherigen Fenster")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _plot_eeg_deviation_index(features_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 5))
    plt.plot(features_df["start_time"], features_df["EEG_Deviation_Index"], linewidth=1.0, color="#E45756")
    plt.title("EEG Deviation Index über Zeit")
    plt.xlabel("Zeit")
    plt.ylabel("EEG Deviation Index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    print("Gespeichert:", path)
    _show_plot()
    plt.close()


def _export_extended_reports(analysis: dict, reports_dir: Path) -> dict:
    """Export extended analysis report as TXT and JSON."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    txt_path = reports_dir / "extended_analysis_report.txt"
    json_path = reports_dir / "extended_analysis_report.json"

    with txt_path.open("w", encoding="utf-8") as f:
        for section in [
            "SIGNAL QUALITY",
            "FEATURE STATISTICS",
            "FEATURE DYNAMICS",
            "COGNITIVE LOAD ANALYSIS",
            "HIGH LOAD PHASES",
            "CORRELATION ANALYSIS",
            "FEATURE IMPORTANCE FOR SCORE",
            "BASELINE DEVIATION",
            "TEMPORAL STABILITY",
            "EEG DEVIATION INDEX",
            "AUTOMATIC INTERPRETATION",
            "IMPORTANT LIMITATIONS",
        ]:
            f.write(f"\n{section}\n")
            f.write("=" * len(section) + "\n")
            f.write(_format_report_value(analysis.get(section, {})))
            f.write("\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=_json_default)

    return {"txt": txt_path, "json": json_path}


def _format_report_value(value, indent: int = 0) -> str:
    """Format nested report values for readable TXT output."""
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_format_report_value(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {item}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(_format_report_value(item, indent + 2))
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{value}\n"


def _json_default(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return str(value)


def _create_windows(df: pd.DataFrame, fs: float, window_seconds: float, overlap: float) -> list[dict]:
    window_size = int(round(window_seconds * fs))
    step_size = int(round(window_size * (1.0 - overlap)))
    windows = []
    for window_id, start_sample in enumerate(range(0, len(df) - window_size + 1, step_size)):
        end_sample = start_sample + window_size
        windows.append(
            {
                "window_id": window_id,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time": df["datetime"].iloc[start_sample],
                "end_time": df["datetime"].iloc[end_sample - 1],
                "signal_segment": df["ch1_filtered"].iloc[start_sample:end_sample].to_numpy(dtype=float),
            }
        )
    return windows


def _compute_artifact_metrics(window: dict) -> dict:
    signal = window["signal_segment"]
    return {
        **_window_metadata(window),
        "p2p": float(np.ptp(signal)),
        "std": float(np.std(signal)),
        "max_abs": float(np.max(np.abs(signal))),
        "energy": float(np.mean(signal**2)),
    }


def _window_metadata(window: dict) -> dict:
    return {
        "window_id": int(window["window_id"]),
        "start_sample": int(window["start_sample"]),
        "end_sample": int(window["end_sample"]),
        "start_time": window["start_time"],
        "end_time": window["end_time"],
    }


def _artifact_thresholds(artifact_df: pd.DataFrame) -> dict:
    return {
        "p2p_threshold": _robust_threshold(artifact_df["p2p"]),
        "std_threshold": _robust_threshold(artifact_df["std"]),
        "max_abs_threshold": _robust_threshold(artifact_df["max_abs"]),
        "energy_threshold": _robust_threshold(artifact_df["energy"]),
    }


def _robust_threshold(values: pd.Series | np.ndarray, factor: float = float(ARTIFACT_CONFIG["mad_factor"])) -> float:
    values = np.asarray(values, dtype=float)
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    return med + factor * 1.4826 * mad


def _bandpower(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    if np.sum(mask) < 2:
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def _spectral_entropy(psd: np.ndarray) -> float:
    total = float(np.sum(psd))
    if total <= EPS:
        return 0.0
    p = psd / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / (np.log2(len(p)) if len(p) > 1 else 1.0))


def _baseline_normalize(features_df: pd.DataFrame, baseline_minutes: float) -> tuple[pd.DataFrame, dict]:
    df = features_df.copy()
    baseline_start = df["start_time"].min()
    baseline_end = baseline_start + pd.Timedelta(minutes=baseline_minutes)
    baseline_df = df[(df["start_time"] >= baseline_start) & (df["start_time"] <= baseline_end)].copy()
    if baseline_df.empty:
        baseline_df = df.head(30).copy()

    df["baseline_window"] = df["window_id"].isin(baseline_df["window_id"])

    normalized = []
    for feature in LOAD_FEATURES:
        mean = float(baseline_df[feature].mean())
        std = float(baseline_df[feature].std())
        if not np.isfinite(std) or std == 0:
            std = EPS
        df[f"{feature}_z"] = (df[feature] - mean) / (std + EPS)
        normalized.append(feature)

    return df, {
        "baseline_windows": int(len(baseline_df)),
        "baseline_start": baseline_df["start_time"].min(),
        "baseline_end": baseline_df["end_time"].max(),
        "normalized_features": normalized,
    }


def _assign_load_states(features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = features_df.copy()
    if "baseline_window" in df.columns and df["baseline_window"].any():
        baseline_scores = df.loc[df["baseline_window"], "Cognitive_Load_Proxy_Score"].dropna()
    else:
        baseline_scores = df["Cognitive_Load_Proxy_Score"].dropna()

    mean = float(baseline_scores.mean())
    std = float(baseline_scores.std())
    if not np.isfinite(std) or std == 0:
        std = EPS
    state_multipliers = PROXY_CONFIG["state_threshold_std_multipliers"]
    elevated = mean + float(state_multipliers["elevated"]) * std
    strongly = mean + float(state_multipliers["strongly_elevated"]) * std
    df["cognitive_load_state"] = "baseline-nah"
    df.loc[df["Cognitive_Load_Proxy_Score"] > elevated, "cognitive_load_state"] = "erhöht"
    df.loc[df["Cognitive_Load_Proxy_Score"] > strongly, "cognitive_load_state"] = "stark erhöht"
    return df, {
        "baseline_score_mean": mean,
        "baseline_score_std": std,
        "elevated_threshold": elevated,
        "strongly_elevated_threshold": strongly,
    }


def _handle_fig(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    if show:
        _show_plot()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main EEG pipeline for IDUN/Guardian CSV files with optional external workload validation."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=None,
        help="CSV-Datei. Wenn leer, wird die erste CSV aus data/raw/ genutzt.",
    )
    parser.add_argument("--output-dir", default=None, help="Optionaler Output-Basisordner.")
    parser.add_argument("--prefix", default=None, help="Prefix fuer Exportdateien.")
    parser.add_argument("--baseline-minutes", type=float, default=BASELINE_MINUTES, help="Baseline-Laenge ab Aufnahmebeginn.")
    parser.add_argument("--rolling-windows", type=int, default=ROLLING_WINDOWS, help="Rolling-Mean-Fenster fuer den Proxy Score.")
    parser.add_argument("--show-plots", action="store_true", help="Plots zusaetzlich anzeigen.")
    parser.add_argument("--no-external-score", action="store_true", help="Externe Score-Validierung deaktivieren.")
    parser.add_argument("--external-score-column", default=EXTERNAL_SCORE_COLUMN, help="Name der externen Score-Spalte.")
    parser.add_argument(
        "--external-score-type",
        default=EXTERNAL_SCORE_TYPE,
        choices=["continuous", "binary"],
        help="Typ des externen Scores.",
    )
    parser.add_argument(
        "--high-workload-threshold",
        type=float,
        default=HIGH_WORKLOAD_THRESHOLD,
        help="Schwelle fuer high_workload beim externen Score.",
    )
    return parser.parse_args()


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    args = parse_args()
    return run_ordered_eeg_pipeline(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        prefix=args.prefix,
        baseline_minutes=args.baseline_minutes,
        rolling_windows=args.rolling_windows,
        show_plots=args.show_plots,
        use_external_score=not args.no_external_score,
        external_score_column=args.external_score_column,
        external_score_type=args.external_score_type,
        high_workload_threshold=args.high_workload_threshold,
    )


if __name__ == "__main__":
    main()
