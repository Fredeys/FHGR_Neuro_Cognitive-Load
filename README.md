# EEG Overload Project

Electroencephalography (EEG) is a non-invasive method for measuring electrical brain activity in real time. EEG signals are widely used in neuroscience, cognitive science, biomedical engineering and human-computer interaction to analyze attention, cognitive workload, fatigue, stress, sleep and other neural processes.

![IDUN Advertisement](src/visualization/Indun_Ad.jpeg)

This project focuses on the analysis of single-channel in-ear EEG data recorded with the IDUN Guardian device. The pipeline provides a scientifically structured workflow for EEG preprocessing, artifact detection, spectral analysis, feature extraction and baseline-relative cognitive load estimation.

The project is designed for exploratory EEG analysis and research-oriented cognitive load investigation using Python-based signal processing techniques.

Python pipeline for IDUN/Guardian EEG CSV files with `timestamps` or `timestamp` and `ch1`.

## Structure

```text
EEG_Overload_Project/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── artifacts/
│   └── features/
├── src/
│   ├── preprocessing/
│   ├── artifact_detection/
│   ├── features/
│   └── visualization/
├── outputs/
│   └── plots/
├── notebooks/
├── pipeline_config.json
├── 00_run_pipeline.py
├── 01_ordered_eeg_pipeline.py
├── 02_first_step_pipeline.py
├── 03_filter_pipeline.py
├── 04_artifact_segmentation_pipeline.py
├── 05_feature_engineering_pipeline.py
└── 06_cognitive_load_proxy.py
```

## Execution

The main pipeline is `01_ordered_eeg_pipeline.py`. `00_run_pipeline.py` is only
a thin wrapper around it.

```bash
cd EEG_Overload_Project
python 01_ordered_eeg_pipeline.py eeg_Work_PC_Morning.csv
```

Plots are saved as PNG files. They can optionally be displayed as well:

```bash
python 01_ordered_eeg_pipeline.py eeg_Work_PC_Morning.csv --show-plots
```

Run without external score validation:

```bash
python 01_ordered_eeg_pipeline.py eeg_Work_PC_Morning.csv --no-external-score
```

Run with a different baseline duration:

```bash
python 01_ordered_eeg_pipeline.py eeg_Work_PC_Morning.csv --baseline-minutes 3
```

## Outputs

- `data/processed/*_cleaned_filtered_signal.csv`
- `data/artifacts/*_artifact_windows.csv`
- `data/artifacts/*_qc_epochs.csv` if optional epoching is enabled
- `data/features/*_features_eeg_only.csv`
- `data/features/*_scores_eeg_only.csv`
- `data/features/*_validation_external_score.csv` if an external score is available
- `outputs/*_summary.json`
- `outputs/plots/*.png`

## Pipeline Logic

The pipeline does not use a single raw FFT as its main method. Frequency
features are computed with Welch PSD. Welch is FFT-based, but averages across
segments and is more stable for EEG bandpower than one FFT over the full window.

Anti-aliasing is not active by default because the pipeline does not downsample
by default. Optional resampling is implemented and disabled in the config. If it
is enabled, an anti-aliasing low-pass filter is applied before downsampling.

The main features are computed from 10-second windows with overlap. These
windows are the basis for artifact marking, Welch PSD, bandpower and the
cognitive-load proxy. Optional 1-second epoching is intended separately for QC
and finer artifact inspection; it does not replace the 10-second feature
windows.

ICA is not implemented. For the current single-channel IDUN/Guardian setup, ICA
is not technically meaningful because ICA typically requires multiple channels
for component separation.

## External Workload Scores

Central parameters are defined in:

```text
pipeline_config.json
```

Sampling rate, filters, window length, PSD, baseline, proxy score and external
score settings are maintained there. The Python pipeline and the notebook read
the same config.

The full analysis pipeline `01_ordered_eeg_pipeline.py` optionally supports an
external reference column. The default config is:

```json
"external_score": {
  "enabled": true,
  "column": "nasa_tlx_score",
  "type": "continuous",
  "high_workload_threshold": 70.0,
  "use_for_calibration": false,
  "use_as_feature": false,
  "export_with_features": false,
  "export_validation_file": true
}
```

If `nasa_tlx_score` is present in the CSV, the score is aggregated per window
and used only for external validation of the `Cognitive_Load_Proxy_Score`. It is
not used as an EEG feature, does not enter the baseline, does not enter
z-normalization and does not enter the cognitive-load score. Without an external
reference, the pipeline does not claim cognitive overload.

## Notebook

The visual full-pipeline analysis is located here:

```text
notebooks/01_full_pipeline_visual_analysis.ipynb
```

The notebook reads files from `data/raw/`, creates Matplotlib visualizations and
saves important PNG files under:

```text
outputs/plots/notebook/
```

## Disclaimer

Codex was used as a development and documentation assistance tool for this
project.
