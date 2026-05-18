# EEG Overload Project

Python-Pipeline fuer IDUN/Guardian EEG-CSV-Dateien mit `timestamps` oder `timestamp` und `ch1`.

## Struktur

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

## Ausfuehrung

Die Hauptpipeline ist `01_ordered_eeg_pipeline.py`. `00_run_pipeline.py` ist nur
ein duenner Wrapper darauf.

```bash
cd EEG_Overload_Project
python 01_ordered_eeg_pipeline.py eeg_Work_PC_Morning.csv
```

Plots werden als PNG gespeichert. Optional kannst du sie zusaetzlich anzeigen:

```bash
python 01_ordered_eeg_pipeline.py eeg_Work_PC_Morning.csv --show-plots
```

Optional ohne externe Score-Validierung:

```bash
python 01_ordered_eeg_pipeline.py eeg_Work_PC_Morning.csv --no-external-score
```

Optional mit anderer Baseline-Laenge:

```bash
python 01_ordered_eeg_pipeline.py eeg_Work_PC_Morning.csv --baseline-minutes 3
```

## Outputs

- `data/processed/*_processed_signal.csv`
- `data/artifacts/*_artifact_windows.csv`
- `data/features/*_features.csv`
- `outputs/*_summary.json`
- `outputs/plots/*.png`

## Externe Workload-Scores

Zentrale Parameter liegen in:

```text
pipeline_config.json
```

Dort werden Samplingrate, Filter, Fensterlaenge, PSD, Baseline, Proxy-Score und
externer Score gepflegt. Die Python-Pipeline und das Notebook lesen dieselbe
Config.

Die vollständige Analysepipeline `01_ordered_eeg_pipeline.py` unterstützt optional eine
externe Referenzspalte. Die Default-Config ist:

```json
"external_score": {
  "use_external_score": true,
  "column": "nasa_tlx_score",
  "type": "continuous",
  "high_workload_threshold": 70.0
}
```

Wenn `nasa_tlx_score` in der CSV vorhanden ist, wird der Score pro Fenster
aggregiert und nur zur externen Validierung des `Cognitive_Load_Proxy_Score`
genutzt. Er wird nicht als EEG-Merkmal verwendet. Ohne externe Referenz wird kein
Cognitive Overload behauptet.

## Notebook

Die visuelle Gesamtanalyse liegt hier:

```text
notebooks/01_full_pipeline_visual_analysis.ipynb
```

Das Notebook liest Dateien aus `data/raw/`, erzeugt Matplotlib-Visualisierungen
und speichert wichtige PNGs unter:

```text
outputs/plots/notebook/
```
