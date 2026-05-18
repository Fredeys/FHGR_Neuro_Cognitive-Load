from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_PIPELINE_PATH = PROJECT_ROOT / "01_ordered_eeg_pipeline.py"


def _load_main_pipeline():
    spec = importlib.util.spec_from_file_location("ordered_eeg_main_pipeline", MAIN_PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Konnte Hauptpipeline nicht laden: {MAIN_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Thin wrapper: the real main pipeline lives in 01_ordered_eeg_pipeline.py."""
    pipeline = _load_main_pipeline()
    return pipeline.main()


if __name__ == "__main__":
    main()
