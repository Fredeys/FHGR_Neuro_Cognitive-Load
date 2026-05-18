from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "pipeline_config.json"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the shared EEG pipeline configuration."""
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()


def section(name: str) -> dict[str, Any]:
    return CONFIG[name]


def frequency_bands() -> dict[str, tuple[float, float]]:
    return {
        name: (float(bounds[0]), float(bounds[1]))
        for name, bounds in CONFIG["frequency_bands"].items()
    }


def load_features() -> list[str]:
    return list(CONFIG["cognitive_load_proxy"]["load_features"])
