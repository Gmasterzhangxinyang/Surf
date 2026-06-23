from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(path: str | Path) -> Path:
    """Create and return output directory."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
