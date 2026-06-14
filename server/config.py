from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json_compatible_yaml(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_configs(base_dir: Path) -> Dict[str, Any]:
    config_dir = base_dir / "configs"
    return {
        "hardware": load_json_compatible_yaml(config_dir / "hardware.yaml"),
        "network_profiles": load_json_compatible_yaml(config_dir / "network_profiles.yaml"),
        "models": load_json_compatible_yaml(config_dir / "models.yaml"),
        "experiment": load_json_compatible_yaml(config_dir / "experiment.yaml"),
    }
