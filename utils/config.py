import os
from typing import Any, Dict, Tuple

import yaml


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(
    root_dir: str, config_path: str | None = None
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ett = load_yaml(os.path.join(root_dir, "configs", "ett.yaml"))
    ompb = load_yaml(os.path.join(root_dir, "configs", "ompb.yaml"))
    models = load_yaml(os.path.join(root_dir, "configs", "models.yaml"))

    if config_path:
        override = load_yaml(config_path)
        if "ett" in override:
            ett = _deep_update(ett, override["ett"])
        if "ompb" in override:
            ompb = _deep_update(ompb, override["ompb"])
        if "models" in override:
            models = _deep_update(models, override["models"])
        if "data" in override:
            ett = _deep_update(ett, override)
        elif "window" in override:
            ett = _deep_update(ett, override)

    return ett, ompb, models


def apply_cli_overrides(args: Dict[str, str], ett: Dict[str, Any], ompb: Dict[str, Any]) -> None:
    """
    Apply simple CLI overrides (k=v) consistently across configs.

    Supported:
      - seq_len: int (updates ompb.seq_len and ett.window.seq_len)
      - pred_len: int (updates ompb.pred_len and ett.window.pred_len)
    """
    if "seq_len" in args:
        v = int(args["seq_len"])
        ompb["seq_len"] = v
        ett.setdefault("window", {})["seq_len"] = v

    if "pred_len" in args:
        v = int(args["pred_len"])
        ompb["pred_len"] = v
        ett.setdefault("window", {})["pred_len"] = v

