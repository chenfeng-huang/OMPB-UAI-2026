import json
import os
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np


def make_run_dir(base_dir: str, run_name: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config_snapshot(run_dir: str, config: Dict) -> None:
    path = os.path.join(run_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        values = [row.get(k, "") for k in keys]
        lines.append(",".join(str(v) for v in values))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_array(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, array)
