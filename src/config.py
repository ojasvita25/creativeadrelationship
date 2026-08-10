"""
src/config.py
-------------
Helper module to load and parse YAML pipeline configuration files.
"""

from __future__ import annotations

import os
from typing import Any
import yaml


def load_config(config_path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Load a YAML configuration file and apply optional CLI parameter overrides.

    Parameters
    ----------
    config_path : str
        Path to the .yaml configuration file.
    overrides : dict | None
        Dictionary of parameter overrides (e.g. sample_size, output_path).

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if overrides:
        dataset_cfg  = config.setdefault("dataset", {})
        pipeline_cfg = config.setdefault("pipeline", {})
        models_cfg   = config.setdefault("models", {})
        thresh_cfg   = config.setdefault("thresholds", {})

        if overrides.get("dataset_name") is not None:
            dataset_cfg["name"] = overrides["dataset_name"]

        if overrides.get("dataset_split") is not None:
            dataset_cfg["split"] = overrides["dataset_split"]

        if overrides.get("sample") is not None:
            val = overrides["sample"]
            if str(val).lower() == "all":
                pipeline_cfg["sample_size"] = None
            else:
                pipeline_cfg["sample_size"] = int(val)

        if overrides.get("output") is not None:
            pipeline_cfg["output_path"] = overrides["output"]

        if overrides.get("device") is not None:
            models_cfg["device"] = overrides["device"]

        if overrides.get("cache_dir") is not None:
            pipeline_cfg["cache_dir"] = overrides["cache_dir"]

        if overrides.get("max_pairs") is not None:
            pipeline_cfg["max_pairs_per_type"] = overrides["max_pairs"]

        if overrides.get("clip_threshold") is not None and "clip_threshold" in thresh_cfg:
            thresh_cfg["clip_threshold"] = overrides["clip_threshold"]

        if overrides.get("resnet_threshold") is not None and "resnet_threshold" in thresh_cfg:
            thresh_cfg["resnet_threshold"] = overrides["resnet_threshold"]

    return config
