"""Command-line and YAML configuration for individual experiment scripts."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def parse_config_path(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="YAML configuration file. Omit it to use the original settings.",
    )
    return parser.parse_args().config


def load_experiment_config(defaults, config_path=None):
    """Load one script's YAML and merge it over its behavior-compatible defaults."""
    config = deepcopy(defaults)
    if config_path is None:
        return config

    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as stream:
        overrides = yaml.safe_load(stream)
    if not isinstance(overrides, dict):
        raise ValueError("YAML root must be a mapping")

    _merge(config, overrides, path.parent, "")
    return config


def _merge(target, overrides, base_dir, location):
    unknown_keys = overrides.keys() - target.keys()
    if unknown_keys:
        prefix = f"{location}." if location else ""
        raise ValueError(f"Unknown configuration keys: {[prefix + key for key in sorted(unknown_keys)]}")

    for key, value in overrides.items():
        field = f"{location}.{key}" if location else key
        default = target[key]
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{field} must be a mapping")
            _merge(default, value, base_dir, field)
        elif isinstance(default, Path):
            path = Path(value)
            target[key] = path if path.is_absolute() else (base_dir / path).resolve()
        else:
            target[key] = value
