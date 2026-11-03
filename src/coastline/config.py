"""Configuration: one place for every path and parameter.

The configuration is read from ``config/default.yaml`` and, if present,
``config/local.yaml`` (your own machine, ignored by Git). Two environment
variables override the base folders, so the same code runs on any computer:

    COASTLINE_DATA_DIR      folder with the input data
    COASTLINE_RESULTS_DIR   folder where the outputs are written
    COASTLINE_CONFIG        another file of local overrides (default config/local.yaml)

Example
-------
>>> from coastline.config import load_config
>>> cfg = load_config()
>>> cfg.input("factors", "LSI")          # absolute Path of the LSI raster
>>> cfg.output("hazard", "weights.csv")  # results/hazard/weights.csv (folder created)
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels above this file (src/coastline/config.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """Return ``base`` updated with ``override``, merging nested dictionaries."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_dotenv() -> None:
    """Load a ``.env`` file at the repository root, if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


@dataclass
class Config:
    """Settings of one run. Build it with :func:`load_config`."""

    raw: dict[str, Any]
    data_dir: Path
    results_dir: Path
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- parameters
    def __getitem__(self, section: str) -> dict[str, Any]:
        """Parameters of one section, e.g. ``cfg["hazard"]["seed"]``."""
        return self.raw[section]

    @property
    def crs(self) -> str:
        return self.raw["crs"]

    @property
    def factor_codes(self) -> list[str]:
        """The Hazard factor codes, in the order of the article (Table 3)."""
        return list(self.raw["inputs"]["factors"].keys())

    # ------------------------------------------------------------------ paths
    def input(self, *keys: str) -> Path:
        """Absolute path of an input, e.g. ``cfg.input("factors", "LSI")``."""
        node: Any = self.raw["inputs"]
        for k in keys:
            node = node[k]
        path = Path(node)
        return path if path.is_absolute() else self.data_dir / path

    def output(self, *parts: str) -> Path:
        """Path inside the results folder; the parent folder is created."""
        path = self.results_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def load_config(local: str | Path | None = None) -> Config:
    """Read the default configuration, the local overrides and the environment.

    Parameters
    ----------
    local
        Optional YAML file with overrides. Default: the file named by the
        environment variable COASTLINE_CONFIG, else ``config/local.yaml`` if it exists.
    """
    _load_dotenv()
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    local_path = Path(local or os.environ.get("COASTLINE_CONFIG", CONFIG_DIR / "local.yaml"))
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})

    def resolve(env_name: str, value: str) -> Path:
        path = Path(os.environ.get(env_name, value)).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path

    return Config(
        raw=raw,
        data_dir=resolve("COASTLINE_DATA_DIR", raw["data_dir"]),
        results_dir=resolve("COASTLINE_RESULTS_DIR", raw["results_dir"]),
    )
