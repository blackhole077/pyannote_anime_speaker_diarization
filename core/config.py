"""Shared YAML → OmegaConf loader.

``get_cfg`` is cached and is only safe for leaf configs in ``configs/`` whose
shape is a plain dict. Composed configs that have ``include``/``overrides``
(currently only ``eval.yaml``) must be resolved by their own loader.
"""

from functools import cache
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from core.constants import CONFIG_ROOT


def load_cfg(config_path: Path) -> DictConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        return OmegaConf.create(yaml.safe_load(f))


@cache
def get_cfg(name: str) -> DictConfig:
    return load_cfg(CONFIG_ROOT / f"{name}.yaml")
