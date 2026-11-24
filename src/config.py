"""YAML config loading with inheritance and command-line overrides.

A config may name a parent with ``base: other.yaml`` (path relative to the child). The child's
values are deep-merged over the parent's, so an ablation config only lists what it changes.

Overrides use dotted keys, e.g. ``--set train.epochs=5 augment=false``; values are parsed as YAML
so numbers, booleans and lists work.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` merged into ``base`` recursively (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, _seen: tuple = ()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Circular 'base:' chain in configs: {' -> '.join(map(str, _seen + (path,)))}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    parent = cfg.pop("base", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent, _seen + (path,)), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.sub=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
