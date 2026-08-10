from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    fed = cfg.get("federated", {})
    if int(fed.get("clients", 0)) < 2:
        raise ValueError("federated.clients must be >= 2")
    if int(fed.get("rounds", 0)) < 1:
        raise ValueError("federated.rounds must be >= 1")
    if fed.get("partition") not in {"iid", "dirichlet"}:
        raise ValueError("federated.partition must be iid or dirichlet")
    if cfg.get("blockchain", {}).get("consensus") != "limited_raft":
        raise ValueError("This implementation intentionally supports Limited-Raft only.")
    bits = int(cfg.get("hce", {}).get("bits", 0))
    if bits <= 0:
        raise ValueError("hce.bits must be positive")


def with_overrides(cfg: dict, overrides: dict) -> dict:
    out = deepcopy(cfg)
    for dotted, value in overrides.items():
        target = out
        parts = dotted.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    validate_config(out)
    return out
