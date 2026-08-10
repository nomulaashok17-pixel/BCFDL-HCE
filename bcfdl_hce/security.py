from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from .privacy import l2_norm


def label_flip(y: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    return (y + 1) % num_classes


def sign_flip(update: dict[str, torch.Tensor], scale: float = 5.0) -> dict[str, torch.Tensor]:
    return {k: -scale*v for k,v in update.items()}


def model_replacement(update: dict[str, torch.Tensor], scale: float = 10.0) -> dict[str, torch.Tensor]:
    return {k: scale*v for k,v in update.items()}


def byzantine_random(update: dict[str, torch.Tensor], seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(v.shape, generator=g, dtype=v.dtype) * (v.float().std()+1e-3) for k,v in update.items()}


def cosine_update(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    av = torch.cat([x.float().flatten() for x in a.values()])
    bv = torch.cat([x.float().flatten() for x in b.values()])
    return float(torch.nn.functional.cosine_similarity(av, bv, dim=0))


def robust_filter(updates: list[dict[str, torch.Tensor]], zmax: float = 3.5, cosine_floor: float = -0.25) -> list[bool]:
    if len(updates) < 3: return [True]*len(updates)
    norms = np.array([float(l2_norm(u)) for u in updates])
    med = np.median(norms)
    mad = np.median(np.abs(norms-med)) + 1e-12
    robust_z = 0.6745*(norms-med)/mad
    # Reference is coordinate-wise mean update; used only as an anomaly screen.
    ref = {k: sum(u[k] for u in updates)/len(updates) for k in updates[0]}
    out=[]
    for z,u in zip(robust_z, updates):
        out.append(bool(abs(z) <= zmax and cosine_update(u, ref) >= cosine_floor))
    if not any(out):
        out[int(np.argmin(np.abs(robust_z)))] = True
    return out


@dataclass
class ReplayGuard:
    window_seconds: float = 120.0
    def __post_init__(self): self.seen = {}
    def accept(self, token: str, timestamp: float | None = None) -> bool:
        now = time.time() if timestamp is None else float(timestamp)
        self.seen = {k:t for k,t in self.seen.items() if now-t <= self.window_seconds}
        if token in self.seen: return False
        self.seen[token] = now
        return True
