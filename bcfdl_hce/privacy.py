from __future__ import annotations
import hashlib
import numpy as np
import torch


def state_delta(local: dict, global_state: dict) -> dict[str, torch.Tensor]:
    return {k: local[k].detach().cpu() - global_state[k].detach().cpu() for k in global_state}


def l2_norm(update: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum((v.float()**2).sum() for v in update.values()))


def clip_update(update: dict[str, torch.Tensor], max_norm: float) -> tuple[dict[str, torch.Tensor], float]:
    norm = float(l2_norm(update))
    scale = min(1.0, float(max_norm)/(norm+1e-12))
    return {k: v*scale for k,v in update.items()}, norm


def add_gaussian_noise(update: dict[str, torch.Tensor], std: float, seed: int) -> dict[str, torch.Tensor]:
    if std <= 0: return {k:v.clone() for k,v in update.items()}
    g = torch.Generator().manual_seed(seed)
    return {k: v + torch.randn(v.shape, generator=g, dtype=v.dtype)*std for k,v in update.items()}


def _pair_seed(a: int, b: int, round_id: int, master_seed: int) -> int:
    x,y = sorted((a,b))
    h = hashlib.sha256(f"{master_seed}:{round_id}:{x}:{y}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**31-1)


def mask_update(update: dict[str, torch.Tensor], client_id: int, participants: list[int],
                round_id: int, master_seed: int) -> dict[str, torch.Tensor]:
    """Pairwise masks cancel exactly when all listed participants contribute."""
    out = {k:v.clone() for k,v in update.items()}
    for peer in participants:
        if peer == client_id: continue
        g = torch.Generator().manual_seed(_pair_seed(client_id, peer, round_id, master_seed))
        sign = 1.0 if client_id < peer else -1.0
        for k,v in out.items():
            out[k] = out[k] + sign*torch.randn(v.shape, generator=g, dtype=v.dtype)
    return out


def secure_aggregate(masked_updates: list[tuple[dict[str, torch.Tensor], int]]) -> dict[str, torch.Tensor]:
    if not masked_updates: raise ValueError("No updates")
    total_w = sum(w for _,w in masked_updates)
    keys = masked_updates[0][0].keys()
    return {k: sum(u[k]*w for u,w in masked_updates)/total_w for k in keys}
