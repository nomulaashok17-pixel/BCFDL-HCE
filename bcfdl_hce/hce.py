from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class HCELoss:
    total: torch.Tensor
    similarity: torch.Tensor
    quantization: torch.Tensor
    prototype: torch.Tensor


class HashContextEncoder(nn.Module):
    """Compact binary representation of learned authentication context embeddings."""
    def __init__(self, embedding_dim: int, bits: int = 64, projection_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim), nn.GELU(),
            nn.Linear(projection_dim, bits), nn.Tanh(),
        )
        self.bits = bits

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)

    @staticmethod
    def binarize(relaxed: torch.Tensor) -> torch.Tensor:
        return torch.where(relaxed >= 0, torch.ones_like(relaxed), -torch.ones_like(relaxed))

    @staticmethod
    def packed_nbytes(batch: int, bits: int) -> int:
        return int(batch * ((bits + 7) // 8))


def hce_loss(relaxed: torch.Tensor, labels: torch.Tensor, prototypes: dict[int, torch.Tensor] | None,
             similarity_weight: float = .2, quantization_weight: float = .05,
             prototype_weight: float = .1) -> HCELoss:
    z = F.normalize(relaxed, dim=1)
    sim = z @ z.T
    same = labels[:, None].eq(labels[None, :]).float()
    # Supervised pairwise target: +1 for same class, -1 for different class.
    target = same * 2.0 - 1.0
    mask = ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    sim_loss = F.mse_loss(sim[mask], target[mask]) if mask.any() else relaxed.new_zeros(())
    quant = ((relaxed.abs() - 1.0) ** 2).mean()
    proto = relaxed.new_zeros(())
    if prototypes:
        terms = []
        for c, p in prototypes.items():
            m = labels == int(c)
            if m.any():
                pp = p.to(relaxed.device).view(1, -1)
                terms.append(1.0 - F.cosine_similarity(relaxed[m], pp.expand(m.sum(), -1), dim=1).mean())
        if terms:
            proto = torch.stack(terms).mean()
    total = similarity_weight*sim_loss + quantization_weight*quant + prototype_weight*proto
    return HCELoss(total, sim_loss, quant, proto)


def class_prototypes(relaxed: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
    out = {}
    for c in labels.unique():
        m = labels == c
        p = relaxed[m].mean(dim=0)
        out[int(c.item())] = torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p)).detach().cpu()
    return out


def aggregate_prototypes(client_prototypes: list[tuple[dict[int, torch.Tensor], dict[int, int]]]) -> dict[int, torch.Tensor]:
    sums, counts = {}, {}
    for protos, cnts in client_prototypes:
        for c, p in protos.items():
            n = int(cnts.get(c, 1))
            sums[c] = sums.get(c, torch.zeros_like(p)) + p.float()*n
            counts[c] = counts.get(c, 0) + n
    return {c: torch.where(v/counts[c] >= 0, torch.ones_like(v), -torch.ones_like(v)) for c, v in sums.items()}
