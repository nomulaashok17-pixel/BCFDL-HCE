from __future__ import annotations
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Factor:
    name: str
    assurance: float
    latency_ms: float
    burden: float
    availability: float = 1.0


DEFAULT_FACTORS = [
    Factor("password", 0.25, 80, 0.10),
    Factor("otp", 0.45, 650, 0.45, 0.95),
    Factor("device_possession", 0.40, 110, 0.10, 0.98),
    Factor("fingerprint", 0.65, 280, 0.25, 0.85),
    Factor("face", 0.60, 350, 0.30, 0.85),
    Factor("voice", 0.50, 700, 0.55, 0.75),
    Factor("keystroke", 0.35, 120, 0.05, 0.80),
]


def assurance_required(risk: float, cfg: dict) -> float:
    if risk < cfg["low_risk_threshold"]: return 0.35
    if risk < cfg["medium_risk_threshold"]: return 0.60
    if risk < cfg["high_risk_threshold"]: return 0.80
    return 0.93


def combined_assurance(selected: list[Factor]) -> float:
    residual = 1.0
    for f in selected:
        residual *= (1.0 - max(0.0, min(1.0, f.assurance)))
    return 1.0 - residual


class AFMFASelector:
    """Genetic risk-adaptive factor selector used by the BCFDL-HCE authentication path."""
    def __init__(self, cfg: dict, factors=None, seed: int = 42):
        self.cfg, self.factors = cfg, list(factors or DEFAULT_FACTORS)
        self.rng = random.Random(seed)

    def _score(self, bits: list[int], required: float) -> float:
        selected = [f for f, b in zip(self.factors, bits) if b and f.availability > 0]
        if len(selected) < int(self.cfg.get("minimum_factors", 1)):
            return 1e6
        a = combined_assurance(selected)
        shortfall = max(0.0, required-a)
        latency = sum(f.latency_ms for f in selected)/1000.0
        burden = sum(f.burden for f in selected)
        availability_penalty = sum(1-f.availability for f in selected)
        return 50.0*shortfall**2 + 0.20*latency + 0.60*burden + 0.40*availability_penalty + 0.05*len(selected)

    def select(self, risk: float) -> dict:
        required = assurance_required(float(risk), self.cfg)
        n = len(self.factors)
        pop_n = int(self.cfg.get("population", 40))
        gens = int(self.cfg.get("generations", 35))
        mut = float(self.cfg.get("mutation_rate", .12))
        pop = [[self.rng.randint(0,1) for _ in range(n)] for _ in range(pop_n)]
        for _ in range(gens):
            pop.sort(key=lambda b: self._score(b, required))
            elite = pop[:max(2, pop_n//5)]
            new = elite.copy()
            while len(new) < pop_n:
                a,b = self.rng.sample(elite, 2)
                cut = self.rng.randint(1, n-1)
                child = a[:cut]+b[cut:]
                child = [1-x if self.rng.random()<mut else x for x in child]
                new.append(child)
            pop = new
        best = min(pop, key=lambda b: self._score(b, required))
        selected = [f for f,b in zip(self.factors,best) if b]
        return {
            "risk": float(risk), "required_assurance": required,
            "factors": [f.name for f in selected],
            "achieved_assurance": combined_assurance(selected),
            "estimated_latency_ms": sum(f.latency_ms for f in selected),
            "objective": self._score(best, required),
        }
