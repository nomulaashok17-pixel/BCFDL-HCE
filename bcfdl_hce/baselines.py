from __future__ import annotations

SUPPORTED = {
    "fedavg": "Federated averaging",
    "fedprox": "FedProx proximal local objective",
    "scaffold": "SCAFFOLD-style client control-variate correction",
    "fednova": "FedNova-style normalized client updates",
    "moon": "MOON-style representation contrastive regularization",
    "feddyn": "FedDyn-style dynamic client regularization",
}


def validate_method(name: str) -> str:
    name=name.lower()
    if name not in SUPPORTED: raise ValueError(f"Unknown baseline {name}; choose {sorted(SUPPORTED)}")
    return name
