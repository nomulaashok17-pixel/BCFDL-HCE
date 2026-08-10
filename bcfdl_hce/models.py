from __future__ import annotations
import torch
from torch import nn


class MNISTNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, embedding_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, classes)
    def forward(self, x, return_embedding=False):
        z = self.features(x)
        logits = self.classifier(z)
        return (logits, z) if return_embedding else logits


class CIFARNet(nn.Module):
    def __init__(self, embedding_dim: int = 256, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128*4*4, embedding_dim), nn.ReLU(), nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(embedding_dim, classes)
    def forward(self, x, return_embedding=False):
        z = self.features(x)
        logits = self.classifier(z)
        return (logits, z) if return_embedding else logits


class AuthenticationNet(nn.Module):
    """Tabular legitimate-vs-malicious classifier for ToN-IoT."""
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, embedding_dim), nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_dim, 2)
    def forward(self, x, return_embedding=False):
        z = self.features(x)
        logits = self.classifier(z)
        return (logits, z) if return_embedding else logits


def build_model(dataset: str, input_dim: int | None = None):
    d = dataset.lower()
    if d == "mnist": return MNISTNet()
    if d == "cifar10": return CIFARNet()
    if d in {"toniot", "toN-iot".lower()}:
        if input_dim is None: raise ValueError("input_dim required for ToN-IoT")
        return AuthenticationNet(input_dim)
    raise ValueError(dataset)
