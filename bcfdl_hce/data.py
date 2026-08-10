from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


BENIGN_TOKENS = {"normal", "benign", "legitimate", "0", "false", "no_attack", "none"}


@dataclass
class TabularBundle:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    feature_dim: int
    preprocessor: ColumnTransformer
    target_column: str


def _binary_target(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
        uniq = np.unique(vals)
        if set(uniq.tolist()).issubset({0, 1}):
            return vals.astype(np.int64)
        return (vals != 0).astype(np.int64)
    s = series.astype(str).str.strip().str.lower()
    return (~s.isin(BENIGN_TOKENS)).astype(np.int64).to_numpy()


def detect_target(df: pd.DataFrame, configured: str | None = None) -> str:
    if configured:
        if configured not in df.columns:
            raise KeyError(f"Configured target column '{configured}' not found.")
        return configured
    lower = {c.lower(): c for c in df.columns}
    for candidate in ["label", "type", "attack", "class", "target", "category"]:
        if candidate in lower:
            return lower[candidate]
    raise ValueError("Could not detect target column. Set datasets.target_column in config.yaml.")


def load_toniot_csv(path: str | Path, *, target_column: str | None = None,
                    test_size: float = 0.2, val_size_from_train: float = 0.2,
                    seed: int = 42, max_rows: int | None = None) -> TabularBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ToN-IoT CSV not found at {path}. Download/provide the dataset and update datasets.toniot_csv."
        )
    df = pd.read_csv(path, nrows=max_rows, low_memory=False)
    target = detect_target(df, target_column)
    y = _binary_target(df[target])
    X = df.drop(columns=[target]).copy()

    # Remove common target aliases and obvious row IDs to reduce leakage.
    aliases = {"label", "type", "attack", "class", "target", "category"}
    drop = [c for c in X.columns if c.lower() in aliases or c.lower().startswith("unnamed:")]
    if drop:
        X = X.drop(columns=drop)
    X = X.replace([np.inf, -np.inf], np.nan)

    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    y_tr = y[tr_idx]
    tr_idx, va_idx = train_test_split(tr_idx, test_size=val_size_from_train,
                                      stratify=y_tr, random_state=seed + 1)

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)),
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop", sparse_threshold=0.0)

    Xtr = prep.fit_transform(X.iloc[tr_idx]).astype("float32")
    Xva = prep.transform(X.iloc[va_idx]).astype("float32")
    Xte = prep.transform(X.iloc[te_idx]).astype("float32")

    def ds(a, b):
        return TensorDataset(torch.from_numpy(a), torch.from_numpy(y[b].astype("int64")))
    return TabularBundle(ds(Xtr, tr_idx), ds(Xva, va_idx), ds(Xte, te_idx), Xtr.shape[1], prep, target)


def load_vision(name: str, data_dir: str | Path, train: bool = True):
    from torchvision import datasets, transforms
    name = name.lower()
    if name == "mnist":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return datasets.MNIST(root=data_dir, train=train, download=True, transform=tfm)
    if name == "cifar10":
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        return datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfm)
    raise ValueError(f"Unsupported vision dataset: {name}")


def labels_from_dataset(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].cpu().numpy()
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return np.asarray(t.cpu().numpy() if torch.is_tensor(t) else t)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))])


def iid_partition(n: int, clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.asarray(x, dtype=int) for x in np.array_split(idx, clients)]


def dirichlet_partition(labels: Sequence[int], clients: int, alpha: float, seed: int,
                        min_size: int = 10, max_tries: int = 1000) -> list[np.ndarray]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_tries):
        parts = [[] for _ in range(clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            p = rng.dirichlet(np.repeat(alpha, clients))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, block in enumerate(np.split(idx, cuts)):
                parts[k].extend(block.tolist())
        if min(map(len, parts)) >= min_size:
            return [np.asarray(sorted(p), dtype=int) for p in parts]
    raise RuntimeError("Unable to create Dirichlet partition satisfying min_size; increase alpha or reduce min_size.")


def make_client_loaders(dataset: Dataset, cfg: dict, seed: int, shuffle: bool = True) -> list[DataLoader]:
    labels = labels_from_dataset(dataset)
    fed = cfg["federated"]
    if fed["partition"] == "iid":
        parts = iid_partition(len(dataset), int(fed["clients"]), seed)
    else:
        parts = dirichlet_partition(labels, int(fed["clients"]), float(fed["dirichlet_alpha"]), seed,
                                    int(fed.get("min_client_samples", 10)))
    bs = int(cfg["optimization"]["batch_size"])
    loaders = []
    for i, idx in enumerate(parts):
        g = torch.Generator().manual_seed(seed + 1000 + i)
        loaders.append(DataLoader(Subset(dataset, idx.tolist()), batch_size=bs, shuffle=shuffle, generator=g))
    return loaders
