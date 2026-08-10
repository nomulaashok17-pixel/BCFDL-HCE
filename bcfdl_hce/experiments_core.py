from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import load_toniot_csv, load_vision, make_client_loaders
from .models import build_model
from .hce import HashContextEncoder
from .federated import FederatedTrainer
from .evaluation import evaluate_model
from .utils import ensure_dir, resolve_device, save_json, seed_everything, environment_snapshot


def prepare(dataset: str, cfg: dict, seed: int):
    if dataset == "toniot":
        d=cfg["datasets"]
        b=load_toniot_csv(d["toniot_csv"],target_column=d.get("target_column"),test_size=float(d.get("test_size",.2)),val_size_from_train=float(d.get("val_size_from_train",.2)),seed=seed,max_rows=d.get("max_rows"))
        train=b.train; test=b.test; input_dim=b.feature_dim
    else:
        train=load_vision(dataset,cfg["datasets"]["data_dir"],True); test=load_vision(dataset,cfg["datasets"]["data_dir"],False); input_dim=None
    clients=make_client_loaders(train,cfg,seed); test_loader=DataLoader(test,batch_size=int(cfg["optimization"]["batch_size"]),shuffle=False)
    return clients,test_loader,input_dim


def run_federated(dataset: str, cfg: dict, method: str="fedavg", rounds: int|None=None, use_hce: bool|None=None, seed: int|None=None) -> dict:
    seed=int(cfg["experiment"]["seed"] if seed is None else seed); seed_everything(seed); device=resolve_device(cfg["experiment"].get("device","auto"))
    clients,test_loader,input_dim=prepare(dataset,cfg,seed); model=build_model(dataset,input_dim)
    hce=None
    if use_hce if use_hce is not None else cfg["hce"].get("enabled",True):
        emb=128 if dataset!="cifar10" else 256
        hce=HashContextEncoder(emb,int(cfg["hce"]["bits"]),int(cfg["hce"].get("projection_dim",128)))
    tr=FederatedTrainer(model,clients,cfg,device,hce,method)
    rr=int(rounds or cfg["federated"]["rounds"]); hist=[]
    for r in range(rr):
        rec=tr.round(r); metrics=evaluate_model(tr.model,test_loader,device); rec.update({f"test_{k}":v for k,v in metrics.items() if k!="confusion_matrix"}); hist.append(rec)
    final=evaluate_model(tr.model,test_loader,device)
    return {"dataset":dataset,"method":method,"seed":seed,"history":hist,"final":final,"environment":environment_snapshot(),"model":tr.model,"test_loader":test_loader}


def serializable(result: dict): return {k:v for k,v in result.items() if k not in {"model","test_loader"}}
