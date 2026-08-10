from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import tempfile
import torch
from torch.utils.data import TensorDataset
from bcfdl_hce.config import load_config
from bcfdl_hce.data import make_client_loaders
from bcfdl_hce.models import AuthenticationNet
from bcfdl_hce.hce import HashContextEncoder
from bcfdl_hce.af_mfa import AFMFASelector
from bcfdl_hce.federated import FederatedTrainer
from bcfdl_hce.privacy import mask_update, secure_aggregate
from bcfdl_hce.ledger import PermissionedLedger, LocalCAS, AuthenticationTransaction
from bcfdl_hce.security import ReplayGuard
from bcfdl_hce.utils import seed_everything
import time, uuid

cfg=load_config("config.yaml"); cfg=deepcopy(cfg); cfg["federated"].update({"clients":4,"rounds":1,"local_epochs":1,"min_client_samples":5,"clusters":2}); cfg["optimization"]["batch_size"]=16; seed_everything(7)
x=torch.randn(120,12); y=(x[:,0]+.5*x[:,1]>0).long(); ds=TensorDataset(x,y); loaders=make_client_loaders(ds,cfg,7); model=AuthenticationNet(12,128); hce=HashContextEncoder(128,64,64); tr=FederatedTrainer(model,loaders,cfg,torch.device("cpu"),hce,"fedavg"); rec=tr.round(0); assert rec["participants"]
selector=AFMFASelector(cfg["af_mfa"],seed=7); low=selector.select(.1); high=selector.select(.95); assert high["required_assurance"]>low["required_assurance"]
u1={"w":torch.randn(10)}; u2={"w":torch.randn(10)}; m1=mask_update(u1,0,[0,1],0,7); m2=mask_update(u2,1,[0,1],0,7); agg=secure_aggregate([(m1,1),(m2,1)]); assert torch.allclose(agg["w"],(u1["w"]+u2["w"])/2,atol=1e-5)
with tempfile.TemporaryDirectory() as td:
    cas=LocalCAS(Path(td)/"cas"); cid=cas.add_bytes(b"model"); ledger=PermissionedLedger(Path(td)/"ledger",5); tx=AuthenticationTransaction(str(uuid.uuid4()),"c","auth","x",cid,time.time(),{}); ledger.append(tx); assert ledger.verify(); assert cas.get_bytes(cid)==b"model"
rg=ReplayGuard(120); assert rg.accept("abc") and not rg.accept("abc")
print("BCFDL-HCE smoke test: PASS")
