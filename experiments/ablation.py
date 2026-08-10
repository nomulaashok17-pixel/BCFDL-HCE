from __future__ import annotations
import argparse, time
from copy import deepcopy
from pathlib import Path
import pandas as pd
from bcfdl_hce.config import load_config
from bcfdl_hce.experiments_core import run_federated
from bcfdl_hce.af_mfa import AFMFASelector
from bcfdl_hce.utils import ensure_dir

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--rounds",type=int)
a=p.parse_args(); cfg=load_config(a.config); out=ensure_dir(Path(cfg["experiment"]["output_dir"])/"ablation")
variants=[
    ("FDL",False,False,False),
    ("FDL + AF-MFA",False,True,False),
    ("FDL + HCE",True,False,False),
    ("FDL + Blockchain",False,False,True),
    ("Full BCFDL-HCE",True,True,True),
]
rows=[]
for name,hce,mfa,chain in variants:
    t=time.perf_counter(); r=run_federated("toniot",cfg,"fedavg",a.rounds,hce); extra=time.perf_counter()-t
    m=r["final"].copy(); m.pop("confusion_matrix",None)
    if mfa: AFMFASelector(cfg["af_mfa"],seed=int(cfg["experiment"]["seed"])).select(0.75)
    rows.append({"variant":name,**m,"communication_bytes":sum(x["upload_bytes"] for x in r["history"]),"elapsed_seconds":extra,"ledger_enabled":chain})
# Cluster scheduling isolation
c2=deepcopy(cfg); c2["federated"]["cluster_training"]=False
r=run_federated("toniot",c2,"fedavg",a.rounds,True); m=r["final"].copy(); m.pop("confusion_matrix",None); rows.append({"variant":"Full minus cluster scheduling",**m,"communication_bytes":sum(x["upload_bytes"] for x in r["history"]),"elapsed_seconds":None,"ledger_enabled":True})
pd.DataFrame(rows).to_csv(out/"ablation.csv",index=False); print(out/"ablation.csv")
