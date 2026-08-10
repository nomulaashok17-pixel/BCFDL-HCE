from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import pandas as pd
from bcfdl_hce.config import load_config
from bcfdl_hce.experiments_core import run_federated
from bcfdl_hce.utils import ensure_dir

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--rounds",type=int)
a=p.parse_args(); cfg=load_config(a.config); out=ensure_dir(Path(cfg["experiment"]["output_dir"])/"privacy_analysis")
variants=[("none",False,False,False,0.0),("clipping",True,False,False,0.0),("clipping+dp",True,True,False,0.15),("clipping+secure_aggregation",True,False,True,0.0),("privacy_enhanced",True,True,True,0.15)]
rows=[]
for name,clip,dp,sec,noise in variants:
    c=deepcopy(cfg); c["privacy"].update({"clipping":clip,"differential_privacy":dp,"secure_aggregation":sec,"noise_multiplier":noise}); r=run_federated("toniot",c,"fedavg",a.rounds,True); m=r["final"].copy(); m.pop("confusion_matrix",None); rows.append({"privacy":name,**m,"communication_bytes":sum(x["upload_bytes"] for x in r["history"])})
pd.DataFrame(rows).to_csv(out/"privacy_analysis.csv",index=False); print(out/"privacy_analysis.csv")
