from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from bcfdl_hce.config import load_config
from bcfdl_hce.experiments_core import run_federated
from bcfdl_hce.utils import ensure_dir

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--dataset",default="toniot",choices=["toniot","mnist","cifar10"]); p.add_argument("--rounds",type=int)
a=p.parse_args(); cfg=load_config(a.config); out=ensure_dir(Path(cfg["experiment"]["output_dir"])/"baseline_comparison")
rows=[]
for m in cfg["baselines"]["methods"]:
    r=run_federated(a.dataset,cfg,m,rounds=a.rounds,use_hce=False); rows.append({"method":m,**{k:v for k,v in r["final"].items() if k!="confusion_matrix"},"communication_bytes":sum(x["upload_bytes"] for x in r["history"])})
pd.DataFrame(rows).to_csv(out/"baseline_comparison.csv",index=False); print(out/"baseline_comparison.csv")
