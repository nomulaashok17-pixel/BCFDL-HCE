from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from bcfdl_hce.config import load_config
from bcfdl_hce.experiments_core import run_federated
from bcfdl_hce.utils import ensure_dir

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--rounds",type=int)
a=p.parse_args(); cfg=load_config(a.config); out=ensure_dir(Path(cfg["experiment"]["output_dir"])/"statistics"); reps=int(cfg["experiment"].get("repetitions",5)); base_seed=int(cfg["experiment"]["seed"])
methods=["fedavg","fedprox","fednova","feddyn"]; records=[]
for m in methods:
    for i in range(reps):
        r=run_federated("toniot",cfg,m,a.rounds,use_hce=(m=="fedavg"),seed=base_seed+i); records.append({"method":m,"rep":i,"accuracy":r["final"]["accuracy"],"f1":r["final"]["f1"],"roc_auc":r["final"].get("roc_auc",np.nan)})
df=pd.DataFrame(records); df.to_csv(out/"repetitions.csv",index=False); rows=[]
for m,g in df.groupby("method"):
    vals=g.accuracy.to_numpy(); ci=stats.t.interval(.95,len(vals)-1,loc=vals.mean(),scale=stats.sem(vals)) if len(vals)>1 else (vals[0],vals[0]); rows.append({"method":m,"mean_accuracy":vals.mean(),"sd_accuracy":vals.std(ddof=1) if len(vals)>1 else 0,"ci95_low":ci[0],"ci95_high":ci[1]})
pd.DataFrame(rows).to_csv(out/"summary.csv",index=False)
if "fedavg" in methods:
    a1=df[df.method=="fedavg"].sort_values("rep").accuracy.to_numpy()
    tests=[]
    for m in methods[1:]:
        b=df[df.method==m].sort_values("rep").accuracy.to_numpy(); t,pv=stats.ttest_rel(a1,b); d=(a1-b).mean()/((a1-b).std(ddof=1)+1e-12); tests.append({"comparison":f"fedavg vs {m}","paired_t":t,"p_value":pv,"paired_cohens_d":d})
    pd.DataFrame(tests).to_csv(out/"paired_tests.csv",index=False)
print(out/"summary.csv")
