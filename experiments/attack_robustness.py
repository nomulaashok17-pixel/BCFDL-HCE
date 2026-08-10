from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from bcfdl_hce.config import load_config
from bcfdl_hce.security import sign_flip, model_replacement, byzantine_random, robust_filter, ReplayGuard
from bcfdl_hce.utils import ensure_dir

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml")
a=p.parse_args(); cfg=load_config(a.config); out=ensure_dir(Path(cfg["experiment"]["output_dir"])/"attack_robustness")
g=torch.Generator().manual_seed(int(cfg["experiment"]["seed"])); base={"w":torch.randn(1024,generator=g)}
benign=[{"w":base["w"]+0.02*torch.randn(1024,generator=g)} for _ in range(8)]
attacks={"sign_flip":sign_flip(base,5),"model_replacement":model_replacement(base,10),"byzantine_random":byzantine_random(base,7)}
rows=[]
for name,a_u in attacks.items():
    ups=benign+[a_u]; keep=robust_filter(ups,float(cfg["security"]["max_update_norm_zscore"]),float(cfg["security"]["cosine_floor"])); rows.append({"attack":name,"malicious_update_rejected":not keep[-1],"accepted_updates":sum(keep)})
rg=ReplayGuard(float(cfg["security"]["replay_window_seconds"])); first=rg.accept("token-1"); second=rg.accept("token-1"); rows.append({"attack":"replay","malicious_update_rejected":bool(first and not second),"accepted_updates":None})
pd.DataFrame(rows).to_csv(out/"attack_robustness.csv",index=False); print(out/"attack_robustness.csv")
