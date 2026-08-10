from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from bcfdl_hce.config import load_config
from bcfdl_hce.experiments_core import run_federated, serializable
from bcfdl_hce.utils import ensure_dir, save_json

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--dataset",choices=["mnist","cifar10"],required=True); p.add_argument("--rounds",type=int)
a=p.parse_args(); cfg=load_config(a.config); out=ensure_dir(Path(cfg["experiment"]["output_dir"])/"fl_benchmarks"/a.dataset)
r=run_federated(a.dataset,cfg,method="fedavg",rounds=a.rounds); save_json(out/"results.json",serializable(r))
xs=[x["round"]+1 for x in r["history"]]; ys=[x["test_accuracy"] for x in r["history"]]
plt.figure(); plt.plot(xs,ys); plt.xlabel("Communication round"); plt.ylabel("Test accuracy"); plt.tight_layout(); plt.savefig(out/"convergence.png",dpi=300); plt.close()
print(out/"results.json")
