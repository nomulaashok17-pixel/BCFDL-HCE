from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from bcfdl_hce.config import load_config

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--skip-vision",action="store_true"); p.add_argument("--quick",action="store_true")
a=p.parse_args(); cfg=load_config(a.config); rounds="3" if a.quick else str(cfg["federated"]["rounds"])
cmds=[]
if not a.skip_vision:
    cmds += [[sys.executable,"experiments/fl_benchmarks.py","--config",a.config,"--dataset","mnist","--rounds",rounds], [sys.executable,"experiments/fl_benchmarks.py","--config",a.config,"--dataset","cifar10","--rounds",rounds]]
cmds += [
 [sys.executable,"experiments/toniot_authentication.py","--config",a.config,"--rounds",rounds],
 [sys.executable,"experiments/baseline_comparison.py","--config",a.config,"--rounds",rounds],
 [sys.executable,"experiments/ablation.py","--config",a.config,"--rounds",rounds],
 [sys.executable,"experiments/attack_robustness.py","--config",a.config],
 [sys.executable,"experiments/privacy_analysis.py","--config",a.config,"--rounds",rounds],
 [sys.executable,"experiments/system_overhead.py","--config",a.config,"--iterations","5" if a.quick else "50"],
 [sys.executable,"experiments/statistical_analysis.py","--config",a.config,"--rounds",rounds],
]
for cmd in cmds:
    print("+", " ".join(cmd)); subprocess.run(cmd,check=True)
