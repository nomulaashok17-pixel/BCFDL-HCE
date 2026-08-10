from __future__ import annotations
import argparse, os, time, uuid
from pathlib import Path
import pandas as pd
from bcfdl_hce.config import load_config
from bcfdl_hce.af_mfa import AFMFASelector
from bcfdl_hce.ledger import PermissionedLedger, LocalCAS, AuthenticationTransaction
from bcfdl_hce.utils import ensure_dir, sha256_bytes

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--iterations",type=int,default=50)
a=p.parse_args(); cfg=load_config(a.config); out=ensure_dir(Path(cfg["experiment"]["output_dir"])/"system_overhead"); ledger=PermissionedLedger(cfg["blockchain"]["ledger_dir"],int(cfg["blockchain"]["validator_nodes"])); cas=LocalCAS(cfg["ipfs"]["local_cas_dir"]); sel=AFMFASelector(cfg["af_mfa"],seed=int(cfg["experiment"]["seed"])); rows=[]
for i in range(a.iterations):
    payload=os.urandom(256*1024); t=time.perf_counter(); h=sha256_bytes(payload); hash_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter(); cid=cas.add_bytes(payload); store_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter(); d=sel.select((i%100)/100); mfa_ms=(time.perf_counter()-t)*1000
    tx=AuthenticationTransaction(str(uuid.uuid4()),"client-0","auth",h,cid,time.time(),{"risk":d["risk"]}); t=time.perf_counter(); ledger.append(tx); ledger_ms=(time.perf_counter()-t)*1000
    t=time.perf_counter(); valid=ledger.verify(); verify_ms=(time.perf_counter()-t)*1000
    rows.append({"hash_ms":hash_ms,"store_ms":store_ms,"af_mfa_ms":mfa_ms,"ledger_commit_ms":ledger_ms,"ledger_verify_ms":verify_ms,"ledger_valid":valid,"payload_bytes":len(payload)})
pd.DataFrame(rows).to_csv(out/"system_overhead.csv",index=False); print(out/"system_overhead.csv")
