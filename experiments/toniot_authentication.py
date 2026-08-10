from __future__ import annotations
import argparse, io, time, uuid
from pathlib import Path
import torch
from bcfdl_hce.config import load_config
from bcfdl_hce.experiments_core import run_federated, serializable
from bcfdl_hce.af_mfa import AFMFASelector
from bcfdl_hce.ledger import PermissionedLedger, LocalCAS, IPFSAdapter, AuthenticationTransaction
from bcfdl_hce.utils import ensure_dir, save_json, sha256_bytes

p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--rounds",type=int)
a=p.parse_args(); cfg=load_config(a.config); out=ensure_dir(Path(cfg["experiment"]["output_dir"])/"toniot_authentication")
r=run_federated("toniot",cfg,"fedavg",rounds=a.rounds,use_hce=True); model=r["model"].eval(); loader=r["test_loader"]
selector=AFMFASelector(cfg["af_mfa"],seed=int(cfg["experiment"]["seed"])); device=next(model.parameters()).device
samples=[]
with torch.no_grad():
    for x,y in loader:
        p1=torch.softmax(model(x.to(device)),1)[:,1].cpu()
        for yy,risk in zip(y.tolist(),p1.tolist()):
            samples.append({"label":int(yy),"risk":risk,"af_mfa":selector.select(risk)})
            if len(samples)>=250: break
        if len(samples)>=250: break
buf=io.BytesIO(); torch.save(model.state_dict(),buf); raw=buf.getvalue(); model_hash=sha256_bytes(raw)
store=IPFSAdapter(cfg["ipfs"]["api_url"]) if cfg["ipfs"].get("enabled",False) else LocalCAS(cfg["ipfs"]["local_cas_dir"]); cid=store.add_bytes(raw)
ledger=PermissionedLedger(cfg["blockchain"]["ledger_dir"],int(cfg["blockchain"]["validator_nodes"])); tx=AuthenticationTransaction(str(uuid.uuid4()),"federated-coordinator","global_model_commit",model_hash,cid,time.time(),{"framework":"BCFDL-HCE","consensus":"limited_raft"}); block=ledger.append(tx)
obj=serializable(r); obj.update({"sample_af_mfa_decisions":samples,"model_hash":model_hash,"content_id":cid,"ledger_block":block.index,"ledger_valid":ledger.verify()}); save_json(out/"results.json",obj); print(out/"results.json")
