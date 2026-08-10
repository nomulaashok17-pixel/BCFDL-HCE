from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import torch
import torch.nn.functional as F
from .hce import HashContextEncoder, hce_loss, class_prototypes, aggregate_prototypes
from .privacy import state_delta, clip_update, add_gaussian_noise, mask_update, secure_aggregate
from .security import robust_filter
from .utils import tensor_state_nbytes


@dataclass
class ClientResult:
    client_id: int
    update: dict
    samples: int
    train_loss: float
    upload_bytes: int
    prototypes: dict | None = None
    prototype_counts: dict | None = None


def clone_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def apply_delta(model, delta):
    s=model.state_dict()
    model.load_state_dict({k: s[k].detach().cpu()+delta[k] for k in s})


def weighted_average(updates: list[tuple[dict,int]]) -> dict:
    total=sum(n for _,n in updates); keys=updates[0][0]
    return {k:sum(u[k]*n for u,n in updates)/total for k in keys}


def _optimizer(model, cfg):
    opt=cfg["optimization"]
    if opt.get("optimizer","adam").lower()=="sgd":
        return torch.optim.SGD(model.parameters(), lr=float(opt["learning_rate"]), momentum=.9, weight_decay=float(opt.get("weight_decay",0)))
    return torch.optim.Adam(model.parameters(), lr=float(opt["learning_rate"]), weight_decay=float(opt.get("weight_decay",0)))


def train_client(model, loader, cfg, device, global_state, client_id: int, round_id: int,
                 method: str = "fedavg", hce: HashContextEncoder | None = None,
                 global_prototypes=None, previous_global=None, client_aux=None) -> ClientResult:
    model=deepcopy(model).to(device); model.train()
    hce=deepcopy(hce).to(device) if hce is not None else None
    params=list(model.parameters()) + (list(hce.parameters()) if hce is not None else [])
    opt=torch.optim.Adam(params, lr=float(cfg["optimization"]["learning_rate"]), weight_decay=float(cfg["optimization"].get("weight_decay",0)))
    ce=torch.nn.CrossEntropyLoss(); losses=[]; local_steps=0
    prox_mu=float(cfg.get("baselines",{}).get("fedprox_mu",.01))
    moon_mu=float(cfg.get("baselines",{}).get("moon_mu",.1))
    dyn_alpha=float(cfg.get("baselines",{}).get("feddyn_alpha",.01))
    pcounts={}; proto_acc=[]

    for _ in range(int(cfg["federated"]["local_epochs"])):
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            logits,z=model(x,return_embedding=True); loss=ce(logits,y)
            if method=="fedprox":
                prox=0.0
                for (name,p) in model.named_parameters():
                    if name in global_state: prox=prox+((p-global_state[name].to(device))**2).sum()
                loss=loss+0.5*prox_mu*prox
            if method=="moon" and previous_global is not None:
                with torch.no_grad(): _,zg=previous_global.to(device)(x,return_embedding=True)
                loss=loss+moon_mu*(1-F.cosine_similarity(z,zg,dim=1).mean())
            if method=="feddyn" and client_aux:
                linear=0.0; quad=0.0
                for name,p in model.named_parameters():
                    if name in global_state:
                        linear += (p*client_aux.get(name,torch.zeros_like(p).cpu()).to(device)).sum()
                        quad += ((p-global_state[name].to(device))**2).sum()
                loss=loss+dyn_alpha*(linear+0.5*quad)
            if hce is not None:
                relaxed=hce(z)
                hl=hce_loss(relaxed,y,global_prototypes,
                            float(cfg["hce"]["similarity_weight"]),float(cfg["hce"]["quantization_weight"]),float(cfg["hce"]["prototype_weight"]))
                loss=loss+hl.total
                pp=class_prototypes(relaxed.detach(),y)
                cnt={int(c.item()):int((y==c).sum().item()) for c in y.unique()}; proto_acc.append((pp,cnt))
                for c,n in cnt.items(): pcounts[c]=pcounts.get(c,0)+n
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu())); local_steps+=1
    local=clone_state(model); delta=state_delta(local,global_state)
    if method=="fednova":
        tau=max(1,local_steps); delta={k:v/tau for k,v in delta.items()}
    privacy=cfg.get("privacy",{})
    if privacy.get("clipping",False): delta,_=clip_update(delta,float(privacy.get("clip_norm",1.0)))
    if privacy.get("differential_privacy",False):
        std=float(privacy.get("noise_multiplier",0))*float(privacy.get("clip_norm",1.0))
        delta=add_gaussian_noise(delta,std,seed=int(cfg["experiment"]["seed"])+round_id*1000+client_id)
    protos=aggregate_prototypes(proto_acc) if proto_acc else None
    return ClientResult(client_id,delta,len(loader.dataset),sum(losses)/max(1,len(losses)),tensor_state_nbytes(delta),protos,pcounts or None)


class FederatedTrainer:
    def __init__(self, model, client_loaders, cfg, device, hce: HashContextEncoder|None=None, method: str|None=None):
        self.model=model.to(device); self.loaders=client_loaders; self.cfg=cfg; self.device=device; self.hce=hce
        self.method=(method or cfg["federated"].get("aggregation","fedavg")).lower(); self.history=[]; self.global_prototypes=None
        self.previous_global=None; self.client_aux={i:{} for i in range(len(client_loaders))}

    def _participants(self, round_id):
        n=len(self.loaders); frac=float(self.cfg["federated"].get("client_fraction",1.0)); k=max(1,math.ceil(n*frac))
        ids=list(range(n))
        if self.cfg["federated"].get("cluster_training",False):
            clusters=max(1,int(self.cfg["federated"].get("clusters",2)))
            buckets=[ids[i::clusters] for i in range(clusters)]
            if self.cfg["federated"].get("odd_even_schedule",False): ids=buckets[round_id%clusters]
        rng=random.Random(int(self.cfg["experiment"]["seed"])+round_id); rng.shuffle(ids)
        return ids[:min(k,len(ids))]

    def round(self, round_id: int):
        global_state=clone_state(self.model); ids=self._participants(round_id); results=[]
        for cid in ids:
            r=train_client(self.model,self.loaders[cid],self.cfg,self.device,global_state,cid,round_id,self.method,self.hce,self.global_prototypes,self.previous_global,self.client_aux[cid])
            results.append(r)
        updates=[r.update for r in results]
        if self.cfg.get("security",{}).get("robust_filter",False) and len(updates)>=3:
            keep=robust_filter(updates,float(self.cfg["security"].get("max_update_norm_zscore",3.5)),float(self.cfg["security"].get("cosine_floor",-.25)))
            results=[r for r,k in zip(results,keep) if k]
        privacy=self.cfg.get("privacy",{})
        if privacy.get("secure_aggregation",False) and len(results)>1:
            pids=[r.client_id for r in results]; masked=[]
            for r in results:
                masked.append((mask_update(r.update,r.client_id,pids,round_id,int(self.cfg["experiment"]["seed"])),r.samples))
            agg=secure_aggregate(masked)
        else: agg=weighted_average([(r.update,r.samples) for r in results])
        if self.method=="fednova":
            # FedNova normalized deltas are rescaled by average local-step proxy.
            agg={k:v*max(1,int(self.cfg["federated"]["local_epochs"])) for k,v in agg.items()}
        self.previous_global=deepcopy(self.model).cpu() if self.method=="moon" else self.previous_global
        apply_delta(self.model,agg); self.model.to(self.device)
        # SCAFFOLD-style correction: accumulate difference between client and aggregate updates.
        if self.method=="scaffold":
            for r in results: self.client_aux[r.client_id]={k:(r.update[k]-agg[k]).detach().cpu() for k in agg}
        if self.method=="feddyn":
            for r in results:
                prev=self.client_aux[r.client_id]
                self.client_aux[r.client_id]={k:prev.get(k,torch.zeros_like(v))+r.update[k] for k,v in r.update.items()}
        proto_items=[(r.prototypes,r.prototype_counts) for r in results if r.prototypes]
        if proto_items: self.global_prototypes=aggregate_prototypes(proto_items)
        record={"round":round_id,"participants":[r.client_id for r in results],"train_loss":sum(r.train_loss for r in results)/max(1,len(results)),"upload_bytes":sum(r.upload_bytes for r in results)}
        self.history.append(record); return record
