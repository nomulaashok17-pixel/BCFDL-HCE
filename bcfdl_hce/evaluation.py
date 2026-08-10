from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    out={
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred,average="binary" if len(np.unique(y_true))==2 else "macro",zero_division=0),
        "recall": recall_score(y_true,y_pred,average="binary" if len(np.unique(y_true))==2 else "macro",zero_division=0),
        "f1": f1_score(y_true,y_pred,average="binary" if len(np.unique(y_true))==2 else "macro",zero_division=0),
        "confusion_matrix": confusion_matrix(y_true,y_pred).tolist(),
    }
    if y_score is not None and len(np.unique(y_true))==2:
        try:
            out["roc_auc"]=roc_auc_score(y_true,y_score)
            out["pr_auc"]=average_precision_score(y_true,y_score)
        except ValueError: pass
    if len(np.unique(y_true))==2:
        cm=confusion_matrix(y_true,y_pred,labels=[0,1])
        tn,fp,fn,tp=cm.ravel()
        out["far"] = fp/max(1,fp+tn)
        out["frr"] = fn/max(1,fn+tp)
    return out


def evaluate_model(model, loader, device) -> dict:
    model.eval(); yt=[]; yp=[]; ys=[]; loss=[]
    ce=torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); logits=model(x); loss.append(float(ce(logits,y)))
            p=torch.softmax(logits,dim=1); pred=p.argmax(1)
            yt.extend(y.cpu().tolist()); yp.extend(pred.cpu().tolist())
            if p.shape[1]==2: ys.extend(p[:,1].cpu().tolist())
    out=classification_metrics(yt,yp,ys if ys else None); out["loss"]=float(np.mean(loss) if loss else np.nan); return out


@contextmanager
def timer():
    box={"seconds":0.0}; start=time.perf_counter()
    try: yield box
    finally: box["seconds"]=time.perf_counter()-start
