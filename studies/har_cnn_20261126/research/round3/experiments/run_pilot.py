#!/usr/bin/env python3
"""UCI HAR development-only MLP pilot. Never opens official test X/y entries."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "uci_har_official.zip"
TRAIN_SUBJECTS = [1,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27]
VAL_SUBJECTS = [3,28,29,30]
CLASS_NAMES = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"]
CONFIG = {
    "protocol_version": "mlp_pilot_v1", "input": "UCI official 561 features",
    "train_subjects": TRAIN_SUBJECTS, "val_subjects": VAL_SUBJECTS,
    "official_test_opened": False, "teacher_hidden": [256,128],
    "student_widths": [16,32,64], "teacher_seed": 20261120,
    "paired_seeds": [20261121,20261122,20261123],
    "epochs": 80, "batch_size": 128, "optimizer": "Adam", "lr": .001,
    "weight_decay": 0., "activation": "ReLU", "dropout": 0.,
    "kd_temperature": 2., "kd_lambda_teacher_weight": .5,
    "kd_temperature_squared": True, "kl_reduction": "batchmean",
    "label_smoothing_uniform_epsilon": .1, "threads": 2,
    "checkpoint_selection": "max validation macro-F1, then min hard-label NLL, then earliest epoch",
    "preprocessing": "StandardScaler fitted on train subjects only; float32 model inputs",
    "early_stopping": False, "device": "cpu", "dtype": "float32",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def lock_configuration():
    path = HERE/"config.json"
    if path.exists() and json.loads(path.read_text()) != CONFIG:
        raise RuntimeError("Existing configuration differs; do not silently overwrite a pilot protocol")
    json_write(path, CONFIG)
    json_write(HERE/"environment.json", {
        "recorded_utc": now(), "python": sys.version, "executable":sys.executable,
        "platform":platform.platform(), "machine":platform.machine(),
        "torch":torch.__version__, "numpy":np.__version__, "pandas":pd.__version__,
        "sklearn":__import__("sklearn").__version__, "cpu_count":os.cpu_count(),
        "torch_threads":torch.get_num_threads(), "torch_interop_threads":torch.get_num_interop_threads(),
        "torch_build_config":torch.__config__.show(),
        "protocol_sha256":digest_file(HERE/"protocol.md"),
        "script_sha256":digest_file(Path(__file__)), "command":sys.argv,
    })


def prepare_data():
    out = HERE/"data"
    out.mkdir(exist_ok=True)
    if (out/"train_val.npz").exists():
        manifest = json.loads((out/"manifest.json").read_text())
        if manifest["train_subjects"] != TRAIN_SUBJECTS or manifest["val_subjects"] != VAL_SUBJECTS:
            raise RuntimeError("Split cache mismatch")
        return
    opened = []
    allowed = {
        "UCI HAR Dataset/train/X_train.txt",
        "UCI HAR Dataset/train/y_train.txt",
        "UCI HAR Dataset/train/subject_train.txt",
    }
    with zipfile.ZipFile(SOURCE) as outer:
        inner_data = outer.read("UCI HAR Dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_data)) as inner:
        def read(name, dtype):
            if name not in allowed or "/test/" in name:
                raise PermissionError("Only predeclared official train entries may be opened")
            opened.append(name)
            return np.loadtxt(io.BytesIO(inner.read(name)), dtype=dtype)
        x = read("UCI HAR Dataset/train/X_train.txt", np.float32)
        y = read("UCI HAR Dataset/train/y_train.txt", np.int64) - 1
        subjects = read("UCI HAR Dataset/train/subject_train.txt", np.int64)
    assert x.shape == (7352,561) and y.shape == (7352,) and subjects.shape == (7352,)
    assert np.isfinite(x).all() and set(np.unique(y)) == set(range(6))
    train_idx = np.flatnonzero(np.isin(subjects, TRAIN_SUBJECTS))
    val_idx = np.flatnonzero(np.isin(subjects, VAL_SUBJECTS))
    assert len(train_idx)==5902 and len(val_idx)==1450
    assert len(np.intersect1d(train_idx,val_idx))==0
    assert not set(subjects[train_idx]) & set(subjects[val_idx])
    assert len(train_idx)+len(val_idx)==len(y)
    scaler = StandardScaler().fit(x[train_idx])
    xtr = np.ascontiguousarray(scaler.transform(x[train_idx]), dtype=np.float32)
    xv = np.ascontiguousarray(scaler.transform(x[val_idx]), dtype=np.float32)
    np.savez_compressed(out/"train_val.npz", x_train=xtr, y_train=y[train_idx],
        subject_train=subjects[train_idx], index_train=train_idx,
        x_val=xv, y_val=y[val_idx], subject_val=subjects[val_idx], index_val=val_idx)
    np.savez(out/"scaler_train_only.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)
    idx = pd.DataFrame({"official_train_row0":np.arange(len(y)),"subject":subjects,
                        "label0":y,"label1":y+1,
                        "split":np.where(np.isin(subjects,VAL_SUBJECTS),"validation","train")})
    idx.to_csv(out/"grouped_indices.csv", index=False)
    idx.groupby(["split","subject","label0"]).size().rename("windows").reset_index().to_csv(out/"subject_class_counts.csv",index=False)
    json_write(out/"manifest.json", {
        "prepared_utc":now(), "source":str(SOURCE), "source_sha256":digest_file(SOURCE),
        "opened_inner_entries":opened, "official_test_feature_label_entries_opened":[],
        "outer_container_note":"The inner ZIP container was read as bytes; official test entries were never opened/decompressed/parsed.",
        "train_subjects":TRAIN_SUBJECTS,"val_subjects":VAL_SUBJECTS,
        "train_shape":list(xtr.shape),"val_shape":list(xv.shape),
        "train_class_counts":np.bincount(y[train_idx],minlength=6).tolist(),
        "val_class_counts":np.bincount(y[val_idx],minlength=6).tolist(),
        "class_names":CLASS_NAMES,"zero_variance_features":int(np.sum(scaler.var_==0)),
        "scaler_fit_rows":train_idx.tolist(),"cached_sha256":digest_file(out/"train_val.npz"),
    })


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        dims = [561]+list(hidden)+[6]
        layers=[]
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            layers.append(nn.Linear(a,b,bias=True))
            if i<len(dims)-2:
                layers.append(nn.ReLU())
        self.layers=nn.Sequential(*layers)
    def forward(self,x):
        return self.layers(x)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def state_hash(state):
    h=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        h.update(key.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def loss_terms(logits,y,method,teacher_logits=None):
    ce=F.cross_entropy(logits,y)
    if method=="ce":
        return ce,ce,None
    if method=="ls":
        return F.cross_entropy(logits,y,label_smoothing=CONFIG["label_smoothing_uniform_epsilon"]),ce,None
    if method=="kd":
        t=CONFIG["kd_temperature"];lam=CONFIG["kd_lambda_teacher_weight"]
        target=F.softmax(teacher_logits.detach()/t,dim=1)
        kl=F.kl_div(F.log_softmax(logits/t,dim=1),target,reduction="batchmean")
        return (1-lam)*ce+lam*t*t*kl,ce,kl
    raise ValueError(method)


def validate_losses():
    torch.manual_seed(12345)
    s=torch.randn(5,6,dtype=torch.float64,requires_grad=True)
    t=torch.randn(5,6,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([0,1,2,3,4]);eps=CONFIG["label_smoothing_uniform_epsilon"]
    target=(1-eps)*F.one_hot(y,6).to(s.dtype)+eps/6
    manual_ls=-(target*F.log_softmax(s,dim=1)).sum(dim=1).mean()
    ls=loss_terms(s,y,"ls")[0]
    temp=CONFIG["kd_temperature"];q=F.softmax(t.detach()/temp,dim=1)
    manual_kl=(q*(q.log()-F.log_softmax(s/temp,dim=1))).sum(dim=1).mean()
    kd,_,kl=loss_terms(s,y,"kd",t)
    kd.backward()
    assert abs(float((ls-manual_ls).detach()))<1e-12
    assert abs(float((kl-manual_kl).detach()))<1e-12
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None
    identical_kl=F.kl_div(F.log_softmax(t.detach()/temp,dim=1),q,reduction="batchmean")
    assert abs(float(identical_kl))<1e-12
    json_write(HERE/"implementation_checks.json",{
        "uniform_ls_matches_manual":True,"kl_batchmean_matches_manual":True,
        "teacher_receives_no_gradient":True,"student_receives_finite_gradient":True,
        "identical_distribution_kl_zero":True,"checked_utc":now()})


def metrics(logits,y):
    if isinstance(logits,np.ndarray):logits=torch.from_numpy(logits)
    if isinstance(y,np.ndarray):y=torch.from_numpy(y)
    pred=logits.argmax(dim=1).numpy();truth=y.numpy()
    cm=np.bincount(truth*6+pred,minlength=36).reshape(6,6)
    tp=cm.diagonal().astype(float); true_count=cm.sum(axis=1);pred_count=cm.sum(axis=0)
    recall=np.divide(tp,true_count,out=np.zeros(6),where=true_count>0)
    precision=np.divide(tp,pred_count,out=np.zeros(6),where=pred_count>0)
    f1=np.divide(2*tp,true_count+pred_count,out=np.zeros(6),where=(true_count+pred_count)>0)
    return {"accuracy":float(tp.sum()/cm.sum()),"macro_f1":float(f1.mean()),
            "nll":float(F.cross_entropy(logits,y)),"confusion":cm.tolist(),
            "class_f1":f1.tolist(),"class_recall":recall.tolist(),"class_precision":precision.tolist()}


@torch.inference_mode()
def predict(model,x,batch=1024):
    model.eval()
    return torch.cat([model(x[i:i+batch]) for i in range(0,len(x),batch)],dim=0)


def epoch_orders(seed,n):
    path=HERE/"data"/f"epoch_orders_seed{seed}.npy"
    if path.exists():
        orders=np.load(path)
    else:
        rng=np.random.default_rng(seed+1000000)
        orders=np.stack([rng.permutation(n) for _ in range(CONFIG["epochs"])]).astype(np.int64)
        np.save(path,orders)
    assert orders.shape==(CONFIG["epochs"],n)
    return torch.from_numpy(orders),digest_file(path)


def measure_cpu(model,x):
    model.eval();one=x[:1]
    with torch.inference_mode():
        for _ in range(100):model(one)
        times=[]
        for _ in range(1000):
            start=time.perf_counter_ns();model(one);times.append((time.perf_counter_ns()-start)/1e6)
    return {"batch_size":1,"warmup":100,"repetitions":1000,"threads":torch.get_num_threads(),
            "median_ms":float(np.median(times)),"p95_ms":float(np.percentile(times,95)),
            "mean_ms":float(np.mean(times)),"scope":"in-memory float32 CPU forward incl. Python call; no feature extraction or loading"}


def export_predictions(path, logits, y, subjects, indices):
    logits_np=logits.numpy();probs=logits.softmax(dim=1).numpy()
    np.savez_compressed(path/"validation_predictions.npz",logits=logits_np,probabilities=probs,
                        y=y.numpy(),subject=subjects,official_train_row0=indices)
    frame=pd.DataFrame({"official_train_row0":indices,"subject":subjects,"y":y.numpy(),
                        "prediction":logits_np.argmax(axis=1)})
    for i in range(6):frame[f"p{i}"]=probs[:,i];frame[f"logit{i}"]=logits_np[:,i]
    frame.to_csv(path/"validation_predictions.csv",index=False)


def run_one(name,hidden,seed,method,data,teacher_logits=None,teacher_hash=None):
    path=HERE/"runs"/name;path.mkdir(parents=True,exist_ok=True)
    if (path/"result.json").exists():
        print(json.dumps({"event":"reuse_completed","run":name}),flush=True)
        return json.loads((path/"result.json").read_text())
    xtr,ytr,xv,yv=data["xtr"],data["ytr"],data["xv"],data["yv"]
    seed_all(seed);model=MLP(hidden)
    initial_hash=state_hash(model.state_dict())
    torch.save(model.state_dict(),path/"initial_state.pt")
    orders,orders_hash=epoch_orders(seed,len(ytr))
    json_write(path/"run_config.json",CONFIG|{
        "run":name,"hidden":hidden,"seed":seed,"method":method,
        "initial_state_sha256":initial_hash,"minibatch_orders_file_sha256":orders_hash,
        "teacher_state_sha256":teacher_hash,"started_utc":now(),"command":sys.argv})
    optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG["lr"],weight_decay=CONFIG["weight_decay"])
    best_key=None;best_state=None;best_epoch=None;curves=[]
    started=time.perf_counter()
    for epoch,order in enumerate(orders,start=1):
        model.train();loss_sum=ce_sum=kl_sum=0.
        epoch_start=time.perf_counter()
        for ids in order.split(CONFIG["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits=model(xtr[ids]);tl=None if teacher_logits is None else teacher_logits[ids]
            loss,ce,kl=loss_terms(logits,ytr[ids],method,tl)
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
            n=len(ids);loss_sum+=float(loss.detach())*n;ce_sum+=float(ce.detach())*n
            if kl is not None:kl_sum+=float(kl.detach())*n
        train_logits=predict(model,xtr);val_logits=predict(model,xv)
        tr=metrics(train_logits,ytr);va=metrics(val_logits,yv)
        row={"epoch":epoch,"objective_loss":loss_sum/len(ytr),"batch_hard_ce":ce_sum/len(ytr),
             "batch_kl_unscaled":kl_sum/len(ytr) if method=="kd" else None,
             "train_accuracy":tr["accuracy"],"train_macro_f1":tr["macro_f1"],"train_nll":tr["nll"],
             "val_accuracy":va["accuracy"],"val_macro_f1":va["macro_f1"],"val_nll":va["nll"],
             "epoch_seconds":time.perf_counter()-epoch_start}
        curves.append(row)
        key=(va["macro_f1"],-va["nll"])
        if best_key is None or key>best_key:
            best_key=key;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if epoch%10==0:
            pd.DataFrame(curves).to_csv(path/"curves.csv",index=False)
    training_seconds=time.perf_counter()-started
    torch.save(model.state_dict(),path/"last_state.pt")
    last_metrics=metrics(predict(model,xv),yv)
    model.load_state_dict(best_state);torch.save(best_state,path/"best_state.pt")
    train_logits=predict(model,xtr);val_logits=predict(model,xv)
    tr=metrics(train_logits,ytr);va=metrics(val_logits,yv)
    export_predictions(path,val_logits,yv,data["sv"],data["iv"])
    per_subject={str(s):metrics(val_logits[data["sv"]==s],yv[data["sv"]==s]) for s in VAL_SUBJECTS}
    pd.DataFrame(curves).to_csv(path/"curves.csv",index=False)
    result={"run":name,"method":method,"hidden":hidden,"width":hidden[0] if len(hidden)==1 else 0,
            "seed":seed,"best_epoch":best_epoch,"epochs_completed":CONFIG["epochs"],
            "optimizer_steps":CONFIG["epochs"]*((len(ytr)+CONFIG["batch_size"]-1)//CONFIG["batch_size"]),
            "train_metrics":tr,"validation_metrics":va,"last_validation_metrics":last_metrics,
            "per_subject_validation":per_subject,"training_seconds":training_seconds,
            "parameters":sum(p.numel() for p in model.parameters()),
            "float32_parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
            "checkpoint_bytes":(path/"best_state.pt").stat().st_size,"cpu_forward":None,
            "initial_state_sha256":initial_hash,"best_state_sha256":state_hash(best_state),
            "minibatch_orders_file_sha256":orders_hash,"teacher_state_sha256":teacher_hash,
            "official_test_evaluated":False,"completed_utc":now()}
    json_write(path/"result.json",result)
    print(json.dumps({"event":"completed","run":name,"parameters":result["parameters"],
        "best_epoch":best_epoch,"train_accuracy":round(tr["accuracy"],6),
        "val_accuracy":round(va["accuracy"],6),"val_macro_f1":round(va["macro_f1"],6),
        "train_seconds":round(training_seconds,2)}),flush=True)
    update_summary()
    return result


def load_data():
    z=np.load(HERE/"data/train_val.npz")
    return {"xtr":torch.from_numpy(z["x_train"]),"ytr":torch.from_numpy(z["y_train"]),
            "xv":torch.from_numpy(z["x_val"]),"yv":torch.from_numpy(z["y_val"]),
            "sv":z["subject_val"],"iv":z["index_val"],"st":z["subject_train"],"it":z["index_train"]}


def update_summary():
    rows=[]
    for path in sorted((HERE/"runs").glob("*/result.json")):
        r=json.loads(path.read_text());va=r["validation_metrics"];tr=r["train_metrics"]
        rows.append({"run":r["run"],"method":r["method"],"width":r["width"],"seed":r["seed"],
            "best_epoch":r["best_epoch"],"parameters":r["parameters"],
            "val_macro_f1":va["macro_f1"],"val_accuracy":va["accuracy"],"val_nll":va["nll"],
            "train_accuracy":tr["accuracy"],"train_nll":tr["nll"],
            "last_val_macro_f1":r["last_validation_metrics"]["macro_f1"],
            "training_seconds":r["training_seconds"],"median_forward_ms":(r.get("cpu_forward") or {}).get("median_ms"),
            "p95_forward_ms":(r.get("cpu_forward") or {}).get("p95_ms"),"checkpoint_bytes":r["checkpoint_bytes"]})
    if not rows:return
    df=pd.DataFrame(rows)
    for column in ["median_forward_ms","p95_forward_ms"]:
        df[column]=pd.to_numeric(df[column],errors="coerce")
    df.to_csv(HERE/"summary_runs.csv",index=False)
    student=df[df.width>0]
    if len(student):
        grouped=student.groupby(["width","method"])[["val_macro_f1","val_accuracy","training_seconds","median_forward_ms"]].agg(["mean","std","count"])
        grouped.to_csv(HERE/"summary_grouped.csv")
        pivot=student.pivot(index=["width","seed"],columns="method",values="val_macro_f1")
        if {"ce","ls","kd"}.issubset(pivot.columns):
            pivot["kd_minus_ce"]=pivot.kd-pivot.ce;pivot["kd_minus_ls"]=pivot.kd-pivot.ls
            pivot.to_csv(HERE/"paired_differences.csv")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","train","summarize","benchmark"])
    parser.add_argument("--seeds",type=int,nargs="+",default=[20261121])
    parser.add_argument("--widths",type=int,nargs="+",default=[16,32,64])
    parser.add_argument("--methods",nargs="+",default=["ce","ls","kd"],choices=["ce","ls","kd"])
    args=parser.parse_args()
    torch.set_num_threads(CONFIG["threads"]);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    (HERE/"runs").mkdir(exist_ok=True)
    lock_configuration()
    if args.action=="summarize":update_summary();return
    prepare_data();validate_losses()
    if args.action=="prepare":
        print(json.dumps({"event":"prepared","train_windows":5902,"val_windows":1450,"test_opened":False}),flush=True);return
    if args.action=="benchmark":
        data=load_data()
        for result_path in sorted((HERE/"runs").glob("*/result.json")):
            result=json.loads(result_path.read_text())
            model=MLP(result["hidden"])
            model.load_state_dict(torch.load(result_path.parent/"best_state.pt",map_location="cpu",weights_only=True))
            result["cpu_forward"]=measure_cpu(model,data["xv"])
            result["cpu_forward"]["measured_utc"]=now()
            result["cpu_forward"]["phase"]="standalone_after_all_training"
            json_write(result_path,result)
            print(json.dumps({"event":"benchmarked","run":result["run"],"median_ms":result["cpu_forward"]["median_ms"]}),flush=True)
        update_summary();return
    if not set(args.seeds)<=set(CONFIG["paired_seeds"]) or not set(args.widths)<=set(CONFIG["student_widths"]):
        raise ValueError("Seed/width outside predeclared protocol")
    data=load_data()
    teacher_name=f"teacher_seed{CONFIG['teacher_seed']}"
    teacher_result=run_one(teacher_name,CONFIG["teacher_hidden"],CONFIG["teacher_seed"],"ce",data)
    teacher=MLP(CONFIG["teacher_hidden"])
    teacher.load_state_dict(torch.load(HERE/"runs"/teacher_name/"best_state.pt",map_location="cpu",weights_only=True))
    teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    teacher_hash=state_hash(teacher.state_dict());teacher_logits=predict(teacher,data["xtr"])
    torch.save(teacher_logits,HERE/"data/teacher_train_logits.pt")
    json_write(HERE/"data/teacher_logits_manifest.json",{
        "teacher_state_sha256":teacher_hash,"rows":len(teacher_logits),
        "row_order":"same as train_val.npz index_train","requires_grad":teacher_logits.requires_grad,
        "official_test_evaluated":False})
    for seed in args.seeds:
        for width in args.widths:
            hashes=[];orders=[]
            for method in args.methods:
                result=run_one(f"student_w{width}_{method}_seed{seed}",[width],seed,method,data,
                               teacher_logits if method=="kd" else None,teacher_hash if method=="kd" else None)
                hashes.append(result["initial_state_sha256"]);orders.append(result["minibatch_orders_file_sha256"])
            assert len(set(hashes))==1 and len(set(orders))==1,"Unpaired initialization or batch order"
    assert teacher_hash==state_hash(teacher.state_dict()),"Teacher mutated during student training"
    update_summary()


if __name__=="__main__":main()
