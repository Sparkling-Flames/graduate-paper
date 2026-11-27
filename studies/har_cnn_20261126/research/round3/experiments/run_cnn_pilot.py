#!/usr/bin/env python3
"""One-seed temporal CNN learning extension; official test entries stay sealed."""
from pathlib import Path
import argparse
import copy
import io
import json
import shutil
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
import run_pilot as base
from run_pilot import (now, digest_file, json_write, seed_all, state_hash,
    loss_terms, metrics, predict, epoch_orders, export_predictions, update_summary, VAL_SUBJECTS)

ROOT = Path(__file__).resolve().parent
HERE = ROOT / "cnn_pilot"
SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
CONFIG = dict(base.CONFIG) | {
    "protocol_version": "cnn_learning_pilot_v1",
    "phase": "single_seed_learning_extension",
    "input": "official train inertial signals; N x 9 x 128; channelwise train-only standardization",
    "teacher_hidden": [32, 64], "student_widths": [8], "paired_seeds": [20261121],
    "teacher_channels": [32, 64], "student_channels": [8, 16],
    "architecture": "Conv1d(k5,p2)-ReLU-MaxPool1d(2)-Conv1d(k5,p2)-ReLU-AdaptiveAvgPool1d(1)-Linear(6)",
    "signals_in_channel_order": SIGNALS,
    "preprocessing": "mean/std per input channel across training windows and time only; float32",
    "teacher_estimated_seconds_stop_threshold": 240,
    "cost_estimate_epoch": 5,
    "pilot_script_sha256": digest_file(Path(__file__)),
}


class ConvHAR(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        c1, c2 = hidden if len(hidden) == 2 else (hidden[0], 2 * hidden[0])
        self.features = nn.Sequential(
            nn.Conv1d(9, c1, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(c2, 6)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


MLP = ConvHAR  # The shared training routine's constructor name; actual model is ConvHAR.


def prepare_data():
    out = HERE / "data"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_val.npz").exists():
        return
    main_data = np.load(ROOT / "data" / "train_val.npz")
    opened = []
    allowed = {f"UCI HAR Dataset/train/Inertial Signals/{signal}_train.txt" for signal in SIGNALS}
    with zipfile.ZipFile(base.SOURCE) as outer:
        inner_bytes = outer.read("UCI HAR Dataset.zip")
    channels = []
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        for signal in SIGNALS:
            entry = f"UCI HAR Dataset/train/Inertial Signals/{signal}_train.txt"
            assert entry in allowed and "/test/" not in entry
            opened.append(entry)
            channels.append(np.loadtxt(io.BytesIO(inner.read(entry)), dtype=np.float32))
    x = np.stack(channels, axis=1)
    assert x.shape == (7352, 9, 128) and np.isfinite(x).all()
    it, iv = main_data["index_train"], main_data["index_val"]
    mean = x[it].mean(axis=(0, 2), dtype=np.float64, keepdims=True)
    scale = x[it].std(axis=(0, 2), dtype=np.float64, keepdims=True)
    scale[scale == 0] = 1
    xtr = np.ascontiguousarray((x[it] - mean) / scale, dtype=np.float32)
    xv = np.ascontiguousarray((x[iv] - mean) / scale, dtype=np.float32)
    assert xtr.shape == (5902, 9, 128) and xv.shape == (1450, 9, 128)
    assert not set(main_data["subject_train"]) & set(main_data["subject_val"])
    np.savez_compressed(out / "train_val.npz", x_train=xtr, x_val=xv,
        **{key: main_data[key] for key in ["y_train", "y_val", "subject_train", "subject_val", "index_train", "index_val"]})
    np.savez(out / "channel_scaler_train_only.npz", mean=mean, scale=scale)
    for filename in ["grouped_indices.csv", "subject_class_counts.csv"]:
        shutil.copyfile(ROOT / "data" / filename, out / filename)
    for seed in [CONFIG["teacher_seed"]] + CONFIG["paired_seeds"]:
        shutil.copyfile(ROOT / "data" / f"epoch_orders_seed{seed}.npy", out / f"epoch_orders_seed{seed}.npy")
    json_write(out / "manifest.json", {
        "prepared_utc": now(), "source_zip_sha256": digest_file(base.SOURCE),
        "opened_inner_entries": opened,
        "official_test_feature_label_entries_opened": [],
        "labels_subjects_indices_source": "../../data/train_val.npz (official train only)",
        "main_data_sha256": digest_file(ROOT / "data" / "train_val.npz"),
        "cached_sha256": digest_file(out / "train_val.npz"),
        "channel_order": SIGNALS, "shape_train": list(xtr.shape), "shape_val": list(xv.shape),
        "train_subjects": CONFIG["train_subjects"], "val_subjects": CONFIG["val_subjects"],
        "mean": mean.flatten().tolist(), "scale": scale.flatten().tolist(),
        "scaler_fit_rows": it.tolist(), "official_test_evaluated": False,
    })


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
        if epoch == CONFIG["cost_estimate_epoch"]:
            elapsed = time.perf_counter() - started
            estimated = elapsed / epoch * CONFIG["epochs"]
            print(json.dumps({"event": "cnn_cost_probe", "run": name,
                "epochs": epoch, "elapsed_seconds": elapsed,
                "estimated_80epoch_seconds": estimated,
                "current_train_accuracy": tr["accuracy"], "current_val_macro_f1": va["macro_f1"]}), flush=True)
            if name.startswith("teacher") and estimated > CONFIG["teacher_estimated_seconds_stop_threshold"]:
                pd.DataFrame(curves).to_csv(path / "curves.csv", index=False)
                torch.save(model.state_dict(), path / "cost_probe_state.pt")
                json_write(HERE / "stopped.json", {"reason": "prespecified_cost_threshold_exceeded",
                    "estimated_seconds": estimated, "threshold_seconds": CONFIG["teacher_estimated_seconds_stop_threshold"]})
                raise RuntimeError("CNN extension stopped at predeclared cost threshold")
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



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "benchmark", "summarize"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    (HERE / "protocol.md").write_text("# CNN学习先导\n\n完整事先决定见../decision_log.md的D3。\n\n"
        + (ROOT / "decision_log.md").read_text().split("## D3：", 1)[1], encoding="utf-8")
    base.HERE, base.CONFIG = HERE, CONFIG
    base.lock_configuration()
    if args.action == "summarize":
        update_summary()
        return
    prepare_data()
    base.validate_losses()
    if args.action == "prepare":
        print(json.dumps({"event": "cnn_prepared", "train_shape": [5902, 9, 128],
                          "val_shape": [1450, 9, 128], "test_opened": False}), flush=True)
        return
    data = base.load_data()
    if args.action == "benchmark":
        for path in sorted((HERE / "runs").glob("*/result.json")):
            result = json.loads(path.read_text())
            model = ConvHAR(result["hidden"])
            model.load_state_dict(torch.load(path.parent / "best_state.pt", weights_only=True))
            result["cpu_forward"] = base.measure_cpu(model, data["xv"])
            result["cpu_forward"].update(measured_utc=now(), phase="standalone_after_all_training")
            json_write(path, result)
            print(json.dumps({"event": "cnn_benchmarked", "run": result["run"],
                              "median_ms": result["cpu_forward"]["median_ms"]}), flush=True)
        update_summary()
        return
    teacher_name = f"teacher_seed{CONFIG['teacher_seed']}"
    teacher_result = run_one(teacher_name, CONFIG["teacher_hidden"], CONFIG["teacher_seed"], "ce", data)
    if teacher_result["train_metrics"]["accuracy"] < .95 or teacher_result["validation_metrics"]["macro_f1"] < .85:
        json_write(HERE / "stopped.json", {"reason": "teacher_training_or_validation_gate_failed",
            "train_accuracy": teacher_result["train_metrics"]["accuracy"],
            "val_macro_f1": teacher_result["validation_metrics"]["macro_f1"],
            "additional_hyperparameter_search": False})
        print(json.dumps({"event": "cnn_stopped", "reason": "teacher gate failed"}), flush=True)
        return
    teacher = ConvHAR(CONFIG["teacher_hidden"])
    teacher.load_state_dict(torch.load(HERE / "runs" / teacher_name / "best_state.pt", weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_hash = state_hash(teacher.state_dict())
    teacher_logits = predict(teacher, data["xtr"])
    torch.save(teacher_logits, HERE / "data" / "teacher_train_logits.pt")
    json_write(HERE / "data" / "teacher_logits_manifest.json", {
        "teacher_state_sha256": teacher_hash, "row_order": "same as data/train_val.npz index_train",
        "rows": len(teacher_logits), "requires_grad": False, "official_test_evaluated": False})
    hashes, orders = [], []
    for method in ["ce", "ls", "kd"]:
        seed = CONFIG["paired_seeds"][0]
        result = run_one(f"student_w8_{method}_seed{seed}", [8], seed, method, data,
                         teacher_logits if method == "kd" else None, teacher_hash if method == "kd" else None)
        hashes.append(result["initial_state_sha256"])
        orders.append(result["minibatch_orders_file_sha256"])
    assert len(set(hashes)) == 1 and len(set(orders)) == 1
    assert teacher_hash == state_hash(teacher.state_dict())
    json_write(HERE / "completion.json", {"models_completed": 4, "paired_initialization_and_order_verified": True,
        "official_test_evaluated": False, "completed_utc": now(), "single_seed_learning_only": True})
    update_summary()


if __name__ == "__main__":
    main()
