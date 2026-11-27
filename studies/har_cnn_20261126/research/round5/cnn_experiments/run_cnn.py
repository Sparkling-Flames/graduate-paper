#!/usr/bin/env python3
"""Fixed-teacher CNN capacity/budget development experiment; no official test reads."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "round3" / "experiments"
OLD_CNN = OLD / "cnn_pilot"
TRAIN_SUBJECTS = [1, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27]
VAL_SUBJECTS = [3, 28, 29, 30]
SEEDS = [20261121, 20261122, 20261123]
CONFIG = {
    "protocol_version": "cnn_fixed_teacher_budget_capacity_v1",
    "input": "official train inertial signals, N x 9 x 128, unchanged audited channel scaler",
    "train_subjects": TRAIN_SUBJECTS, "val_subjects": VAL_SUBJECTS,
    "student_widths": [4, 8, 16], "main_methods": ["ce", "ls", "kd_ce"],
    "diagnostic_width": 8, "diagnostic_method": "kd_ls", "paired_seeds": SEEDS,
    "student_epochs": 160, "budgets": [80, 160], "teacher_epochs": 80,
    "teacher_seed": 20261120, "teacher_channels": [32, 64],
    "ce_teacher_source": "frozen round3 best80, epoch17, no retraining",
    "ls_teacher_source": "new 80epoch LS teacher; freeze best80",
    "architecture": "Conv9-w(k5,p2)-ReLU-MaxPool2-Convw-2w(k5,p2)-ReLU-GAP-Linear6",
    "batch_size": 128, "optimizer": "Adam", "lr": 0.001, "weight_decay": 0.0,
    "dropout": 0.0, "batch_norm": False, "augmentation": False,
    "kd_temperature": 2.0, "kd_lambda_teacher_weight": 0.5,
    "kl_reduction": "batchmean", "kd_temperature_squared": True,
    "label_smoothing_uniform_epsilon": 0.1, "threads": 2, "interop_threads": 1,
    "checkpoint_selection": "max val macro-F1, then min hard-label NLL, then earliest epoch",
    "early_stopping": False, "device": "cpu", "dtype": "float32",
    "official_test_opened": False, "benchmark": False,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lock_configuration():
    path = HERE / "config.json"
    if path.exists() and json.loads(path.read_text()) != CONFIG:
        raise RuntimeError("Refusing to overwrite a different experimental configuration")
    json_write(path, CONFIG)
    env = HERE / "environment.json"
    if not env.exists():
        json_write(env, {"recorded_utc": now(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "protocol_sha256": file_hash(HERE / "protocol.md"),
            "script_sha256": file_hash(__file__), "command": sys.argv})


class ConvHAR(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(9, width, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(width, 2 * width, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(2 * width, 6)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


def loss_terms(logits, y, method, teacher_logits=None):
    ce = F.cross_entropy(logits, y)
    if method == "ce":
        return ce, ce, None
    if method == "ls":
        return F.cross_entropy(logits, y, label_smoothing=0.1), ce, None
    if method in {"kd_ce", "kd_ls"}:
        if teacher_logits is None:
            raise ValueError("KD requires the frozen teacher's aligned logits")
        t, lam = CONFIG["kd_temperature"], CONFIG["kd_lambda_teacher_weight"]
        q = F.softmax(teacher_logits.detach() / t, dim=1)
        kl = F.kl_div(F.log_softmax(logits / t, dim=1), q, reduction="batchmean")
        return (1 - lam) * ce + lam * t * t * kl, ce, kl
    raise ValueError(method)


def metrics(logits, y):
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    pred, truth = logits.argmax(dim=1).numpy(), y.numpy()
    cm = np.bincount(truth * 6 + pred, minlength=36).reshape(6, 6)
    tp = cm.diagonal().astype(float)
    true_count, pred_count = cm.sum(1), cm.sum(0)
    recall = np.divide(tp, true_count, out=np.zeros(6), where=true_count > 0)
    precision = np.divide(tp, pred_count, out=np.zeros(6), where=pred_count > 0)
    f1 = np.divide(2 * tp, true_count + pred_count, out=np.zeros(6), where=true_count + pred_count > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
        "nll": float(F.cross_entropy(logits, y)), "confusion": cm.tolist(),
        "class_f1": f1.tolist(), "class_recall": recall.tolist(), "class_precision": precision.tolist()}


@torch.inference_mode()
def predict(model, x, batch=1024):
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)], dim=0)


def prepare_data():
    out = HERE / "data"
    out.mkdir(parents=True, exist_ok=True)
    source = OLD_CNN / "data"
    original_manifest = json.loads((source / "manifest.json").read_text())
    assert original_manifest["official_test_evaluated"] is False
    assert original_manifest["official_test_feature_label_entries_opened"] == []
    assert all("/train/" in name and "/test/" not in name for name in original_manifest["opened_inner_entries"])
    assert file_hash(source / "train_val.npz") == original_manifest["cached_sha256"]
    for name in ["train_val.npz", "channel_scaler_train_only.npz", "grouped_indices.csv", "subject_class_counts.csv"]:
        target = out / name
        if target.exists():
            assert file_hash(target) == file_hash(source / name), ("Cache mismatch", name)
        else:
            shutil.copyfile(source / name, target)
    z = np.load(out / "train_val.npz", allow_pickle=False)
    assert z["x_train"].shape == (5902, 9, 128) and z["x_val"].shape == (1450, 9, 128)
    assert z["x_train"].dtype == z["x_val"].dtype == np.float32
    assert np.isfinite(z["x_train"]).all() and np.isfinite(z["x_val"]).all()
    assert set(z["subject_train"]) == set(TRAIN_SUBJECTS) and set(z["subject_val"]) == set(VAL_SUBJECTS)
    assert not set(z["subject_train"]) & set(z["subject_val"])
    assert len(np.intersect1d(z["index_train"], z["index_val"])) == 0
    assert set(np.r_[z["index_train"], z["index_val"]]) == set(range(7352))
    assert set(z["y_train"]) == set(z["y_val"]) == set(range(6))
    source_mlp = np.load(OLD / "data/train_val.npz", allow_pickle=False)
    for key in ["y_train", "y_val", "subject_train", "subject_val", "index_train", "index_val"]:
        assert np.array_equal(z[key], source_mlp[key]), key
    mean_err = float(np.abs(z["x_train"].mean(axis=(0, 2), dtype=np.float64)).max())
    std_err = float(np.abs(z["x_train"].std(axis=(0, 2), dtype=np.float64) - 1).max())
    assert mean_err < 1e-6 and std_err < 1e-6
    order_provenance = {}
    for seed in [CONFIG["teacher_seed"]] + SEEDS:
        path = out / f"epoch_orders_seed{seed}.npy"
        if not path.exists():
            rng = np.random.default_rng(seed + 1000000)
            orders = np.stack([rng.permutation(5902) for _ in range(160)]).astype(np.int64)
            np.save(path, orders)
        orders = np.load(path, allow_pickle=False)
        assert orders.shape == (160, 5902)
        assert np.array_equal(np.sort(orders, axis=1), np.broadcast_to(np.arange(5902), orders.shape))
        old_orders = np.load(OLD / "data" / path.name, allow_pickle=False)
        assert np.array_equal(orders[:80], old_orders)
        order_provenance[str(seed)] = {"file_sha256": file_hash(path),
            "first80_content_sha256": array_hash(orders[:80]), "old80_content_sha256": array_hash(old_orders)}
    json_write(out / "manifest.json", {"prepared_utc": now(), "source_cache": str(source / "train_val.npz"),
        "source_cache_sha256": file_hash(source / "train_val.npz"), "cached_sha256": file_hash(out / "train_val.npz"),
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "shape_train": [5902, 9, 128], "shape_val": [1450, 9, 128],
        "train_subjects": TRAIN_SUBJECTS, "val_subjects": VAL_SUBJECTS,
        "channel_order": original_manifest["channel_order"],
        "scaler_sha256": file_hash(out / "channel_scaler_train_only.npz"),
        "train_channel_max_abs_mean": mean_err, "train_channel_max_abs_std_minus_one": std_err,
        "row_alignment_with_official_train_feature_cache": True,
        "index_train_sha256": array_hash(z["index_train"]), "epoch_orders": order_provenance,
        "opened_raw_dataset_entries_this_round": [], "official_test_feature_label_entries_opened": [],
        "official_test_evaluated": False})
    return {"xtr": torch.from_numpy(z["x_train"]), "ytr": torch.from_numpy(z["y_train"]),
        "xv": torch.from_numpy(z["x_val"]), "yv": torch.from_numpy(z["y_val"]),
        "sv": z["subject_val"], "iv": z["index_val"], "it": z["index_train"]}


def preflight(data):
    seed_all(12345)
    s = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    t = torch.randn(5, 6, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 1, 2, 3, 4])
    target = 0.9 * F.one_hot(y, 6).to(s.dtype) + 0.1 / 6
    ls = loss_terms(s, y, "ls")[0]
    assert abs(float((ls + (target * F.log_softmax(s, 1)).sum(1).mean()).detach())) < 1e-12
    kd, _, kl = loss_terms(s, y, "kd_ce", t)
    q = F.softmax(t.detach() / 2, 1)
    assert abs(float((kl - (q * (q.log() - F.log_softmax(s / 2, 1))).sum(1).mean()).detach())) < 1e-12
    expected = (0.5 * (s.softmax(1) - F.one_hot(y, 6)) + (s.div(2).softmax(1) - q)) / 5
    kd.backward()
    err = float((s.grad - expected).abs().max())
    assert err < 1e-12 and t.grad is None
    counts = {}
    for width in [4, 8, 16, 32]:
        model = ConvHAR(width)
        counts[str(width)] = sum(p.numel() for p in model.parameters())
        assert counts[str(width)] == 10 * width * width + 60 * width + 6
        assert model(torch.zeros(2, 9, 128)).shape == (2, 6)
    old_init_checks = {}
    for name, width, seed in [("teacher_seed20261120", 32, 20261120), ("student_w8_ce_seed20261121", 8, 20261121)]:
        seed_all(seed)
        new_initial = state_hash(ConvHAR(width).state_dict())
        old_initial = state_hash(torch.load(OLD_CNN / "runs" / name / "initial_state.pt", weights_only=True))
        assert new_initial == old_initial
        old_init_checks[name] = new_initial
    teacher_path = OLD_CNN / "runs/teacher_seed20261120"
    teacher_result = json.loads((teacher_path / "result.json").read_text())
    teacher = ConvHAR(32)
    teacher.load_state_dict(torch.load(teacher_path / "best_state.pt", weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    h = state_hash(teacher.state_dict())
    assert h == teacher_result["best_state_sha256"] and teacher_result["best_epoch"] == 17
    cache = torch.load(OLD_CNN / "data/teacher_train_logits.pt", weights_only=True)
    recomputed = predict(teacher, data["xtr"])
    logit_error = float((cache - recomputed).abs().max())
    assert logit_error < 1e-6 and cache.shape == (5902, 6) and not cache.requires_grad
    va = metrics(predict(teacher, data["xv"]), data["yv"])
    assert abs(va["macro_f1"] - teacher_result["validation_metrics"]["macro_f1"]) < 1e-12
    result = {"checked_utc": now(), "script_sha256": file_hash(__file__),
        "uniform_ls_and_temperature_scaled_kl_validated": True, "max_kd_gradient_error": err,
        "teacher_receives_no_gradient": True, "parameter_counts": counts,
        "old_initialization_hashes_match": old_init_checks, "old_teacher_cache_max_abs_error": logit_error,
        "old_teacher_validation_macro_f1": va["macro_f1"], "official_test_opened": False}
    json_write(HERE / "preflight_checks.json", result)
    print(json.dumps({"event": "preflight_passed", "teacher_logit_max_error": logit_error,
        "parameter_counts": counts, "new_models_trained": 0}), flush=True)


def export_checkpoint(path, tag, model, data, epoch):
    state = copy.deepcopy(model.state_dict())
    torch.save(state, path / f"{tag}_state.pt")
    train_logits, val_logits = predict(model, data["xtr"]), predict(model, data["xv"])
    tr, va = metrics(train_logits, data["ytr"]), metrics(val_logits, data["yv"])
    np.savez_compressed(path / f"validation_predictions_{tag}.npz",
        logits=val_logits.numpy(), probabilities=val_logits.softmax(1).numpy(), y=data["yv"].numpy(),
        subject=data["sv"], official_train_row0=data["iv"])
    return {"epoch": int(epoch), "state_sha256": state_hash(state), "train_metrics": tr,
        "validation_metrics": va, "per_subject_validation": {
            str(s): metrics(val_logits[data["sv"] == s], data["yv"][data["sv"] == s]) for s in VAL_SUBJECTS}}


def completed_result(path):
    result_file = path / "result.json"
    if not result_file.exists():
        return None
    r = json.loads(result_file.read_text())
    assert r["script_sha256"] == file_hash(__file__), "Completed run belongs to a different script revision"
    assert r["configuration"] == CONFIG
    return r


def import_frozen_ce(data):
    path = HERE / "runs/teacher_ce_frozen_round3"
    existing = completed_result(path)
    if existing is not None:
        return existing
    path.mkdir(parents=True, exist_ok=True)
    source = OLD_CNN / "runs/teacher_seed20261120"
    shutil.copytree(source, path / "source", dirs_exist_ok=True)
    original = json.loads((source / "result.json").read_text())
    shutil.copyfile(source / "curves.csv", path / "curves.csv")
    shutil.copyfile(source / "initial_state.pt", path / "initial_state.pt")
    teacher = ConvHAR(32)
    records = {}
    for tag, fname, epoch in [("best80", "best_state.pt", 17), ("last80", "last_state.pt", 80)]:
        teacher.load_state_dict(torch.load(source / fname, weights_only=True))
        records[tag] = export_checkpoint(path, tag, teacher, data, epoch)
    assert records["best80"]["state_sha256"] == original["best_state_sha256"]
    for target, oldname in [("best_state.pt", "best_state.pt"), ("last_state.pt", "last_state.pt")]:
        shutil.copyfile(source / oldname, path / target)
    r = {"run": path.name, "role": "teacher", "method": "ce", "width": 32,
        "seed": 20261120, "epochs_completed": 80, "newly_trained": False,
        "checkpoints": records, "parameters": 12166, "training_seconds": 0.0,
        "historical_training_seconds": original["training_seconds"],
        "initial_state_sha256": original["initial_state_sha256"],
        "source_run_result_sha256": file_hash(source / "result.json"),
        "original_minibatch_orders_file_sha256": original["minibatch_orders_file_sha256"],
        "configuration": CONFIG, "script_sha256": file_hash(__file__), "official_test_evaluated": False}
    json_write(path / "result.json", r)
    json_write(path / "frozen_teacher_identity.json", {
        "source_path": str(source), "copied_full_original_run_to": "source/",
        "selected_epoch": 17, "selected_budget": 80, "state_sha256": original["best_state_sha256"],
        "source_config_sha256": file_hash(source / "run_config.json"),
        "source_cached_logits_sha256": file_hash(OLD_CNN / "data/teacher_train_logits.pt")})
    return r


def teacher_targets(objective, result, data):
    name = result["run"]
    path = HERE / "runs" / name
    model = ConvHAR(32)
    model.load_state_dict(torch.load(path / "best80_state.pt", weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    h = state_hash(model.state_dict())
    assert h == result["checkpoints"]["best80"]["state_sha256"]
    cache_path = HERE / "data" / f"teacher_{objective}_train_logits.pt"
    if objective == "ce":
        logits = torch.load(OLD_CNN / "data/teacher_train_logits.pt", weights_only=True)
    else:
        logits = predict(model, data["xtr"])
    assert logits.shape == (5902, 6) and torch.isfinite(logits).all() and not logits.requires_grad
    torch.save(logits, cache_path)
    entropy = {}
    for t in [1.0, 2.0]:
        lp = F.log_softmax(logits.double() / t, 1)
        entropy[str(t)] = {"mean_entropy_nats": float(-(lp.exp() * lp).sum(1).mean()),
            "mean_max_probability": float(lp.exp().max(1).values.mean())}
    identity = {"objective": objective, "run": name, "selected_budget": 80,
        "selected_epoch": result["checkpoints"]["best80"]["epoch"], "state_sha256": h,
        "cached_logits_file_sha256": file_hash(cache_path), "index_train_sha256": array_hash(data["it"]),
        "row_count": len(logits), "requires_grad": False, "training_target_statistics": entropy,
        "teacher_used_for_all_student_epochs": "same fixed best80 snapshot", "official_test_evaluated": False}
    assert state_hash(model.state_dict()) == h
    json_write(HERE / "data" / f"teacher_{objective}_logits_manifest.json", identity)
    return logits, identity


def verify_old_prefix(path, method, curves, initial_hash, orders):
    old_method = "kd" if method == "kd_ce" else method
    old_path = OLD_CNN / "runs" / f"student_w8_{old_method}_seed20261121"
    old = pd.read_csv(old_path / "curves.csv")
    new = pd.DataFrame(curves).iloc[:80]
    cols = ["train_accuracy", "train_macro_f1", "train_nll", "val_accuracy", "val_macro_f1", "val_nll",
        "objective_loss", "batch_hard_ce"]
    differences = {col: float(np.abs(old[col].to_numpy() - new[col].to_numpy()).max()) for col in cols}
    old_result = json.loads((old_path / "result.json").read_text())
    identity_ok = initial_hash == old_result["initial_state_sha256"]
    order_ok = np.array_equal(orders[:80], np.load(OLD_CNN / "data/epoch_orders_seed20261121.npy"))
    numeric_ok = all(np.allclose(old[col], new[col], rtol=1e-6, atol=1e-7) for col in cols)
    passed = bool(identity_ok and order_ok and numeric_ok)
    json_write(path / "round3_prefix_reproduction.json", {"passed": passed,
        "old_run": str(old_path), "initial_state_equal": identity_ok, "first80_minibatch_order_equal": order_ok,
        "max_abs_metric_differences": differences, "comparison_excludes_wall_clock_time": True})
    print(json.dumps({"event": "old80_reproduction", "run": path.name, "passed": passed,
        "max_abs_metric_difference": max(differences.values())}), flush=True)
    if not passed:
        raise RuntimeError("Old first80 reproduction differs materially; stop before expanding matrix")


def run_one(name, width, seed, method, data, epochs=160, role="student", teacher_logits=None, teacher_identity=None):
    path = HERE / "runs" / name
    existing = completed_result(path)
    if existing is not None:
        print(json.dumps({"event": "reuse_completed", "run": name}), flush=True)
        return existing
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Unfinished run preserved at {path}; archive explicitly before restarting")
    path.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = ConvHAR(width)
    initial_hash = state_hash(model.state_dict())
    torch.save(model.state_dict(), path / "initial_state.pt")
    orders_path = HERE / "data" / f"epoch_orders_seed{seed}.npy"
    orders_np = np.load(orders_path, allow_pickle=False)
    orders = torch.from_numpy(orders_np[:epochs])
    orders_hash = file_hash(orders_path)
    if method.startswith("kd"):
        assert teacher_identity is not None and teacher_logits is not None and not teacher_logits.requires_grad
        assert teacher_identity["index_train_sha256"] == array_hash(data["it"])
    config = CONFIG | {"run": name, "role": role, "width": width, "seed": seed, "method": method,
        "epochs_this_run": epochs, "initial_state_sha256": initial_hash,
        "minibatch_orders_file_sha256": orders_hash, "used_orders_content_sha256": array_hash(orders_np[:epochs]),
        "teacher_identity": teacher_identity, "script_sha256": file_hash(__file__), "started_utc": now()}
    json_write(path / "run_config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    best_key, best_state, best_epoch = None, None, None
    curves, checkpoints = [], {}
    elapsed_at80 = None
    started = time.perf_counter()
    for epoch, order in enumerate(orders, 1):
        model.train()
        sums = [0.0, 0.0, 0.0]
        ep_start = time.perf_counter()
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            logits = model(data["xtr"][ids])
            tl = None if teacher_logits is None else teacher_logits[ids]
            loss, ce, kl = loss_terms(logits, data["ytr"][ids], method, tl)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            n = len(ids)
            sums[0] += float(loss.detach()) * n
            sums[1] += float(ce.detach()) * n
            if kl is not None:
                sums[2] += float(kl.detach()) * n
        tr = metrics(predict(model, data["xtr"]), data["ytr"])
        va = metrics(predict(model, data["xv"]), data["yv"])
        curves.append({"epoch": epoch, "objective_loss": sums[0] / 5902, "batch_hard_ce": sums[1] / 5902,
            "batch_kl_unscaled": sums[2] / 5902 if method.startswith("kd") else None,
            "train_accuracy": tr["accuracy"], "train_macro_f1": tr["macro_f1"], "train_nll": tr["nll"],
            "val_accuracy": va["accuracy"], "val_macro_f1": va["macro_f1"], "val_nll": va["nll"],
            "epoch_seconds": time.perf_counter() - ep_start})
        key = (va["macro_f1"], -va["nll"])
        if best_key is None or key > best_key:
            best_key, best_epoch, best_state = key, epoch, copy.deepcopy(model.state_dict())
        if epoch == 5:
            estimate = (time.perf_counter() - started) / epoch * epochs
            print(json.dumps({"event": "cost_probe", "run": name, "epochs": epochs,
                "estimated_training_seconds": round(estimate, 2)}), flush=True)
        if epoch % 10 == 0:
            pd.DataFrame(curves).to_csv(path / "curves.csv", index=False)
            json_write(HERE / "progress.json", {"run": name, "epoch": epoch, "budget": epochs,
                "elapsed_seconds_this_run": time.perf_counter() - started, "updated_utc": now()})
        if epoch in [80, 160]:
            current = copy.deepcopy(model.state_dict())
            checkpoints[f"last{epoch}"] = export_checkpoint(path, f"last{epoch}", model, data, epoch)
            model.load_state_dict(best_state)
            checkpoints[f"best{epoch}"] = export_checkpoint(path, f"best{epoch}", model, data, best_epoch)
            model.load_state_dict(current)
            if epoch == 80:
                elapsed_at80 = time.perf_counter() - started
                if role == "student" and width == 8 and seed == 20261121 and method in {"ce", "ls", "kd_ce"}:
                    verify_old_prefix(path, method, curves, initial_hash, orders_np)
    training_seconds = time.perf_counter() - started
    if teacher_identity is not None:
        teacher_file = HERE / "runs" / teacher_identity["run"] / "best80_state.pt"
        assert state_hash(torch.load(teacher_file, weights_only=True)) == teacher_identity["state_sha256"]
    shutil.copyfile(path / f"best{epochs}_state.pt", path / "best_state.pt")
    shutil.copyfile(path / f"last{epochs}_state.pt", path / "last_state.pt")
    pd.DataFrame(curves).to_csv(path / "curves.csv", index=False)
    result = {"run": name, "role": role, "method": method, "width": width, "seed": seed,
        "epochs_completed": epochs, "newly_trained": True, "optimizer_steps": epochs * 47,
        "checkpoints": checkpoints, "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": training_seconds, "elapsed_seconds_through80": elapsed_at80,
        "initial_state_sha256": initial_hash, "minibatch_orders_file_sha256": orders_hash,
        "used_orders_content_sha256": array_hash(orders_np[:epochs]),
        "teacher_identity": teacher_identity, "configuration": CONFIG, "script_sha256": file_hash(__file__),
        "official_test_evaluated": False, "completed_utc": now()}
    json_write(path / "result.json", result)
    print(json.dumps({"event": "completed", "run": name, "best80_f1": checkpoints["best80"]["validation_metrics"]["macro_f1"],
        "best160_f1": checkpoints.get("best160", {}).get("validation_metrics", {}).get("macro_f1"),
        "training_seconds": round(training_seconds, 2)}), flush=True)
    summarize()
    return result


def summarize():
    rows, people, confusion = [], [], []
    for p in sorted((HERE / "runs").glob("*/result.json")):
        r = json.loads(p.read_text())
        curves = pd.read_csv(p.parent / "curves.csv")
        common = {k: r[k] for k in ["run", "role", "method", "width", "seed", "parameters"]}
        for tag, cp in r["checkpoints"].items():
            selection = "best" if tag.startswith("best") else "last"
            budget = int(tag.replace(selection, ""))
            row = common | {"budget": budget, "selection": selection, "checkpoint_epoch": cp["epoch"],
                "train_accuracy": cp["train_metrics"]["accuracy"], "train_nll": cp["train_metrics"]["nll"],
                "val_accuracy": cp["validation_metrics"]["accuracy"], "val_macro_f1": cp["validation_metrics"]["macro_f1"],
                "val_nll": cp["validation_metrics"]["nll"], "training_seconds_full_run": r["training_seconds"],
                "initial_state_sha256": r["initial_state_sha256"],
                "orders_file_sha256": r.get("minibatch_orders_file_sha256", "historical_source")}
            rows.append(row)
            for person, m in cp["per_subject_validation"].items():
                people.append(common | {"budget": budget, "selection": selection, "subject": int(person),
                    "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "nll": m["nll"],
                    "n": int(np.asarray(m["confusion"]).sum())})
            for true_class in range(6):
                for predicted_class in range(6):
                    confusion.append(common | {"budget": budget, "selection": selection, "true_class0": true_class,
                        "predicted_class0": predicted_class, "count": cp["validation_metrics"]["confusion"][true_class][predicted_class]})
        for budget in [b for b in [80, 160] if b <= r["epochs_completed"]]:
            tail = curves[(curves.epoch > budget - 10) & (curves.epoch <= budget)]
            rows.append(common | {"budget": budget, "selection": "last10_mean", "checkpoint_epoch": None,
                **{k: float(tail[k].mean()) for k in ["train_accuracy", "train_nll", "val_accuracy", "val_macro_f1", "val_nll"]},
                "training_seconds_full_run": r["training_seconds"], "initial_state_sha256": r["initial_state_sha256"],
                "orders_file_sha256": r.get("minibatch_orders_file_sha256", "historical_source")})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "summary_checkpoints.csv", index=False)
    pd.DataFrame(people).to_csv(HERE / "per_subject_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(HERE / "class_confusions.csv", index=False)
    students = df[df.role == "student"]
    aggregates = students.groupby(["width", "method", "budget", "selection"]).agg(
        n_seeds=("seed", "count"), mean_macro_f1=("val_macro_f1", "mean"), sd_macro_f1=("val_macro_f1", "std"),
        mean_accuracy=("val_accuracy", "mean"), sd_accuracy=("val_accuracy", "std"), mean_nll=("val_nll", "mean"))
    aggregates.to_csv(HERE / "aggregate_metrics.csv")
    pairs = []
    for (width, seed, budget, selection), group in students.groupby(["width", "seed", "budget", "selection"]):
        indexed = group.set_index("method")
        for left, right in [("kd_ce", "ce"), ("kd_ce", "ls"), ("ls", "ce"),
                            ("kd_ls", "ce"), ("kd_ls", "ls"), ("kd_ls", "kd_ce")]:
            if left not in indexed.index or right not in indexed.index:
                continue
            a, b = indexed.loc[left], indexed.loc[right]
            assert a.initial_state_sha256 == b.initial_state_sha256 and a.orders_file_sha256 == b.orders_file_sha256
            pairs.append({"width": width, "seed": seed, "budget": budget, "selection": selection,
                "contrast": left + "_minus_" + right, "delta_macro_f1_pp": 100 * (a.val_macro_f1 - b.val_macro_f1),
                "delta_accuracy_pp": 100 * (a.val_accuracy - b.val_accuracy), "delta_nll": a.val_nll - b.val_nll,
                "paired_initialization_equal": True, "paired_minibatch_order_equal": True})
    pd.DataFrame(pairs).to_csv(HERE / "paired_differences.csv", index=False)
    if pairs:
        pd.DataFrame(pairs).groupby(["width", "budget", "selection", "contrast"]).agg(
            n_pairs=("seed", "count"), mean_delta_f1_pp=("delta_macro_f1_pp", "mean"),
            sd_delta_f1_pp=("delta_macro_f1_pp", "std"),
            positive_pairs=("delta_macro_f1_pp", lambda x: int((x > 0).sum()))).to_csv(HERE / "paired_aggregate.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "summarize"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    (HERE / "runs").mkdir(parents=True, exist_ok=True)
    lock_configuration()
    if args.action == "summarize":
        summarize()
        return
    data = prepare_data()
    check_path = HERE / "preflight_checks.json"
    if not check_path.exists() or json.loads(check_path.read_text())["script_sha256"] != file_hash(__file__):
        preflight(data)
    if args.action == "prepare":
        print(json.dumps({"event": "prepared", "student_matrix": 30, "new_teacher": 1, "training_started": False}), flush=True)
        return
    ce_result = import_frozen_ce(data)
    ce_logits, ce_identity = teacher_targets("ce", ce_result, data)
    # Teacher objective is the only teacher-training change; source CE initial state/order are preflight checked.
    ls_result = run_one("teacher_ls_seed20261120", 32, 20261120, "ls", data, epochs=80, role="teacher")
    assert ls_result["initial_state_sha256"] == ce_result["initial_state_sha256"]
    assert ls_result["checkpoints"]["best80"]["train_metrics"]["accuracy"] >= 0.95
    assert ls_result["checkpoints"]["best80"]["validation_metrics"]["macro_f1"] >= 0.85
    ls_logits, ls_identity = teacher_targets("ls", ls_result, data)
    for width in [8, 4, 16]:
        for seed in SEEDS:
            for method in ["ce", "ls", "kd_ce"]:
                run_one(f"student_w{width}_{method}_seed{seed}", width, seed, method, data,
                    teacher_logits=ce_logits if method == "kd_ce" else None,
                    teacher_identity=ce_identity if method == "kd_ce" else None)
    for seed in SEEDS:
        run_one(f"student_w8_kd_ls_seed{seed}", 8, seed, "kd_ls", data,
            teacher_logits=ls_logits, teacher_identity=ls_identity)
    summarize()
    finished = list((HERE / "runs").glob("*/result.json"))
    assert len(finished) == 32
    json_write(HERE / "completion.json", {"completed_utc": now(), "new_students": 30,
        "new_teachers": 1, "frozen_old_ce_teachers": 1, "all_students_160_epochs": True,
        "fixed_teacher_across_80_and_160_budgets": True, "paired_initialization_and_order_verified": True,
        "old_width8_seed20261121_first80_reproduction_passed": True, "official_test_evaluated": False,
        "scope": "development evidence on one four-person validation split; no deployment benchmark"})
    print(json.dumps({"event": "matrix_complete", "new_models": 31, "official_test_opened": False}), flush=True)


if __name__ == "__main__":
    main()
