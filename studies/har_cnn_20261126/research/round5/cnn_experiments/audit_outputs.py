#!/usr/bin/env python3
"""Read-only numerical audit of the completed fixed-teacher CNN experiment."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "round3/experiments/cnn_pilot"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest(), sum(x.numel() for x in state.values())


def calculate(z, mask=None):
    y = z["y"] if mask is None else z["y"][mask]
    logits = z["logits"] if mask is None else z["logits"][mask]
    p = logits.argmax(1)
    a = logits.astype(np.float64)
    maxima = a.max(1)
    nll = (maxima + np.log(np.exp(a - maxima[:, None]).sum(1)) - a[np.arange(len(y)), y]).mean()
    return {"accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=range(6), average="macro", zero_division=0)),
        "nll": float(nll), "confusion": confusion_matrix(y, p, labels=range(6)).tolist()}


def check_metrics(actual, reported):
    assert abs(actual["accuracy"] - reported["accuracy"]) < 1e-12
    assert abs(actual["macro_f1"] - reported["macro_f1"]) < 1e-12
    assert abs(actual["nll"] - reported["nll"]) < 2e-6
    assert actual["confusion"] == reported["confusion"]


def main():
    torch.set_num_threads(2)
    completion = json.loads((HERE / "completion.json").read_text())
    assert completion["new_students"] == 30 and completion["new_teachers"] == 1
    data = np.load(HERE / "data/train_val.npz", allow_pickle=False)
    results = {}
    recalculated = []
    expected_configs = {(w, m, s) for w in [4, 8, 16] for m in ["ce", "ls", "kd_ce"] for s in [20261121, 20261122, 20261123]}
    expected_configs |= {(8, "kd_ls", s) for s in [20261121, 20261122, 20261123]}
    observed_configs = set()
    cp_count = 0
    script_hash = file_hash(HERE / "run_cnn.py")
    for path in sorted((HERE / "runs").glob("*/result.json")):
        r = json.loads(path.read_text())
        results[r["run"]] = r
        assert r["official_test_evaluated"] is False and r["script_sha256"] == script_hash
        c = pd.read_csv(path.parent / "curves.csv")
        assert c.epoch.tolist() == list(range(1, r["epochs_completed"] + 1))
        initial_hash, n = state_hash(path.parent / "initial_state.pt")
        assert initial_hash == r["initial_state_sha256"] and n == r["parameters"]
        assert n == 10 * r["width"] ** 2 + 60 * r["width"] + 6
        if r["role"] == "student":
            observed_configs.add((r["width"], r["method"], r["seed"]))
            assert r["epochs_completed"] == 160 and r["optimizer_steps"] == 7520
            assert set(r["checkpoints"]) == {"best80", "last80", "best160", "last160"}
        for tag, cp in r["checkpoints"].items():
            cp_count += 1
            selection = "best" if tag.startswith("best") else "last"
            budget = int(tag.replace(selection, ""))
            c_prefix = c[c.epoch <= budget]
            chosen = c_prefix.sort_values(["val_macro_f1", "val_nll", "epoch"], ascending=[False, True, True]).iloc[0]
            expected_epoch = int(chosen.epoch) if selection == "best" else budget
            assert cp["epoch"] == expected_epoch
            ck_hash, _ = state_hash(path.parent / f"{tag}_state.pt")
            assert ck_hash == cp["state_sha256"]
            z = np.load(path.parent / f"validation_predictions_{tag}.npz", allow_pickle=False)
            for out_key, data_key in [("y", "y_val"), ("subject", "subject_val"), ("official_train_row0", "index_val")]:
                assert np.array_equal(z[out_key], data[data_key])
            m = calculate(z)
            check_metrics(m, cp["validation_metrics"])
            row = c[c.epoch == expected_epoch].iloc[0]
            for key in ["accuracy", "macro_f1", "nll"]:
                assert abs(float(row[f"val_{key}"]) - cp["validation_metrics"][key]) < 2e-6
                assert abs(float(row[f"train_{key}"]) - cp["train_metrics"][key]) < 2e-6
            for subject, reported in cp["per_subject_validation"].items():
                check_metrics(calculate(z, z["subject"] == int(subject)), reported)
            recalculated.append({"run": r["run"], "role": r["role"], "width": r["width"],
                "method": r["method"], "seed": r["seed"], "budget": budget, "selection": selection,
                "epoch": expected_epoch, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "nll": m["nll"]})
    assert observed_configs == expected_configs and len(results) == 32 and cp_count == 124
    pair_sets = {}
    for r in results.values():
        if r["role"] == "student":
            key = (r["width"], r["seed"])
            pair_sets.setdefault(key, []).append(r)
            if r["method"].startswith("kd"):
                identity = r["teacher_identity"]
                teacher = results[identity["run"]]
                assert identity["selected_budget"] == 80
                assert identity["state_sha256"] == teacher["checkpoints"]["best80"]["state_sha256"]
    for group in pair_sets.values():
        assert len({x["initial_state_sha256"] for x in group}) == 1
        assert len({x["minibatch_orders_file_sha256"] for x in group}) == 1
        assert len({x["used_orders_content_sha256"] for x in group}) == 1
    old_ce = json.loads((OLD / "runs/teacher_seed20261120/result.json").read_text())
    ce = results["teacher_ce_frozen_round3"]
    ls = results["teacher_ls_seed20261120"]
    assert ce["checkpoints"]["best80"]["state_sha256"] == old_ce["best_state_sha256"]
    assert ce["initial_state_sha256"] == ls["initial_state_sha256"]
    old_teacher_orders = np.load(OLD / "data/epoch_orders_seed20261120.npy")
    new_teacher_orders = np.load(HERE / "data/epoch_orders_seed20261120.npy")
    assert np.array_equal(old_teacher_orders, new_teacher_orders[:80])
    for method in ["ce", "ls", "kd_ce"]:
        check = json.loads((HERE / "runs" / f"student_w8_{method}_seed20261121/round3_prefix_reproduction.json").read_text())
        assert check["passed"]
    D = pd.DataFrame(recalculated)
    D.to_csv(HERE / "independent_recomputed_metrics.csv", index=False)
    audit = {"runs_checked": 32, "new_students_checked": 30, "checkpoints_and_predictions_checked": cp_count,
        "independent_sklearn_metrics_and_float64_nll_match": True,
        "per_subject_and_confusion_recomputation_match": True, "checkpoint_selection_matches_curves": True,
        "all_student_80_160_paths_share_fixed_teacher": True, "paired_initialization_and_order_match": True,
        "ce_teacher_original_hash_and_ls_teacher_initialization_order_match": True,
        "old_first80_three_method_reproduction_passed": True,
        "official_test_evaluated": False, "extra_training_or_model_selection_performed_by_audit": False}
    (HERE / "audit_complete.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
