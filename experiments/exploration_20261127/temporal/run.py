"""Paired MotionSense time-span pilot, using only the archived development split."""
import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "studies/har_cnn_20261126/research/round5/motionsense_experiments"
sys.path.insert(0, str(FROZEN))
from run_motionsense import CHANNELS, CLASSES, CNN, TEST, TRAIN, VAL, metrics, predict

SEEDS = (20261121, 20261122)
EPOCHS = 12


def load_development_data():
    trials = pd.read_csv(FROZEN / "data/trial_manifest.csv")
    data = {}
    with np.load(FROZEN / "data/channel_scaler_train_only.npz") as scaler:
        mean, std = scaler["mean"], scaler["std"]
    with ZipFile(ROOT / "data/MotionSense_A_DeviceMotion.zip") as archive:
        for split, people in (("train", TRAIN), ("val", VAL)):
            rows = trials[trials.split == split]
            assert set(rows.subject) == set(people) and set(people).isdisjoint(TEST)
            features, labels, subjects = [], [], []
            for row in rows.itertuples():
                assert row.source_entry.endswith(f"/sub_{row.subject}.csv")
                with archive.open(row.source_entry) as stream:
                    frame = pd.read_csv(stream, usecols=CHANNELS)
                signal = frame[CHANNELS].to_numpy(dtype=np.float32)
                windows = np.lib.stride_tricks.sliding_window_view(signal, 128, axis=0)[::64]
                assert len(windows) == row.windows
                features.append(windows)
                labels.extend([CLASSES.index(row.activity)] * len(windows))
                subjects.extend([row.subject] * len(windows))
            values = np.ascontiguousarray((np.concatenate(features) - mean) / std, dtype=np.float32)
            data[split] = torch.from_numpy(values), torch.tensor(labels), np.array(subjects)
    return data


def evaluate(model, x, y, subjects):
    logits = predict(model, x)
    scores = metrics(logits, y)
    scores["per_subject_f1"] = {
        str(subject): metrics(logits[subjects == subject], y[subjects == subject])["macro_f1"]
        for subject in VAL
    }
    return scores


def main(method, followup):
    torch.set_num_threads(1)
    epochs = 40 if followup else EPOCHS
    data = load_development_data()
    train_x, train_y, _ = data["train"]
    val_x, val_y, val_subjects = data["val"]
    assert (len(train_x), len(val_x)) == (14189, 3709)
    max_teacher_difference = None
    if method == "kd":
        teacher = CNN(32)
        checkpoint = FROZEN / "runs/teacher_ce_seed20261120/best80_state.pt"
        teacher.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        targets = torch.load(FROZEN / "data/teacher_train_logits.pt", map_location="cpu", weights_only=True)
        fresh = predict(teacher, train_x)
        max_teacher_difference = float((fresh - targets).abs().max())
        torch.testing.assert_close(fresh, targets, rtol=1e-5, atol=2e-5)
        soft_targets = F.softmax(targets / 2, dim=1)
    results = []
    for seed in SEEDS:
        for dilation in (1, 4):
            torch.manual_seed(seed)
            model = CNN(8)
            conv = model.features[3]
            conv.dilation = (dilation,)
            conv.padding = (2 * dilation,)
            assert sum(p.numel() for p in model.parameters()) == 1126
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            orders = np.random.default_rng(seed + 1000000)
            for epoch in range(1, epochs + 1):
                model.train()
                for ids in torch.from_numpy(orders.permutation(len(train_y))).split(128):
                    opt.zero_grad(set_to_none=True)
                    logits = model(train_x[ids])
                    if method == "ls":
                        loss = F.cross_entropy(logits, train_y[ids], label_smoothing=.1)
                    else:
                        divergence = F.kl_div(F.log_softmax(logits / 2, dim=1),
                                              soft_targets[ids], reduction="batchmean")
                        loss = .5 * F.cross_entropy(logits, train_y[ids]) + 2 * divergence
                    loss.backward()
                    opt.step()
                if epoch in (4, EPOCHS, epochs):
                    result = evaluate(model, val_x, val_y, val_subjects)
                    result.update(seed=seed, dilation=dilation, epoch=epoch)
                    results.append(result)
                    print(seed, dilation, epoch, round(result["macro_f1"], 4), flush=True)
    output = dict(torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__,
                  class_order=CLASSES, train_subjects=TRAIN, validation_subjects=VAL,
                  windows=dict(train=len(train_x), validation=len(val_x)),
                  primary_epoch=EPOCHS, run_epochs=epochs, posthoc_followup=followup,
                  parameters=1126, method=method,
                  max_teacher_logit_difference=max_teacher_difference, results=results)
    suffix = "_followup" if followup else ""
    (HERE / f"results_{method}{suffix}.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["ls", "kd"])
    parser.add_argument("--followup", action="store_true", help="Post-hoc 40-epoch budget diagnostic")
    args = parser.parse_args()
    main(args.method, args.followup)
