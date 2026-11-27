"""UCI development pilots for training-person coverage and probability calibration."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data" / "UCI HAR Dataset" / "train"
GROUPS = {"A": [3, 28, 29, 30], "B": [5, 11, 21, 26]}
SEEDS = range(11, 17)


def model_and_optimizer(seed, widths):
    torch.manual_seed(seed)
    layers = []
    n_in = 561
    for width in widths:
        layers.extend([nn.Linear(n_in, width), nn.ReLU()])
        n_in = width
    layers.append(nn.Linear(n_in, 6))
    model = nn.Sequential(*layers)
    return model, torch.optim.Adam(model.parameters(), lr=0.001)


def standardize(x, rows):
    mean = x[rows].mean(axis=0, dtype=np.float64)
    sd = x[rows].std(axis=0, dtype=np.float64)
    sd[sd < 1e-8] = 1
    return torch.from_numpy(((x - mean) / sd).astype(np.float32))


def update(model, opt, x, y, rows):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x[rows]), y[rows])
    loss.backward()
    opt.step()


def losses(logits, y, temp):
    scores = logits.astype(np.float64) / temp
    return np.logaddexp.reduce(scores, axis=1) - scores[np.arange(len(y)), y]


def scores(logits, y, temp=1.0):
    pred = logits.argmax(1)
    cm = np.bincount(6 * y + pred, minlength=36).reshape(6, 6)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(6), where=denom != 0)
    return {"f1": f1.mean(), "nll": losses(logits, y, temp).mean()}


def predictions(model, x):
    with torch.inference_mode():
        return model(x).numpy()


def diversity(x, y, subject, train_people):
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        people = rng.permutation(train_people)[:12]
        pool = {(p, c): rng.permutation(np.flatnonzero((subject == p) & (y == c)))
                for p in people for c in range(6)}
        for n_people, quota in [(6, 36), (12, 18), (12, 36)]:
            # Each class and each person contributes equally. The larger arm contains the smaller samples.
            rows = np.concatenate([pool[p, c][:quota] for p in people[:n_people] for c in range(6)])
            assert len(rows) == n_people * 6 * quota and not set(subject[rows]) & set(sum(GROUPS.values(), []))
            xn = standardize(x, rows)
            model, opt = model_and_optimizer(seed, [32])
            batches = np.random.default_rng(seed + 100)
            labels = torch.from_numpy(y)
            for _ in range(400):
                update(model, opt, xn, labels, rows[batches.integers(len(rows), size=128)])
            logits = predictions(model, xn)
            for group, ids in GROUPS.items():
                for p in ids:
                    m = subject == p
                    out.append(dict(seed=seed, group=group, person=p, people=n_people, windows=len(rows),
                                    training_ids=" ".join(map(str, people[:n_people])), **scores(logits[m], y[m])))
    return out


def calibrate(x, y, subject, train_people):
    rows = np.flatnonzero(np.isin(subject, train_people))
    xn = standardize(x, rows)
    labels = torch.from_numpy(y)
    temps = np.unique(np.r_[1.0, np.geomspace(0.25, 8, 101)])
    out = []
    for seed in [11, 12, 13]:
        model, opt = model_and_optimizer(seed, [256, 128])
        rng = np.random.default_rng(seed)
        for epoch in range(1, 81):
            perm = rng.permutation(rows)
            for start in range(0, len(rows), 128):
                update(model, opt, xn, labels, perm[start:start+128])
            if epoch not in (25, 80):
                continue
            logits = predictions(model, xn)
            for group, other in [("A", "B"), ("B", "A")]:
                # Fit one scalar using the other four people, weighting people equally.
                masks = [subject == p for p in GROUPS[other]]
                objective = [np.mean([losses(logits[m], y[m], t).mean() for m in masks]) for t in temps]
                t = temps[np.argmin(objective)]
                for p in GROUPS[group]:
                    m = subject == p
                    raw = scores(logits[m], y[m])
                    out.append(dict(seed=seed, epoch=epoch, group=group, person=p, temperature=t,
                                    f1=raw["f1"], raw_nll=raw["nll"], scaled_nll=losses(logits[m],y[m],t).mean()))
    return out


def save(name, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    torch.set_num_threads(1)
    x = np.loadtxt(DATA / "X_train.txt", dtype=np.float32)
    y = np.loadtxt(DATA / "y_train.txt", dtype=np.int64) - 1
    subject = np.loadtxt(DATA / "subject_train.txt", dtype=int)
    train_people = sorted(set(subject) - set(GROUPS["A"]) - set(GROUPS["B"]))
    assert len(train_people) == 13
    save("diversity.csv", diversity(x, y, subject, train_people))
    save("calibration.csv", calibrate(x, y, subject, train_people))
