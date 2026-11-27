"""Compare PAMAP2 acceleration placements on exactly the same development windows."""
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[2] / "data" / "PAMAP2_Dataset.zip"
TRAIN = (101, 102, 103, 104)
VALID = (105, 106)
ACTIVITIES = {2: "sitting", 3: "standing", 4: "walking", 12: "upstairs", 13: "downstairs"}
LABELS = list(ACTIVITIES)
SITES = ("wrist", "chest", "ankle")
WINDOW = 256


def features(windows):
    # Windows x time x placement x axis; norm is independent of axis orientation.
    norm = np.linalg.norm(windows, axis=3, keepdims=True)
    signals = np.concatenate((windows, norm), axis=3)
    return np.concatenate((
        signals.mean(axis=1), signals.std(axis=1),
        np.quantile(signals, .10, axis=1), np.quantile(signals, .90, axis=1),
        np.sqrt(np.mean(np.diff(signals, axis=1) ** 2, axis=1)),
    ), axis=2)


def select_windows(windows, allow_short_gaps):
    complete = np.isfinite(windows).all(axis=(2, 3))
    keep = complete.all(axis=1)
    if allow_short_gaps:
        repair = ~keep & complete[:, 0] & complete[:, -1] & ((~complete).sum(axis=1) <= 10)
        for index in np.flatnonzero(repair):
            for channel in windows[index].reshape(WINDOW, 9).T:
                finite = np.isfinite(channel)
                if not finite.all():
                    channel[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), channel[finite])
        keep |= repair
    chosen = windows[keep]
    assert np.isfinite(chosen).all()
    return chosen


def subject_windows(archive, subject, allow_short_gaps):
    cols = [0, 1, 4, 5, 6, 21, 22, 23, 38, 39, 40]
    with archive.open(f"PAMAP2_Dataset/Protocol/subject{subject}.dat") as stream:
        records = pd.read_csv(stream, header=None, sep=r"\s+", usecols=cols)
    timestamps = records[0].to_numpy()
    labels = records[1].to_numpy(dtype=int)
    acceleration = np.ascontiguousarray(records[cols[2:]].to_numpy(dtype=np.float64))
    dt = np.diff(timestamps)
    breaks = np.flatnonzero((dt <= 0) | (dt > .015) | (np.diff(labels) != 0)) + 1
    bounds = np.r_[0, breaks, len(labels)]
    blocks, out_labels, audit = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = (b-a) // WINDOW
        label = int(labels[a])
        if label not in ACTIVITIES or n == 0:
            continue
        windows = acceleration[a:a+n*WINDOW].reshape(n, WINDOW, 3, 3)
        selected = select_windows(windows, allow_short_gaps)
        audit.append((subject, ACTIVITIES[label], n, len(selected)))
        if len(selected):
            blocks.append(features(selected))
            out_labels.extend([label] * len(selected))
    return np.concatenate(blocks), np.array(out_labels), audit


def run(allow_short_gaps):
    output = HERE / "short_gaps" if allow_short_gaps else HERE
    output.mkdir(exist_ok=True)
    blocks, labels, subjects, audits = [], [], [], []
    with ZipFile(ARCHIVE) as archive:
        for subject in TRAIN + VALID:
            x, y, audit = subject_windows(archive, subject, allow_short_gaps)
            blocks.append(x)
            labels.extend(y)
            subjects.extend([subject] * len(y))
            audits.extend(audit)
    X, y, ids = np.concatenate(blocks), np.array(labels), np.array(subjects)
    train, valid = np.isin(ids, TRAIN), np.isin(ids, VALID)
    assert not set(ids[train]) & set(ids[valid])
    assert all(set(y[ids == s]) == set(LABELS) for s in TRAIN + VALID)
    audit_df = pd.DataFrame(audits, columns=["subject", "activity", "candidate", "retained"])
    audit_df.groupby(["subject", "activity"]).sum().to_csv(output / "windows.csv")
    scores, class_scores, corrections = [], [], []
    predictions = {}
    for name, sites in (("wrist", [0]), ("chest", [1]), ("ankle", [2]),
                        ("wrist_copied_3x", [0, 0, 0]), ("all_three", [0, 1, 2])):
        values = X[:, sites, :].reshape(len(X), -1)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=1500, solver="lbfgs", class_weight="balanced"))
        model.fit(values[train], y[train])
        probs = model.predict_proba(values[valid])
        pred = model.classes_[probs.argmax(axis=1)]
        predictions[name] = pred
        for group in ["pooled"] + list(VALID):
            rows = np.ones(valid.sum(), dtype=bool) if group == "pooled" else ids[valid] == group
            actual, guesses = y[valid][rows], pred[rows]
            f1 = f1_score(actual, guesses, labels=LABELS, average=None, zero_division=0)
            scores.append(dict(input=name, group=group, windows=int(rows.sum()),
                               macro_f1=float(f1.mean()), nll=float(log_loss(actual, probs[rows], labels=LABELS))))
            for k, label in enumerate(LABELS):
                class_scores.append(dict(input=name, group=group, activity=ACTIVITIES[label],
                                         support=int((actual==label).sum()), f1=float(f1[k])))
    wrist, multi = predictions["wrist"], predictions["all_three"]
    actual = y[valid]
    for label in LABELS:
        rows = actual == label
        corrections.append(dict(activity=ACTIVITIES[label], support=int(rows.sum()),
                                wrist_wrong_multi_right=int((rows & (wrist != actual) & (multi == actual)).sum()),
                                wrist_right_multi_wrong=int((rows & (wrist == actual) & (multi != actual)).sum())))
    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(class_scores).to_csv(output / "class_f1.csv", index=False)
    pd.DataFrame(corrections).to_csv(output / "paired_errors.csv", index=False)
    matrices = {name: confusion_matrix(actual, p, labels=LABELS).tolist() for name, p in predictions.items()}
    (output / "confusions.json").write_text(json.dumps({"activities": list(ACTIVITIES.values()),
                                                        "counts": matrices}, indent=2), encoding="utf-8")
    print(output)
    print(pd.DataFrame(scores).to_string(index=False))
    print(pd.DataFrame(corrections).to_string(index=False))
    print("train / validation windows:", train.sum(), valid.sum())


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run(allow_short_gaps=False)
        run(allow_short_gaps=True)
