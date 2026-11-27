"""Five-activity MLP pilot using only development subjects in UCI and MotionSense."""
from pathlib import Path
from zipfile import ZipFile
import re

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"
CLASSES = ("walking", "upstairs", "downstairs", "sitting", "standing")
MOTION_CODES = ("wlk", "ups", "dws", "sit", "std")
UCI_VAL = {3, 28, 29, 30}
MOTION_TRAIN = {1, 4, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 20, 21, 23, 24}
MOTION_VAL = {2, 5, 14, 19}
SEEDS = (42, 43, 44)


def features(body, gyro, gravity):
    """N x time x 3 arrays, body and gravity in g and gyro in rad/s."""
    columns = []
    for vector in (body, gyro):
        magnitude = np.linalg.norm(vector, axis=2)
        columns.extend([magnitude.mean(1), magnitude.std(1),
                        *np.quantile(magnitude, [.1, .9], axis=1),
                        np.sqrt(np.mean(np.diff(magnitude, axis=1) ** 2, axis=1))])
    direction = gravity.mean(1)
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
    columns.extend(np.abs(direction).T)
    x = np.column_stack(columns)
    assert x.shape == (len(body), 13) and np.isfinite(x).all()
    return x


def read_uci(extract=features):
    path = DATA / "UCI HAR Dataset" / "train"
    subject = np.loadtxt(path / "subject_train.txt", dtype=int)
    y = np.loadtxt(path / "y_train.txt", dtype=int)
    keep = (y >= 1) & (y <= len(CLASSES))
    y, subject = y[keep] - 1, subject[keep]

    def signal(name):
        axes = [np.loadtxt(path / "Inertial Signals" / f"{name}_{axis}_train.txt",
                           dtype=np.float32)[keep] for axis in "xyz"]
        return np.stack(axes, axis=2)

    body = signal("body_acc")
    x = extract(body, signal("body_gyro"), signal("total_acc") - body)
    return x, y, subject


def read_motion(extract=features):
    columns = [f"{name}.{axis}" for name in ("userAcceleration", "rotationRate", "gravity")
               for axis in "xyz"]
    blocks, targets, subjects = [], [], []
    with ZipFile(DATA / "MotionSense_A_DeviceMotion.zip") as z:
        for name in sorted(z.namelist()):
            match = re.fullmatch(r"A_DeviceMotion_data/([a-z]+)_\d+/sub_(\d+)\.csv", name)
            if not match:
                continue
            activity, person = match[1], int(match[2])
            if activity not in MOTION_CODES or person not in MOTION_TRAIN | MOTION_VAL:
                continue
            with z.open(name) as f:
                frame = pd.read_csv(f, usecols=columns)
            readings = frame[columns].to_numpy(dtype=np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(readings, 128, axis=0)[::64]
            windows = windows.transpose(0, 2, 1)
            blocks.append(extract(windows[:, :, :3], windows[:, :, 3:6], windows[:, :, 6:]))
            targets.extend([MOTION_CODES.index(activity)] * len(windows))
            subjects.extend([person] * len(windows))
    return np.concatenate(blocks), np.array(targets), np.array(subjects)


def train_models(xu, yu, xm, ym, seed):
    settings = dict(hidden_layer_sizes=(32,), solver="adam", alpha=0.,
                    learning_rate_init=.001, shuffle=False, random_state=seed)
    uci = MLPClassifier(batch_size=100, **settings)
    mixed = MLPClassifier(batch_size=100, **settings)
    half = MLPClassifier(batch_size=50, **settings)
    rng = np.random.default_rng(seed)
    labels = np.arange(len(CLASSES))
    groups_u = [np.flatnonzero(yu == c) for c in labels]
    groups_m = [np.flatnonzero(ym == c) for c in labels]
    for _ in range(800):
        indices = np.stack([rng.choice(ids, 20) for ids in groups_u])
        same_half, other_half = indices[:, :10].ravel(), indices[:, 10:].ravel()
        motion_half = np.concatenate([rng.choice(ids, 10) for ids in groups_m])
        batch_u = np.concatenate((xu[same_half], xu[other_half]))
        batch_mix = np.concatenate((xu[same_half], xm[motion_half]))
        batch_y = np.concatenate((yu[same_half], yu[other_half]))
        assert np.array_equal(yu[other_half], ym[motion_half])
        uci.partial_fit(batch_u, batch_y, classes=labels)
        mixed.partial_fit(batch_mix, batch_y, classes=labels)
        half.partial_fit(xu[same_half], yu[same_half], classes=labels)
    return {"uci_only": uci, "uci_motion_50_50": mixed, "uci_same_50": half}


def evaluate(model, x, y):
    hidden = np.maximum(0, x @ model.coefs_[0] + model.intercepts_[0])
    logits = hidden @ model.coefs_[1] + model.intercepts_[1]
    f1 = f1_score(y, logits.argmax(1), labels=np.arange(5), average=None, zero_division=0)
    nll = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y])
    return dict(windows=len(y), macro_f1=np.mean(f1), nll=nll,
                **{f"f1_{name}": v for name, v in zip(CLASSES, f1)})


def main():
    xu, yu, su = read_uci()
    xm, ym, sm = read_motion()
    uv, mv = np.isin(su, list(UCI_VAL)), np.isin(sm, list(MOTION_VAL))
    assert (int((~uv).sum()), int((~mv).sum())) == (4776, 12851)
    assert len(set(su[~uv])) == 17 and set(sm[~mv]) == MOTION_TRAIN
    assert set(sm[mv]) == MOTION_VAL and set(su[uv]) == UCI_VAL
    scaler = StandardScaler().fit(xu[~uv])
    xu, xm = scaler.transform(xu), scaler.transform(xm)
    validation = {"uci": (xu[uv], yu[uv], su[uv]), "motion": (xm[mv], ym[mv], sm[mv])}
    rows = []
    for seed in SEEDS:
        for arm, model in train_models(xu[~uv], yu[~uv], xm[~mv], ym[~mv], seed).items():
            for dataset, (x, y, people) in validation.items():
                masks = {"all": np.ones(len(y), dtype=bool)}
                masks.update({str(s): people == s for s in np.unique(people)})
                for subject, mask in masks.items():
                    rows.append(dict(seed=seed, arm=arm, dataset=dataset, subject=subject,
                                     **evaluate(model, x[mask], y[mask])))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results.csv", index=False)
    print(df[df.subject == "all"].groupby(["dataset", "arm"])[["macro_f1", "nll"]].mean())


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
