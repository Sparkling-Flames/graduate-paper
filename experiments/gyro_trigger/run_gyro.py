"""Request gyro for the second half of a window from first-half acceleration."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "activity_budget"))
from pilot import UCI_DEV, MOTION_VAL, read_motion, read_uci


def stats(a):
    x = np.concatenate((a, np.linalg.norm(a, axis=-1, keepdims=True)), axis=-1)
    return np.column_stack((x.mean(1), x.std(1), np.abs(np.diff(x, axis=1)).mean(1)))


def extract(body, gyro, gravity):
    acc = body + gravity
    return np.column_stack((stats(acc[:, :64]), stats(acc), stats(gyro[:, 64:]), stats(gyro)))


def predict(x, y, train, test):
    inputs = [x[:, :12], x[:, 12:24], x[:, 12:36], x[:, np.r_[12:24, 36:48]]]
    return [make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1200))
            .fit(a[train], y[train]).predict_proba(a[test]) for a in inputs]


def f1(y, pred):
    cm = np.bincount(5*y+pred, minlength=25).reshape(5, 5)
    return (2*cm.diagonal() / (cm.sum(0)+cm.sum(1))).mean()


def run(name, data, held):
    x, y, p = data
    train = np.flatnonzero(~np.isin(p, list(held)))
    valid = np.flatnonzero(np.isin(p, list(held)))
    oof = np.zeros((len(x), 5))
    gain = np.zeros(len(x))
    for a, b in GroupKFold(4).split(train, groups=p[train]):
        rows = train[b]
        pre, acc, fused, _ = predict(x, y, train[a], rows)
        oof[rows] = pre
        gain[rows] = (fused.argmax(1)==y[rows]).astype(float) - (acc.argmax(1)==y[rows])
    pre, acc, fused, full = predict(x, y, train, valid)
    learned = make_pipeline(StandardScaler(), Ridge(alpha=20))
    learned.fit(np.column_stack((x[train,:12], oof[train])), gain[train])
    rules = dict(uncertain=(1-oof[train].max(1), 1-pre.max(1)),
                 learned=(learned.predict(np.column_stack((x[train,:12],oof[train]))),
                          learned.predict(np.column_stack((x[valid,:12],pre)))))
    rows = []
    for who in sorted(held):
        mask = p[valid] == who
        target = y[valid][mask]
        base, extra, always = (v[mask].argmax(1) for v in [acc, fused, full])
        assert len(np.unique(target)) == 5
        for rule,(cal,score) in rules.items():
            for budget in (.25,.50):
                select = score[mask] >= np.quantile(cal,1-budget)
                mixed = np.where(select,extra,base)
                rng = np.random.default_rng(1127)
                random_f1 = np.mean([f1(target,np.where(rng.permutation(select),extra,base)) for _ in range(200)])
                rows.append(dict(dataset=name, person=who, rule=rule, budget=budget, windows=len(target),
                                 triggered=select.sum(), rate=select.mean(), acc=f1(target,base),
                                 gyro_suffix=f1(target,extra), gyro_full=f1(target,always),
                                 gated=f1(target,mixed), random=random_f1))
    return rows


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        rows = run("UCI", read_uci(extract), UCI_DEV) + run("MotionSense", read_motion(extract), MOTION_VAL)
        df = pd.DataFrame(rows)
        df.to_csv(HERE / "results.csv",index=False)
        print(df.groupby(["dataset","rule","budget"])[["rate","acc","gyro_suffix","gyro_full","gated","random"]].mean().round(4))
