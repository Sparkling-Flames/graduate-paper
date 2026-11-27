"""Two-donor acquisition protocols for cross-dataset smartphone HAR."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exploration_20261127" / "pooling"))
from run_pooling import CLASSES, MOTION_VAL, read_motion, read_uci

UCI_DEV = {3, 5, 11, 21, 26, 28, 29, 30}


def fit(x, y, weights=None):
    return LogisticRegression(max_iter=1500, C=1, class_weight=None if weights is not None else "balanced").fit(
        x, y, sample_weight=weights)


def heads(x, y):
    dynamic = y < 3
    return (fit(x, dynamic), fit(x[dynamic], y[dynamic]), fit(x[~dynamic], y[~dynamic]))


def predict_heads(models, x):
    gate, motion, posture = models
    return np.where(gate.predict(x), motion.predict(x), posture.predict(x))


def domain_weights(y):
    # Unit total mass per domain; equal weight for each class within each domain.
    return 1 / (5 * np.bincount(y)[y])


def run(direction, source, target):
    sx, sy, sp, sd = source
    tx, ty, tp, td = target
    train = ~np.isin(sp, list(sd))
    valid = np.isin(tp, list(td))
    sx, sy = sx[train], sy[train]
    raw_target = tx
    scaler = StandardScaler().fit(sx)
    sx, tx = scaler.transform(sx), scaler.transform(tx)
    vx = tx[valid]
    frozen = heads(sx, sy)
    source_pred = predict_heads(frozen, vx)
    flat_source = fit(sx, sy).predict(vx)
    full_scale = StandardScaler().fit(raw_target[~valid])
    full_target = predict_heads(heads(full_scale.transform(raw_target[~valid]), ty[~valid]),
                                full_scale.transform(raw_target[valid]))
    donors = np.random.default_rng(1127).permutation(np.unique(tp[~valid]))
    results, draws = [], []
    # Each available donor appears at least once. Wrap the final odd pair.
    for pair in range((len(donors) + 1)//2):
        people = donors[np.array([2*pair, 2*pair+1]) % len(donors)]
        rng = np.random.default_rng(200 + pair)
        order = {}
        for c in range(5):
            pools = [rng.permutation(np.flatnonzero((tp == p) & (ty == c))[::2]) for p in people]
            # Interleave the donors. Each selected window is an actual labelled example.
            order[c] = np.array([pool[k] for k in range(max(map(len, pools))) for pool in pools if k < len(pool)])
        for budget in (30, 60):
            predictions = dict(source_hierarchy=source_pred, source_flat=flat_source, full_target=full_target)
            policies = {"balanced": range(5), "posture": (3, 4), "motion": (0, 1, 2),
                        "level_ground": (0, 3, 4), "skip_up": (0, 2, 3, 4), "skip_down": (0, 1, 3, 4)}
            for arm, classes in policies.items():
                quota, extra = divmod(budget, len(classes))
                chosen = np.concatenate([order[c][:quota + ((j - pair) % len(classes) < extra)]
                                         for j, c in enumerate(classes)])
                assert len(np.unique(chosen)) == budget and not valid[chosen].any()
                assert set(tp[chosen]) == set(people)
                draws.extend(dict(direction=direction, pair=pair, budget=budget, policy=arm, row=int(row),
                                  donor=int(tp[row]), activity=CLASSES[ty[row]]) for row in chosen)
                x, y = tx[chosen], ty[chosen]
                # Each class has the same total mass. For sampled classes, transfer half
                # its mass to target examples; unseen target classes retain source labels.
                source_mass = np.where(np.isin(sy, classes), .5, 1.) * domain_weights(sy)
                target_mass = .5 / (5 * np.bincount(y, minlength=5)[y])
                weights = 2 * budget * np.r_[source_mass, target_mass]
                predictions["pool_" + arm] = fit(np.vstack((sx, x)), np.r_[sy, y], weights).predict(vx)
                if arm in ("balanced", "level_ground"):
                    # All rows enter each tree. Bootstrap here would under-sample the few target rows.
                    forest = RandomForestClassifier(n_estimators=48, max_depth=8, bootstrap=False,
                                                    random_state=23, n_jobs=1)
                    forest.fit(np.vstack((sx, x)), np.r_[sy, y], sample_weight=weights)
                    predictions["forest_" + arm] = forest.predict(vx)
                    if arm == "balanced":
                        direct = RandomForestClassifier(n_estimators=48, max_depth=8, bootstrap=False,
                                                        random_state=23, n_jobs=1)
                        predictions["forest_target_only"] = direct.fit(x, y).predict(vx)
                if arm not in ("balanced", "posture", "motion"):
                    continue
                gate, motion, posture = frozen
                if arm != "posture":
                    motion = fit(x[y < 3], y[y < 3])
                if arm != "motion":
                    posture = fit(x[y >= 3], y[y >= 3])
                predictions[arm] = predict_heads((gate, motion, posture), vx)
                if arm == "balanced":
                    target_scale = StandardScaler().fit(raw_target[chosen])
                    few = target_scale.transform(raw_target[chosen])
                    test = target_scale.transform(raw_target[valid])
                    predictions["target_only_flat"] = fit(few, y).predict(test)
                    predictions["target_only_hierarchy"] = predict_heads(heads(few, y), test)
            for method, pred in predictions.items():
                for person in sorted(td):
                    mask = tp[valid] == person
                    f1 = f1_score(ty[valid][mask], pred[mask], labels=np.arange(5), average=None, zero_division=0)
                    results.append(dict(direction=direction, pair=pair, donors="-".join(map(str, people)),
                                        budget=budget, method=method, person=person, macro_f1=f1.mean(),
                                        **dict(zip(CLASSES, f1))))
    return results, draws


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        uci = (*read_uci(), UCI_DEV)
        motion = (*read_motion(), MOTION_VAL)
        results, draws = [], []
        for args in (("waist_to_pocket", uci, motion), ("pocket_to_waist", motion, uci)):
            rows, selections = run(*args)
            results.extend(rows)
            draws.extend(selections)
        df = pd.DataFrame(results)
        df.to_csv(HERE / "results.csv", index=False)
        pd.DataFrame(draws).to_csv(HERE / "selections.csv", index=False)
        print((100 * df.groupby(["direction","budget","method"]).macro_f1.mean().unstack()).round(2).T.to_string())
