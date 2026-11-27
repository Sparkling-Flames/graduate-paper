"""Source-only transfer of acceleration measured along the estimated gravity axis."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "activity_budget"))
from pilot import UCI_DEV, fit
from run_pooling import CLASSES, MOTION_VAL, features, read_motion, read_uci

def temporal(signal):
    """Time summaries for one scalar signal sampled at 50 Hz."""
    mean = signal.mean(axis=1)
    centered = signal - mean[:, None]
    variance = np.mean(centered**2, axis=1)
    # Stabilize ratios for near-constant signals; 0.02 is in native signal units.
    std_floor = np.sqrt(variance + .02**2)
    skew = np.mean((centered / std_floor[:, None])**3, axis=1)
    correlations = [np.mean(centered[:, :-lag] * centered[:, lag:], axis=1) / std_floor**2
                    for lag in (12, 24)]
    return np.column_stack([mean, np.sqrt(variance), *np.quantile(signal, [.1, .5, .9], axis=1),
                            np.sqrt((np.diff(signal, axis=1)**2).mean(axis=1)), skew, *correlations])


def extract(body, gyro, gravity):
    unit_g = gravity / np.maximum(np.linalg.norm(gravity, axis=2, keepdims=True), 1e-8)
    vertical = np.sum(body * unit_g, axis=2)
    horizontal = np.linalg.norm(body - vertical[:, :, None] * unit_g, axis=2)
    base = features(body, gyro, gravity)
    return np.column_stack([base, temporal(np.linalg.norm(body, axis=2)), temporal(np.linalg.norm(gyro, axis=2)),
                            temporal(horizontal), temporal(np.abs(vertical)), temporal(vertical)])


def arm_features(x):
    base, body, gyro, horizontal, unsigned, signed = np.split(x, [13, 22, 31, 40, 49], axis=1)
    # Original features already include the mean, SD, outer quantiles and RMS difference of each magnitude.
    extra = [3, 6, 7, 8]  # median, skew, autocorrelations at 12 and 24 samples
    return {"baseline13": base, "magnitude_time": np.column_stack((base, body, gyro)),
            "magnitude_time_unique": np.column_stack((base, body[:, extra], gyro[:, extra])),
            "unsigned_projection": np.column_stack((base, horizontal, unsigned)),
            "signed_projection": np.column_stack((base, horizontal, signed))}


def run(direction, source, target):
    sx, sy, sp, sd = source
    tx, ty, tp, td = target
    train = ~np.isin(sp, list(sd))
    valid = np.isin(tp, list(td))
    assert set(tp[valid]) == td
    scores, confusions = [], []
    source_arms = arm_features(sx[train])
    for arm, target_x in arm_features(tx[valid]).items():
        x = source_arms[arm]
        scaler = StandardScaler().fit(x)
        model = fit(scaler.transform(x), sy[train])
        prediction = model.predict(scaler.transform(target_x))
        for p in sorted(td):
            rows = tp[valid] == p
            y, pred = ty[valid][rows], prediction[rows]
            matrix = confusion_matrix(y, pred, labels=np.arange(5))
            f1 = f1_score(y, pred, average=None, labels=np.arange(5), zero_division=0)
            common = dict(direction=direction, arm=arm, person=p)
            scores.append(dict(**common, windows=len(y), macro_f1=f1.mean(), stairs_f1=f1[1:3].mean(),
                               posture_f1=f1[3:5].mean(), **dict(zip(CLASSES, f1))))
            confusions.extend(dict(**common, actual=a, predicted=b, count=matrix[i, j])
                              for i, a in enumerate(CLASSES) for j, b in enumerate(CLASSES))
    return scores, confusions


def main():
    uci = (*read_uci(extract), UCI_DEV)
    motion = (*read_motion(extract), MOTION_VAL)
    stats, counts = [], []
    for name, source, target in (("uci_to_motion", uci, motion), ("motion_to_uci", motion, uci)):
        scores, confusion = run(name, source, target)
        stats.extend(scores)
        counts.extend(confusion)
    df = pd.DataFrame(stats)
    df.to_csv(HERE / "scores.csv", index=False)
    pd.DataFrame(counts).to_csv(HERE / "confusions.csv", index=False)
    print(df.groupby(["direction", "arm"])[["macro_f1", *CLASSES]].mean().round(4).to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
