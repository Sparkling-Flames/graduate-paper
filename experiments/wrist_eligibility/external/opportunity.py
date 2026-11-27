"""Audit original UCI OPPORTUNITY ADL sensor eligibility and score paired predictions."""
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from evaluate import evaluate_fold

sys.path.insert(0, str(ROOT / "experiments" / "exploration_20261127" / "sensors"))
from run import features

ARCHIVE = ROOT / "data" / "OPPORTUNITY.zip"
FOLDER = "OpportunityUCIDataset/dataset/"
PEOPLE = (1, 2, 3)  # S4 is reserved. Open only the five ADL runs of each development subject.
ACTIVITIES = {1: "stand", 2: "walk", 4: "sit", 5: "lie"}
CONFIGS = {  # 1-based; first site is always the deployment sensor.
    "imu_forearm": ("InertialMeasurementUnit", {"RLA": 64, "BACK": 38, "LLA": 90}),
    "acc_wrist": ("Accelerometer", {"RWR": 23, "BACK": 17, "RKN_": 20}),
}
WINDOW = 77
RULES = {"strict": 0, "short_gaps": 3}


def usable(windows, allowance):
    # Count time points where at least one of the requested axes is missing.
    observed = np.isfinite(windows).all(axis=(2, 3))
    return observed[:, 0] & observed[:, -1] & ((~observed).sum(axis=1) <= allowance)


def window_features(windows):
    windows = windows.copy()
    for i in np.flatnonzero(~np.isfinite(windows).all(axis=(1, 2, 3))):
        for channel in windows[i].reshape(WINDOW, -1).T:
            ok = np.isfinite(channel)
            if not ok.all():
                channel[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), channel[ok])
    assert np.isfinite(windows).all()
    return features(windows)


def records(source, subject, adl, sites):
    cols = [i - 1 for start in sites.values() for i in range(start, start+3)]
    with source.open(f"{FOLDER}S{subject}-ADL{adl}.dat") as stream:
        frame = pd.read_csv(stream, sep=r"\s+", header=None, usecols=[0, *cols, 243])
    time = frame[0].to_numpy()
    y = frame[243].to_numpy(dtype=int)
    assert np.isfinite(time).all() and set(np.unique(y)) <= {0, *ACTIVITIES}
    values = np.ascontiguousarray(frame[cols].to_numpy(dtype=float) / 1000)  # milli-g to g
    return time, y, values.reshape(-1, 3, 3)


def audit(source, device, sites):
    raw, coverage = [], []
    blocks = {rule: [] for rule in RULES}
    columns = source.read(FOLDER + "column_names.txt").decode("latin1").splitlines()
    for site, start in sites.items():
        for number, axis in zip(range(start, start+3), "XYZ"):
            assert any(line.startswith(f"Column: {number} {device} {site} acc{axis};")
                       and "milli g" in line for line in columns)
    assert any(line.strip() == "Column: 244 Locomotion" for line in columns)
    for person in PEOPLE:
        for adl in range(1, 6):
            time, labels, signal = records(source, person, adl, sites)
            delta = np.diff(time)
            breaks = np.flatnonzero((np.diff(labels) != 0) | (delta <= 0) | (delta > 50)) + 1
            bounds = np.r_[0, breaks, len(time)]
            ok = np.isfinite(signal).all(axis=2)
            for label, activity in {0: "unlabelled", **ACTIVITIES}.items():
                curr = ok[labels == label]
                raw.append(dict(person=person, adl=adl, activity=activity, rows=len(curr),
                                target_missing_rows=(~curr[:, 0]).sum(), back_missing_rows=(~curr[:, 1]).sum(),
                                other_aux_missing_rows=(~curr[:, 2]).sum(),
                                target_ok_aux_missing_rows=(curr[:, 0] & ~curr.all(axis=1)).sum()))
            for a, b in zip(bounds[:-1], bounds[1:]):
                label = labels[a]
                n = (b - a) // WINDOW
                if label not in ACTIVITIES or not n:
                    continue
                w = signal[a:a+n*WINDOW].reshape(n, WINDOW, 3, 3)
                for rule, allowance in RULES.items():
                    wrist = usable(w[:, :, :1], allowance)
                    joint = usable(w, allowance)
                    assert not (joint & ~wrist).any()
                    coverage.append(dict(rule=rule, person=person, activity=ACTIVITIES[label],
                                         candidate=n, three_usable=joint.sum(),
                                         target_ok_aux_ineligible=(wrist & ~joint).sum(),
                                         target_unusable=(~wrist).sum()))
                    if not wrist.any():
                        continue
                    forearm_x = window_features(w[wrist, :, :1])
                    x = np.full((wrist.sum(), 3, forearm_x.shape[-1]), np.nan)
                    x[:, :1] = forearm_x
                    if joint.any():
                        fused = window_features(w[joint])
                        np.testing.assert_allclose(fused[:, 0], forearm_x[joint[wrist], 0])
                        x[joint[wrist]] = fused
                    blocks[rule].append((x, np.full(len(x), list(ACTIVITIES).index(label)),
                                         np.full(len(x), person), joint[wrist]))
    coverage = pd.DataFrame(coverage).groupby(["rule", "person", "activity"], as_index=False).sum()
    assert (coverage.candidate == coverage.three_usable + coverage.target_ok_aux_ineligible
            + coverage.target_unusable).all()
    return pd.DataFrame(raw), coverage, blocks


def score(config, blocks):
    metrics, byclass = [], []
    for rule, chunks in blocks.items():
        x, y, person, complete = (np.concatenate(col) for col in zip(*chunks))
        for held in PEOPLE:
            train = tuple(p for p in PEOPLE if p != held)
            for model in ("linear", "nystroem_rbf"):
                scores, individual = evaluate_fold(x, y, person, complete, train, (held,),
                                                   nonlinear=model == "nystroem_rbf")
                metrics.extend(dict(config=config, rule=rule, model=model, **row) for row in scores if row["subject"] == held)
                byclass.extend(dict(config=config, rule=rule, model=model, activity=list(ACTIVITIES.values())[row["label"]], **row)
                               for row in individual if row["subject"] == held)
    return pd.DataFrame(metrics), pd.DataFrame(byclass).drop(columns="label")


def main():
    all_raw, all_coverage, all_metrics, all_classes = [], [], [], []
    with ZipFile(ARCHIVE) as source:
        for config, (device, sites) in CONFIGS.items():
            raw, coverage, blocks = audit(source, device, sites)
            metrics, byclass = score(config, blocks)
            all_raw.append(raw.assign(config=config))
            all_coverage.append(coverage.assign(config=config))
            all_metrics.append(metrics)
            all_classes.append(byclass)
    for name, frames in (("raw_missing", all_raw), ("coverage", all_coverage),
                         ("metrics", all_metrics), ("class_metrics", all_classes)):
        pd.concat(frames, ignore_index=True).to_csv(HERE / f"{name}.csv", index=False)
    print(pd.concat(all_coverage).groupby(["config", "rule"]).sum(numeric_only=True).drop(columns="person"))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
