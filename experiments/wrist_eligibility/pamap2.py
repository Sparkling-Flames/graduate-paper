"""Test deployment-wrist eligibility on the six PAMAP2 development participants."""
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate import evaluate_fold

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "exploration_20261127" / "sensors"
sys.path.insert(0, str(PREVIOUS))
from run import ACTIVITIES, ARCHIVE, LABELS, TRAIN, VALID, eligible, features, raw_windows, select_windows

PEOPLE = TRAIN + VALID


def load_windows():
    records = {"strict": [], "short_gaps": []}
    counts = []
    with ZipFile(ARCHIVE) as source:
        for subject in PEOPLE:
            for label, windows in raw_windows(source, subject):
                for rule, blocks in records.items():
                    repair = rule == "short_gaps"
                    wrist = windows[:, :, :1, :]
                    target = eligible(wrist, repair)
                    joint = eligible(windows, repair)
                    assert not (joint & ~target).any()
                    counts.append(dict(rule=rule, subject=subject, activity=ACTIVITIES[label],
                                       candidate=len(joint), complete=joint.sum(),
                                       wrist_ok_aux_missing=(target & ~joint).sum(),
                                       wrist_unusable=(~target).sum()))
                    if not target.any():
                        continue
                    single = features(select_windows(wrist, repair))
                    x = np.full((len(single), 3, single.shape[2]), np.nan)
                    x[:, 0, :] = single[:, 0, :]
                    if joint.any():
                        fused = features(select_windows(windows, repair))
                        np.testing.assert_allclose(fused[:, 0], x[joint[target], 0])
                        x[joint[target]] = fused
                    blocks.append((x, np.full(len(x), LABELS.index(label)), np.full(len(x), subject),
                                   joint[target]))
    counts = pd.DataFrame(counts).groupby(["rule", "subject", "activity"], as_index=False).sum()
    assert (counts.candidate == counts.complete + counts.wrist_ok_aux_missing + counts.wrist_unusable).all()
    for rule, folder in (("strict", PREVIOUS), ("short_gaps", PREVIOUS / "short_gaps")):
        old = pd.read_csv(folder / "windows.csv").set_index(["subject", "activity"]).sort_index()
        new = counts[counts.rule == rule].set_index(["subject", "activity"]).sort_index()
        pd.testing.assert_index_equal(old.index, new.index)
        np.testing.assert_array_equal(old[["candidate", "retained"]], new[["candidate", "complete"]])
    return {rule: tuple(np.concatenate(col) for col in zip(*blocks)) for rule, blocks in records.items()}, counts


def main():
    data, counts = load_windows()
    measurements, activities = [], []
    folds = [("original", TRAIN, VALID)]
    folds.extend(("loso", tuple(p for p in PEOPLE if p != test), (test,)) for test in PEOPLE)
    for rule, (x, y, person, complete) in data.items():
        for split, train_ids, valid_ids in folds:
            for model in ("linear", "rbf"):
                stats, per_class = evaluate_fold(x, y, person, complete, train_ids, valid_ids,
                                                 nonlinear=model == "rbf")
                fold = dict(model=model, rule=rule, split=split, held_out="+".join(map(str, valid_ids)))
                measurements.extend(dict(**fold, **row) for row in stats)
                activities.extend(dict(**fold, activity=ACTIVITIES[LABELS[row["label"]]], **row)
                                  for row in per_class)
    results = pd.DataFrame(measurements)
    counts.to_csv(HERE / "pamap_coverage.csv", index=False)
    results.to_csv(HERE / "pamap_metrics.csv", index=False)
    pd.DataFrame(activities).drop(columns="label").to_csv(HERE / "pamap_activity.csv", index=False)
    individuals = results[results.subject != "pooled"]
    print(counts[counts.subject.isin(VALID)].groupby("rule").sum(numeric_only=True).drop(columns="subject"))
    means = individuals.groupby(["model", "rule", "split", "scope", "method"]).macro_f1.mean().unstack("method")
    print((100 * means).round(2).to_string())


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
