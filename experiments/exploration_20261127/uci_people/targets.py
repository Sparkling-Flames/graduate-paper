"""Does the teacher's distribution need to be paired with the right window?"""
import numpy as np
import torch
from torch.nn import functional as F

from run import DATA, GROUPS, model_and_optimizer, predictions, save, scores, standardize


def train_teacher(x, y):
    model, opt = model_and_optimizer(20261120, [256, 128])
    rng = np.random.default_rng(20261120)
    for _ in range(40):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 128):
            rows = order[start:start+128]
            opt.zero_grad()
            F.cross_entropy(model(x[rows]), y[rows], label_smoothing=0.1).backward()
            opt.step()
    with torch.inference_mode():
        return F.softmax(model(x) / 2, dim=1)


def target_variants(probs, y, seed):
    means = torch.empty_like(probs)
    shuffled = torch.empty_like(probs)
    rng = np.random.default_rng(seed + 500)
    for c in range(6):
        order = rng.permutation(np.flatnonzero(y.numpy() == c))
        means[order] = probs[order].mean(0)
        # A within-class cycle gives every window a different window's complete target.
        shuffled[order] = probs[np.roll(order, 1)]
        torch.testing.assert_close(shuffled[order].mean(0), means[order[0]])
    return {"direct_ls": None, "paired": probs, "class_mean": means, "class_shuffle": shuffled}


def main():
    torch.set_num_threads(1)
    x = np.loadtxt(DATA / "X_train.txt", dtype=np.float32)
    y = np.loadtxt(DATA / "y_train.txt", dtype=np.int64) - 1
    persons = np.loadtxt(DATA / "subject_train.txt", dtype=int)
    train = np.flatnonzero(~np.isin(persons, GROUPS["A"] + GROUPS["B"]))
    xn = standardize(x, train)
    xt, yt = xn[train], torch.from_numpy(y[train])
    teacher_probs = train_teacher(xt, yt)
    result = []
    for seed in (11, 12, 13):
        for name, targets in target_variants(teacher_probs, yt, seed).items():
            model, opt = model_and_optimizer(seed, [32])
            rng = np.random.default_rng(seed)
            for _ in range(40):
                order = rng.permutation(len(yt))
                for start in range(0, len(yt), 128):
                    rows = order[start:start+128]
                    logits = model(xt[rows])
                    if targets is None:
                        loss = F.cross_entropy(logits, yt[rows], label_smoothing=0.1)
                    else:
                        kl = F.kl_div(F.log_softmax(logits/2, dim=1), targets[rows], reduction="batchmean")
                        loss = 0.5 * F.cross_entropy(logits, yt[rows]) + 2 * kl
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            logits = predictions(model, xn)
            for group, ids in GROUPS.items():
                for person in ids:
                    mask = persons == person
                    result.append(dict(seed=seed, method=name, group=group, person=person,
                                       **scores(logits[mask], y[mask])))
    save("targets.csv", result)


if __name__ == "__main__":
    main()
