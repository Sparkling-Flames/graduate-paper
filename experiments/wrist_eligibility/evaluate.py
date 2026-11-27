"""Compare the same wrist-only predictions under two sensor eligibility rules."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler


def design(train, test, nonlinear=False):
    scaler = StandardScaler().fit(train)
    train, test = scaler.transform(train), scaler.transform(test)
    if nonlinear:
        kernel = Nystroem(kernel="rbf", gamma=1 / train.shape[1], n_components=min(128, len(train)),
                          random_state=23)
        train = kernel.fit_transform(train)
        test = kernel.transform(test)
    return tuple(np.column_stack((a, np.ones(len(a)))) for a in (train, test))


def classifier(x, labels, classes, teacher_logits=None):
    """Balanced multinomial logistic regression, optional temperature-2 distillation."""
    hard = np.eye(classes)[labels]
    weights = len(labels) / (classes * np.bincount(labels, minlength=classes))
    row_weights = weights[labels, None]
    temperature = 2.
    soft = None if teacher_logits is None else softmax(teacher_logits / temperature, axis=1)

    def objective(flat):
        coef = flat.reshape(x.shape[1], classes)
        logits = x @ coef
        logp = log_softmax(logits, axis=1)
        losses = -(hard * logp).sum(axis=1)
        residual = np.exp(logp) - hard
        if soft is not None:
            logp_t = log_softmax(logits / temperature, axis=1)
            losses = .5 * (losses - temperature ** 2 * (soft * logp_t).sum(axis=1))
            residual = .5 * (residual + temperature * (np.exp(logp_t) - soft))
        loss = row_weights[:, 0] @ losses + .5 * np.square(coef[:-1]).sum()
        grad = x.T @ (row_weights * residual)
        grad[:-1] += coef[:-1]  # C=1, unpenalized intercept for every route.
        return loss, grad.ravel()

    result = minimize(objective, np.zeros(x.shape[1] * classes), method="L-BFGS-B",
                      jac=True, options={"maxiter": 1500, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.reshape(x.shape[1], classes)


def evaluate_fold(x, y, person, complete, train_ids, valid_ids, nonlinear=False):
    """x: wrist-eligible windows × placements × features; y: contiguous class IDs."""
    classes = np.arange(y.max() + 1)
    train = np.isin(person, train_ids)
    valid = np.isin(person, valid_ids)
    paired = train & complete
    assert not (train & valid).any() and np.array_equal(np.unique(y[paired]), classes)
    assert np.isfinite(x[:, 0]).all() and np.isfinite(x[complete]).all()
    wrist_train, wrist_valid = design(x[paired, 0], x[valid, 0], nonlinear)
    test_complete = complete[valid]
    all_train, all_valid = design(x[paired].reshape(paired.sum(), -1),
                                  x[valid & complete].reshape((valid & complete).sum(), -1), nonlinear)
    count = len(classes)
    baseline = classifier(wrist_train, y[paired], count)
    teacher = classifier(all_train, y[paired], count)
    kd = classifier(wrist_train, y[paired], count, all_train @ teacher)
    wrist_kd = classifier(wrist_train, y[paired], count, wrist_train @ baseline)
    extra_train, extra_valid = design(x[train, 0], x[valid, 0], nonlinear)
    all_wrist_baseline = classifier(extra_train, y[train], count)
    logits = {"wrist_ce": wrist_valid @ baseline, "multi_kd": wrist_valid @ kd,
              "wrist_kd": wrist_valid @ wrist_kd, "wrist_ce_all_train": extra_valid @ all_wrist_baseline}
    oracle = np.full_like(logits["wrist_ce"], np.nan)
    oracle[test_complete] = all_valid @ teacher
    logits["multi_teacher"] = oracle
    metrics, byclass = [], []
    for subject in ["pooled", *valid_ids]:
        subject_mask = np.ones(valid.sum(), bool) if subject == "pooled" else person[valid] == subject
        for scope, subset in (("complete", test_complete), ("wrist_eligible", np.ones(valid.sum(), bool)),
                              ("aux_missing", ~test_complete)):
            rows = subject_mask & subset
            if not rows.any():
                continue
            actual = y[valid][rows]
            for method, values in logits.items():
                if method == "multi_teacher" and scope != "complete":
                    continue
                probs = softmax(values[rows], axis=1)
                matrix = confusion_matrix(actual, probs.argmax(axis=1), labels=classes)
                support = matrix.sum(axis=1)
                denom = support + matrix.sum(axis=0)
                f1 = np.divide(2 * matrix.diagonal(), denom, out=np.zeros(count), where=denom > 0)
                common = dict(subject=subject, scope=scope, method=method)
                metrics.append(dict(**common, windows=len(actual),
                                    macro_f1=f1.mean() if support.min() else np.nan,
                                    accuracy=matrix.trace()/len(actual),
                                    nll=log_loss(actual, probs, labels=classes)))
                for c in classes:
                    byclass.append(dict(**common, label=c, support=support[c], correct=matrix[c, c],
                                        predicted=matrix[:, c].sum(), f1=f1[c] if support[c] else np.nan))
    return metrics, byclass
