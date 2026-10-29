"""Class weighting versus threshold choice, on a small synthetic two-feature dataset.

The features imitate two ECG rhythm summaries but are synthetic: they are not
computed from any ECG. Within each class they are Gaussian with a shared
covariance, so the true log-odds of AF are linear in the features and an
unweighted logistic regression is correctly specified. Class weighting then
mainly moves the intercept; it changes which recordings sit above 0.5, not how
well the model ranks them.
"""

import numpy as np

from . import reference
from .comparison import auc
from .logistic import fit_logistic_regression, predict_logistic

FEATURES = ("rr_irregularity", "p_wave_absence")
AF_MEAN = np.array([1.6, 1.2])
COVARIANCE = np.array([[1.0, .3], [.3, 1.0]])


def synthetic_rhythm_features(n=2000, seed=20261106, prevalence=.1):
    """Deterministic synthetic features: non-AF ~ N((0, 0), S), AF ~ N((1.6, 1.2), S).

    S has unit variances and correlation 0.3. Exactly round(prevalence * n)
    recordings have AF. Alternate recordings within each class are assigned to
    "train" and "test". Returns {"X", "y", "split", "feature_names"}.
    """
    rng = np.random.default_rng(seed)
    positives = int(round(prevalence * n))
    y = np.r_[np.ones(positives), np.zeros(n - positives)]
    X = rng.multivariate_normal(np.zeros(2), COVARIANCE, size=n) + np.outer(y, AF_MEAN)
    split = np.empty(n, dtype="U5")
    for label in (0, 1):
        rows = np.flatnonzero(y == label)
        split[rows] = np.where(np.arange(len(rows)) % 2 == 0, "train", "test")
    order = rng.permutation(n)
    return {"X": X[order], "y": y[order], "split": split[order], "feature_names": FEATURES}


def compare_class_weighting(dataset, costs=None, class_weight="balanced"):
    """Fit unweighted and class-weighted logistic regressions on train; compare them on test.

    Reports for each model: intercept, coefficients, test AUC, reviews and loss
    at the default 0.5 cutoff, and the minimum loss over every distinct cutoff
    (``reference.sweep_thresholds``). Also reports the cosine between the two
    coefficient vectors (1 means the same direction in feature space).
    """
    costs = reference.DEFAULT_COSTS if costs is None else costs
    train, test = dataset["split"] == "train", dataset["split"] == "test"
    result = {}
    for name, weight in (("unweighted", None), ("weighted", class_weight)):
        model = fit_logistic_regression(dataset["X"][train], dataset["y"][train], weight)
        scores = predict_logistic(model, dataset["X"][test])
        records = [{"score": float(s), "af_present": bool(t)}
                   for s, t in zip(scores, dataset["y"][test])]
        at_half = reference.confusion_counts(reference.evaluate_policy(records, .5, costs))
        best = reference.best_thresholds(reference.sweep_thresholds(records, costs))[0]
        result[name] = {"intercept": model["intercept"],
                        "coefficients": [float(c) for c in model["coefficients"]],
                        "auc": auc(scores, dataset["y"][test]),
                        "reviews_at_0_5": at_half["TP"] + at_half["FP"],
                        "loss_at_0_5": float(reference.matrix_loss(at_half, costs)),
                        "min_loss": best["total_loss"], "min_loss_threshold": best["threshold"],
                        "min_loss_reviews": best["reviews"]}
    a = np.array(result["unweighted"]["coefficients"])
    b = np.array(result["weighted"]["coefficients"])
    result["coefficient_cosine"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    result["test_recordings"] = int(test.sum())
    return result
