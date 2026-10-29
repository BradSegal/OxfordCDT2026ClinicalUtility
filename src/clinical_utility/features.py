"""A small synthetic two-feature dataset for the class-weighting extension.

The features imitate two ECG rhythm summaries but are synthetic: they are not
computed from any ECG. Within each class they are Gaussian with a shared
covariance, so the true log-odds of AF are linear in the features and an
unweighted logistic regression is correctly specified.
"""

import numpy as np
import pandas as pd

FEATURES = ("rr_irregularity", "p_wave_absence")
AF_MEAN = np.array([1.6, 1.2])
COVARIANCE = np.array([[1.0, .3], [.3, 1.0]])
SPLIT_PATTERN = ("train", "train", "validation", "test")


def synthetic_rhythm_features(n=2000, seed=20261106, prevalence=.1):
    """Deterministic synthetic features: non-AF ~ N((0, 0), S), AF ~ N((1.6, 1.2), S).

    S has unit variances and correlation 0.3. Exactly round(prevalence * n)
    recordings have AF. Within each class, recordings are assigned in turn to
    train, train, validation and test, so the split is 50/25/25 and stratified.
    Returns a DataFrame with the two features, ``af`` and ``split``.
    """
    rng = np.random.default_rng(seed)
    positives = int(round(prevalence * n))
    y = np.r_[np.ones(positives, dtype=bool), np.zeros(n - positives, dtype=bool)]
    X = rng.multivariate_normal(np.zeros(2), COVARIANCE, size=n) + np.outer(y, AF_MEAN)
    split = np.empty(n, dtype=object)
    for label in (False, True):
        rows = np.flatnonzero(y == label)
        split[rows] = [SPLIT_PATTERN[i % 4] for i in range(len(rows))]
    order = rng.permutation(n)
    frame = pd.DataFrame(X[order], columns=list(FEATURES))
    frame["af"], frame["split"] = y[order], split[order].astype(str)
    return frame


def fit_models(train, features=FEATURES):
    """Unpenalised logistic regressions on the training rows, without and with balanced weights.

    The penalty is off (``C=np.inf``) because class weights rescale the
    log-likelihood: with scikit-learn's default L2 penalty (C=1) the weights
    would also change how strongly the coefficients are shrunk, and the
    intercept-only correction below would then be approximate.
    """
    from sklearn.linear_model import LogisticRegression
    X, y = train[list(features)], train["af"]
    return {"unweighted": LogisticRegression(C=np.inf, solver="newton-cholesky").fit(X, y),
            "weighted": LogisticRegression(C=np.inf, solver="newton-cholesky",
                                           class_weight="balanced").fit(X, y)}


def test_report(models, validation, test, costs, features=FEATURES):
    """Each model at the cutoff chosen on validation by loss, assessed on test, per 1,000 recordings."""
    from .reference import cheapest, counts, operating_points, policy_cost
    rows = []
    for name, model in models.items():
        cut = float(cheapest(operating_points(validation["af"],
                                              model.predict_proba(validation[list(features)])[:, 1],
                                              costs))["threshold"])
        review = model.predict_proba(test[list(features)])[:, 1] >= cut
        c, n = counts(test["af"], review), len(test)
        rows.append({"model": name, "cutoff": cut, "reviews": 1000 * (c["TP"] + c["FP"]) / n,
                     "af_detected": 1000 * c["TP"] / n, "af_missed": 1000 * c["FN"] / n,
                     "loss": policy_cost(test["af"], review, costs)})
    return pd.DataFrame(rows).set_index("model")


def tuned_threshold(train, costs, features=FEATURES):
    """scikit-learn's cost-based threshold tuning, by 5-fold cross-validation on the training rows.

    The cross-validation is internal to the tuner: it chooses the threshold
    without touching the validation or test rows. Use the fitted tuner through
    ``.predict()``, which applies the tuned threshold.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import TunedThresholdClassifierCV

    from .reference import policy_cost
    scorer = make_scorer(policy_cost, greater_is_better=False, costs=costs)
    return TunedThresholdClassifierCV(LogisticRegression(C=np.inf, solver="newton-cholesky"),
                                      scoring=scorer, cv=5).fit(
        train[list(features)], train["af"])


def undo_weighting(q, positive_to_negative_weight):
    """Original-population probability from a class-weighted model's output.

    Weighting AF by w multiplies the fitted odds by w, so subtract log(w) on
    the log-odds scale.
    """
    from scipy.special import expit, logit
    return expit(logit(np.asarray(q, dtype=float)) - np.log(positive_to_negative_weight))
