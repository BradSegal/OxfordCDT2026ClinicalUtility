"""Probability quality: recalibration maps, calibration summaries and prior shift.

Every function takes plain arrays of probabilities (or scores) and 0/1 reference
labels. A recalibration map is fitted on one set of recordings and applied to
others; fitting and assessing on the same records flatters the map.
"""

import numpy as np

from .logistic import fit_logistic_regression, logit, sigmoid

__all__ = ["logit", "sigmoid", "brier_score", "reliability_summary", "platt_fit", "platt_apply",
           "isotonic_fit", "isotonic_apply", "calibration_in_the_large", "calibration_slope",
           "calibration_intercept_slope", "recalibrate_in_the_large", "top_fraction_gap"]


def _labels(y):
    y = np.asarray(y, dtype=float)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Reference labels must be 0 or 1.")
    return y


def brier_score(probabilities, y):
    """Brier = mean((p_i - y_i)^2). Smaller is better; NaN when there are no records."""
    p, y = np.asarray(probabilities, dtype=float), _labels(y)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def reliability_summary(probabilities, y, bins=8):
    """Equal-count bins of stated probabilities, and the Brier score over every recording.

    Recordings are sorted by probability, with their labels, and split into
    ``min(bins, n)`` groups. Returns arrays "predicted" (mean probability),
    "observed" (AF fraction) and "n" (size), one entry per group, and "brier".
    """
    p, y = np.asarray(probabilities, dtype=float), _labels(y)
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, min(bins, len(p)))
    return {"predicted": np.array([p[g].mean() for g in groups]),
            "observed": np.array([y[g].mean() for g in groups]),
            "n": np.array([len(g) for g in groups]), "brier": brier_score(p, y)}


def platt_fit(scores, y):
    """Platt scaling: fit P(y = 1) = sigmoid(a + b logit(score)) by maximum likelihood.

    This is a one-feature logistic regression of y on logit(score). Scores of
    exactly 0 or 1 have infinite logits and are rejected. Returns
    {"intercept": a, "slope": b}. A positive slope preserves the ranking.
    """
    z = logit(scores)
    if not np.isfinite(z).all():
        raise ValueError("Platt scaling needs scores strictly between 0 and 1.")
    model = fit_logistic_regression(z, _labels(y))
    return {"intercept": model["intercept"], "slope": float(model["coefficients"][0])}


def platt_apply(scores, params):
    """sigmoid(a + b logit(score)) with a, b from ``platt_fit``."""
    return sigmoid(params["intercept"] + params["slope"] * logit(scores))


def isotonic_fit(scores, y):
    """Isotonic recalibration by pool-adjacent-violators (PAV).

    Sort records by score. Records sharing a score form one block. Start with
    each block's observed AF fraction; while a block's fraction exceeds the next
    block's, merge them into their size-weighted mean. The result is the
    non-decreasing step function closest to the labels in squared error.
    Returns {"lower": first score in each block, "value": fitted fraction}.
    """
    s, y = np.asarray(scores, dtype=float), _labels(y)
    if not len(s) or not np.isfinite(s).all():
        raise ValueError("Isotonic fitting needs at least one finite score.")
    unique, inverse = np.unique(s, return_inverse=True)
    blocks = [[float(u), float(np.sum(y[inverse == i])), float(np.sum(inverse == i))]
              for i, u in enumerate(unique)]  # lower score, events, records
    pooled = []
    for block in blocks:
        pooled.append(block)
        while len(pooled) > 1 and pooled[-2][1] / pooled[-2][2] > pooled[-1][1] / pooled[-1][2]:
            lower, events, count = pooled.pop()
            pooled[-1][1] += events
            pooled[-1][2] += count
    return {"lower": np.array([b[0] for b in pooled]),
            "value": np.array([b[1] / b[2] for b in pooled])}


def isotonic_apply(scores, model):
    """Value of the last block whose lower score is <= the score (a right-continuous step).

    Scores below the first block take the first block's value.
    """
    index = np.searchsorted(model["lower"], np.asarray(scores, dtype=float), side="right") - 1
    return model["value"][np.clip(index, 0, None)]


def calibration_in_the_large(probabilities, y):
    """Intercept a of logit P(y = 1) = a + logit(p), with the slope fixed at 1.

    Ideal a = 0; a < 0 means the probabilities are too high on average. NaN
    when there are no records, when every label is the same, or when a
    probability is 0 or 1. Raises ValueError if the fit has no finite answer.
    """
    z, y = logit(probabilities), _labels(y)
    if not len(y) or not np.isfinite(z).all() or y.min() == y.max():
        return float("nan")
    return fit_logistic_regression(np.zeros((len(y), 0)), y, offset=z)["intercept"]


def calibration_slope(probabilities, y):
    """Slope b of logit P(y = 1) = a + b logit(p), with both free.

    Ideal b = 1; b < 1 means the probabilities are too extreme (overconfident).
    NaN as for ``calibration_in_the_large`` and when every logit(p) is equal.
    Raises ValueError when the probabilities separate the two labels, because
    the maximum-likelihood slope is then infinite.
    """
    z, y = logit(probabilities), _labels(y)
    if not len(y) or not np.isfinite(z).all() or y.min() == y.max() or np.ptp(z) == 0:
        return float("nan")
    return float(fit_logistic_regression(z, y)["coefficients"][0])


def calibration_intercept_slope(probabilities, y):
    """Logistic calibration summary of stated probabilities p against outcomes y.

    Slope b: fit logit P(y = 1) = a + b logit(p). Ideal b = 1; b < 1 means the
    probabilities are too extreme (overconfident). Intercept (calibration in
    the large) a: fit logit P(y = 1) = a + logit(p), with the slope fixed at 1.
    Ideal a = 0; a < 0 means the probabilities are too high on average. The
    slope is NaN when every logit(p) is equal; both are NaN for p of 0 or 1.
    """
    return {"intercept": calibration_in_the_large(probabilities, y),
            "slope": calibration_slope(probabilities, y)}


def recalibrate_in_the_large(probabilities, old_prevalence, new_prevalence):
    """Move probabilities to a population with a different AF prevalence (prior shift).

    new logit = logit(p) + logit(new prevalence) - logit(old prevalence).
    This is Bayes' rule when only the prevalence changes and the score
    distribution within AF and within non-AF recordings stays the same.
    """
    for value in (old_prevalence, new_prevalence):
        if not 0 < value < 1:
            raise ValueError("Prevalences must lie strictly between 0 and 1.")
    shift = logit(new_prevalence) - logit(old_prevalence)
    return sigmoid(logit(probabilities) + shift)


def top_fraction_gap(probabilities, y, fraction=0.1):
    """Mean stated probability minus observed AF fraction among the top-scoring records.

    The top ceil(fraction * n) records by probability are used (ties at the edge
    broken by input order). A positive gap means overconfidence where the model
    is most certain.
    """
    p, y = np.asarray(probabilities, dtype=float), _labels(y)
    k = int(np.ceil(fraction * len(p)))
    if k == 0:
        return float("nan")
    top = np.argsort(-p, kind="stable")[:k]
    return float(p[top].mean() - y[top].mean())
