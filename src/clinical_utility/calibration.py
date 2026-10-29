"""Probability quality: the Platt map, calibration intercept and slope, and prior shift.

Library tools do the fitting: scikit-learn's ``LogisticRegression`` for the
Platt map and statsmodels' binomial GLM for the calibration intercept and
slope. This module fixes the specification once, so the data generator, the
checkers and the reference answers agree, and adds the guards that a
learner's own scores need: statsmodels only warns on perfectly separated data
and returns enormous coefficients, so separation is detected before fitting.
"""

import warnings

import numpy as np
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

__all__ = ["platt", "intercept_slope", "shift_prior", "calibration_report", "brier_difference"]


def platt():
    """An unfitted Platt map: logistic regression of the label on logit(score), unpenalised.

    ``C=np.inf`` switches the penalty off (``penalty=None`` is deprecated from
    scikit-learn 1.8). The Newton-Cholesky solver reaches the exact maximum
    likelihood; the default lbfgs solver stops about 1e-4 away from it.
    """
    return make_pipeline(FunctionTransformer(logit),
                         LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10))


def _separated(z, y):
    """True when one threshold on z puts every AF recording on one side (MLE slope infinite)."""
    positive, negative = z[y], z[~y]
    return positive.min() >= negative.max() or negative.min() >= positive.max()


def intercept_slope(y, p):
    """Calibration-in-the-large and calibration slope of stated probabilities ``p``.

    Intercept: ``logit P(AF) = a + logit(p)``, a GLM with ``logit(p)`` as an
    offset (slope fixed at 1). Ideal 0; below 0 means the probabilities are too
    high. Slope: ``logit P(AF) = a + b logit(p)``. Ideal 1; below 1 means they
    are too extreme. Read only ``b`` from the second fit.

    Returns ``{"intercept", "slope", "reasons"}``. A value that cannot be
    estimated is None and ``reasons`` says why: one outcome class, a probability
    of exactly 0 or 1, every probability equal, or (for the slope) scores that
    separate the classes completely.
    """
    y = np.asarray(y, dtype=bool)
    z = logit(np.asarray(p, dtype=float))
    result = {"intercept": None, "slope": None, "reasons": []}
    if not len(y) or y.all() or not y.any():
        result["reasons"].append("these recordings have only one outcome")
        return result
    if not np.isfinite(z).all():
        result["reasons"].append("a probability of exactly 0 or 1 has infinite log-odds")
        return result
    outcome = y.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(outcome, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit()
    result["intercept"] = float(fit.params[0])
    if np.ptp(z) == 0:
        result["reasons"].append("every score is the same, so there is no slope to fit")
    elif _separated(z, y):
        result["reasons"].append("the scores separate AF from non-AF completely, so the "
                                 "slope has no finite estimate")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(outcome, sm.add_constant(z), family=sm.families.Binomial()).fit()
        result["slope"] = float(fit.params[1])
    return result


def shift_prior(p, old_prevalence, new_prevalence):
    """Move calibrated probabilities to a population with another AF prevalence.

    ``logit(p_new) = logit(p) + logit(new) - logit(old)``: Bayes' rule when only
    the prevalence changes and the score distribution within each class stays
    the same.
    """
    for value in (old_prevalence, new_prevalence):
        if not 0 < value < 1:
            raise ValueError("Prevalences must lie strictly between 0 and 1.")
    return expit(logit(np.asarray(p, dtype=float)) + logit(new_prevalence) - logit(old_prevalence))


def calibration_report(y, p, bins=10):
    """Brier score, reliability points, calibration-in-the-large and slope of probabilities ``p``.

    The reliability points come from ``calibration_curve`` with equal-count
    (quantile) bins; it returns the observed AF fraction first and the mean
    stated probability second. The intercept and slope are None, with the
    reason in ``reasons``, when they cannot be estimated; for example an
    isotonic map that outputs exactly 0 or 1 has infinite log-odds there, while
    its Brier score and reliability points are still defined.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    y = np.asarray(y, dtype=bool)
    observed, predicted = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    fit = intercept_slope(y, p)
    return {"brier": float(brier_score_loss(y, p)), "observed": observed, "predicted": predicted,
            "intercept": fit["intercept"], "slope": fit["slope"], "reasons": fit["reasons"]}


def brier_difference(y, first, second, resamples=2000, seed=20261106):
    """Brier(second) - Brier(first) on the same recordings, with a 95% paired bootstrap interval.

    A positive difference means ``first`` has the lower (better) Brier score.
    Returns ``{"estimate", "low", "high"}``.
    """
    from .reference import paired_bootstrap
    y = np.asarray(y, dtype=float)
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)

    def statistic(y, first, second, axis=-1):
        return np.mean((second - y) ** 2, axis=axis) - np.mean((first - y) ** 2, axis=axis)
    result = paired_bootstrap((y, first, second), statistic, resamples, seed)
    return {"estimate": float(statistic(y, first, second)),
            "low": float(result.confidence_interval.low),
            "high": float(result.confidence_interval.high)}
