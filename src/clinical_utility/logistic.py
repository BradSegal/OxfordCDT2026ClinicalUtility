"""Logistic regression written out in numpy, for recalibration and weighting demos.

The model is P(y = 1 | x) = sigmoid(b0 + x . b). Coefficients maximise the
(optionally weighted) log-likelihood by Newton's method, also called iteratively
reweighted least squares (IRLS). There is no penalty, so perfectly separable
data have no finite answer; the fit then stops with an error instead of
returning enormous coefficients. Separation is judged on the fitted
probabilities, not on coefficient size, because a coefficient's size depends on
the units of its feature. For the same reason, convergence is judged on how far
a step moves the linear predictor, not on how far it moves the coefficients.
"""

import numpy as np

# When every row is fitted to within this distance, treat the classes as separable.
SEPARATION_TOLERANCE = 1e-6


def sigmoid(x):
    """sigmoid(x) = 1 / (1 + exp(-x)), computed without overflow."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=float)))


def logit(p):
    """logit(p) = log(p / (1 - p)).

    logit(0) = -inf and logit(1) = +inf. Values outside [0, 1] or NaN give NaN.
    """
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(p) - np.log1p(-p)
    return np.where((p >= 0) & (p <= 1), result, np.nan)


def class_weights(y, class_weight):
    """Per-row weights for a class-weight rule.

    None gives every row weight 1. "balanced" gives class c the weight
    n / (2 n_c), so both classes carry equal total weight. A dictionary
    {0: w0, 1: w1} is used as supplied.
    """
    y = np.asarray(y, dtype=int)
    if class_weight is None:
        return np.ones(len(y))
    if class_weight == "balanced":
        counts = np.bincount(y, minlength=2)
        if counts.min() == 0:
            raise ValueError("Balanced weights need both classes.")
        class_weight = {c: len(y) / (2 * counts[c]) for c in (0, 1)}
    return np.where(y == 1, float(class_weight[1]), float(class_weight[0]))


def fit_logistic_regression(X, y, class_weight=None, offset=None, max_iter=100, tol=1e-10,
                            row_weights=None):
    """Maximise sum_i w_i [y_i z_i - log(1 + exp(z_i))], z_i = offset_i + b0 + x_i . b.

    One Newton step solves (X'WX) step = X'w(y - p), with W = diag(w p (1 - p))
    and an intercept column added to X; a step is halved until the
    log-likelihood does not fall. The fit has converged when a step changes no
    row's linear predictor z_i by more than ``tol``, which does not depend on
    the units of the features. ``class_weight`` follows ``class_weights``;
    ``row_weights``, one per row, multiply those class weights. ``offset`` is a
    known term with coefficient fixed at 1 (used for calibration-in-the-large).
    Returns {"intercept", "coefficients", "iterations"}.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    if len(X) != len(y):
        raise ValueError("X and y need the same number of rows.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1.")
    if not np.isfinite(X).all():
        raise ValueError("Features must be finite.")
    weights = class_weights(y, class_weight)
    if row_weights is not None:
        row_weights = np.asarray(row_weights, dtype=float)
        if row_weights.shape != y.shape or not np.all(np.isfinite(row_weights) & (row_weights >= 0)):
            raise ValueError("row_weights needs one finite, non-negative weight per row.")
        weights = weights * row_weights
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    design = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(design.shape[1])

    def log_likelihood(b):
        z = offset + design @ b
        return float(weights @ (y * z - np.logaddexp(0.0, z)))

    current = log_likelihood(beta)
    for iteration in range(1, max_iter + 1):
        p = sigmoid(offset + design @ beta)
        gradient = design.T @ (weights * (y - p))
        hessian = design.T @ ((weights * p * (1 - p))[:, None] * design)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            raise ValueError("The information matrix became singular; the classes may be "
                             "separable or a feature may be constant.") from None
        scale = 1.0
        while log_likelihood(beta + scale * step) < current - 1e-12 and scale > 1e-6:
            scale /= 2
        beta = beta + scale * step
        current = log_likelihood(beta)
        if np.max(np.abs(design @ (scale * step))) < tol:
            break
    else:
        raise ValueError("Logistic regression did not converge; the classes may be separable.")
    p = sigmoid(offset + design @ beta)
    if np.all(np.abs(y - p)[weights > 0] < SEPARATION_TOLERANCE):
        raise ValueError("Every row is fitted almost exactly: the classes are separable and the "
                         "maximum-likelihood coefficients are infinite.")
    return {"intercept": float(beta[0]), "coefficients": beta[1:], "iterations": iteration}


def predict_logistic(model, X):
    """sigmoid(b0 + x . b) for each row of X."""
    X = np.asarray(X, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    return sigmoid(model["intercept"] + X @ model["coefficients"])
