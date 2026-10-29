"""Feedback for extension exercises, including small counterexamples to common errors.

Each case gives the arguments, the expected answer and a default hint. A case
may also list other valid answers (``accept``) and plausible wrong answers with
their own hints (``diagnose``), so a learner hears which misconception they hit.
A returned value of the wrong kind, or a dictionary without the expected keys,
is reported before any calculation is compared. Learner functions are called
only inside the feedback boundary. Normal notebook use never raises;
UTILITY_STRICT=1 makes worked-copy verification fail visibly. Reference
calculations live in the existing semantic modules.

``ready`` lets a later cell wait for a learner function, naming the problem,
how to unblock it and how to recover (the core's ``RECOVERY``: Runtime > Run
before, then the cell itself); ``waiting`` prints the same kind of line for
anything else a cell needs. ``use_reference`` is the stuck switch that supplies
a reference version of a function later cells use.
"""

import ast
import inspect
import math
import os
import textwrap
from functools import lru_cache

import numpy as np

from . import calibration, comparison, data, deployment, logistic, own_scores, reference, windows
from .checks import RECOVERY, waiting  # noqa: F401  (re-exported for the notebooks)

WAITING = "Replace each ... with your answer and run the cell again."


def _unfinished(value):
    if value is Ellipsis:
        return True
    if isinstance(value, dict):
        return any(_unfinished(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_unfinished(v) for v in value)
    return False


def _holds_ellipsis(code):
    """Whether compiled code, or code nested inside it, uses the `...` constant at all."""
    return any(const is Ellipsis or (inspect.iscode(const) and _holds_ellipsis(const))
               for const in code.co_consts)


def _placeholder_in_source(function):
    """True when the function body still holds a `...` placeholder statement.

    Only placeholders in statement, assignment or return position count, so a
    learner's ``x[..., 0]`` is not mistaken for unfinished work. Scaffolds can
    therefore leave ``gradient = ...`` without guards.
    """
    code = getattr(function, "__code__", None)
    if code is None or not _holds_ellipsis(code):
        return False
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, SyntaxError):
        return False
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if (isinstance(node, (ast.Expr, ast.Assign, ast.Return)) and isinstance(value, ast.Constant)
                and value.value is Ellipsis):
            return True
    return False


def _equal(actual, expected, rtol=1e-6):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() <= actual.keys()
                and all(_equal(actual[k], v, rtol) for k, v in expected.items()))
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, (tuple, list)):
        return (isinstance(actual, (tuple, list)) and len(actual) == len(expected)
                and all(_equal(a, b, rtol) for a, b in zip(actual, expected)))
    try:
        a, b = np.asarray(actual), np.asarray(expected)
        if b.dtype.kind in "USO":
            return a.shape == b.shape and np.array_equal(a, b)
        if a.dtype.kind not in "biuf":
            return False
        return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=1e-8, equal_nan=True)
    except (TypeError, ValueError, OverflowError):
        return False


def _kind(value):
    if value is None:
        return "None"
    name = type(value).__name__
    return f"{'an' if name[0] in 'aeiou' else 'a'} {name}"


def _shape_problem(actual, expected, where=None):
    """A hint when the answer is the wrong kind of value or lacks expected keys; else None."""
    subject, verb, tense = (f"The entry {where!r}", "be", "is") if where else (
        "The function", "return", "returned")
    if isinstance(expected, dict):
        keys = ", ".join(repr(k) for k in expected)
        if not isinstance(actual, dict):
            return f"{subject} should {verb} a dictionary with the keys {keys}; it {tense} {_kind(actual)}."
        missing = [k for k in expected if k not in actual]
        if missing:
            plural = "s" if len(missing) > 1 else ""
            return (f"{subject} {tense} a dictionary without the key{plural} "
                    f"{', '.join(repr(k) for k in missing)}. Use the keys {keys}.")
        for key, value in expected.items():
            problem = _shape_problem(actual[key], value, key)
            if problem:
                return problem
        return None
    if isinstance(expected, str):
        return None if isinstance(actual, str) else f"{subject} should {verb} a string; it {tense} {_kind(actual)}."
    numeric = isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool)
    if numeric and (actual is None or isinstance(actual, (dict, str, list, tuple))):
        return f"{subject} should {verb} one number; it {tense} {_kind(actual)}."
    return None


def _case(args, expected, hint, accept=(), diagnose=(), rtol=1e-6):
    return {"args": args, "expected": expected, "hint": hint, "accept": tuple(accept),
            "diagnose": tuple(diagnose), "rtol": rtol}


# --- Reference variants: valid alternatives and known mistakes ----------------


def _summary(p, y, bins, sort_labels=True, brier="records"):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    labels = [y[g] if sort_labels else y[i * len(g):(i + 1) * len(g)] for i, g in enumerate(groups)]
    value = {"records": np.mean((p - y) ** 2), "squared_mean": np.mean(p - y) ** 2}[brier]
    return {"predicted": np.array([p[g].mean() for g in groups]),
            "observed": np.array([lab.mean() for lab in labels]),
            "n": np.array([len(g) for g in groups]), "brier": value}


def _contribution(old, new, y, p_t, weight=None):
    y = np.asarray(y, dtype=bool)
    weight = p_t / (1 - p_t) if weight is None else weight
    return ((np.asarray(new) >= p_t).astype(float) - (np.asarray(old) >= p_t)) * np.where(y, 1., -weight)


def _bootstrap(old, new, y, p_t, resamples, seed, draw="rows", quantiles=(.025, .975), weight=None):
    """The paired bootstrap, with one of several valid ways of drawing the indices."""
    contribution = _contribution(old, new, y, p_t, weight)
    n, rng = len(contribution), np.random.default_rng(seed)
    if draw == "rows":
        indices = rng.integers(0, n, size=(resamples, n))
    elif draw == "columns":
        indices = rng.integers(0, n, size=(n, resamples)).T
    elif draw == "loop":
        indices = np.array([rng.integers(0, n, n) for _ in range(resamples)])
    elif draw == "choice":
        indices = rng.choice(n, size=(resamples, n))
    else:
        indices = np.array([rng.choice(n, n) for _ in range(resamples)])
    means = contribution[indices].mean(axis=1)
    low, high = np.quantile(means, quantiles)
    return {"estimate": contribution.mean(), "low": low, "high": high}


def _unpaired_bootstrap(old, new, y, p_t, resamples, seed):
    rng = np.random.default_rng(seed)
    old, new, y = np.asarray(old), np.asarray(new), np.asarray(y, dtype=bool)
    draws = []
    for _ in range(resamples):
        a, b = rng.integers(0, len(y), len(y)), rng.integers(0, len(y), len(y))
        draws.append(comparison.policy_net_benefit(new[b], y[b], p_t)
                     - comparison.policy_net_benefit(old[a], y[a], p_t))
    low, high = np.quantile(draws, [.025, .975])
    return {"estimate": comparison.policy_net_benefit(new, y, p_t)
            - comparison.policy_net_benefit(old, y, p_t), "low": low, "high": high}


def _changed(ids, old, new, y, p_t, strict=False, weight=None, any_change=False):
    old, new = np.asarray(old), np.asarray(new)
    a, b = (old > p_t, new > p_t) if strict else (old >= p_t, new >= p_t)
    changed = (old != new) if any_change else (a != b)
    price = p_t / (1 - p_t) if weight is None else weight
    contribution = (b.astype(float) - a) * np.where(y, 1., -price)
    return {"recording": np.asarray(ids)[changed], "direction": np.where(b[changed], "up", "down"),
            "contribution": contribution[changed]}


def _weighted_fit(X, y, w):
    """Weighted logistic coefficients, intercept first, via integer replication of rows."""
    model = logistic.fit_logistic_regression(np.repeat(X, w.astype(int), axis=0),
                                             np.repeat(y, w.astype(int)))
    return np.r_[model["intercept"], model["coefficients"]]


def _queue(mean, rate, capacity, days, seed, loop=False, at_least=False):
    rng = np.random.default_rng(seed)
    if loop:
        reviews = []
        for _ in range(days):
            arrivals = rng.poisson(mean)
            reviews.append(rng.binomial(arrivals, rate))
        reviews = np.array(reviews)
    else:
        reviews = rng.binomial(rng.poisson(mean, size=days), rate)
    return float(np.mean(reviews >= capacity if at_least else reviews > capacity))


def _platt_brier(splits, fit_split, assess_split):
    fitted = calibration.platt_fit(splits[fit_split]["resnet"], splits[fit_split]["af_present"])
    scores = calibration.platt_apply(splits[assess_split]["resnet"], fitted)
    return calibration.brier_score(scores, splits[assess_split]["af_present"])


def _repeat(p, costs, repeat_cost, q, se, sp, delay, variant="right"):
    review = p * costs["TP"] + (1 - p) * costs["FP"]
    accept = p * costs["FN"] + (1 - p) * costs["TN"]
    readable = p * (se * costs["TP"] + (1 - se) * costs["FN"]) + (1 - p) * (
        (1 - sp) * costs["FP"] + sp * costs["TN"])
    fallback = accept if variant == "no_review_fallback" else review
    once = repeat_cost + delay
    if variant == "no_fallback":
        repeat = once + q * readable
    elif variant == "charged_twice":
        repeat = once + q * (once + readable) + (1 - q) * (once + fallback)
    else:
        repeat = once + q * readable + (1 - q) * fallback
    losses = {"review_now": review, "no_review_now": accept, "repeat": repeat}
    return {**losses, "best": min(losses, key=losses.get)}


def _recording_sensitivity(window_positive, af, rule):
    return float(np.mean(rule(window_positive)[np.asarray(af, dtype=bool)]))


@lru_cache(maxsize=1)
def capacity_example():
    """The mean-feasible cutoff at 5% prevalence, and Extension 4's prevalence answer."""
    rows = []
    for row in reference.sweep_thresholds(data.load_records("random_forest", "development"),
                                          reference.DEFAULT_COSTS):
        counts = reference.projected_counts(row, .05)
        rows.append({"row": row, "reviews": counts["TP"] + counts["FP"],
                     "loss": reference.matrix_loss(counts, reference.DEFAULT_COSTS)})
    chosen = min((r for r in rows if r["reviews"] <= 50), key=lambda r: r["loss"])["row"]
    grid = [k / 200 for k in range(2, 21)]
    safe = []
    for prevalence in grid:
        counts = reference.projected_counts(chosen, prevalence)
        if deployment.overflow_probability(counts["TP"] + counts["FP"], 50) <= .1:
            safe.append(prevalence)
    return {"threshold": chosen["threshold"], "grid": grid, "highest_safe": max(safe)}


@lru_cache(maxsize=1)
def cases():
    """Small test inputs; full-cohort outputs are checked by the scenario tests."""
    old = np.array([.1, .2, .8, .4, .6, .2])
    new = np.array([.2, .1, .9, .1, .7, .2])
    y = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
    p = .2
    ids = np.array(["a", "b", "c", "d", "e", "f"])
    splits = {split: data.load_arrays(split) for split in data.SPLITS}
    cal, dev = splits["calibration"], splits["development"]
    params = calibration.platt_fit(cal["resnet"], cal["af_present"])
    probability_feature = logistic.fit_logistic_regression(cal["resnet"], cal["af_present"])
    log_feature = logistic.fit_logistic_regression(np.log(cal["resnet"]), cal["af_present"])
    rows = [dict(threshold=.1, reviews=70., loss=80., overflow=.5),
            dict(threshold=.3, reviews=49., loss=90., overflow=.45),
            dict(threshold=.5, reviews=35., loss=110., overflow=.03),
            dict(threshold=.8, reviews=10., loss=170., overflow=.0)]
    positive = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0]], bool)
    repeat_inputs = (.3, reference.DEFAULT_COSTS, .2, .8, .76, .95, .0)
    weight_X = np.array([[-1., 0.], [0., 1.], [1., 0.], [0., -1.], [2., 1.], [-2., 1.]])
    weight_y = np.array([0, 1, 1, 0, 0, 1])
    weight_w = np.array([1., 3., 3., 1., 1., 3.])
    capacity = capacity_example()
    summary_p, summary_y = np.array([.8, .1, .6, .2]), np.array([1, 0, 1, 0])
    review = np.array([True, True, True, False, False, False])
    review_af = np.array([True, True, False, False, False, False])
    calls = np.array([[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0]], bool)
    calls_af = np.array([True, True, True, False])

    def nb_winner(o, n, truth, cut):
        a = comparison.policy_net_benefit(o, truth, cut)
        b = comparison.policy_net_benefit(n, truth, cut)
        return {"winner": "resnet" if b > a else "random_forest", "basis": "net_benefit",
                "preference": cut}

    return {
        # Extension 1
        "cal_summary": [
            _case((summary_p, summary_y, 2), _summary(summary_p, summary_y, 2),
                  "Sort scores and labels together, use equal-count bins, and square each "
                  "residual before averaging.",
                  diagnose=[(_summary(summary_p, summary_y, 2, brier="squared_mean"),
                             "Square each recording's residual, then average. The square of the "
                             "mean residual is not the Brier score."),
                            (_summary(summary_p, summary_y, 2, sort_labels=False),
                             "You sorted the scores but not the labels with them. Use one order, "
                             "from np.argsort(p), for both.")]),
            _case((np.array([.2, .8, .4]), np.array([0, 1, 1]), 2),
                  {"predicted": np.array([.3, .8]), "observed": np.array([.5, 1.]),
                   "n": np.array([2, 1]), "brier": (.04 + .04 + .36) / 3},
                  "Brier uses every recording equally, not an unweighted mean of bin errors.")],
        "cal_predict": [_case((), "same", "A positive Platt slope preserves order and ties, so "
                              "every achievable policy, and the minimum loss, is unchanged. It "
                              "changes the probabilities, not the ranking.")],
        "cal_fit": [_case((cal["resnet"], cal["af_present"]), params,
                          "Use log(p/(1 − p)) as the feature with an intercept column. The "
                          "gradient is X.T @ (y − p) and the information is "
                          "X.T @ ((p * (1 − p))[:, None] * X).",
                          diagnose=[({"intercept": probability_feature["intercept"],
                                      "slope": probability_feature["coefficients"][0]},
                                     "Your feature is the probability itself. Regressing on the "
                                     "probability is a legitimate calibration model, but it does "
                                     "not implement the logit-linear specification used here, "
                                     "which regresses on logit(score)."),
                                    ({"intercept": log_feature["intercept"],
                                      "slope": log_feature["coefficients"][0]},
                                     "Your feature is log(p). The log-odds is log(p/(1 − p)).")],
                          rtol=1e-5)],
        "cal_split": [_case((splits,), _platt_brier(splits, "calibration", "evaluation"),
                            "Fit the map on the calibration split and compute the Brier score on "
                            "the evaluation split.",
                            diagnose=[(_platt_brier(splits, "evaluation", "evaluation"),
                                       "The map is still fitted on the recordings it is judged "
                                       "on. That leaks the evaluation outcomes into the fit."),
                                      (_platt_brier(splits, "calibration", "calibration"),
                                       "That assesses the map on the recordings that fitted it. "
                                       "Assess on the evaluation split."),
                                      (_platt_brier(splits, "development", "evaluation"),
                                       "Fit the map on the calibration split; the development "
                                       "split is for choosing policies."),
                                      (calibration.brier_score(splits["evaluation"]["resnet"],
                                                               splits["evaluation"]["af_present"]),
                                       "That is the raw ResNet's Brier score: the map is no "
                                       "longer applied.")])],
        "cal_shift": [_case((np.array([.1, .4]), .1, .02),
                            calibration.recalibrate_in_the_large([.1, .4], .1, .02),
                            "Add the new log-odds of prevalence minus the old; multiplying "
                            "probabilities is not a prior shift.",
                            diagnose=[(calibration.recalibrate_in_the_large([.1, .4], .02, .1),
                                       "The shift runs the wrong way: add logit(new) − "
                                       "logit(old), not the reverse."),
                                      (np.array([.1, .4]) * .02 / .1,
                                       "Multiplying probabilities by the prevalence ratio is not "
                                       "a prior shift: shift the log-odds.")])],
        # Extension 2
        "cmp_predict": [_case((), "within_outcome", "Each NRI component uses its own "
                              "outcome-group denominator, not the whole cohort.")],
        "cmp_categorical": [
            _case((old, new, y, p), comparison.categorical_nri(old, new, y, [p]),
                  "Use >= at the cut; divide up-minus-down within AF and down-minus-up within "
                  "non-AF."),
            _case((np.array([.1, .2]), np.array([.2, .1]), np.array([True, False]), .2),
                  {"event": 1., "non_event": 1., "total": 2.},
                  "Use >= at the cut; a score equal to the threshold enters review.")],
        "cmp_continuous": [_case((old, new, y), {"continuous": comparison.continuous_nri(old, new, y),
                                                 "idi": comparison.idi(old, new, y)},
                                 "Continuous NRI counts strict movements; IDI uses their "
                                 "magnitudes. Ties count neither way.")],
        "cmp_bootstrap": [_case(
            (old, new, y, p, 200, 17), _bootstrap(old, new, y, p, 200, 17),
            "Resample the same recording indices for both models and labels, and compute new "
            "minus old NB using p_t/(1 − p_t).",
            accept=[_bootstrap(old, new, y, p, 200, 17, draw) for draw in
                    ("columns", "loop", "choice", "choice_loop")],
            diagnose=[(_unpaired_bootstrap(old, new, y, p, 200, 17),
                       "Resample the same recording indices for both models and labels: the "
                       "comparison is paired."),
                      (_bootstrap(old, new, y, p, 200, 17, quantiles=(.05, .95)),
                       "That is a 90% interval. Take the 2.5th and 97.5th percentiles for 95%."),
                      (_bootstrap(old, new, y, p, 200, 17, weight=p),
                       "Weight a false positive by the odds p_t/(1 − p_t), not by p_t.")])],
        "cmp_changed": [_case(
            (ids, old, new, y, p),
            {"recording": np.array(["a", "b", "d"]), "direction": np.array(["up", "down", "down"]),
             "contribution": np.array([1., -1., .25])},
            "Select only changes in review action, retain recording IDs, and price direction "
            "using reference AF.",
            diagnose=[(_changed(ids, old, new, y, p, any_change=True),
                       "Include only recordings whose review action changes, not every "
                       "recording whose score moved."),
                      (_changed(ids, old, new, y, p, strict=True),
                       "Use >= at the cut: a score equal to p_t enters review."),
                      (_changed(ids, old, new, y, p, weight=1.),
                       "Price a change for a recording without AF by p_t/(1 − p_t), not 1.")])],
        "cmp_choice": [
            _case((dev["random_forest"], dev["resnet"], dev["af_present"], .2),
                  nb_winner(dev["random_forest"], dev["resnet"], dev["af_present"], .2),
                  "Choose the winner by net benefit at the stated preference.",
                  diagnose=[({"winner": "resnet", "basis": "nri", "preference": .2},
                             "Positive NRI does not establish utility; compare net benefit at "
                             "the stated preference.")]),
            _case((np.array([.1, .1, .1, .1]), np.array([.3, .3, .3, .1]),
                   np.array([True, True, False, False]), .2),
                  {"winner": "resnet", "basis": "net_benefit", "preference": .2},
                  "Compute the winner from the two net benefits; a fixed answer fails on other "
                  "data.")],
        # Extension 3
        "weight_predict": [_case((), "intercept", "Positive class weights mostly shift the "
                                 "intercept in this correctly specified example.")],
        "weight_weights": [_case((np.array([1, 0, 0, 0]),), np.array([2., 2 / 3, 2 / 3, 2 / 3]),
                                 "Balanced weights give each class equal total weight: "
                                 "n/(2*n_class).",
                                 diagnose=[(np.array([3., 1., 1., 1.]),
                                            "Those weights have the right ratio but not the "
                                            "balanced scale n/(2*n_class).")])],
        "weight_fit": [_case((weight_X, weight_y, weight_w), _weighted_fit(weight_X, weight_y, weight_w),
                             "Multiply both the gradient and the information by row_weights.",
                             diagnose=[(_weighted_fit(weight_X, weight_y, np.ones(6)),
                                        "row_weights do not reach the gradient, so this is the "
                                        "unweighted fit. The gradient defines where the optimum "
                                        "is.")],
                             rtol=1e-5)],
        "weight_optimum": [_case((np.array([.1, .2, .2, .8]), np.array([0, 1, 0, 1], bool),
                                  reference.DEFAULT_COSTS), 3.,
                                 "Sweep every distinct score plus review-all and review-none, "
                                 "and charge TP reviews as well as FP.",
                                 diagnose=[(1., "Charge correct reviews (TP) as well as false "
                                                "positives.")])],
        "weight_correct": [_case((np.array([.5, .9]), 9.),
                                 logistic.sigmoid(logistic.logit([.5, .9]) - np.log(9)),
                                 "Subtract log(weight ratio) from the log-odds, then apply the "
                                 "sigmoid.",
                                 diagnose=[(np.array([.5, .9]) / 9,
                                            "Dividing the probability by the weight ratio is the "
                                            "bug. Undo the weighting on the log-odds scale."),
                                           (np.array([.5, .9]) - np.log(9),
                                            "Subtract log(weight ratio) from the log-odds, not "
                                            "from the probability."),
                                           (logistic.sigmoid(logistic.logit([.5, .9]) + np.log(9)),
                                            "Wrong direction: weighting AF up added log(w) to "
                                            "the log-odds, so subtract it.")])],
        # Extension 4
        "queue_predict": [_case((), "sometimes", "Capacity bounds a daily count; a mean below it "
                                "still allows random high-demand days.")],
        "queue_counts": [_case((.8, .9, .05, 1000), {"TP": 40., "FN": 10., "FP": 95., "TN": 855.},
                               "Apply sensitivity to AF recordings and the false-positive rate "
                               "to non-AF recordings at the new prevalence.",
                               diagnose=[({"TP": 40., "FN": 10., "FP": 100., "TN": 900.},
                                          "The false-positive rate applies to recordings without "
                                          "AF, population × (1 − prevalence), not to all "
                                          "recordings.")])],
        "queue_select": [_case((rows, 50.), .3, "Filter by expected reviews before minimising "
                               "loss; the unconstrained optimum is infeasible.")],
        "queue_simulate": [
            _case((1000, .0497, 50, 365, 19), _queue(1000, .0497, 50, 365, 19),
                  "Draw Poisson arrivals, then binomial reviews, and return the fraction of days "
                  "with reviews strictly above capacity.",
                  accept=[_queue(1000, .0497, 50, 365, 19, loop=True)],
                  diagnose=[(_queue(1000, .0497, 50, 365, 19, at_least=True),
                             "A day exactly at capacity is not overflow: use >, not >=."),
                            (_queue(1000, .0497, 50, 365, 19, loop=True, at_least=True),
                             "A day exactly at capacity is not overflow: use >, not >=.")]),
            _case((5, 1., 5, 400, 5), _queue(5, 1., 5, 400, 5),
                  "A day exactly at capacity is not overflow; use strictly greater than.",
                  accept=[_queue(5, 1., 5, 400, 5, loop=True)])],
        "queue_tolerance": [_case((rows, .1), .5, "Filter by the stated overflow tolerance, then "
                                  "minimise loss.",
                                  diagnose=[(.3, "That is the mean-feasible policy. A mean "
                                                 "within capacity is not acceptable daily "
                                                 "service: filter on overflow."),
                                            (.1, "Filter on overflow before minimising loss; "
                                                 "the cheapest policy overflows on half the "
                                                 "days.")])],
        "queue_prevalence": [_case((), capacity["highest_safe"],
                                   "Keep the Problem C cutoff fixed, project its reviews at each "
                                   "prevalence on the grid, and keep the highest prevalence whose "
                                   "overflow probability is at most 10%.",
                                   diagnose=[(.05, "At the planning prevalence the overflow "
                                                   "probability is far above 10%. Go lower."),
                                             (100 * capacity["highest_safe"],
                                              "Give the prevalence as a proportion, not a "
                                              "percentage."),
                                             (capacity["highest_safe"] + .005,
                                              "That is the first prevalence that breaks the "
                                              "tolerance. Keep the last one that meets it.")])],
        # Extension 5
        "window_predict": [_case((), "no", "One recording label cannot tell which windows contain "
                                 "AF; interval truth is needed.")],
        "window_rules": [_case((positive, 2), {"any": positive.any(axis=1),
                                               "majority": positive.sum(axis=1) > 2,
                                               "count": positive.sum(axis=1) >= 2},
                               "Reduce across windows within each recording; majority is strictly "
                               "more than half and count includes equality.")],
        "window_persistence": [
            _case((positive, 2), np.array([False, True, False, True]),
                  "Reset the run after every negative window; separated positives do not "
                  "establish persistence."),
            _case((np.array([[1, 0, 1], [1, 1, 1]], bool), 3), np.array([False, True]),
                  "Reset the run after a negative and include a run ending at the last window.")],
        "window_counts": [
            _case((review, review_af), {"TP": 2, "FP": 1, "FN": 0, "TN": 3},
                  "Count recording decisions against recording truth, one cell per recording.",
                  diagnose=[({"TP": 2, "FP": 0, "FN": 1, "TN": 3},
                             "FP and FN are swapped: FP is a review without AF; FN is AF without "
                             "review."),
                            ({"TP": 2, "FP": 1, "FN": 3, "TN": 0},
                             "FN and TN are swapped: FN is AF without review; TN is neither.")]),
            _case((np.array([True, False, True, False]), np.array([True, True, False, False])),
                  {"TP": 1, "FP": 1, "FN": 1, "TN": 1},
                  "Count recording decisions against recording truth; use logical complements.")],
        "window_sensitivity": [
            _case((calls, calls_af, windows.any_positive),
                  _recording_sensitivity(calls, calls_af, windows.any_positive),
                  "Apply the rule to get one decision per recording, then take the fraction of "
                  "AF recordings reviewed.",
                  diagnose=[(calls[calls_af].sum() / calls_af.sum(),
                             "That counts positive windows, not reviewed recordings, so it can "
                             "exceed 1. Apply the rule first."),
                            (calls[calls_af].mean(),
                             "That is a window-level rate. The unit of analysis is the "
                             "recording: apply the rule first.")]),
            _case((calls, calls_af, windows.persistence(2)),
                  _recording_sensitivity(calls, calls_af, windows.persistence(2)),
                  "Use the rule passed in, not any-positive.",
                  diagnose=[(_recording_sensitivity(calls, calls_af, windows.any_positive),
                             "Use the rule passed in; this answer applies any-positive.")])],
        "window_repeat": [
            _case(repeat_inputs, _repeat(*repeat_inputs),
                  "Pay repeat and delay once, and weight the readable and unreadable branches by "
                  "q and 1 − q.",
                  diagnose=[(_repeat(*repeat_inputs, variant="no_fallback"),
                             "You left out the unreadable branch: with probability 1 − q the "
                             "repeat fails and the fallback, review, applies."),
                            (_repeat(*repeat_inputs, variant="no_review_fallback"),
                             "The fallback here is review, priced p·C_TP + (1 − p)·C_FP."),
                            (_repeat(*repeat_inputs, variant="charged_twice"),
                             "Charge the repeat and delay once, not again inside each branch.")]),
            _case((.3, reference.DEFAULT_COSTS, .2, 0., .76, .95, .1),
                  {"review_now": 1., "no_review_now": 1.5, "repeat": 1.3, "best": "review_now"},
                  "If no repeats are readable, everyone reaches the fallback and still pays "
                  "acquisition and delay.")],
    }


# --- Extension 6: checks that read the learner's own data ---------------------
# Each returns (state, problem). state is True while the answer is unfinished,
# False once it can be judged, or UNAVAILABLE when the learner's data cannot
# support the check; problem None means correct.

UNAVAILABLE = "unavailable"
JOIN_SCORES = [{"recording": "A00001", "score": .2}, {"recording": "A00004", "score": .9},
               {"recording": "A00002", "score": .4}, {"recording": "A00002", "score": .5},
               {"recording": "Z99999", "score": .3}, {"recording": "A00003", "score": 1.4},
               {"recording": "A00005", "score": float("nan")}, {"recording": "A00006", "score": .7},
               {"recording": "A00007", "score": .1}]
JOIN_LABELS = {"A00001": "N", "A00002": "N", "A00003": "A", "A00004": "A", "A00005": "O",
               "A00006": "~", "A00007": "O"}
JOIN_EXPECTED = {"records": [{"recording": "A00001", "score": .2, "af_present": False},
                             {"recording": "A00004", "score": .9, "af_present": True},
                             {"recording": "A00007", "score": .1, "af_present": False}],
                 "duplicates": ["A00002"], "unknown": ["Z99999"],
                 "out_of_range": ["A00003", "A00005"], "noise": 1}
NO_DEVELOPMENT = ("There are no development recordings among your joined scores. Finish Problem A "
                  "(or use its stuck switch), or bring scores for recordings in the development "
                  "split.")


def _is_false(value):
    """False as a value: Python's False or numpy's False, not 0 or None."""
    return isinstance(value, (bool, np.bool_)) and not bool(value)


def _own_join(function):
    if not callable(function):
        return False, "Expected your join_scores function."
    result = function([dict(row) for row in JOIN_SCORES], dict(JOIN_LABELS))
    if _unfinished(result):
        return True, WAITING
    shape = _shape_problem(result, JOIN_EXPECTED)
    if shape:
        return False, shape
    kept = {r.get("recording"): r for r in result["records"] if isinstance(r, dict)}
    if "A00006" in kept:
        return False, ("A noise recording (~) has no binary AF reference. Exclude it and count it "
                       "in 'noise'; do not keep it as a recording without AF.")
    if "A00002" in kept:
        return False, ("A00002 appears twice with different scores. Exclude every copy of a "
                       "duplicated ID and list the ID once in 'duplicates'.")
    if "A00003" in kept or "A00005" in kept:
        return False, ("Scores must be finite and between 0 and 1. List the others in "
                       "'out_of_range'; a NaN score fails both comparisons.")
    if "A00007" in kept and not _is_false(kept["A00007"].get("af_present")):
        return False, "Only reference label A is AF. N and O are AF-negative."
    for key in ("duplicates", "unknown", "out_of_range"):
        if list(result[key]) != JOIN_EXPECTED[key]:
            return False, (f"'{key}' should be {JOIN_EXPECTED[key]}, received {list(result[key])}. "
                           "Check each row in the stated order: duplicate, unknown, out of range, "
                           "noise.")
    if result["noise"] != JOIN_EXPECTED["noise"]:
        return False, "Count the noise recordings you excluded in 'noise'."
    if not _equal(result["records"], JOIN_EXPECTED["records"]):
        return False, ("Keep the remaining rows in input order as dictionaries with 'recording', "
                       "'score' and 'af_present'.")
    return False, None


def _relation(j, loss):
    return "lower" if j < loss else "higher" if j > loss else "same"


def _own_j(answer, development):
    if answer is Ellipsis:
        return True, WAITING
    if answer not in ("lower", "higher", "same"):
        return False, "Answer 'lower', 'higher' or 'same'."
    if not development:
        return UNAVAILABLE, NO_DEVELOPMENT
    n_pos = sum(r["af_present"] for r in development)
    if n_pos in (0, len(development)):
        return UNAVAILABLE, (f"Every development recording {'has' if n_pos else 'lacks'} AF, so "
                             f"{'specificity' if n_pos else 'sensitivity'}, and therefore J, is "
                             "undefined. There is nothing to compare.")
    rows = reference.sweep_thresholds(development, reference.DEFAULT_COSTS)
    j_cuts = [r["threshold"] for r in reference.best_thresholds(rows, "youden_j", minimise=False)]
    loss_cuts = [r["threshold"] for r in reference.best_thresholds(rows)]
    # Tied optima are equally valid choices, so any relation one pair of them shows is correct.
    attainable = {_relation(j, loss) for j in j_cuts for loss in loss_cuts}
    if answer in attainable:
        return False, None
    expected = " or ".join(sorted(attainable))
    ratio = (len(development) - n_pos) / n_pos
    benefit = reference.DEFAULT_COSTS["FN"] - reference.DEFAULT_COSTS["TP"]
    return False, (f"On your development recordings J's cutoff is {expected} than the "
                   f"minimum-loss cutoff. J prices a miss at n_neg/n_pos = {ratio:.1f} false "
                   f"referrals against {benefit:g} in the matrix; your scores' calibration decides "
                   "where each cutoff falls.")


def _is_number(value):
    """A real number, Python or NumPy (np.float32(.5), np.int64(2)), but not a boolean."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _own_freeze(answer, development, evaluation):
    threshold, predicted = answer
    if threshold is Ellipsis or predicted is Ellipsis:
        return True, WAITING
    if not development:
        return UNAVAILABLE, NO_DEVELOPMENT + " A policy is frozen on development recordings."
    if not _is_number(threshold) or not 0 <= threshold <= 1.01:
        return False, "Set final_threshold to a cutoff between 0 and 1.01, chosen on development."
    if not _is_number(predicted) or not math.isfinite(predicted) or predicted < 0:
        return False, "Set predicted_evaluation_loss to one number: the total loss you expect."
    threshold, predicted = float(threshold), float(predicted)
    dev = reference.policy_summary(development, threshold, reference.DEFAULT_COSTS)
    # Only a positive total can be copied unscaled: a development total of zero scales to zero.
    if (dev["total_loss"] > 0 and math.isclose(predicted, dev["total_loss"])
            and len(evaluation) != len(development)):
        return False, (f"That is the development total over {len(development):,} recordings; "
                       f"evaluation has {len(evaluation):,}. Scale it before predicting.")
    return False, None


CUSTOM = {"own_join": _own_join, "own_j": _own_j, "own_freeze": _own_freeze}


def names():
    """Every checker name a notebook may pass to ``run``."""
    return set(cases()) | set(CUSTOM)


# --- One entry point -----------------------------------------------------------


def _verdict(name, answer, context):
    """(waiting, problem) for one answer; (False, None) means correct."""
    if callable(answer) and _placeholder_in_source(answer):
        return True, WAITING
    if name in CUSTOM:
        return CUSTOM[name](answer, *context)
    for case in cases()[name]:
        try:
            actual = answer(*case["args"]) if callable(answer) else answer
        except Exception as error:
            return False, f"{case['hint']} ({type(error).__name__}: {error})"
        if _unfinished(actual):
            return True, WAITING
        shape = _shape_problem(actual, case["expected"])
        if shape and not any(_equal(actual, valid, case["rtol"]) for valid in case["accept"]):
            return False, shape
        if any(_equal(actual, valid, case["rtol"]) for valid in (case["expected"], *case["accept"])):
            continue
        hint = next((h for wrong, h in case["diagnose"] if _equal(actual, wrong, case["rtol"])),
                    case["hint"])
        return False, hint
    return False, None


def run(name, answer, *context):
    """Check one prediction or learner function; return success for guarded follow-up cells.

    ``context`` passes the learner's own data to the Extension 6 checks.
    """
    try:
        unattempted, problem = _verdict(name, answer, context)
    except Exception as error:
        unattempted, problem = False, (f"The checker could not read this answer "
                                   f"({type(error).__name__}: {error}).")
    if unattempted == UNAVAILABLE:
        print(f"Not available: {name}. {problem}")
        return False
    if problem is None:
        where = "on your own recordings" if name in ("own_j", "own_freeze") else \
            "including a small counterexample"
        print(f"✓ {name}: checked, {where}.")
        return True
    print(f"{'Not attempted yet' if unattempted else 'Not yet'}: {name}. {problem}")
    if os.environ.get("UTILITY_STRICT") == "1":
        raise AssertionError(f"{name}: {problem}")
    return False


# --- Waiting for a learner function, and the stuck switches ---------------------


def _expected_counts(se, sp, prevalence, population=1000):
    af, non_af = population * prevalence, population * (1 - prevalence)
    return {"TP": af * se, "FN": af * (1 - se), "FP": non_af * (1 - sp), "TN": non_af * sp}


def _select_mean_feasible(rows, capacity):
    return min((row for row in rows if row["reviews"] <= capacity), key=lambda row: row["loss"])["threshold"]


def _minimum_loss(scores, y, costs):
    cuts = np.r_[0., np.unique(scores), 1.01]
    return float(min(reference.matrix_loss(comparison.counts_at_threshold(scores, y, cut), costs)
                     for cut in cuts))


def _fit_weighted(features, y, row_weights):
    model = logistic.fit_logistic_regression(features, y, row_weights=row_weights)
    return np.r_[model["intercept"], model["coefficients"]]


def _recording_counts(review, af_present):
    review, af = np.asarray(review, dtype=bool), np.asarray(af_present, dtype=bool)
    return {"TP": int(np.sum(review & af)), "FP": int(np.sum(review & ~af)),
            "FN": int(np.sum(~review & af)), "TN": int(np.sum(~review & ~af))}


# Learner functions that later cells use: (checker, problem, reference version or None).
# A Fix-the-bug function has no switch: the repair is the exercise.
FUNCTIONS = {
    "probability_summary": ("cal_summary", "Problem A", calibration.reliability_summary),
    "fit_platt": ("cal_fit", "Problem C", calibration.platt_fit),
    "shift_prior": ("cal_shift", "Problem E", calibration.recalibrate_in_the_large),
    "categorical_nri": ("cmp_categorical", "Problem B",
                        lambda old, new, y, cut: comparison.categorical_nri(old, new, y, [cut])),
    "movement_indices": ("cmp_continuous", "Problem C",
                         lambda old, new, y: {"continuous": comparison.continuous_nri(old, new, y),
                                              "idi": comparison.idi(old, new, y)}),
    "paired_bootstrap": ("cmp_bootstrap", "Problem D", comparison.delta_net_benefit_bootstrap),
    "changed_recordings": ("cmp_changed", "Problem E",
                           lambda ids, old, new, y, preference: _changed(
                               ids, old, new, np.asarray(y, dtype=bool), preference)),
    "balanced_weights": ("weight_weights", "Problem B",
                         lambda y: logistic.class_weights(y, "balanced")),
    "fit_weighted": ("weight_fit", "Problem C", _fit_weighted),
    "minimum_loss": ("weight_optimum", "Problem D", _minimum_loss),
    "undo_weighting": ("weight_correct", "Problem E", None),
    "expected_counts": ("queue_counts", "Problem B", _expected_counts),
    "select_mean_feasible": ("queue_select", "Problem C", _select_mean_feasible),
    "fraction_over_capacity": ("queue_simulate", "Problem D",
                               lambda arrival_mean, review_rate, capacity, days=365, seed=20261106:
                               _queue(arrival_mean, review_rate, capacity, days, seed)),
    "select_policy": ("queue_tolerance", "Problem E", None),
    "aggregate": ("window_rules", "Problem B",
                  lambda positive, k: {"any": windows.any_positive(positive),
                                       "majority": windows.majority(positive),
                                       "count": windows.count_at_least(k)(positive)}),
    "persistent": ("window_persistence", "Problem C",
                   lambda positive, run_length: windows.persistence(run_length)(positive)),
    "recording_counts": ("window_counts", "Problem D", _recording_counts),
    "recording_sensitivity": ("window_sensitivity", "Problem E", None),
    "repeat_losses": ("window_repeat", "Problem F",
                      lambda p, costs, repeat_cost, q, se, sp, delay_cost=0.: _repeat(
                          p, costs, repeat_cost, q, se, sp, delay_cost)),
    "join_scores": ("own_join", "Problem A", own_scores.join_scores),
}


def ready(**answers):
    """True when every named learner function passes its checker; one line per wait otherwise.

    ``check.ready(probability_summary=probability_summary)`` lets a later cell
    wait for an unfinished function, saying which problem it needs and how to
    unblock it. Nothing is printed when all pass.
    """
    pending = []
    for name, answer in answers.items():
        checker, problem, switch = FUNCTIONS[name]
        try:
            state, verdict = _verdict(checker, answer, ())
            passed = verdict is None and state is False
        except Exception:
            passed = False
        if not passed:
            pending.append(name)
            unblock = ("finish it and run its checker, or delete the # in the stuck cell below it "
                       "to use the reference version" if switch else
                       "fix it and run its checker; the worked copy shows one repair")
            print(f"Waiting for {name} ({problem}): {unblock}. {RECOVERY}")
    if pending and os.environ.get("UTILITY_STRICT") == "1":
        raise AssertionError(f"Not ready: {', '.join(pending)}")
    return not pending


def use_reference(name):
    """The stuck switch: return the reference version of a function later cells use, and say so."""
    function = FUNCTIONS.get(name, (None, None, None))[2]
    if function is None:
        print(f"There is no reference version of {name!r}. Use the worked copy in "
              "extension/solutions/ for this problem.")
        return None
    print(f"Using the reference {name} from now on. Come back to your own version later and "
          "compare it with the worked copy.")
    return function
