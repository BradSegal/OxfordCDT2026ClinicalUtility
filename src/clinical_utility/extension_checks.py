"""Feedback for the extension exercises, with counterexamples to common errors.

Each case gives the arguments, the expected answer and a default hint. A case
may also list other valid answers (``accept``) and plausible wrong answers with
their own hints (``diagnose``), so a learner hears which misconception they hit.
A returned value of the wrong kind, or a dictionary or table without the
expected keys or columns, is reported before any calculation is compared.
Answers that are fitted scikit-learn models or tables get their own checks.
Learner code is called only inside the feedback boundary. Normal notebook use
never raises; UTILITY_STRICT=1 makes worked-copy verification fail visibly.

``ready`` lets a later cell wait for a learner answer, naming the problem, how
to unblock it and how to recover (the core's ``RECOVERY``: Runtime > Run
before, then the cell itself); ``waiting`` prints the same kind of line for
anything else a cell needs. ``use_reference`` is the stuck switch that supplies
a reference version of an answer later cells use.
"""

import ast
import inspect
import math
import os
import textwrap
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import calibration, comparison, data, deployment, features, own_scores, reference, windows
from .checks import RECOVERY, show, waiting  # noqa: F401  (re-exported for the notebooks)

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
    if isinstance(expected, pd.DataFrame):
        if not isinstance(actual, pd.DataFrame) or list(actual.shape) != list(expected.shape):
            return False
        return all(_equal(actual[c].to_numpy(), expected[c].to_numpy(), rtol)
                   for c in expected.columns if c in actual.columns) and \
            set(expected.columns) <= set(actual.columns)
    if isinstance(expected, pd.Series):
        expected = expected.to_numpy()
    if isinstance(actual, pd.Series):
        actual = actual.to_numpy()
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


UNAVAILABLE = "unavailable"
COSTS = reference.DEFAULT_COSTS
UNUSUAL_COSTS = {"TP": 2.0, "FP": 3.0, "FN": 11.0, "TN": 0.5}


def _split(name):
    frame = data.load_scores()
    return frame[frame["split"] == name].reset_index(drop=True)


def _call(function, *args):
    """(result, error) of learner code, with its warnings silenced."""
    import contextlib
    import io
    import warnings
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args), None
    except Exception as error:
        return None, error


def _table_problem(actual, columns, what="table"):
    if not isinstance(actual, pd.DataFrame):
        return f"Return a pandas DataFrame; received {_kind(actual)}."
    missing = [c for c in columns if c not in actual.columns]
    if missing:
        return f"The {what} has no column {', '.join(map(repr, missing))}."
    return None


# --- Small fixed inputs -------------------------------------------------------------


@lru_cache(maxsize=1)
def _fixtures():
    development, calibration_split, evaluation = (_split(s) for s in
                                                   ("development", "calibration", "evaluation"))
    # Two AF recordings and five without, with unequal up and down moves in each group, so
    # that a wrong denominator or a swapped direction gives a different answer.
    old = np.array([.1, .2, .8, .4, .6, .1, .3])
    new = np.array([.3, .1, .9, .1, .7, .5, .1])
    y = np.array([1, 1, 0, 0, 0, 0, 0], dtype=bool)
    rhythm = features.synthetic_rhythm_features()
    parts = {s: rhythm[rhythm["split"] == s].reset_index(drop=True)
             for s in ("train", "validation", "test")}
    window_positive = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0]], bool)
    window_valid = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]], bool)
    return {"development": development, "calibration": calibration_split,
            "evaluation": evaluation, "old": old, "new": new, "y": y, "rhythm": parts,
            "window_positive": window_positive, "window_valid": window_valid}


@lru_cache(maxsize=1)
def _platt_reference():
    fitting = _fixtures()["calibration"]
    return calibration.platt().fit(fitting[["resnet"]], fitting["af"])


@lru_cache(maxsize=1)
def _isotonic_reference():
    fitting = _fixtures()["calibration"]
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(fitting["resnet"],
                                                                           fitting["af"])


def _compare_on(frame, fit_split, assess_split="development"):
    """Brier scores on ``assess_split`` of the Platt and isotonic maps, both fitted on ``fit_split``."""
    from sklearn.metrics import brier_score_loss
    fitting, assessed = frame[frame["split"] == fit_split], frame[frame["split"] == assess_split]
    platt = calibration.platt().fit(fitting[["resnet"]], fitting["af"])
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(fitting["resnet"], fitting["af"])
    return {"platt": float(brier_score_loss(assessed["af"],
                                            platt.predict_proba(assessed[["resnet"]])[:, 1])),
            "isotonic": float(brier_score_loss(assessed["af"], isotonic.predict(assessed["resnet"])))}


def _compare_reference(frame):
    return _compare_on(frame, "calibration", "development")


# --- Custom checks: (state, problem); state True means not attempted ---------------


def _fitted_probabilities(name, model, X, columns_hint):
    if model is Ellipsis:
        return None, (True, WAITING)
    method = "predict_proba" if hasattr(model, "predict_proba") else "predict"
    if not hasattr(model, method):
        return None, (False, f"Expected a fitted scikit-learn model; received {_kind(model)}.")
    got, error = _call(lambda: getattr(model, method)(X))
    if isinstance(error, NotFittedError):
        return None, (False, "The model is not fitted yet: call .fit(...) on the calibration split.")
    if error is not None:
        return None, (False, f"{method} raised {type(error).__name__}: {error}. {columns_hint}")
    got = np.asarray(got, dtype=float)
    return (got[:, 1] if got.ndim == 2 else got), None


def _cal_report(function):
    if not callable(function):
        return False, "Expected your calibration_report function."
    fx = _fixtures()
    for column in ("random_forest", "resnet"):
        y, p = fx["development"]["af"], fx["development"][column]
        got, error = _call(function, y, p)
        if error is not None:
            return False, f"It raised {type(error).__name__}: {error}."
        if _unfinished(got):
            return True, WAITING
        expected = calibration.calibration_report(y, p)
        shape = _shape_problem(got, {k: 0.0 for k in ("brier", "observed", "predicted",
                                                     "intercept", "slope")})
        if shape:
            return False, shape
        if not _equal(got["brier"], expected["brier"]):
            return False, "Use brier_score_loss(y, p): the mean squared difference over recordings."
        if _equal(got["observed"], expected["predicted"]) and _equal(got["predicted"],
                                                                      expected["observed"]):
            return False, ("calibration_curve returns (prob_true, prob_pred): the observed AF "
                           "fraction first, then the mean stated probability.")
        if not (_equal(got["observed"], expected["observed"])
                and _equal(got["predicted"], expected["predicted"])):
            return False, ("Use calibration_curve(y, p, n_bins=10, strategy='quantile') for "
                           "equal-count bins.")
        if not _equal(got["intercept"], expected["intercept"], 1e-5):
            from statsmodels.api import GLM, add_constant, families
            z = logit(np.asarray(p, dtype=float))
            free = GLM(np.asarray(y, float), add_constant(z), family=families.Binomial()).fit()
            if _equal(got["intercept"], float(np.asarray(free.params)[0]), 1e-5):
                return False, ("That intercept comes from the fit with a free slope. "
                               "Calibration-in-the-large fixes the slope at 1: fit an intercept "
                               "only, with logit(p) as an offset.")
            return False, ("Fit logit P(AF) = a + logit(p): a binomial GLM with a column of ones "
                           "and offset=logit(p).")
        if not _equal(got["slope"], expected["slope"], 1e-5):
            from statsmodels.api import GLM, families
            origin = GLM(np.asarray(y, float), logit(np.asarray(p, dtype=float))[:, None],
                         family=families.Binomial()).fit()
            if _equal(got["slope"], float(np.asarray(origin.params)[0]), 1e-5):
                return False, "The slope fit needs an intercept too: use add_constant(logit(p))."
            return False, "Fit logit P(AF) = a + b·logit(p) and report b."
    return False, None


def _cal_platt(model):
    fx = _fixtures()
    got, problem = _fitted_probabilities("platt_map", model, fx["evaluation"][["resnet"]],
                                         "Fit on a one-column DataFrame, calibration_split[['resnet']].")
    if problem:
        return problem
    expected = _platt_reference().predict_proba(fx["evaluation"][["resnet"]])[:, 1]
    if np.allclose(got, expected, atol=5e-4, rtol=0):
        return False, None
    X_cal, y_cal = fx["calibration"][["resnet"]], fx["calibration"]["af"]
    variants = [
        (LogisticRegression(solver="newton-cholesky").fit(logit(X_cal), y_cal).predict_proba(
            logit(fx["evaluation"][["resnet"]]))[:, 1],
         "LogisticRegression shrinks its coefficients by default (C=1). Use C=np.inf."),
        (LogisticRegression(C=np.inf, solver="newton-cholesky").fit(X_cal, y_cal).predict_proba(
            fx["evaluation"][["resnet"]])[:, 1],
         "This regresses on the raw score. Platt scaling uses logit(score): put "
         "FunctionTransformer(logit) before the logistic regression."),
    ]
    for split in ("development", "evaluation"):
        frame = fx[split]
        variants.append((calibration.platt().fit(frame[["resnet"]], frame["af"]).predict_proba(
            fx["evaluation"][["resnet"]])[:, 1],
            f"This map was fitted on the {split} split. Fit it on the calibration split only."))
    for i, (wrong, hint) in enumerate(variants):
        # The raw-score fit (second) is far from the answer, and solvers disagree more on it.
        if np.allclose(got, wrong, atol=2e-3 if i == 1 else 5e-4, rtol=0):
            return False, hint
    return False, "Fit logit(resnet) against AF on the calibration split, with no penalty."


def _cal_isotonic(model):
    fx = _fixtures()
    got, problem = _fitted_probabilities("isotonic_map", model, fx["evaluation"]["resnet"],
                                         "Fit and predict on the score column itself.")
    if problem:
        return problem
    if np.isnan(got).any():
        return False, ("Some evaluation scores lie outside the calibration scores' range and get "
                       "NaN. Set out_of_bounds='clip'.")
    expected = _isotonic_reference().predict(fx["evaluation"]["resnet"])
    if np.allclose(got, expected, atol=1e-9):
        return False, None
    for split in ("development", "evaluation"):
        frame = fx[split]
        wrong = IsotonicRegression(out_of_bounds="clip").fit(frame["resnet"], frame["af"]).predict(
            fx["evaluation"]["resnet"])
        if np.allclose(got, wrong, atol=1e-9):
            return False, f"This map was fitted on the {split} split. Fit it on calibration only."
    return False, "Fit IsotonicRegression(out_of_bounds='clip') on the calibration split's resnet scores."


def _cal_leak(function):
    if not callable(function):
        return False, "Expected your compare_recalibrations function."
    frame = data.load_scores()
    got, error = _call(function, frame)
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    shape = _shape_problem(got, {"platt": 0.0, "isotonic": 0.0})
    if shape:
        return False, shape
    # Brier scores are compared at 1e-4: solvers at their default tolerances differ by about 1e-6.
    if _equal(got, _compare_reference(frame), 1e-4):
        return False, None
    if _equal(got, _compare_on(frame, "evaluation", "evaluation"), 1e-4):
        return False, ("Both maps are still fitted on the evaluation recordings they are judged "
                       "on. The flexible isotonic map gains most from that, which is why it looked "
                       "better.")
    if _equal(got, _compare_on(frame, "calibration", "evaluation"), 1e-4):
        return False, ("The maps are now fitted honestly, but the choice is still made on the "
                       "evaluation split. Choose on development; evaluation only reports the "
                       "frozen choice.")
    if _equal(got, _compare_on(frame, "development", "development"), 1e-4):
        return False, "Fit the maps on the calibration split, not development."
    return False, ("Fit both maps on the calibration split, then compare their Brier scores on the "
                   "development split.")


def _cmp_bootstrap(function):
    if not callable(function):
        return False, "Expected your delta_net_benefit function."
    fx = _fixtures()
    d = fx["development"]
    y, old_review, new_review = d["af"].to_numpy(), (d["random_forest"] >= .2).to_numpy(), \
        (d["resnet"] >= .2).to_numpy()
    got, error = _call(function, y, old_review, new_review, .2)
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    expected = comparison.delta_net_benefit(y, old_review, new_review, .2)
    shape = _shape_problem(got, {"estimate": 0.0, "low": 0.0, "high": 0.0})
    if shape:
        return False, shape
    if not _equal(got["estimate"], expected["estimate"]):
        weight_p = np.mean((new_review.astype(float) - old_review) * np.where(y, 1, -.2))
        if _equal(got["estimate"], weight_p):
            return False, "Weight a false referral by the odds p_t/(1 - p_t), not by p_t."
        return False, "The estimate is the mean of the per-recording contributions, new minus old."
    width = expected["high"] - expected["low"]
    if not all(abs(got[k] - expected[k]) <= .1 * width for k in ("low", "high")):
        if got["high"] - got["low"] > 1.3 * width:
            return False, ("The interval is too wide: resample the recordings with both models' "
                           "decisions together (paired=True).")
        return False, ("Use scipy.stats.bootstrap with paired=True, method='percentile' and the "
                       "default 95% level.")
    return False, None


CHANGED_COLUMNS = ("recording", "reference_label", "direction", "contribution")


def _cmp_changed(function):
    if not callable(function):
        return False, "Expected your changed_decisions function."
    d = _fixtures()["development"]
    got, error = _call(function, d, "random_forest", "resnet", .2, .2)
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    problem = _table_problem(got, CHANGED_COLUMNS)
    if problem:
        return False, problem
    expected = comparison.changed_decisions(d, "random_forest", "resnet", .2, .2)
    if len(got) != len(expected):
        moved = int((d["resnet"] != d["random_forest"]).sum())
        if len(got) == moved:
            return False, ("Keep only recordings whose review decision changes, not every recording "
                           "whose score moved.")
        return False, (f"Expected {len(expected)} changed recordings, received {len(got)}. Review "
                       "when the score is at or above the cut (>=).")
    got = got.reset_index(drop=True)
    if list(got["recording"]) != list(expected["recording"]):
        return False, "Keep the recordings in input order."
    if list(got["direction"]) != list(expected["direction"]):
        return False, "'up' means the new model reviews a recording that the old one did not."
    for column in ("random_forest", "resnet"):
        if column not in got.columns or not _equal(got[column], expected[column]):
            return False, f"Keep the score column {column!r}, unchanged, for each changed recording."
    if list(got["reference_label"]) != list(expected["reference_label"]):
        return False, "Keep each recording's reference label (A, N or O) from the frame."
    if not _equal(got["contribution"], expected["contribution"]):
        return False, ("An AF recording that moves up contributes +1, and one without AF that "
                       "moves up contributes -p_t/(1 - p_t); moving down reverses the sign.")
    return False, None


def _cmp_preference(function):
    if not callable(function):
        return False, "Expected your compare_net_benefit function."
    d = _fixtures()["development"]
    cutoffs = {m: float(reference.cheapest(reference.operating_points(d["af"], d[m], COSTS))["threshold"])
               for m in ("random_forest", "resnet")}
    got, error = _call(function, d, cutoffs, .2)
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    expected = comparison.compare_net_benefit(d, cutoffs, .2)
    shape = _shape_problem(got, {k: 0.0 for k in expected})
    if shape:
        return False, shape
    if _equal(got, expected):
        return False, None
    return False, ("Value every model's decisions at the service's preference p_t. Each cutoff "
                   "decides who is reviewed; p_t alone sets the weight of a false referral.")


def _weight_fit(function):
    if not callable(function):
        return False, "Expected your fit_models function."
    train = _fixtures()["rhythm"]["train"]
    got, error = _call(function, train)
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    shape = _shape_problem(got, {"unweighted": 0.0, "weighted": 0.0})
    if shape and not isinstance(got, dict):
        return False, shape
    if not isinstance(got, dict) or set(got) < {"unweighted", "weighted"}:
        return False, "Return a dictionary with the keys 'unweighted' and 'weighted'."
    expected = features.fit_models(train)
    X = train[list(features.FEATURES)]
    for key in ("unweighted", "weighted"):
        model = got[key]
        if not hasattr(model, "predict_proba"):
            return False, f"'{key}' should be a fitted LogisticRegression."
        probability, error = _call(lambda m=model: m.predict_proba(X)[:, 1])
        if error is not None:
            return False, f"'{key}' could not predict: {type(error).__name__}: {error}."
        if not np.allclose(probability, expected[key].predict_proba(X)[:, 1], atol=1e-3):
            weights = "balanced" if key == "weighted" else None
            penalised = LogisticRegression(class_weight=weights, solver="newton-cholesky").fit(X, train["af"])
            if np.allclose(probability, penalised.predict_proba(X)[:, 1], atol=1e-3):
                return False, ("This fit keeps scikit-learn's default L2 penalty (C=1). Class weights "
                               "rescale the log-likelihood, so they would also change how hard "
                               "the coefficients are shrunk. Switch the penalty off with "
                               "C=np.inf.")
            if key == "weighted":
                return False, ("Give the weighted model class_weight='balanced', so both classes "
                               "carry the same total weight.")
            return False, ("Fit the unweighted model with LogisticRegression(C=np.inf, "
                           "solver=\"newton-cholesky\") on the training rows.")
    return False, None


REPORT_TABLE = ("cutoff", "reviews", "af_detected", "af_missed", "loss")


def _report_problem(got, models, parts):
    """None when each cutoff attains the minimum validation loss and its row reports its test results.

    Always semantic: the decisions the submitted cutoff makes are computed on the
    validation rows (they must attain the minimum loss there) and on the test
    rows (the reported columns must be what they give). A cutoff that merely
    looks close to the reference is not enough; an equivalent or tied one is.
    """
    X = list(features.FEATURES)
    cutoffs = {}
    for name in models:
        try:
            cutoffs[name] = float(got.loc[name, "cutoff"])
        except (TypeError, ValueError, KeyError):
            return f"The cutoff for {name} should be one number."
    for name, model in models.items():
        cutoff = cutoffs[name]
        validation_p = model.predict_proba(parts["validation"][X])[:, 1]
        best = reference.cheapest(reference.operating_points(parts["validation"]["af"], validation_p,
                                                             COSTS))["loss"]
        attained = reference.policy_cost(parts["validation"]["af"], validation_p >= cutoff, COSTS)
        if not math.isclose(attained, best, rel_tol=1e-9):
            if all(math.isclose(c, .5) for c in cutoffs.values()):
                return ("predict() still applies its default boundary of 0.5 to both models. "
                        "Threshold predict_proba(X)[:, 1] at each model's cutoff, chosen on "
                        "validation.")
            return (f"On the validation rows the cutoff given for {name} has loss {attained:g}, not "
                    f"the minimum {best:g}. Choose each model's cutoff on validation by loss, and use "
                    "the threshold as it is, not rounded.")
        review = model.predict_proba(parts["test"][X])[:, 1] >= cutoff
        c, n = reference.counts(parts["test"]["af"], review), len(parts["test"])
        actual = {"reviews": 1000 * (c["TP"] + c["FP"]) / n, "af_detected": 1000 * c["TP"] / n,
                  "af_missed": 1000 * c["FN"] / n,
                  "loss": reference.policy_cost(parts["test"]["af"], review, COSTS)}
        for key, value in actual.items():
            reported = pd.to_numeric(pd.Series([got.loc[name, key]]), errors="coerce").iloc[0]
            if not _equal(reported, value):
                return (f"Column {key!r} differs for {name}: on the test rows that cutoff gives "
                        f"{value:g}. Report its consequences on the test rows per 1,000.")
    return None


def _weight_report(function):
    if not callable(function):
        return False, "Expected your test_report function."
    parts = _fixtures()["rhythm"]
    models = features.fit_models(parts["train"])
    got, error = _call(function, models, parts["validation"], parts["test"], COSTS)
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    problem = _table_problem(got, REPORT_TABLE)
    if problem:
        return False, problem
    try:
        got = got.loc[list(models)]
    except KeyError:
        return False, "Index the table by model name, as the original does."
    return False, _report_problem(got, models, parts)


def _weight_tuned(model):
    """A fitted TunedThresholdClassifierCV whose objective is the priced loss."""
    if model is Ellipsis:
        return True, WAITING
    if not hasattr(model, "best_threshold_") or not hasattr(model, "scoring"):
        return False, ("Expected a fitted TunedThresholdClassifierCV; it has best_threshold_ after "
                       ".fit(...).")
    train = _fixtures()["rhythm"]["train"]
    X, y = train[list(features.FEATURES)], train["af"]
    predicted, error = _call(model.predict, X)
    if error is not None:
        return False, f"predict raised {type(error).__name__}: {error}. Fit it on the training rows."
    from sklearn.metrics import get_scorer
    scorer = get_scorer(model.scoring) if isinstance(model.scoring, str) else model.scoring
    if scorer is None:
        return False, ("Give the tuner a scoring argument built from policy_cost; without one it "
                       "maximises balanced accuracy, which ignores the prices.")
    score, error = _call(scorer, model, X, y)
    if error is not None:
        return False, f"Its scorer could not be applied: {type(error).__name__}: {error}."
    loss = reference.policy_cost(y, np.asarray(predicted, dtype=bool), COSTS)
    if math.isclose(score, loss, rel_tol=1e-9):
        return False, ("The tuner maximised the loss. Build the scorer with "
                       "make_scorer(policy_cost, greater_is_better=False, costs=costs).")
    if not math.isclose(score, -loss, rel_tol=1e-9):
        return False, ("Its objective is not the priced loss: score with make_scorer(policy_cost, "
                       "greater_is_better=False, costs=costs).")
    fixed = reference.policy_cost(y, model.estimator_.predict_proba(X)[:, 1] >= .5, COSTS)
    if loss > fixed:
        return False, ("The tuned threshold costs more on the training rows than 0.5 does, so the "
                       "tuning did not minimise the loss. Check the scorer and the estimator.")
    return False, None


def _own_join(function):
    if not callable(function):
        return False, "Expected your join_scores function."
    got, error = _call(function, JOIN_SCORES.copy(), JOIN_LABELS.copy())
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    shape = _shape_problem(got, {"records": 0.0, "duplicates": 0.0, "unknown": 0.0,
                                 "out_of_range": 0.0, "noise": 0.0})
    if shape and not isinstance(got, dict):
        return False, shape
    keys = ("records", "duplicates", "unknown", "out_of_range", "noise")
    missing = [k for k in keys if not isinstance(got, dict) or k not in got]
    if missing:
        return False, f"The result has no {', '.join(map(repr, missing))}. Use the keys {keys}."
    records = got["records"]
    problem = _table_problem(records, ("recording", "score", "af"), "'records' table")
    if problem:
        return False, problem
    kept = dict(zip(records["recording"], records["af"]))
    if "A00006" in kept:
        return False, ("A noise recording (~) has no binary AF reference. Exclude it and count it "
                       "in 'noise'; do not keep it as a recording without AF.")
    if "A00002" in kept:
        return False, ("A00002 appears twice with different scores. Exclude every copy of a "
                       "duplicated ID and list the ID once in 'duplicates'.")
    if "A00003" in kept or "A00005" in kept:
        return False, ("Scores must be between 0 and 1. List the others in 'out_of_range'; a NaN "
                       "score is not between 0 and 1 either.")
    if "A00007" in kept and bool(kept["A00007"]):
        return False, "Only reference label A is AF. N and O are AF-negative."
    expected = own_scores.join_scores(JOIN_SCORES, JOIN_LABELS)
    for key in ("duplicates", "unknown", "out_of_range"):
        if list(got[key]) != expected[key]:
            return False, (f"'{key}' should be {expected[key]}, received {list(got[key])}. Check "
                           "each row in the stated order: duplicate, unknown, out of range, noise.")
    if got["noise"] != expected["noise"]:
        return False, "Count the noise recordings you excluded in 'noise'."
    got_records = records.reset_index(drop=True)
    if (list(got_records["recording"]) != list(expected["records"]["recording"])
            or not _equal(got_records["score"], expected["records"]["score"])
            or list(map(bool, got_records["af"])) != list(expected["records"]["af"])):
        return False, "Keep the remaining rows in input order with recording, score and af."
    return False, None


NO_DEVELOPMENT = ("There are no development recordings among your joined scores. Finish Problem A "
                  "(or use its stuck switch), or bring scores for recordings in the development "
                  "split.")


def _own_freeze(frozen, development):
    """Both cutoffs frozen on development, with supported prices; evaluation is never read.

    ``stated`` must be the threshold the prices imply, and ``optimum`` must make
    the development minimum-loss decisions (any equivalent or tied cutoff).
    """
    if _unfinished(frozen):
        return True, WAITING
    keys = {"stated", "optimum", "costs", "p_t"}
    if not isinstance(frozen, dict) or not keys <= set(frozen):
        return False, ("Set frozen to a dictionary with the keys 'stated', 'optimum', 'costs' and "
                       "'p_t'.")
    if development is None or len(development) == 0:
        return UNAVAILABLE, NO_DEVELOPMENT + " A policy is frozen on development recordings."
    costs = frozen["costs"]
    if not (isinstance(costs, dict) and set(costs) == {"TP", "FP", "FN", "TN"}
            and all(_is_number(v) and math.isfinite(v) for v in costs.values())):
        return False, "Set costs to the four finite prices, for example dict(costs)."
    try:
        implied = reference.threshold_probability(costs)
    except ValueError:
        return False, ("These prices imply no threshold probability. Choose prices with "
                       "C_FP > C_TN and C_FN > C_TP.")
    p_t = frozen["p_t"]
    if not (_is_number(p_t) and math.isfinite(p_t) and 0 < p_t < 1
            and math.isclose(p_t, implied, rel_tol=1e-6)):
        return False, f"p_t should follow from the prices: {implied:.4f}."
    for key in ("stated", "optimum"):
        if not _is_number(frozen[key]) or not (0 <= frozen[key] <= 1 or math.isinf(frozen[key])):
            return False, f"Set {key} to one cutoff: a number between 0 and 1, or infinity."
    if not math.isclose(frozen["stated"], implied, rel_tol=1e-6):
        return False, (f"stated is the threshold the prices imply, {implied:.4f}; the development "
                       "optimum goes in optimum.")
    y = development["af"].to_numpy(dtype=bool)
    scores = development["score"].to_numpy(dtype=float)
    best = reference.cheapest(reference.operating_points(y, scores, costs))
    attained = reference.policy_cost(y, scores >= frozen["optimum"], costs)
    if not math.isclose(attained, best["loss"], rel_tol=1e-12, abs_tol=1e-12):
        return False, ("optimum should be the development minimum-loss cutoff, best['threshold'] "
                       "(or any cutoff that makes the same decisions).")
    return False, None


def _is_number(value):
    """A real number, Python or NumPy, but not a boolean."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_))


JOIN_SCORES = pd.DataFrame({
    "recording": ["A00001", "A00004", "A00002", "Z99999", "A00002", "A00006", "A00007", "A00003",
                  "A00005", "A00008"],
    "score": [.2, .9, .5, .4, .6, .3, .7, 1.4, np.nan, .1]})
JOIN_LABELS = pd.DataFrame({
    "recording": ["A00001", "A00002", "A00003", "A00004", "A00005", "A00006", "A00007", "A00008"],
    "reference_label": ["N", "N", "A", "A", "O", "~", "O", "N"]})


CUSTOM = {"cal_report": _cal_report, "cal_platt": _cal_platt, "cal_isotonic": _cal_isotonic,
          "cal_leak": _cal_leak, "cmp_bootstrap": _cmp_bootstrap, "cmp_changed": _cmp_changed,
          "cmp_preference": _cmp_preference, "weight_fit": _weight_fit,
          "weight_report": _weight_report, "weight_tuned": _weight_tuned,
          "own_join": _own_join, "own_freeze": _own_freeze}


# --- Table-driven cases for functions with small inputs ----------------------------


# Two false positives and one miss, so a swap of FP and FN changes the answer.
TABLE_DECISIONS = {"any": np.array([True, True, False, True, True]),
                   "none": np.zeros(5, dtype=bool)}
TABLE_AF = np.array([True, True, True, False, False])
TABLE_KIND = np.array(["sustained", "paroxysmal", "paroxysmal", "artefact", "no_af"])


def _swapped_rule_table():
    table = windows.rule_table(TABLE_DECISIONS, TABLE_AF, TABLE_KIND, COSTS, 1000)
    from sklearn.metrics import confusion_matrix
    for name, review in TABLE_DECISIONS.items():
        tn, fp, fn, tp = confusion_matrix(review, TABLE_AF, labels=[False, True]).ravel()
        table.loc[name, "af_missed"] = 1000 * fn / len(TABLE_AF)
        table.loc[name, "reviews"] = 1000 * (tp + fp) / len(TABLE_AF)
        table.loc[name, "af_detected"] = 1000 * tp / len(TABLE_AF)
        table.loc[name, "loss"] = (tp * COSTS["TP"] + fp * COSTS["FP"] + fn * COSTS["FN"]
                                   + tn * COSTS["TN"])
    return table


UNREADABLE = {"over_read_success": .5, "repeat_readable": .6, "repeat_cost": .2, "delay_cost": .1,
              "refer_cost": 3.0, "sensitivity": .76, "specificity": .95}


def _any_rule(positive):
    return np.asarray(positive, dtype=bool).any(axis=1)


def _af_denominator_nri(old, new, y, cut):
    before, after = old >= cut, new >= cut
    up, down, n_af = after & ~before, before & ~after, y.sum()
    event = (up & y).sum() / n_af - (down & y).sum() / n_af
    non_event = (down & ~y).sum() / n_af - (up & ~y).sum() / n_af
    return {"event": event, "non_event": non_event, "total": event + non_event}


def _whole_cohort_nri(old, new, y, cut):
    before, after = old >= cut, new >= cut
    up, down, n = after & ~before, before & ~after, len(y)
    event = (up & y).sum() / n - (down & y).sum() / n
    non_event = (down & ~y).sum() / n - (up & ~y).sum() / n
    return {"event": event, "non_event": non_event, "total": event + non_event}


def _strict_nri(old, new, y, cut):
    return comparison.categorical_nri(old - 1e-12, new - 1e-12, y, cut)


def _queue_capacity(function):
    """The cheapest row within the overflow tolerance, returned with its overflow probability."""
    from scipy import stats
    if not callable(function):
        return False, "Expected your capacity_policy function."
    rows = pd.DataFrame({"threshold": [.1, .3, .5, .8], "reviews": [70., 49., 35., 10.],
                         "loss": [80., 90., 110., 170.]})
    got, error = _call(function, rows, 50, .1)
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    if not isinstance(got, pd.Series) or not {"threshold", "reviews", "loss"} <= set(got.index):
        return False, "Return the chosen row of the table (a pandas Series), not only its cutoff."
    if math.isclose(float(got["threshold"]), .3):
        return False, ("Mean reviews within capacity is not daily service: a mean of 49 against "
                       "a capacity of 50 overflows on about 45% of days. Filter on "
                       "poisson.sf(capacity, reviews) <= tolerance.")
    if math.isclose(float(got["threshold"]), .1):
        return False, "Filter by the overflow tolerance before choosing the cheapest row."
    if not math.isclose(float(got["threshold"]), .5):
        return False, ("Add overflow = poisson.sf(capacity, reviews), keep the rows within the "
                       "tolerance and return the cheapest.")
    boundary = pd.DataFrame({"threshold": [.2, .4, .6], "reviews": [45.0, 41.3, 35.0],
                             "loss": [80., 90., 110.]})
    chosen, error = _call(function, boundary, 50, .1)
    if error is None and isinstance(chosen, pd.Series) and "threshold" in chosen.index \
            and math.isclose(float(chosen["threshold"]), .6):
        return False, ("Overflow means more than capacity, P(X > c) = poisson.sf(capacity, "
                       "mean). poisson.sf(capacity - 1, mean) is P(X >= c), which counts a "
                       "day exactly at capacity as overflow and rejects a feasible policy.")
    if not (error is None and isinstance(chosen, pd.Series) and "threshold" in chosen.index
            and math.isclose(float(chosen["threshold"]), .4)):
        return False, "Keep the cheapest row whose overflow probability is at most the tolerance."
    if "overflow" not in got.index:
        return False, ("The chosen row has no 'overflow' value. Add the column with "
                       "poisson.sf(capacity, reviews) before filtering, so the row carries it.")
    if not math.isclose(float(got["overflow"]), stats.poisson.sf(50, float(got["reviews"])),
                        rel_tol=1e-6, abs_tol=1e-12):
        return False, "The row's overflow should be poisson.sf(capacity, reviews) for that row."
    return False, None


def _bad_project(points, prevalence, costs, per=1000):
    """Applies the false-positive rate to every recording, not only those without AF."""
    table = deployment.project(points, prevalence, costs, per)
    table["FP"] = per * (1 - points["specificity"])
    table["reviews"] = table["TP"] + table["FP"]
    return table


@lru_cache(maxsize=1)
def cases():
    """Small test inputs; full-cohort outputs are checked by the scenario tests."""
    fx = _fixtures()
    old, new, y = fx["old"], fx["new"], fx["y"]
    toy_points = pd.DataFrame({"threshold": [np.inf, .8, .5, .2], "sensitivity": [0, .5, .8, 1.],
                               "specificity": [1., .95, .9, .6]})
    positive, valid = fx["window_positive"], fx["window_valid"]
    calls_af = np.array([True, True, True, False])
    return {
        # Extension 1
        "cal_shift": [_case((np.array([.1, .4]), .1, .02),
                            calibration.shift_prior(np.array([.1, .4]), .1, .02),
                            "Add logit(new) - logit(old) to logit(p), then apply the sigmoid.",
                            diagnose=[(np.array([.1, .4]) * .02 / .1,
                                       "Multiplying probabilities by the prevalence ratio is not "
                                       "Bayes' rule; shift the log-odds instead."),
                                      (calibration.shift_prior(np.array([.1, .4]), .02, .1),
                                       "The shift runs the wrong way: add logit(new) and subtract "
                                       "logit(old).")])],
        # Extension 2
        "cmp_nri": [
            _case((old, new, y, .2), comparison.categorical_nri(old, new, y, .2),
                  "Each NRI component is a proportion within its own outcome group, and a score "
                  "equal to the cut is reviewed.",
                  diagnose=[(_af_denominator_nri(old, new, y, .2),
                             "Each component divides by its own outcome group: the recordings "
                             "without AF for the non-event component, not the AF recordings."),
                            (_whole_cohort_nri(old, new, y, .2),
                             "Divide each component by its own outcome group, not the whole "
                             "cohort."),
                            (_strict_nri(old, new, y, .2),
                             "A score equal to the cut is reviewed: use >=, not >.")])],
        # Extension 4
        "queue_project": [
            _case((toy_points, .05, COSTS, 1000), deployment.project(toy_points, .05, COSTS, 1000),
                  "Apply the sensitivity to the AF group (per × prevalence) and the false-positive "
                  "rate to the rest; keep fractional counts and price all four.",
                  diagnose=[(_bad_project(toy_points, .05, COSTS, 1000),
                             "The false-positive rate applies only to the recordings without AF, "
                             "per × (1 - prevalence), not to every recording.")])],
        # Extension 5
        "window_rules": [_case((positive, valid, 2), windows.aggregate(positive, valid, 2),
                               "Reduce over each recording's windows: any, more than half of its "
                               "own windows, and at least k positive.",
                               diagnose=[({**windows.aggregate(positive, valid, 2),
                                           "majority": positive.sum(axis=1) >= valid.sum(axis=1) / 2},
                                          "Majority means strictly more than half."),
                                         ({**windows.aggregate(positive, valid, 2),
                                           "majority": positive.sum(axis=1) > positive.shape[1] / 2},
                                          "Count only the recording's own windows: recordings "
                                          "differ in length, so use valid.sum(axis=1).")])],
        "window_persistence": [
            _case((positive, 2), windows.persistent(positive, 2),
                  "A run is consecutive positive windows: sliding_window_view(positive, k, axis=1)"
                  ".all(axis=-1).any(axis=1).",
                  diagnose=[(positive.sum(axis=1) >= 2,
                             "That counts positive windows anywhere; separated spikes are not a "
                             "run.")])],
        "window_sensitivity": [
            _case((positive, calls_af, _any_rule),
                  windows.recording_sensitivity(positive, calls_af, _any_rule),
                  "Apply the rule to get one decision per recording, then recall_score against "
                  "the recording labels.",
                  diagnose=[(positive[calls_af].sum() / calls_af.sum(),
                             "That counts positive windows, not recordings. Apply the rule first.")])],
        "weight_correct": [
            _case((np.array([.5, .9]), 9.), features.undo_weighting(np.array([.5, .9]), 9.),
                  "Subtract log(w) on the log-odds scale, then apply the sigmoid.",
                  diagnose=[(np.array([.5, .9]) / 9., "Dividing a probability by the weight is not "
                                                      "the inverse of weighting the odds."),
                            (features.undo_weighting(np.array([.5, .9]), 1 / 9.),
                             "The correction runs the wrong way: subtract log(w), do not add it.")])],
        "window_unreadable": [
            _case((.3, COSTS, UNREADABLE), deployment.unreadable_losses(.3, COSTS, UNREADABLE),
                  "Price all four pathways. The repeat pays its costs once, then weights the "
                  "readable branch by repeat_readable and the over-read fallback by the rest; an "
                  "unresolved over-read leaves AF undetected after using a review.",
                  diagnose=[({**deployment.unreadable_losses(.3, COSTS, UNREADABLE),
                              "over_read": .3 * COSTS["TP"] + .7 * COSTS["FP"]},
                             "An over-read is not an ordinary review: it is conclusive only with "
                             "probability over_read_success, and an unresolved AF recording is "
                             "still undetected.")])],
    }


def names():
    """Every checker name a notebook may pass to ``run``."""
    return set(cases()) | set(CUSTOM) | {"queue_prevalence"}


def _queue_prevalence(answer):
    if answer is Ellipsis:
        return True, WAITING
    fx = _fixtures()
    d = fx["development"]
    points = reference.operating_points(d["af"], d["random_forest"], COSTS)
    chosen = deployment.capacity_policy(deployment.project(points, .05, COSTS), 50, .1)
    grid = [k / 200 for k in range(2, 21)]
    expected = deployment.highest_safe_prevalence(points, chosen["threshold"], COSTS, 50, .1, grid)
    if _is_number(answer) and math.isclose(answer, expected):
        return False, None
    if _is_number(answer) and math.isclose(answer, 100 * expected):
        return False, "Store the prevalence as a proportion, not a percentage."
    if _is_number(answer) and math.isclose(answer, expected + .005):
        return False, "That is the first prevalence that breaks the tolerance; report the last safe one."
    return False, ("Hold the cutoff fixed, project it at each prevalence on the grid, and keep the "
                   "highest prevalence whose overflow probability is at most the tolerance.")


CUSTOM["queue_prevalence"] = _queue_prevalence
CUSTOM["queue_capacity"] = _queue_capacity


def _window_table(function):
    """One row per rule, labelled by the rule's name, compared rule by rule."""
    if not callable(function):
        return False, "Expected your rule_table function."
    got, error = _call(function, TABLE_DECISIONS, TABLE_AF, TABLE_KIND, COSTS, 1000)
    if error is not None:
        return False, f"It raised {type(error).__name__}: {error}."
    if _unfinished(got):
        return True, WAITING
    expected = windows.rule_table(TABLE_DECISIONS, TABLE_AF, TABLE_KIND, COSTS, 1000)
    problem = _table_problem(got, list(expected.columns))
    if problem:
        return False, problem
    labels = list(got.index)
    if len(set(labels)) != len(labels) or set(labels) != set(expected.index):
        return False, ("Index the table by rule name, one row per rule in decisions "
                       f"({', '.join(map(repr, expected.index))}).")
    got = got.loc[list(expected.index)]
    if _equal(got[list(expected.columns)], expected):
        return False, None
    if _equal(got[list(expected.columns)], _swapped_rule_table()):
        return False, ("FP and FN are swapped: pass the reference labels first, "
                       "confusion_matrix(af_present, review, ...).")
    return False, ("One row per rule, counts per 1,000 recordings from confusion_matrix, and the "
                   "loss over all four outcomes.")


CUSTOM["window_table"] = _window_table


# --- One entry point ----------------------------------------------------------------


def _verdict(name, answer, context):
    """(waiting, problem) for one answer; (False, None) means correct."""
    if callable(answer) and not hasattr(answer, "predict") and _placeholder_in_source(answer):
        return True, WAITING
    if name in CUSTOM:
        return CUSTOM[name](answer, *context)
    for case in cases()[name]:
        if callable(answer):
            actual, error = _call(answer, *case["args"])
            if error is not None:
                return False, f"{case['hint']} ({type(error).__name__}: {error})"
        else:
            actual = answer
        if _unfinished(actual):
            return True, WAITING
        expected = case["expected"]
        if isinstance(expected, pd.DataFrame):
            shape = _table_problem(actual, list(expected.columns))
        else:
            shape = _shape_problem(actual, expected)
        if shape and not any(_equal(actual, valid, case["rtol"]) for valid in case["accept"]):
            return False, shape
        if any(_equal(actual, valid, case["rtol"]) for valid in (expected, *case["accept"])):
            continue
        hint = next((h for wrong, h in case["diagnose"] if _equal(actual, wrong, case["rtol"])),
                    case["hint"])
        return False, hint
    return False, None


def run(name, answer, *context):
    """Check one learner answer; return success for guarded follow-up cells.

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
        where = "on your own recordings" if name == "own_freeze" else "on the checker's cases"
        print(f"✓ {name}: checked {where}.")
        return True
    print(f"{'Not attempted yet' if unattempted else 'Not yet'}: {name}. {problem}")
    if os.environ.get("UTILITY_STRICT") == "1":
        raise AssertionError(f"{name}: {problem}")
    return False


# --- Waiting for a learner answer, and the stuck switches ---------------------------


def _fit_models_reference(train, features_=features.FEATURES):
    return features.fit_models(train, features_)


# Learner answers that later cells use: (checker, problem, reference or None).
# A Fix-the-bug function has no switch: the repair is the exercise. A fitted model's
# reference is a function of no arguments that fits it.
FUNCTIONS = {
    "calibration_report": ("cal_report", "Problem A", calibration.calibration_report),
    "platt_map": ("cal_platt", "Problem B", _platt_reference),
    "isotonic_map": ("cal_isotonic", "Problem C", _isotonic_reference),
    "compare_recalibrations": ("cal_leak", "Problem D", None),
    "shift_prior": ("cal_shift", "Problem E", calibration.shift_prior),
    "categorical_nri": ("cmp_nri", "Problem A", comparison.categorical_nri),
    "delta_net_benefit": ("cmp_bootstrap", "Problem B", comparison.delta_net_benefit),
    "changed_decisions": ("cmp_changed", "Problem C", comparison.changed_decisions),
    "compare_net_benefit": ("cmp_preference", "Problem D", None),
    "fit_models": ("weight_fit", "Problem A", _fit_models_reference),
    "test_report": ("weight_report", "Problem B", None),
    "tuned": ("weight_tuned", "Problem C",
              lambda: features.tuned_threshold(_fixtures()["rhythm"]["train"], COSTS)),
    "undo_weighting": ("weight_correct", "Problem D", features.undo_weighting),
    "project": ("queue_project", "Problem A", deployment.project),
    "capacity_policy": ("queue_capacity", "Problem B", deployment.capacity_policy),
    "aggregate": ("window_rules", "Problem A", windows.aggregate),
    "persistent": ("window_persistence", "Problem B", windows.persistent),
    "rule_table": ("window_table", "Problem C", windows.rule_table),
    "recording_sensitivity": ("window_sensitivity", "Problem D", None),
    "unreadable_losses": ("window_unreadable", "Problem E", deployment.unreadable_losses),
    "join_scores": ("own_join", "Problem A", own_scores.join_scores),
}
FITTED = {"platt_map", "isotonic_map", "tuned"}


def ready(**answers):
    """True when every named learner answer passes its checker; one line per wait otherwise.

    ``check.ready(platt_map=platt_map)`` lets a later cell wait for an
    unfinished answer, saying which problem it needs and how to unblock it.
    Nothing is printed when all pass.
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
    """The stuck switch: return the reference version of an answer later cells use, and say so."""
    function = FUNCTIONS.get(name, (None, None, None))[2]
    if function is None:
        print(f"There is no reference version of {name!r}. Use the worked copy in "
              "extension/solutions/ for this problem.")
        return None
    print(f"Using the reference {name} from now on. Come back to your own version later and "
          "compare it with the worked copy.")
    return function() if name in FITTED else function
