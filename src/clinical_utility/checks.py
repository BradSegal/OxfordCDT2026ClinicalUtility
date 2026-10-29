"""Specific, non-blocking feedback for the learner's calculations in the core notebook.

Every checker prints one verdict and returns a ``CheckResult`` that is truthy
when the answer is right. ``✓`` means correct, ``Not yet`` names the likely
mistake and ``Not attempted yet`` means the ``...`` placeholder is still there.
A checker never raises in a student notebook. With ``UTILITY_STRICT=1`` (set
when the worked notebooks are executed automatically) a failure raises
``AssertionError`` instead, so an incorrect worked answer cannot go unnoticed.

Expected answers come from ``reference``; mistakes are recognised by running
the common wrong versions of each calculation on the same inputs.
"""

import contextlib
import functools
import inspect
import io
import math
import numbers
import os
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from . import calibration, data, reference

MAX_PROBLEMS_SHOWN = 3
# No two cells share a price and none equals its default, so each mistake is identifiable.
UNUSUAL_COSTS = {"TP": 2.0, "FP": 3.0, "FN": 11.0, "TN": 0.5}
DEFAULT_COSTS = reference.DEFAULT_COSTS


class CheckResult:
    """Pass or fail, without a stray True/False echoed under the feedback."""

    __slots__ = ("passed",)

    def __init__(self, passed):
        self.passed = bool(passed)

    def __bool__(self):
        return self.passed

    def __eq__(self, other):
        if isinstance(other, (bool, CheckResult)):
            return self.passed == bool(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.passed)

    def __repr__(self):
        return repr(self.passed)

    def _ipython_display_(self):
        pass


def _strict():
    return os.environ.get("UTILITY_STRICT") == "1" and not _QUIET["active"]


def _report(name, problems, waiting=False, success="correct"):
    if not problems:
        print(f"✓ {name}: {success}.")
        return CheckResult(True)
    if waiting:
        print(f"Not attempted yet: {name}. {problems[0]}")
    else:
        print(f"Not yet: {name}.")
        for problem in problems[:MAX_PROBLEMS_SHOWN]:
            print(f"  {problem}")
    if _strict():
        raise AssertionError(f"{name}: {'; '.join(problems)}")
    return CheckResult(False)


def _waiting(name):
    return _report(name, ["Replace the ... and run this cell again."], waiting=True)


def _not_function(name, function):
    return _report(name, [f"Expected a function, received {type(function).__name__}."])


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _near(actual, expected, tolerance=1e-6):
    if not _is_number(actual) or not _is_number(expected):
        return False
    try:
        return math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance)
    except (OverflowError, ValueError):
        return False


def _matches(actual, expected):
    if isinstance(expected, bool):
        return actual is expected or (type(actual).__name__ == "bool_" and bool(actual) == expected)
    return _near(actual, expected)


def _value(name, actual, expected, hints=(), default_hint=""):
    """Check one number; ``hints`` are (wrong value, specific hint) pairs tried in order."""
    if actual is Ellipsis:
        return _waiting(name)
    if _matches(actual, expected):
        return _report(name, [])
    for wrong, hint in hints:
        if _matches(actual, wrong):
            return _report(name, [f"Received {actual!r}. {hint}"])
    return _report(name, [f"Received {actual!r}. {default_hint}".strip()])


# --- Shared inputs and helpers ------------------------------------------------


def _scores():
    return data.load_scores()


def _split(name):
    frame = _scores()
    return frame[frame["split"] == name].reset_index(drop=True)


def _six():
    return data.six_recordings()


def _unfinished(value):
    """True for the starter's ``...``, or a structure that still holds one."""
    if value is Ellipsis:
        return True
    if isinstance(value, dict):
        return any(_unfinished(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_unfinished(v) for v in value)
    return False


def _call(function, *args):
    """(result, error) of the learner's function, without letting it stop the notebook."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args), None
    except Exception as error:  # the learner's code may raise anything
        return None, error


def _same(actual, expected, tolerance=1e-9):
    """Numbers, NaNs, dicts and sequences compared value by value."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(expected) <= set(actual)
                and all(_same(actual[k], expected[k], tolerance) for k in expected))
    if isinstance(expected, float) and math.isnan(expected):
        return _is_number(actual) and math.isnan(float(actual))
    return _near(actual, expected, tolerance)


def _missing_keys(name, actual, keys):
    if not isinstance(actual, dict):
        return _report(name, [f"Return a dictionary with the keys {', '.join(map(repr, keys))}; "
                              f"received {type(actual).__name__}."])
    missing = [k for k in keys if k not in actual]
    if missing:
        return _report(name, [f"The dictionary has no {', '.join(map(repr, missing))}."])
    return None


def _first_match(actual, variants, compare=_same):
    for wrong, hint in variants:
        if compare(actual, wrong):
            return hint
    return None


# --- Section 1 ----------------------------------------------------------------


def threshold_probability(function):
    """Problem 1: p_t = harm / (harm + benefit) from any price matrix, and a refusal when it does not apply."""
    name = "threshold_probability"
    if not callable(function):
        return _not_function(name, function)
    cases = [DEFAULT_COSTS, {**DEFAULT_COSTS, "FN": 10.0}, UNUSUAL_COSTS]
    for costs in cases:
        got, error = _call(function, costs)
        if error is not None:
            return _report(name, [f"It raised {type(error).__name__}: {error}."])
        if got is Ellipsis:
            return _waiting(name)
        expected = reference.threshold_probability(costs)
        if _near(got, expected):
            continue
        harm, benefit = costs["FP"] - costs["TN"], costs["FN"] - costs["TP"]
        hint = _first_match(got, [
            (costs["FP"] / (costs["FP"] + costs["FN"]),
             "Only the price differences matter: harm = C_FP - C_TN and benefit = C_FN - C_TP. "
             "A correct review still costs a review, so subtract C_TP."),
            (1 - expected, "That is the probability of no AF at the crossing point."),
            (harm / benefit, "That is harm / benefit, the odds at the threshold. The threshold "
                             "is harm / (harm + benefit)."),
            (100 * expected, "Give the threshold as a proportion, not a percentage."),
            (benefit / (harm + benefit), "Harm and benefit are the wrong way round: the "
                                         "numerator is the harm of a needless review."),
        ])
        return _report(name, [f"With prices {costs} it returned {got!r}; expected "
                              f"{expected:.4f}. " + (hint or "Read every price from costs.")])
    _, error = _call(function, {"TP": 1.0, "FP": 0.0, "FN": 5.0, "TN": 0.0})
    if not isinstance(error, ValueError):
        return _report(name, ["When C_FP = C_TN a review costs nothing extra, so review is never "
                              "worse and the threshold rule does not apply. Raise a ValueError "
                              "when harm or benefit is not positive."])
    return _report(name, [])


# --- Section 2 ----------------------------------------------------------------

COUNT_KEYS = ("TP", "FP", "FN", "TN")


def _count_cases():
    six = _six()
    one_class = (np.zeros(3, dtype=bool), np.zeros(3, dtype=bool))
    return [(six["af"], six["random_forest"] >= .2), (six["af"], six["random_forest"] >= .5),
            one_class]


def six_counts(function):
    """Problem 2: TP, FP, FN and TN from confusion_matrix, in the right order and for one class too."""
    name = "six_counts"
    if not callable(function):
        return _not_function(name, function)
    for y, review in _count_cases():
        got, error = _call(function, y, review)
        expected = reference.counts(y, review)
        if error is not None:
            if len(set(np.asarray(y))) == 1:
                return _report(name, [
                    "With only one class present, confusion_matrix(y, review) is 1 × 1 and ravel() "
                    "has one value to unpack. Pass labels=[False, True] so the matrix is always "
                    "2 × 2."])
            return _report(name, [f"It raised {type(error).__name__}: {error}."])
        if _unfinished(got):
            return _waiting(name)
        problem = _missing_keys(name, got, COUNT_KEYS)
        if problem is not None:
            return problem
        got = {k: got[k] for k in COUNT_KEYS}
        if _same(got, expected):
            continue
        e = expected
        hint = _first_match(got, [
            ({"TP": e["TN"], "FP": e["FP"], "FN": e["FN"], "TN": e["TP"]},
             "TP and TN are swapped. confusion_matrix puts the true labels in rows and the "
             "decisions in columns, both in the order [False, True], so ravel() returns "
             "tn, fp, fn, tp."),
            ({"TP": e["TN"], "FP": e["FN"], "FN": e["FP"], "TN": e["TP"]},
             "This reads the matrix as [[TP, FN], [FP, TN]], the textbook layout. scikit-learn's "
             "ravel() returns tn, fp, fn, tp."),
            ({"TP": e["TP"], "FP": e["FN"], "FN": e["FP"], "TN": e["TN"]},
             "FP and FN are swapped: call confusion_matrix(y_true, decisions), with the reference "
             "labels first."),
        ])
        return _report(name, [f"Expected {expected}, received {got}. " + (hint or "")])
    return _report(name, [])


# --- Section 3 ----------------------------------------------------------------


def _cost_variants(y, review, costs):
    c = reference.counts(y, review)

    def price(tp, fp, fn, tn, prices=costs):
        return tp * prices["TP"] + fp * prices["FP"] + fn * prices["FN"] + tn * prices["TN"]
    return [
        (price(c["TN"], c["FP"], c["FN"], c["TP"]),
         "TP and TN are swapped: confusion_matrix(...).ravel() returns tn, fp, fn, tp."),
        (price(c["TN"], c["FN"], c["FP"], c["TP"]),
         "This reads the matrix as [[TP, FN], [FP, TN]]. scikit-learn's ravel() returns "
         "tn, fp, fn, tp."),
        (price(c["TP"], c["FN"], c["FP"], c["TN"]),
         "FP and FN are swapped: pass the reference labels first, confusion_matrix(y, review)."),
        (price(0, c["FP"], c["FN"], 0),
         "Every outcome has a price, including the two correct ones. A correct review still "
         "costs a review."),
        (price(c["TP"], c["FP"], c["FN"], c["TN"], DEFAULT_COSTS),
         "The prices look typed in. Read each one from the costs argument."),
    ]


def policy_cost(function):
    """Problem 3: the total loss of a set of decisions from confusion_matrix and all four prices."""
    name = "policy_cost"
    if not callable(function):
        return _not_function(name, function)
    six = _six()
    cases = [(six["af"], six["random_forest"] >= .2, DEFAULT_COSTS),
             (six["af"], six["random_forest"] >= .5, DEFAULT_COSTS),
             (six["af"], six["random_forest"] >= .2, UNUSUAL_COSTS),
             (np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), UNUSUAL_COSTS)]
    for y, review, costs in cases:
        got, error = _call(function, y, review, costs)
        if error is not None:
            if len(set(np.asarray(y))) == 1:
                return _report(name, ["With one class present the confusion matrix is 1 × 1. "
                                      "Pass labels=[False, True]."])
            return _report(name, [f"It raised {type(error).__name__}: {error}."])
        if got is Ellipsis:
            return _waiting(name)
        expected = reference.policy_cost(y, review, costs)
        if _near(got, expected):
            continue
        hint = _first_match(got, _cost_variants(y, review, costs))
        return _report(name, [f"With prices {costs} expected {expected:g}, received {got!r}. "
                              + (hint or "Multiply each count by its price and add the four.")])
    return _report(name, [])


# --- Section 4 ----------------------------------------------------------------


def _frame_problem(name, got, columns):
    if not isinstance(got, pd.DataFrame):
        return _report(name, [f"Return a pandas DataFrame; received {type(got).__name__}."])
    missing = [c for c in columns if c not in got.columns]
    if missing:
        return _report(name, [f"The table has no column {', '.join(map(repr, missing))}."])
    return None


def _columns_match(got, expected, columns, tolerance=1e-9):
    for column in columns:
        a = pd.to_numeric(got[column], errors="coerce").to_numpy(dtype=float)
        b = expected[column].to_numpy(dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=tolerance, atol=tolerance,
                                                 equal_nan=True):
            return column
    return None



COMPARE_COLUMNS = ("cutoff", "reviews", "af_detected", "af_missed", "loss")
MODELS = ("random_forest", "resnet")


def _consequences_at(frame, model, cutoff, costs):
    """Reviews, AF detected and missed per 1,000, and the loss, of one model at one cutoff."""
    review = frame[model] >= cutoff
    c, n = reference.counts(frame["af"], review), len(frame)
    return {"reviews": 1000 * (c["TP"] + c["FP"]) / n, "af_detected": 1000 * c["TP"] / n,
            "af_missed": 1000 * c["FN"] / n, "loss": reference.policy_cost(frame["af"], review, costs)}


def _cutoff_problem(got, frame, costs, expected):
    """None when each model's cutoff attains the minimum loss and its row reports what it does.

    The check is semantic: the decisions the submitted cutoff makes on the frame
    are computed, so a cutoff that merely looks close to the reference is not
    enough. Several cutoffs can make the same decisions, and different decisions
    can tie on loss; any of them is accepted, as the guides promise. Otherwise the
    hint names the first problem, comparing each row with its own cutoff.
    """
    cutoffs = {}
    for model in got.index:
        try:
            cutoffs[model] = float(got.loc[model, "cutoff"])
        except (TypeError, ValueError):
            return f"The cutoff for {model} should be one number."
    for model, cutoff in cutoffs.items():
        actual = _consequences_at(frame, model, cutoff, costs)
        if not math.isclose(actual["loss"], float(expected.loc[model, "loss"]), rel_tol=1e-9):
            if all(math.isclose(c, .5) for c in cutoffs.values()):
                return ("Both models are still judged at 0.5, the default boundary of predict(). "
                        "Their scores are on different scales: find each model's cheapest cutoff "
                        "in its own operating-point table.")
            return (f"At the cutoff given for {model} the loss is {actual['loss']:g}, not the minimum "
                    f"{float(expected.loc[model, 'loss']):g}. Choose each model's row with the lowest "
                    "loss (fewest reviews on a tie), using its threshold as it is, not rounded.")
        for key, value in actual.items():
            if not _near(pd.to_numeric(got.loc[model, key], errors="coerce"), value):
                return (f"Column {key!r} differs for {model}: at that cutoff it should be "
                        f"{value:g}. Report the counts per 1,000 recordings that the cutoff gives.")
    return None


def compare_models(function):
    """Problem 4: each model at its own minimum-loss cutoff under the same prices."""
    name = "compare_models"
    if not callable(function):
        return _not_function(name, function)
    development = _split("development")
    for costs in (DEFAULT_COSTS, UNUSUAL_COSTS):
        got, error = _call(function, development, list(MODELS), costs)
        if error is not None:
            return _report(name, [f"It raised {type(error).__name__}: {error}."])
        if got is Ellipsis:
            return _waiting(name)
        problem = _frame_problem(name, got, COMPARE_COLUMNS)
        if problem is not None:
            return problem
        expected = reference.compare_models(development, list(MODELS), costs)
        try:
            got = got.loc[list(MODELS)]
        except KeyError:
            return _report(name, ["Index the table by model name, as the original does."])
        problem = _cutoff_problem(got, development, costs, expected)
        if problem is not None:
            return _report(name, [problem])
    return _report(name, [])


# --- Section 6 ----------------------------------------------------------------


def _reference_recalibration():
    fitting = _split("calibration")
    return calibration.platt().fit(fitting[["resnet"]], fitting["af"])


def recalibration(model):
    """Problem 5: the Platt map, fitted on the calibration split only."""
    name = "recalibration"
    if model is Ellipsis:
        return _waiting(name)
    if not hasattr(model, "predict_proba"):
        return _report(name, ["Expected a fitted scikit-learn model with predict_proba, for "
                              "example a pipeline of FunctionTransformer(logit) and "
                              "LogisticRegression."])
    development = _split("development")
    got, error = _call(lambda: model.predict_proba(development[["resnet"]])[:, 1])
    if isinstance(error, NotFittedError):
        return _report(name, ["The model is not fitted yet: call .fit(...) on the calibration "
                              "split's resnet scores and AF labels."])
    if error is not None:
        return _report(name, [f"predict_proba raised {type(error).__name__}: {error}. Fit on a "
                              "one-column DataFrame, calibration_split[['resnet']]."])
    expected = _reference_recalibration().predict_proba(development[["resnet"]])[:, 1]
    if np.allclose(got, expected, atol=5e-4, rtol=0):
        return _report(name, [])
    fits = {}
    for split in ("development", "evaluation"):
        frame = _split(split)
        fits[f"the {split} split"] = calibration.platt().fit(frame[["resnet"]], frame["af"])
    fitting = _split("calibration")
    fits["penalty"] = LogisticRegression(solver="newton-cholesky").fit(logit(fitting[["resnet"]]), fitting["af"])
    fits["raw"] = LogisticRegression(C=np.inf, solver="newton-cholesky").fit(fitting[["resnet"]], fitting["af"])
    for label, fit in fits.items():
        x = logit(development[["resnet"]]) if label == "penalty" else development[["resnet"]]
        # The raw-score fit is far from the answer, and solvers disagree more on unscaled scores.
        tolerance = 2e-3 if label == "raw" else 5e-4
        if np.allclose(got, fit.predict_proba(x)[:, 1], atol=tolerance, rtol=0):
            if label == "penalty":
                return _report(name, ["LogisticRegression shrinks its coefficients by default "
                                      "(C=1). Switch the penalty off with C=np.inf."])
            if label == "raw":
                return _report(name, ["This regresses on the raw score. Platt scaling uses "
                                      "logit(score) as the feature: put "
                                      "FunctionTransformer(logit) first in the pipeline."])
            return _report(name, [f"This map was fitted on {label}. Fit it on the calibration "
                                  "split only: development chooses policies and evaluation "
                                  "assesses them."])
    return _report(name, ["The fitted probabilities differ from a Platt map fitted on the "
                          "calibration split. Use logit(resnet) as the one feature and no penalty."])


def net_benefit(function):
    """Problem 6: TP/n - FP/n × p_t/(1 - p_t)."""
    name = "net_benefit"
    if not callable(function):
        return _not_function(name, function)
    six, development = _six(), _split("development")
    cases = [(six["af"], six["random_forest"] >= .2, .2),
             (development["af"], development["random_forest"] >= .1, .1),
             (development["af"], development["resnet"] >= .4, .4)]
    for y, review, p_t in cases:
        got, error = _call(function, y, review, p_t)
        if error is not None:
            return _report(name, [f"It raised {type(error).__name__}: {error}."])
        if got is Ellipsis:
            return _waiting(name)
        expected = reference.net_benefit(y, review, p_t)
        if _near(got, expected):
            continue
        c = reference.counts(y, review)
        n, odds = sum(c.values()), p_t / (1 - p_t)
        hint = _first_match(got, [
            (c["TP"] - c["FP"] * odds, "Divide by n, the number of recordings."),
            ((c["TP"] - c["FP"] * p_t) / n, "Weight false positives by the odds p_t/(1 - p_t), "
                                            "not by p_t."),
            ((c["TP"] - c["FP"] / odds) / n, "The odds are inverted: use p_t/(1 - p_t)."),
            ((c["TP"] - c["FN"] * odds) / n, "Net benefit charges false positives, not misses."),
            ((c["TP"] - c["FP"] * odds) / max(c["TP"] + c["FP"], 1),
             "Divide by all recordings, not only the reviewed ones."),
        ])
        return _report(name, [f"At p_t = {p_t} expected {expected:.4f}, received {got!r}. "
                              + (hint or "")])
    return _report(name, [])


# --- Section 7 ----------------------------------------------------------------

FROZEN_KEYS = ("model", "cutoff", "costs", "p_t", "chosen_on")


def frozen_policy(frozen):
    """Problem 7: a complete frozen policy whose preference follows from its prices.

    The checker never reads the evaluation split and cannot judge the choice:
    a cutoff derived from the prices is as legitimate as one read from the
    development table. It checks completeness and that ``p_t`` follows from
    the prices, and records where the cutoff was chosen.
    """
    name = "Frozen policy"
    if _unfinished(frozen):
        return _waiting(name)
    problem = _missing_keys(name, frozen, FROZEN_KEYS)
    if problem is not None:
        return problem
    model, cutoff, costs, p_t, chosen_on = (frozen[k] for k in FROZEN_KEYS)
    problems = []
    if model not in data.MODEL_NAMES:
        problems.append(f"Set model to one of {', '.join(map(repr, data.MODEL_NAMES))}.")
    if not (isinstance(costs, dict) and set(costs) == set(COUNT_KEYS)
            and all(_is_number(v) and math.isfinite(v) for v in costs.values())):
        problems.append("Set costs to a dictionary of four finite prices with the keys 'TP', "
                        "'FP', 'FN' and 'TN'.")
    if not _is_number(cutoff) or not (0 <= cutoff <= 1 or math.isinf(cutoff)):
        problems.append("Set cutoff to one number between 0 and 1 on that model's score scale.")
    if not isinstance(chosen_on, str) or not chosen_on.strip():
        problems.append("Say in chosen_on how the cutoff was chosen, for example 'development "
                        "minimum loss' or 'derived from the prices'.")
    if problems:
        return _report(name, problems)
    try:
        implied = reference.threshold_probability(costs)
    except ValueError:
        return _report(name, ["These prices give no threshold probability; choose prices with "
                              "C_FP > C_TN and C_FN > C_TP."])
    if not _near(p_t, implied):
        return _report(name, [f"p_t should follow from the prices: {implied:.4f}."])
    return _report(name, [], success=f"frozen ({chosen_on.strip()})")


# --- Section 8 ----------------------------------------------------------------

RECORD_TEXT = ("action", "price_rationale", "reopen_if")


def decision_record(record, frozen, reports):
    """Problem 7, second part: the three sentences of the decision record, then the whole record.

    The learner writes ``action``, ``price_rationale`` and ``reopen_if``; the
    checker adds the frozen policy and the two reports, so the record is
    consistent by construction, and prints it. It cannot judge the valuation.
    """
    name = "Decision record"
    if _unfinished(record):
        return _waiting(name)
    problem = _missing_keys(name, record, RECORD_TEXT)
    if problem is not None:
        return problem
    problems = [f"Write {key} in a sentence or two." for key in RECORD_TEXT
                if not isinstance(record[key], str) or not record[key].strip()]
    if not (isinstance(frozen, dict) and isinstance(reports, dict)
            and {"development", "evaluation"} <= set(reports)):
        problems.append("Freeze your policy and run the reveal first.")
    if problems:
        return _report(name, problems)
    full = {"action": record["action"], "prices": frozen.get("costs"), "model": frozen.get("model"),
            "cutoff": frozen.get("cutoff"), "chosen_on": frozen.get("chosen_on"),
            "price_rationale": record["price_rationale"], "reopen_if": record["reopen_if"]}
    for key, value in full.items():
        print(f"{key}: {value}")
    for split in ("development", "evaluation"):
        print(f"{split} report:")
        print(reports[split].round(3).to_string())
    return _report(name, [], success="recorded. A checker cannot judge a valuation; read it back "
                                     "as the service would")


# --- Previews of learner code ---------------------------------------------------


def show(function, *args):
    """Run learner code and display what it returns, without letting it stop the notebook.

    A preview sits inside the same no-raise boundary as the checkers: an
    unfinished answer or an error is reported in one line, and the checker in
    the same cell still runs.
    """
    from IPython.display import display
    if not callable(function):
        print(f"Nothing to preview yet: expected a function, received {type(function).__name__}.")
        return None
    result, error = _call(function, *args)
    if error is not None:
        print(f"Preview not available yet: {type(error).__name__}: {error}. The checker below says "
              "what to change.")
        return None
    if _unfinished(result):
        print("Preview not available yet: the function still returns ... somewhere.")
        return None
    display(result)
    return result


# --- Quiet guard for cells that build on earlier answers -------------------

_QUIET = {"active": False}


# How to recover once the missing answer is in place. Rerunning only the waiting
# cell is not enough when an earlier cell skipped its own work for the same reason,
# so the learner reruns everything above it, then the cell itself.
RECOVERY = ("Then click in this cell, choose Runtime > Run before to rerun every cell above it, "
            "and run this cell.")


def waiting(what):
    """Print one waiting line for something other than a checked function, with the recovery."""
    print(f"Waiting for {what}. {RECOVERY}")


# Functions later cells wait for: (checker, problem, has an I'm-stuck switch).
WAITED_FOR = {"threshold_probability": ("threshold_probability", "Problem 1", True),
              "six_counts": ("six_counts", "Problem 2", False),
              "policy_cost": ("policy_cost", "Problem 3", True),
              "compare_models": ("compare_models", "Problem 4", False),
              "recalibration": ("recalibration", "Problem 5", True),
              "net_benefit": ("net_benefit", "Problem 6", True)}


def ready(**answers):
    """True when every named answer passes its checker; prints one line per wait otherwise.

    ``check.ready(policy_cost=policy_cost)`` lets a later cell wait for an
    unfinished function instead of failing. Each line names the problem, how to
    unblock it and how to recover (``RECOVERY``). Nothing is printed when all pass.
    """
    waiting = []
    for name, answer in answers.items():
        checker_name, problem, switch = WAITED_FOR.get(name, (name, None, False))
        checker = globals()[checker_name]
        _QUIET["active"] = True
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                passed = bool(checker(answer))
        except Exception:
            passed = False
        finally:
            _QUIET["active"] = False
        if not passed:
            waiting.append(name)
            label = f"{name} ({problem})" if problem else name
            unblock = ("finish it and check it above, or delete the # in the stuck cell below it "
                       "to use the reference version" if switch else
                       "finish it and check it above; the worked copy shows one answer")
            print(f"Waiting for {label}: {unblock}. {RECOVERY}")
    if waiting and _strict():
        raise AssertionError(f"Not ready: {', '.join(waiting)}")
    return CheckResult(not waiting)


REFERENCE_VERSIONS = {"threshold_probability": reference.threshold_probability,
                      "policy_cost": reference.policy_cost,
                      "net_benefit": reference.net_benefit}


def use_reference(name):
    """Return the reference version of a function later cells depend on, and say so.

    For a learner who is stuck: ``policy_cost = check.use_reference("policy_cost")``
    lets the rest of the notebook run. Their own attempt is untouched until that
    line runs, so they can come back to it later.
    """
    if name == "recalibration":
        print("Using the reference recalibration, fitted on the calibration split, from now on. "
              "Come back to your own version later and compare it with the worked copy.")
        return _reference_recalibration()
    if name not in REFERENCE_VERSIONS:
        listed = ", ".join(repr(k) for k in [*REFERENCE_VERSIONS, "recalibration"])
        print(f"There is no reference version of {name!r}. Choose one of {listed}.")
        return None
    print(f"Using the reference {name} from now on. Come back to your own version later and "
          "compare it with the worked copy.")
    return REFERENCE_VERSIONS[name]


def _guarded(checker):
    @functools.wraps(checker)
    def guarded(*args, **kwargs):
        try:
            return checker(*args, **kwargs)
        except AssertionError:
            raise
        except Exception as error:
            if _strict():
                raise
            label = checker.__name__.replace("_", " ").capitalize()
            print(f"Not yet: {label}. The checker could not read this answer "
                  f"({type(error).__name__}: {error}). Check it is the kind of value the problem "
                  "asks for, then run the cell again.")
            return CheckResult(False)
    guarded.utility_checker = True
    return guarded


for _name, _function in list(globals().items()):
    if (inspect.isfunction(_function) and _function.__module__ == __name__
            and not _name.startswith("_")
            and _name not in ("ready", "use_reference", "waiting", "show")):
        globals()[_name] = _guarded(_function)
del _name, _function
