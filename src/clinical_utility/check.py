"""Feedback on the notebook's tasks.

Each checker prints one verdict and returns a ``Result`` that is truthy when the
answer is right: ``✓`` means correct, ``Not yet`` names the likely mistake and
``Not attempted yet`` means a ``...`` is still in the answer. A checker never
raises in a student notebook. With ``UTILITY_STRICT=1`` (set when the worked
notebook is run automatically) a failed check raises ``AssertionError``.

Every check runs the learner's code on small made-up inputs stored in
``assets/checks.json``, with the expected outputs and the outputs of common
mistakes. None of them are the practical's own patients, so the stored numbers
are not the answers to the session's questions, and this module contains no
worked answer.
"""

import contextlib
import functools
import io
import json
import math
import numbers
import os
import types
from functools import lru_cache
from importlib.resources import files

import numpy as np
import pandas as pd

TASKS = {
    "outcome_counts": "Task 1, outcome_counts",
    "policy_cost": "Task 2, policy_cost",
    "roc_points": "Task 3, roc_points",
    "price_cutoffs": "Task 4, price_cutoffs",
    "threshold_probability": "Task 5, threshold_probability",
    "net_benefit": "Task 6, net_benefit",
    "recalibrate": "Task 7, recalibrate",
    "recommendation": "Task 8, recommendation",
    "cheapest_within_capacity": "Extension A, cheapest_within_capacity",
    "share_of_days_over": "Extension B, share_of_days_over",
    "cheapest_reliable": "Extension C, cheapest_reliable",
}
OUTCOMES = ("TP", "FP", "FN", "TN")
ROC_FIELDS = ("cutoff", "fpr", "tpr")
TABLE_FIELDS = ("cutoff", "flags", "af_found", "af_missed", "cost")
MODELS = ("random_forest", "resnet", "resnet_recalibrated")
RECOMMENDATION_FIELDS = ("model", "threshold", "reasoning", "reopen_if")

HINTS = {
    "ravel_tp_first": "The counts are in the wrong places. Check the order in which .ravel() returns them.",
    "ravel_textbook": "The counts are in the wrong places. Check the order in which .ravel() returns them.",
    "ravel_transposed": "Two of the counts have swapped places. Check how you called confusion_matrix.",
    "raw_total": "The figure is on the wrong scale. What unit should the cost be in?",
    "per_patient": "The figure is on the wrong scale. What unit should the cost be in?",
    "flags_only": "Some outcomes are missing from the cost.",
    "no_inf_row": "One policy is missing from the table.",
    "swapped_rates": "Two of the columns look swapped.",
    "fpr_as_flags": "Check which patients count as flagged.",
    "rates_not_counts": "These look like rates rather than figures per 1,000 patients.",
    "missed_as_rate": "Check how AF missed is worked out.",
    "costs_ignored_tn": "Check that every outcome's cost is included.",
    "without_correct_outcomes": "Look again at the inequality: what changes, for each kind of patient, when they are flagged?",
    "complement": "This looks like the answer for the opposite decision.",
    "odds": "This is built from the right quantities, but it isn't a probability.",
    "nb_counts": "The figure is on the wrong scale.",
    "nb_weight_pt": "Check how each false positive is weighted.",
    "nb_inverted": "Check how each false positive is weighted.",
    "nb_missed": "Check which outcome net benefit penalises.",
    "nb_clipped": "Can a policy do worse than flagging no one?",
    "nb_absolute": "Can a policy do worse than flagging no one?",
    "corners": "Some cutoffs are missing. Look at roc_curve's options.",
    "no_logit": "Check what the logistic regression is fitted on.",
    "penalised": "Check LogisticRegression's default settings.",
    "labels_not_probabilities": "These look like labels rather than probabilities.",
    "wrong_column": "Check which column you are returning.",
    "at_or_over": "Check what counts as a day over capacity.",
    "count_not_share": "Check what the function should return.",
    "not_per_1000": "Check the probability that a single strip is flagged.",
    "cheapest_overall": "Check that every condition in the task is applied.",
    "average_only": "A policy that fits the average capacity can still overflow often.",
    "ignores_capacity": "Check that the limit is applied.",
    "strict_capacity": "Check the boundary: is a policy exactly at the limit allowed?",
    "most_flags": "Check what you are choosing between the allowed policies on.",
}


class Result:
    """Pass or fail, without a stray True or False echoed under the feedback."""

    __slots__ = ("passed",)

    def __init__(self, passed):
        self.passed = bool(passed)

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return repr(self.passed)

    def _ipython_display_(self):
        pass


_QUIET = {"active": False}


def _strict():
    return os.environ.get("UTILITY_STRICT") == "1" and not _QUIET["active"]


def _report(task, problems, attempted=True):
    label = TASKS[task]
    if not problems:
        print(f"✓ {label}: correct.")
        return Result(True)
    if not attempted:
        print(f"Not attempted yet: {label}. Replace every ... and run this cell again.")
    else:
        print(f"Not yet: {label}.")
        for problem in problems[:3]:
            print(f"  {problem}")
    if _strict():
        raise AssertionError(f"{label}: {'; '.join(problems)}")
    return Result(False)


def _safe(task):
    """Turn any unexpected error while checking into feedback instead of a traceback."""
    def decorate(checker):
        @functools.wraps(checker)
        def run(value):
            try:
                return checker(value)
            except AssertionError:
                raise
            except Exception as error:  # an unusual answer must not stop the notebook
                return _report(task, [f"Your answer could not be checked: {type(error).__name__}: "
                                      f"{error}. Check what it returns."])
        return run
    return decorate


@lru_cache(maxsize=1)
def _fixtures():
    return json.loads(files("clinical_utility").joinpath("assets/checks.json").read_text())


# --- Recognising unfinished and malformed answers ------------------------------------------


def _code_has_ellipsis(code):
    return any(c is Ellipsis or (isinstance(c, types.CodeType) and _code_has_ellipsis(c))
               for c in code.co_consts)


def _unfinished(value):
    if value is Ellipsis:
        return True
    if isinstance(value, types.FunctionType):
        return _code_has_ellipsis(value.__code__)
    if isinstance(value, dict):
        return any(_unfinished(v) for v in value.values())
    return False


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Real):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _close(a, b, tolerance=1e-6):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=tolerance,
                                                           abs_tol=tolerance)


def _record(output, fields):
    """A dict of floats from a Series or dict holding ``fields``, or an error message."""
    if isinstance(output, pd.Series):
        if output.index.duplicated().any():
            return None, "Each figure should appear once."
        output = output.to_dict()
    if not isinstance(output, dict):
        return None, f"Expected a pandas Series or dict, received {type(output).__name__}."
    missing = [f for f in fields if f not in output]
    if missing:
        return None, f"Missing {', '.join(missing)}: return {', '.join(fields)}."
    values = {f: _number(output[f]) for f in fields}
    bad = [f for f, v in values.items() if v is None]
    if bad:
        return None, f"{', '.join(bad)} should be a finite number."
    return values, None


def _same(values, expected, fields):
    return all(_close(values[f], expected[f]) for f in fields)


def _call(function, *args):
    """The function's output, or an error message naming the exception."""
    try:
        with _warnings_quiet():
            return function(*args), None
    except Exception as error:  # a learner's bug must not stop the notebook
        return None, f"Calling it raised {type(error).__name__}: {error}"


@contextlib.contextmanager
def _warnings_quiet():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _needs(function, *names):
    """Earlier tasks that ``function`` uses from the notebook and that are not passing yet."""
    namespace = getattr(function, "__globals__", {})
    return [TASKS[name] for name in names
            if name not in namespace or not _quiet(name, namespace[name])]


def _waiting(task, function, *names):
    """Print and return a waiting result if ``function`` depends on unfinished tasks."""
    waiting = _needs(function, *names)
    if not waiting:
        return None
    print(f"Waiting for {' and '.join(waiting)}. Finish {'them' if len(waiting) > 1 else 'it'} "
          "above, then run this cell again.")
    if _strict():
        raise AssertionError(f"{TASKS[task]} is waiting for {', '.join(waiting)}")
    return Result(False)


def _uses(function, name):
    """True if ``function``'s code refers to the notebook name ``name``."""
    code = getattr(function, "__code__", None)

    def names(c):
        found = set(c.co_names)
        for const in c.co_consts:
            if isinstance(const, types.CodeType):
                found |= names(const)
        return found
    return code is not None and name in names(code)


def _dependencies(task, function, *names):
    """Waiting result for the earlier tasks this function actually calls, if any are unfinished."""
    used = [name for name in names if _uses(function, name)]
    return _waiting(task, function, *used) if used else None


def _inputs(case):
    """A case's labels and flags or scores, as arrays or as pandas Series."""
    y = np.array(case["y"], dtype=bool)
    second = np.array(case["flagged"] if "flagged" in case else case["score"],
                      dtype=bool if "flagged" in case else float)
    if case.get("pandas"):
        index = pd.RangeIndex(100, 100 + len(y))
        return pd.Series(y, index=index, name="af"), pd.Series(second, index=index, name="x")
    return y, second


def _start(task, function):
    """Feedback for an answer that is not a finished function, or None to carry on."""
    if not callable(function):
        return _report(task, [f"Expected a function, received {type(function).__name__}."])
    if _unfinished(function):
        return _report(task, ["..."], attempted=False)
    return None


def _frame(output, fields):
    """The learner's table as floats sorted by cutoff, highest first, or a problem."""
    if not isinstance(output, pd.DataFrame):
        return None, f"Expected a pandas DataFrame, received {type(output).__name__}."
    missing = [f for f in fields if f not in output.columns]
    if missing:
        return None, f"Missing columns {', '.join(missing)}."
    if output.columns.duplicated().any():
        return None, "Each column should appear once."
    if output.empty:
        return None, "The table has no rows."
    try:
        table = output[list(fields)].astype(float)
    except (TypeError, ValueError):
        return None, "Every column should hold numbers."
    if table.drop(columns="cutoff").isna().any().any() or table["cutoff"].isna().any():
        return None, "Some figures are missing (NaN)."
    return table.sort_values("cutoff", ascending=False, kind="stable").reset_index(drop=True), None


def _matches(table, expected, fields):
    expected = pd.DataFrame(expected, columns=fields).astype(float)
    expected = expected.sort_values("cutoff", ascending=False, kind="stable").reset_index(drop=True)
    return len(table) == len(expected) and all(
        np.allclose(table[f], expected[f], rtol=1e-6, atol=1e-6) for f in fields)


def _table_task(task, function, fields, call_args):
    """Shared logic for the two table-returning tasks."""
    for case in _fixtures()[task]["cases"]:
        output, error = _call(function, *call_args(case))
        if error:
            return _report(task, [f"For {case['label']}: {error}"])
        table, problem = _frame(output, fields)
        if problem:
            return _report(task, [f"For {case['label']}: {problem}"])
        if any(_matches(table, rows, fields) for rows in case["accepted"]):
            continue
        full = case["accepted"][0]
        if len(table) == len(full) - 1 and not np.isinf(table["cutoff"]).any():
            return _report(task, [HINTS["no_inf_row"]])
        hint = next((HINTS[m["id"]] for m in case.get("mistakes", [])
                     if _matches(table, m["output"], fields)), None)
        return _report(task, [f"The figures for {case['label']} don't match.",
                              hint or "Check one row by hand."])
    return _report(task, [])


# --- Task checkers ------------------------------------------------------------------------


@_safe("outcome_counts")
def outcome_counts(function):
    """Task 1: the four outcome counts."""
    task = "outcome_counts"
    early = _start(task, function)
    if early is not None:
        return early
    for case in _fixtures()[task]["cases"]:
        y, flagged = _inputs(case)
        output, error = _call(function, y, flagged)
        if error:
            return _report(task, [f"For {case['label']}: {error}"])
        values, problem = _record(output, OUTCOMES)
        if problem:
            return _report(task, [f"For {case['label']}: {problem}"])
        if not _same(values, case["expected"], OUTCOMES):
            hint = next((HINTS[m["id"]] for m in case["mistakes"]
                         if _same(values, m["output"], OUTCOMES)), None)
            shown = ", ".join(f"{k} {values[k]:g}" for k in OUTCOMES)
            return _report(task, [f"For {case['label']}, received {shown}.",
                                  hint or "Count one of them by hand to check."])
    return _report(task, [])


@_safe("policy_cost")
def policy_cost(function):
    """Task 2: the cost of a policy in dollars per 1,000 patients."""
    task = "policy_cost"
    early = _start(task, function)
    if early is None:
        early = _dependencies(task, function, "outcome_counts")
    if early is not None:
        return early
    for case in _fixtures()[task]["cases"]:
        y, flagged = _inputs(case)
        output, error = _call(function, y, flagged, dict(case["costs"]))
        if error:
            return _report(task, [f"For {case['label']}: {error}"])
        value = _number(output)
        if value is None:
            return _report(task, [f"Return one number, received {type(output).__name__}."])
        if not _close(value, case["expected"]):
            hint = next((HINTS[m["id"]] for m in case["mistakes"] if _close(value, m["output"])),
                        None)
            return _report(task, [f"For {case['label']}, received {value:,.2f}.",
                                  hint or "Work out one of the figures by hand to check."])
    return _report(task, [])


@_safe("roc_points")
def roc_points(function):
    """Task 3: every cutoff with its false and true positive rates."""
    task = "roc_points"
    early = _start(task, function)
    if early is not None:
        return early
    return _table_task(task, function, ROC_FIELDS, lambda case: _inputs(case))


@_safe("price_cutoffs")
def price_cutoffs(function):
    """Task 4: the cost of every cutoff, built on roc_points."""
    task = "price_cutoffs"
    early = _start(task, function)
    if early is None:
        early = _dependencies(task, function, "roc_points")
    if early is not None:
        return early
    return _table_task(task, function, TABLE_FIELDS,
                       lambda case: (*_inputs(case), dict(case["costs"])))


@_safe("threshold_probability")
def threshold_probability(function):
    """Task 5: the threshold probability the costs imply."""
    task = "threshold_probability"
    early = _start(task, function)
    if early is not None:
        return early
    for case in _fixtures()[task]["cases"]:
        output, error = _call(function, dict(case["costs"]))
        if error:
            return _report(task, [error])
        value = _number(output)
        if value is None:
            return _report(task, [f"Return one number, received {type(output).__name__}."])
        if not _close(value, case["expected"]):
            hint = next((HINTS[m["id"]] for m in case["mistakes"] if _close(value, m["output"])),
                        None)
            return _report(task, [f"For the costs {case['costs']}, received {value:.4f}.",
                                  hint or "Solve the inequality for p."])
    return _report(task, [])


@_safe("net_benefit")
def net_benefit(function):
    """Task 6: net benefit at a threshold probability."""
    task = "net_benefit"
    early = _start(task, function)
    if early is None:
        early = _dependencies(task, function, "outcome_counts")
    if early is not None:
        return early
    for case in _fixtures()[task]["cases"]:
        y, flagged = _inputs(case)
        output, error = _call(function, y, flagged, case["p_t"])
        if error:
            return _report(task, [f"For {case['label']}: {error}"])
        value = _number(output)
        if value is None:
            return _report(task, [f"Return one number, received {type(output).__name__}."])
        if not _close(value, case["expected"]):
            hint = next((HINTS[m["id"]] for m in case["mistakes"] if _close(value, m["output"])),
                        None)
            return _report(task, [f"For {case['label']} at p_t = {case['p_t']}, received "
                                  f"{value:.4f}.", hint or "Compare your code with the formula above."])
    return _report(task, [])


@_safe("recalibrate")
def recalibrate(function):
    """Task 7: a Platt map fitted on one set of scores and applied to another."""
    task = "recalibrate"
    early = _start(task, function)
    if early is not None:
        return early
    for case in _fixtures()[task]["cases"]:
        args = [np.array(case["fit_score"]), np.array(case["fit_y"], dtype=bool),
                np.array(case["score"])]
        if case.get("pandas"):
            args = [pd.Series(a, index=pd.RangeIndex(50, 50 + len(a))) for a in args]
        output, error = _call(function, *args)
        if error:
            return _report(task, [f"For {case['label']}: {error}"])
        try:
            values = np.asarray(output, dtype=float).ravel()
        except (TypeError, ValueError):
            return _report(task, [f"Return one probability per score, received "
                                  f"{type(output).__name__}."])
        if values.shape != (len(case["expected"]),):
            return _report(task, [f"Return one probability per score: expected "
                                  f"{len(case['expected'])}, received {values.size}."])
        if np.allclose(values, case["expected"], atol=2e-3):
            continue
        hint = next((HINTS[m["id"]] for m in case["mistakes"]
                     if np.allclose(values, m["output"], atol=2e-3)), None)
        return _report(task, [f"The probabilities for {case['label']} don't match.",
                              hint or "Check which scores you fit on and which you transform."])
    return _report(task, [])


@_safe("recommendation")
def recommendation(record):
    """Task 7: the written recommendation. The checker sees only that it is complete."""
    task = "recommendation"
    if not isinstance(record, dict):
        return _report(task, [f"Expected a dict, received {type(record).__name__}."])
    if _unfinished(record):
        return _report(task, ["..."], attempted=False)
    problems = []
    if set(record) != set(RECOMMENDATION_FIELDS):
        problems.append(f"Use exactly the keys {', '.join(RECOMMENDATION_FIELDS)}.")
    model = record.get("model")
    if not (isinstance(model, str) and model in MODELS):
        problems.append(f"model should be one of {', '.join(repr(m) for m in MODELS)}.")
    threshold = _number(record.get("threshold"))
    if threshold is None or not 0 < threshold < 1:
        problems.append("threshold should be a probability between 0 and 1.")
    for field in ("reasoning", "reopen_if"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            problems.append(f"Write {field} as a sentence or two.")
    result = _report(task, problems)
    if result:
        print("  Your demonstrator will read it; the checker can't judge the reasoning.")
    return result


@_safe("cheapest_within_capacity")
def cheapest_within_capacity(function):
    """Stretch: the cheapest policy that stays within a limit on flags."""
    task = "cheapest_within_capacity"
    early = _start(task, function)
    if early is not None:
        return early
    for case in _fixtures()[task]["cases"]:
        table = pd.DataFrame(case["table"], columns=TABLE_FIELDS)
        output, error = _call(function, table.copy(), case["max_flags"])
        if error:
            return _report(task, [error])
        values, problem = _record(output, TABLE_FIELDS)
        if problem:
            return _report(task, [problem])
        if not _same(values, case["expected"], TABLE_FIELDS):
            hint = next((HINTS[m["id"]] for m in case["mistakes"]
                         if _same(values, m["output"], TABLE_FIELDS)), None)
            return _report(task, [hint or "Check your answer against the table by hand."])
    return _report(task, [])


@_safe("share_of_days_over")
def share_of_days_over(function):
    """Extension B: the share of days on which a policy's flags exceed capacity."""
    task = "share_of_days_over"
    early = _start(task, function)
    if early is not None:
        return early
    for case in _fixtures()[task]["cases"]:
        output, error = _call(function, case["flags_per_1000"], case["strips"], case["reviews"])
        if error:
            return _report(task, [error])
        value = _number(output)
        if value is None:
            return _report(task, [f"Return one number, received {type(output).__name__}."])
        if abs(value - case["expected"]) <= case["tolerance"]:
            continue
        hint = next((HINTS[m["id"]] for m in case["mistakes"]
                     if abs(value - m["output"]) <= case["tolerance"]), None)
        if value > 1:
            hint = HINTS["count_not_share"]
        return _report(task, [f"For {case['flags_per_1000']} flags per 1,000, {case['strips']} strips "
                              f"and {case['reviews']} reviews a day, received {value:.3f}.",
                              hint or "Check how you simulate a day and what you count."])
    return _report(task, [])


@_safe("cheapest_reliable")
def cheapest_reliable(function):
    """Extension C: the cheapest policy that rarely overflows."""
    task = "cheapest_reliable"
    early = _start(task, function)
    if early is None:
        early = _dependencies(task, function, "share_of_days_over")
    if early is not None:
        return early
    for case in _fixtures()[task]["cases"]:
        table = pd.DataFrame(case["table"], columns=TABLE_FIELDS)
        output, error = _call(function, table.copy(), case["strips"], case["reviews"], case["max_share"])
        if error:
            return _report(task, [error])
        values, problem = _record(output, TABLE_FIELDS)
        if problem:
            return _report(task, [problem])
        if not _same(values, case["expected"], TABLE_FIELDS):
            hint = next((HINTS[m["id"]] for m in case["mistakes"]
                         if _same(values, m["output"], TABLE_FIELDS)), None)
            return _report(task, [hint or "Check your answer against share_of_days_over for each "
                                          "candidate."])
    return _report(task, [])


CHECKERS = {"outcome_counts": outcome_counts, "policy_cost": policy_cost,
            "roc_points": roc_points, "price_cutoffs": price_cutoffs,
            "threshold_probability": threshold_probability, "net_benefit": net_benefit,
            "recalibrate": recalibrate, "recommendation": recommendation,
            "cheapest_within_capacity": cheapest_within_capacity,
            "share_of_days_over": share_of_days_over, "cheapest_reliable": cheapest_reliable}


def _quiet(name, value):
    """Run a checker without printing and without strict mode."""
    previous = _QUIET["active"]
    _QUIET["active"] = True
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return bool(CHECKERS[name](value))
    except Exception:
        return False
    finally:
        _QUIET["active"] = previous


def ready(**answers):
    """True when every named answer passes its checker; otherwise say which task is waiting."""
    for name, value in answers.items():
        if not _quiet(name, value):
            print(f"Waiting for {TASKS[name]}. Finish it above, then run this cell again.")
            return False
    return True


def progress(namespace):
    """Which tasks pass, from the notebook's own variables. Pass ``globals()``."""
    rows = []
    for name, label in TASKS.items():
        if name not in namespace:
            status = "not run yet"
        elif _unfinished(namespace[name]):
            status = "not attempted yet"
        elif _quiet(name, namespace[name]):
            status = "written" if name == "recommendation" else "✓"
        else:
            status = "not yet"
        rows.append((label, status))
    core = [status for label, status in rows if not label.startswith("Extension")]
    done = sum(status in ("✓", "written") for status in core)
    width = max(len(label) for label, _ in rows)
    for label, status in rows:
        print(f"{label:<{width}}  {status}")
    print(f"\n{done} of {len(core)} core answers complete.")
