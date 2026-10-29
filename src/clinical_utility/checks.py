"""Specific, non-blocking feedback for the learner's calculations.

Every checker prints one verdict and returns a ``CheckResult`` that is truthy
when the answer is right. ``✓`` means correct, ``Not yet`` names the likely
mistake and ``Not attempted yet`` means the ``...`` placeholder is still there.
Two predictions are hypotheses whose premises come later: their checkers only
record a valid answer (``Recorded``), and a ``reconcile_`` function prints the
result beside it once the reveal cell has run.
A checker never raises in a student notebook. With ``UTILITY_STRICT=1`` (set
when the worked notebooks are executed automatically) a failure raises
``AssertionError`` instead, so an incorrect worked answer cannot go unnoticed.

Expected answers come from ``reference``; no price or count is restated here.
"""

import contextlib
import functools
import io
import inspect
import math
import numbers
import os

from . import data, reference

MAX_PROBLEMS_SHOWN = 3
OUTCOMES = (("TP", True, True), ("FP", False, True), ("FN", True, False), ("TN", False, False))
# No two cells share a price and none equals its default, so each mistake is identifiable.
UNUSUAL_COSTS = {"TP": 2.0, "FP": 3.0, "FN": 11.0, "TN": 0.5}


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


def _near(actual, expected, tolerance=1e-9):
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


def _choice(name, answer, expected, hints, options=None):
    """A Predict answered with one of a few words; ``hints`` maps each wrong option to its hint."""
    options = tuple(options or (expected, *hints))
    if answer is Ellipsis:
        return _waiting(name)
    if not isinstance(answer, str) or answer.strip().lower() not in options:
        listed = ", ".join(repr(o) for o in options)
        return _report(name, [f"Answer with one of {listed}, as a string in quotes."])
    answer = answer.strip().lower()
    if answer == expected:
        return _report(name, [])
    return _report(name, [hints[answer]])


# --- Section 1 ---------------------------------------------------------------

PREDICT_MISS_PRICE = 10.0


def predicted_threshold(answer):
    """Problem 1: the threshold probability p_t when C_FN = 10 and the other prices are unchanged."""
    costs = {**reference.DEFAULT_COSTS, "FN": PREDICT_MISS_PRICE}
    expected = reference.probability_threshold(costs)
    original = reference.probability_threshold(reference.DEFAULT_COSTS)
    return _value("Threshold when a miss costs more", answer, expected, hints=(
        (original, "That is the threshold at the original prices. A dearer miss makes 'no "
                   "review' more expensive at every probability, so the crossing point moves."),
        (reference.probability_threshold({**costs, "TP": 0.0}),
         "You set C_TP to 0. A review takes the same clinician time when it finds AF, so the "
         "expected loss of review is 1 at every probability."),
        (1 - expected, "That is the probability of no AF at the crossing point. Solve for p, "
                       "the AF probability."),
        (2 * original, "The threshold moved the wrong way. When a miss costs more, weaker "
                       "evidence of AF is enough to justify a review."),
        (100 * expected, "Give the threshold as a proportion between 0 and 1, not a percentage."),
    ), default_hint="Set the expected loss of review equal to that of no review and solve for p.")


# --- Section 2 ---------------------------------------------------------------


def calls_at_half(reviews, misses):
    """Problem 2: reviews and missed AF among the six cases at a cutoff of 0.5."""
    name = "Calls at 0.5"
    if reviews is Ellipsis or misses is Ellipsis:
        return _waiting(name)
    counts = reference.confusion_counts(
        reference.evaluate_policy(data.teaching_cases(), .5, reference.DEFAULT_COSTS))
    problems = []
    if not _matches(reviews, counts["TP"] + counts["FP"]):
        if _matches(reviews, counts["TP"]):
            problems.append("You counted only the AF recordings that were reviewed. Teaching B "
                            "has no AF but its score meets the cutoff, so it is reviewed too.")
        else:
            problems.append("Count every recording whose score is at or above 0.5, whatever its "
                            "reference label.")
    if not _matches(misses, counts["FN"]):
        if _matches(misses, 0):
            problems.append("Teaching E contains AF but scores below 0.5, so it is missed.")
        elif _matches(misses, counts["FN"] + counts["TN"]):
            problems.append("A miss is an AF recording that is not reviewed. Unreviewed "
                            "recordings without AF are true negatives.")
        else:
            problems.append("Count the AF recordings whose score is below 0.5.")
    return _report(name, problems)


def npv_review_none(answer):
    """Problem 3: NPV when no development recording is reviewed."""
    name = "NPV when nobody is reviewed"
    rows = data.load_records("random_forest", "development")
    counts = reference.confusion_counts(reference.evaluate_policy(rows, 1.01, reference.DEFAULT_COSTS))
    m = reference.metrics(counts)
    if _is_number(answer) and answer != answer:
        return _report(name, [
            "PPV is undefined here because nobody is reviewed. NPV is defined: every recording "
            "is unreviewed, so its denominator is the whole cohort."])
    return _value(name, answer, m["npv"], hints=(
        (counts["FN"] / counts["TN"],
         "You divided misses by true negatives. NPV is TN/(TN + FN): the AF-negative fraction "
         "of everyone not reviewed."),
        (1 - m["npv"], "That is the fraction of unreviewed recordings that contain AF. NPV is "
                       "the fraction that do not."),
        (100 * m["npv"], "Give NPV as a proportion between 0 and 1, not a percentage."),
        (m["specificity"], "That is specificity, TN/(TN + FP). NPV conditions on the decision: "
                           "TN/(TN + FN)."),
    ), default_hint="NPV = TN/(TN + FN), and every development recording is unreviewed.")


# --- Sections 3 and 6: pricing all four cells -------------------------------


def _cell_hint(key, got, costs):
    other = [k for k, v in costs.items() if k != key and _matches(got, v)]
    partner = {"FP": "FN", "FN": "FP"}.get(key)
    if not _is_number(got):
        return "Return a number from the costs dictionary."
    if key == "TP" and _matches(got, 0):
        return "A correct positive still uses a review: return costs['TP'], not 0."
    if key == "TN" and _matches(got, 0):
        return f"Read the TN price from the matrix; it is not always zero (here {costs['TN']!r})."
    if costs is not reference.DEFAULT_COSTS and _matches(got, reference.DEFAULT_COSTS[key]):
        return (f"You returned the default {key} price. Read prices from the costs argument so "
                "the function works for any matrix.")
    if partner in other:
        return "FP and FN are swapped. FP is a review without AF; FN is AF with no review."
    if other:
        return f"That is the {other[0]} price. Check which branch handles {key}."
    return f"The {key} branch should return costs['{key}'] unchanged."


def _price_cells(name, function, cost_sets):
    if function is Ellipsis:
        return _waiting(name)
    if not callable(function):
        return _not_function(name, function)
    problems = []
    for costs in cost_sets:
        for key, actual, review in OUTCOMES:
            try:
                got = function(actual, review, dict(costs))
            except Exception as error:
                return _report(name, [f"Calling it raised {type(error).__name__}: {error}. Use "
                                      "the keys 'TP', 'FP', 'FN' and 'TN' of costs."])
            if got is Ellipsis:
                return _waiting(name)
            if not _matches(got, costs[key]):
                problems.append(f"With costs {costs}, actual_af={actual} and review={review} "
                                f"({key}) returned {got!r}; expected {costs[key]!r}. "
                                + _cell_hint(key, got, costs))
    return _report(name, problems)


def decision_cost(function):
    """Problem 4: all four cells, at the default and at unusual prices."""
    return _price_cells("Decision-cost function", function,
                        (reference.DEFAULT_COSTS, UNUSUAL_COSTS))


# --- Section 4 ---------------------------------------------------------------


def policy(function):
    """Problem 5: an inspectable ledger: review when score >= cutoff, priced by decision_cost."""
    name = "Policy ledger"
    if function is Ellipsis:
        return _waiting(name)
    if not callable(function):
        return _not_function(name, function)
    records = data.teaching_cases()
    for costs in (reference.DEFAULT_COSTS, UNUSUAL_COSTS):
        # 0.25 is Teaching E's score, so a strict > comparison is caught.
        for threshold in (.5, .25, .2, 1.01):
            try:
                ledger = function([dict(r) for r in records], threshold, dict(costs))
            except Exception as error:
                return _report(name, [f"{type(error).__name__}: {error}. Read record['score'] "
                                      "and record['af_present'] from each dictionary."])
            if ledger is Ellipsis:
                return _waiting(name)
            if ledger is None:
                return _report(name, ["It returned None. Return the ledger after the loop."])
            if not isinstance(ledger, list) or len(ledger) != len(records):
                return _report(name, ["Return a list with one dictionary per recording. Check "
                                      "that `return` is after the loop, not inside it."])
            for row, answer in zip(ledger, reference.evaluate_policy(records, threshold, costs)):
                if not isinstance(row, dict) or not {"recording", "review", "loss"} <= set(row):
                    return _report(name, ["Each entry needs the keys 'recording', 'review' and "
                                          "'loss'."])
                where = f"{answer['recording']} at cutoff {threshold} with costs {costs}"
                if row["recording"] != answer["recording"]:
                    return _report(name, ["Keep the records in their original order."])
                if not _matches(row["review"], answer["review"]):
                    return _report(name, [f"{where}: review should be {answer['review']}. Review "
                                          "when record['score'] >= threshold: a score exactly at "
                                          "the cutoff is reviewed."])
                if not _near(row["loss"], answer["loss"]):
                    return _report(name, [f"{where}: loss should be {answer['loss']:g}. Pass "
                                          "af_present, the review decision and the costs argument "
                                          "to decision_cost."])
    return _report(name, [])


@functools.lru_cache(maxsize=1)
def _development_sweep():
    return reference.sweep_thresholds(data.load_records("random_forest", "development"),
                                      reference.DEFAULT_COSTS)


def _hypothesis(name, answer, options):
    """Record a prediction without marking it; the result is revealed and compared later."""
    if answer is Ellipsis:
        return _waiting(name)
    if not isinstance(answer, str) or answer.strip().lower() not in options:
        listed = ", ".join(repr(o) for o in options)
        return _report(name, [f"Answer with one of {listed}, as a string in quotes."])
    print(f"Recorded: {name}, {answer.strip().lower()!r}. It is not marked. Run the next cell "
          "and compare the result with it.")
    return CheckResult(True)


def _reconcile(answer, expected, result, reasons):
    """Print the revealed result beside the learner's recorded prediction."""
    given = answer.strip().lower() if isinstance(answer, str) else None
    if given == expected:
        print(f"Your prediction, {given!r}, matches the result: {result}")
    elif given in reasons:
        print(f"You predicted {given!r}; the result is {expected!r}: {result} {reasons[given]}")
    else:
        print(f"No prediction was recorded. The result: {result}")
    return CheckResult(given == expected)


J_OPTIONS = ("lower", "higher", "same")


def _j_relation():
    rows = _development_sweep()
    j_row = reference.best_thresholds(rows, "youden_j", minimise=False)[0]
    loss_row = reference.best_thresholds(rows)[0]
    relation = ("lower" if j_row["threshold"] < loss_row["threshold"] else
                "higher" if j_row["threshold"] > loss_row["threshold"] else "same")
    return relation, j_row, loss_row


def j_cutoff(answer):
    """Problem 7: record a prediction of J's cutoff against the minimum-loss cutoff."""
    return _hypothesis("Prediction of J's cutoff", answer, J_OPTIONS)


def reconcile_j_cutoff(answer):
    """After Problem 7's reveal: the result beside the recorded prediction, with the reason."""
    expected, j_row, loss_row = _j_relation()
    ratio = (j_row["FP"] + j_row["TN"]) / (j_row["TP"] + j_row["FN"])
    benefit = reference.DEFAULT_COSTS["FN"] - reference.DEFAULT_COSTS["TP"]
    result = (f"J's cutoff, {j_row['threshold']:.3f}, is {expected} than the minimum-loss cutoff, "
              f"{loss_row['threshold']:.3f}.")
    reason = (f"J prices one miss at n_neg/n_pos = {ratio:.1f} false referrals; the matrix prices "
              f"it at C_FN − C_TP = {benefit:g}.")
    return _reconcile(answer, expected, result, {
        "higher": reason + " A dearer miss justifies reviewing on weaker evidence: a lower cutoff.",
        "lower": reason + " Here that makes J's cutoff higher.",
        "same": "The two objectives price a miss differently. " + reason,
    })


def _counts(tp, fp, fn, tn):
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def youden_j(function):
    """Problem 6: J = sensitivity + specificity − 1 from four counts."""
    name = "Youden's J"
    if function is Ellipsis:
        return _waiting(name)
    if not callable(function):
        return _not_function(name, function)
    # Ordered so the first case that fails also tells the misconceptions apart.
    for case in (_counts(2, 1, 1, 2), _counts(260, 150, 81, 3220), _counts(3, 2, 0, 1),
                 _counts(9, 1, 1, 89)):
        m = reference.metrics(case)
        se, sp = m["sensitivity"], m["specificity"]
        try:
            got = function(dict(case))
        except Exception as error:
            return _report(name, [f"Calling it raised {type(error).__name__}: {error}. Read "
                                  "counts['TP'], counts['FP'], counts['FN'] and counts['TN']."])
        if got is Ellipsis:
            return _waiting(name)
        if _matches(got, m["youden_j"]):
            continue
        wrong = (
            (se + sp, "You added sensitivity and specificity but did not subtract 1."),
            (se + m["ppv"] - 1, "You used PPV in place of specificity. Specificity is "
                                "TN/(TN + FP): it conditions on the reference, not the decision."),
            (m["ppv"] + m["npv"] - 1, "You used the predictive values. J uses sensitivity "
                                      "TP/(TP + FN) and specificity TN/(TN + FP)."),
            (se + (1 - sp) - 1, "You used the false-positive rate, FP/(FP + TN), in place of "
                                "specificity, TN/(TN + FP)."),
        )
        hint = next((h for w, h in wrong if _matches(got, w)),
                    "Sensitivity is TP/(TP + FN) and specificity is TN/(TN + FP).")
        return _report(name, [f"For {case}, expected {m['youden_j']:.4f}, received {got!r}. {hint}"])
    return _report(name, [])


# --- Section 5 ---------------------------------------------------------------

FLIP_PRICES = range(2, 31)


@functools.lru_cache(maxsize=1)
def first_flip_price():
    """First whole C_FN (C_TP=1, C_FP=1, C_TN=0) at which the raw ResNet's minimum dev loss <= the forest's."""
    rows = [reference.sweep_thresholds(data.load_records(model, "development"),
                                       reference.DEFAULT_COSTS)
            for model in ("random_forest", "resnet")]
    for miss in FLIP_PRICES:
        costs = {**reference.DEFAULT_COSTS, "FN": float(miss)}
        forest, resnet = (min(reference.matrix_loss(row, costs) for row in r) for r in rows)
        if resnet <= forest:
            return miss
    return None


def flip_price(answer):
    """Problem 8: the price at which the preference flips from the forest to the ResNet."""
    name = "Price at which the preference flips"
    flip = first_flip_price()
    if answer is None:
        return _report(name, ["Your loop never set the answer. The flip is where the ResNet's "
                              "minimum loss becomes at most the forest's."])
    return _value(name, answer, flip, hints=(
        (flip + 1, "At the price just below your answer the two minimum losses are exactly "
                   "equal. Use <= so a tie counts as the point where the forest stops being "
                   "preferred."),
        (flip - 1, "At that price the forest is still strictly cheaper. Stop at the first price "
                   "where the ResNet is no worse."),
        (FLIP_PRICES[0], "Your loop stopped on its first price. Compare the ResNet with the "
                         "forest and break only when the ResNet is no worse."),
        (FLIP_PRICES[-1], "Your loop ran to the end without breaking. Check the direction of the "
                          "comparison and where the break sits."),
    ), default_hint="Loop over whole-number miss prices with C_TP=1, C_FP=1 and C_TN=0, and stop "
                    "at the first price where the ResNet's minimum loss is at most the forest's.")


def flip_on_new_sample(answer):
    """Problem 9: would the flip price move on another sample of the same size?"""
    return _choice("Flip price on another sample", answer, "could_move", {
        "same": "The development split has a few hundred AF recordings, and near the flip the two "
                "minimum losses differ by only a few units. Another sample of the same size could "
                "move the flip. The next cell shows how far."},
        options=("same", "could_move"))


# --- Section 6 ---------------------------------------------------------------


def colleague_cutoff(answer):
    """Problem 10: is the colleague's chosen cutoff lower, equal or higher than yours?"""
    records = data.load_records("random_forest", "development")

    def free_correct_decisions(actual_af, review, costs):
        if actual_af and not review:
            return costs["FN"]
        if review and not actual_af:
            return costs["FP"]
        return 0

    theirs = reference.best_thresholds(
        reference.sweep_thresholds(records, reference.DEFAULT_COSTS, free_correct_decisions))[0]
    yours = reference.best_thresholds(_development_sweep())[0]
    expected = ("lower" if theirs["threshold"] < yours["threshold"] else
                "higher" if theirs["threshold"] > yours["threshold"] else "equal")
    reason = ("The colleague's scorer charges nothing for a correct review, so every review "
              "looks cheaper than it is.")
    return _choice("The colleague's cutoff", answer, expected, {
        "higher": reason + " Cheaper reviews pull the cutoff down, not up.",
        "equal": reason + " That changes the balance between reviewing and not reviewing, so the "
                          "cheapest cutoff can move.",
        "lower": reason + " Here the cutoff does not move down.",
    } if expected != "lower" else {
        "higher": reason + " Cheaper reviews pull the cutoff down, not up.",
        "equal": reason + " That changes the balance between reviewing and not reviewing, so the "
                          "cheapest cutoff can move.",
    }, options=("lower", "equal", "higher"))


def colleague_cost(function):
    """Problem 11: the repaired scorer must price all four cells from any matrix."""
    return _price_cells("Repaired colleague's scorer", function, (UNUSUAL_COSTS,))


# --- Section 7 ---------------------------------------------------------------


def net_benefit(function):
    """Problem 12: NB = TP/n − FP/n · p_t/(1 − p_t)."""
    name = "Net benefit"
    if function is Ellipsis:
        return _waiting(name)
    if not callable(function):
        return _not_function(name, function)
    records = data.teaching_cases()
    cases = []
    for threshold in (.5, .2):
        counts = reference.confusion_counts(
            reference.evaluate_policy(records, threshold, reference.DEFAULT_COSTS))
        cases += [(counts, p) for p in (.2, .1, .4)]
    cases.append((_counts(260, 150, 81, 3220), .2))
    for counts, p in cases:
        expected = reference.net_benefit(counts, p)
        n, tp, fp, fn = sum(counts.values()), counts["TP"], counts["FP"], counts["FN"]
        try:
            got = function(dict(counts), p)
        except Exception as error:
            return _report(name, [f"Calling it raised {type(error).__name__}: {error}. Read the "
                                  "four counts and the threshold probability p_t."])
        if got is Ellipsis:
            return _waiting(name)
        if _matches(got, expected):
            continue
        wrong = (
            (tp - fp * p / (1 - p), "Divide by n, the number of recordings, so net benefit is "
                                    "per recording."),
            ((tp - fp * p) / n, "Weight false positives by the odds p_t/(1 − p_t), not by p_t itself."),
            ((tp - fp * (1 - p) / p) / n, "The odds are inverted: use p_t/(1 − p_t)."),
            ((tp - fn * p / (1 - p)) / n, "Net benefit charges false positives (unnecessary "
                                          "reviews), not misses."),
            ((tp - fp * p / (1 - p)) / (tp + fp), "Divide by every recording, n = TP + FP + FN + "
                                                  "TN, not only those reviewed."),
        )
        hint = next((h for w, h in wrong if _matches(got, w)), "NB = TP/n − (FP/n) × p_t/(1 − p_t).")
        return _report(name, [f"For {counts} at p_t={p}, expected {expected:.4f}, received {got!r}. "
                              + hint])
    return _report(name, [])


def _highest_net_benefit_model():
    table = data.load_arrays("development")
    from .comparison import policy_net_benefit
    values = {model: policy_net_benefit(table[model], table["af_present"], .2)
              for model in data.MODEL_NAMES}
    return max(values, key=values.get)


def highest_net_benefit(answer):
    """Problem 13: record a prediction of the model with the highest net benefit at p_t = 0.2."""
    return _hypothesis("Prediction of the highest net benefit at p_t = 0.2", answer,
                       tuple(data.MODEL_NAMES))


def reconcile_highest_net_benefit(answer):
    """After Problem 13's reveal: the result beside the recorded prediction, with the reason."""
    expected = _highest_net_benefit_model()
    reasons = {
        "resnet": "The raw ResNet's scores are too high, so a stated 0.2 sends far more "
                  "recordings for review than its best cutoff would.",
        "resnet_calibrated": "Recalibration puts the ResNet's 0.2 close to its best cutoff, but "
                             "Section 5 found the forest's best policy cheaper at C_FN = 5, which "
                             "is the matrix behind p_t = 0.2.",
        "random_forest": "At this preference another model is higher; compare the three at the "
                         "same p_t.",
    }
    result = f"{data.MODEL_NAMES[expected]} has the highest net benefit at p_t = 0.2."
    return _reconcile(answer, expected, result, {k: v for k, v in reasons.items() if k != expected})


# --- Section 8 ---------------------------------------------------------------

REFLECTION_MIN_WORDS = 40


def frozen_policy(model, threshold, costs, predicted_loss):
    """Problem 14: a complete frozen policy and a predicted evaluation loss, before the reveal.

    The checker cannot judge the choice or the prediction. It checks that the
    policy is complete and catches two slips: predicting the unscaled
    development total, and predicting a loss per recording.
    """
    name = "Frozen policy"
    if any(value is Ellipsis for value in (model, threshold, costs, predicted_loss)):
        return _waiting(name)
    problems = []
    if model not in data.MODEL_NAMES:
        problems.append(f"Set final_model to one of {', '.join(repr(m) for m in data.MODEL_NAMES)}.")
    if not _is_number(threshold) or not 0 <= threshold <= 1.01:
        problems.append("Set final_threshold to a score cutoff between 0 and 1.01, chosen on the "
                        "development split.")
    if not isinstance(costs, dict) or set(costs) != {"TP", "FP", "FN", "TN"} or not all(
            _is_number(v) and math.isfinite(v) for v in costs.values()):
        problems.append("Set final_costs to a dictionary of four finite prices with the keys "
                        "'TP', 'FP', 'FN' and 'TN', for example dict(costs).")
    if not _is_number(predicted_loss) or not math.isfinite(predicted_loss) or predicted_loss < 0:
        problems.append("Set predicted_evaluation_loss to one number: the total loss you expect "
                        "on the evaluation split.")
    if problems:
        return _report(name, problems)
    development = reference.policy_summary(data.load_records(model, "development"), threshold, costs)
    evaluation_size = len(data.load_records(model, "evaluation"))
    # Only a positive development total can be copied unscaled: a total of zero scales to zero.
    if (development["total_loss"] > 0 and _near(predicted_loss, development["total_loss"])
            and evaluation_size != development["recordings"]):
        return _report(name, [
            f"That is the development total over {development['recordings']:,} recordings. The "
            f"evaluation split has {evaluation_size:,}, so scale the total before predicting."])
    if development["total_loss"] > 0 and 0 < predicted_loss <= 10 * development["mean_loss"]:
        return _report(name, ["That looks like a loss per recording. Predict the total over the "
                              "evaluation recordings."])
    return _report(name, [], success="frozen. The checker cannot judge your choice or your "
                                     "prediction; the next cell reveals the evaluation result")


def reflection(answer):
    """Optional decision statement: records a substantive statement; it does not grade the valuation."""
    name = "Decision statement"
    if answer is Ellipsis:
        return _waiting(name)
    if not isinstance(answer, str) or len(answer.split()) < REFLECTION_MIN_WORDS:
        return _report(name, ["Write four or five sentences: the action, your prices, the policy, "
                              "the evidence for it and one assumption that would change it."])
    return _report(name, [], success="recorded. A keyword rule cannot judge a valuation; discuss "
                                     "the reasoning with your neighbour")


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
WAITED_FOR = {"decision_cost": ("decision_cost", "Problem 4", True),
              "youden_j": ("youden_j", "Problem 6", True),
              "repaired_colleague_cost": ("colleague_cost", "Problem 11", False),
              "net_benefit": ("net_benefit", "Problem 12", True)}


def ready(**answers):
    """True when every named answer passes its checker; prints one line per wait otherwise.

    ``check.ready(decision_cost=decision_cost)`` lets a later cell wait for an
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


def _reference_youden_j(counts):
    """Reference J: sensitivity + specificity − 1."""
    return reference.metrics(counts)["youden_j"]


REFERENCE_VERSIONS = {"decision_cost": reference.decision_cost,
                      "youden_j": _reference_youden_j,
                      "net_benefit": reference.net_benefit}


def use_reference(name):
    """Return the reference version of a function later cells depend on, and say so.

    For a pair that is stuck: ``decision_cost = check.use_reference("decision_cost")``
    lets the rest of the notebook run. Their own attempt is untouched until that
    line runs, so they can come back to it later.
    """
    if name not in REFERENCE_VERSIONS:
        listed = ", ".join(repr(k) for k in REFERENCE_VERSIONS)
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
            and _name not in ("first_flip_price", "ready", "use_reference", "waiting")):
        globals()[_name] = _guarded(_function)
del _name, _function
