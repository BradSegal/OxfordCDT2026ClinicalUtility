"""Reference decisions and consequences; prices are illustrative loss units.

The functions are intentionally readable. A recording is the unit of analysis.
There is no physiological, treatment-outcome or window-localisation model.
"""

import math

import numpy as np

DEFAULT_COSTS = {"TP": 1.0, "FP": 1.0, "FN": 5.0, "TN": 0.0}


def should_review(score, threshold):
    """Review when the recording score meets the chosen threshold."""
    return score >= threshold


def decision_cost(actual_af, review, costs):
    """Look up the realised loss in the four-cell decision matrix."""
    if review:
        outcome = "TP" if actual_af else "FP"
    else:
        outcome = "FN" if actual_af else "TN"
    return costs[outcome]


def _finite_scores(records):
    """Scores as a float array; NaN or infinite scores are refused, never silently missed."""
    scores = np.array([float(row["score"]) for row in records])
    if not np.isfinite(scores).all():
        raise ValueError("Every score must be a finite number; remove or repair NaN or infinite scores.")
    return scores


def evaluate_policy(records, threshold, costs, cost_function=decision_cost):
    """Return a decision ledger, using the supplied scoring function on every row."""
    _finite_scores(records)
    ledger = []
    for record in records:
        review = should_review(record["score"], threshold)
        actual = record["af_present"]
        outcome = ("TP" if actual else "FP") if review else ("FN" if actual else "TN")
        loss = cost_function(actual, review, costs)
        if loss is Ellipsis or loss is None:
            raise ValueError("The cost function is unfinished; return a number for every outcome.")
        if not isinstance(loss, (int, float, np.number)) or not math.isfinite(float(loss)):
            raise ValueError(f"The cost function returned {loss!r}; return a finite loss.")
        ledger.append({**record, "review": review, "outcome": outcome, "loss": float(loss)})
    return ledger


def policy_summary(records, threshold, costs, cost_function=decision_cost):
    """Recordings, reviews, missed AF and loss of one fixed policy on one set of records."""
    counts = confusion_counts(evaluate_policy(records, threshold, costs, cost_function))
    total = float(sum(counts[k] * cost_function(actual, review, costs) for k, actual, review in (
        ("TP", True, True), ("FP", False, True), ("FN", True, False), ("TN", False, False))))
    return {"recordings": len(records), "reviews": counts["TP"] + counts["FP"],
            "missed_af": counts["FN"], "total_loss": total, "mean_loss": total / len(records)}


def confusion_counts(ledger):
    """Counts use reference labels, not assumptions about subsequent diagnosis."""
    counts = dict.fromkeys(("TP", "FP", "FN", "TN"), 0)
    for row in ledger:
        counts[row["outcome"]] += 1
    return counts


def divide(numerator, denominator):
    """Undefined ratios remain undefined rather than becoming zero."""
    return numerator / denominator if denominator else float("nan")


def metrics(counts):
    tp, fp, fn, tn = (counts[k] for k in ("TP", "FP", "FN", "TN"))
    se, sp = divide(tp, tp + fn), divide(tn, tn + fp)
    return {
        "sensitivity": se, "specificity": sp, "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn), "youden_j": se + sp - 1,
        "reviews": tp + fp, "missed_af": fn,
    }


def matrix_loss(counts, costs):
    return sum(counts[k] * costs[k] for k in ("TP", "FP", "FN", "TN"))


def expected_cost(probability, action, costs):
    """Expected loss before knowing the recording's reference outcome."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if action == "review":
        return probability * costs["TP"] + (1 - probability) * costs["FP"]
    if action == "no_review":
        return probability * costs["FN"] + (1 - probability) * costs["TN"]
    raise ValueError("Action must be 'review' or 'no_review'.")


def probability_threshold(costs):
    """Indifference probability when each state has the intended preferred action."""
    benefit = costs["FN"] - costs["TP"]
    harm = costs["FP"] - costs["TN"]
    if benefit <= 0 or harm <= 0:
        raise ValueError("This formula requires FN > TP and FP > TN; compare actions directly.")
    return harm / (benefit + harm)


def threshold_grid(records):
    """All distinct decisions, including review all and review none."""
    return sorted({0.0, 1.01, *(float(row["score"]) for row in records)})


def sweep_thresholds(records, costs, cost_function=decision_cost, thresholds=None):
    """Exact operating points by default; a supplied grid is an explicit approximation.

    Prices are evaluated once for each outcome using the learner's function;
    cumulative counts then give exactly the same loss as a full per-row ledger.
    """
    if not records:
        raise ValueError("At least one assessable recording is needed.")
    # Call the learner on each outcome. This keeps plots coupled to their code.
    effective = {key: cost_function(actual, review, costs) for key, actual, review in (
        ("TP", True, True), ("FP", False, True), ("FN", True, False), ("TN", False, False))}
    if any(v is None or v is Ellipsis for v in effective.values()):
        raise ValueError("Finish all four outcomes in the cost function first.")
    if not all(isinstance(v, (int, float, np.number)) and math.isfinite(float(v))
               for v in effective.values()):
        raise ValueError("The cost function must return finite numbers.")
    scores = _finite_scores(records)
    truth = np.array([r["af_present"] for r in records], dtype=bool)
    order = np.argsort(scores)
    ordered = scores[order]
    prefix = np.r_[0, np.cumsum(truth[order])]
    positives, n = int(truth.sum()), len(records)
    rows = []
    for threshold in threshold_grid(records) if thresholds is None else thresholds:
        below = int(np.searchsorted(ordered, threshold, side="left"))
        fn = int(prefix[below])
        counts = {"TP": positives - fn, "FN": fn,
                  "TN": below - fn, "FP": n - positives - below + fn}
        rows.append({"threshold": float(threshold), **counts, **metrics(counts),
                     "total_loss": float(matrix_loss(counts, effective)),
                     "mean_loss": float(matrix_loss(counts, effective)) / n})
    return rows


def best_thresholds(rows, objective="total_loss", minimise=True):
    """Return every tied grid optimum; there need not be one uniquely best cutoff."""
    finite = [r for r in rows if math.isfinite(r[objective])]
    if not finite:
        raise ValueError("No defined values for this objective.")
    target = (min if minimise else max)(r[objective] for r in finite)
    return [r for r in finite if math.isclose(r[objective], target, abs_tol=1e-10)]


def projected_counts(counts, prevalence, population=1000):
    """Change prevalence only, holding estimated sensitivity and specificity fixed."""
    if not 0 < prevalence < 1 or population <= 0:
        raise ValueError("Use 0 < prevalence < 1 and a positive population.")
    m = metrics(counts)
    if not math.isfinite(m["sensitivity"]) or not math.isfinite(m["specificity"]):
        raise ValueError("Both AF and non-AF records are needed to project a population.")
    positives, negatives = population * prevalence, population * (1 - prevalence)
    return {"TP": positives * m["sensitivity"], "FN": positives * (1 - m["sensitivity"]),
            "FP": negatives * (1 - m["specificity"]), "TN": negatives * m["specificity"]}


def net_benefit(counts, threshold_probability):
    p = threshold_probability
    if not 0 < p < 1:
        raise ValueError("Decision-curve threshold probability must be strictly between 0 and 1.")
    return divide(counts["TP"] - counts["FP"] * p / (1 - p), sum(counts.values()))


def decision_curve(records, preferences):
    return [{"preference": float(p), "net_benefit": net_benefit(
        confusion_counts(evaluate_policy(records, p, DEFAULT_COSTS)), p)} for p in preferences]


def calibration_summary(records, bins=8):
    """Quantile bins and Brier score; small bins are descriptive, not proof of calibration."""
    if not records:
        raise ValueError("No records to assess.")
    ordered = sorted(records, key=lambda row: row["score"])
    groups = np.array_split(np.arange(len(ordered)), min(bins, len(ordered)))
    points = []
    for group in groups:
        rows = [ordered[int(i)] for i in group]
        points.append({"predicted": float(np.mean([r["score"] for r in rows])),
                       "observed": float(np.mean([r["af_present"] for r in rows])),
                       "n": len(rows)})
    brier = float(np.mean([(r["score"] - r["af_present"]) ** 2 for r in records]))
    return {"points": points, "brier": brier}
