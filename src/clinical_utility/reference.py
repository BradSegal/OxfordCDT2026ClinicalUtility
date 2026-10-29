"""Reference answers for the core notebook's decision calculations.

Counts come from scikit-learn's ``confusion_matrix`` and every operating point
from ``roc_curve``; uncertainty intervals come from ``scipy.stats.bootstrap``.
What no library provides is written out here because it is the point of the
practical: pricing a decision, the threshold that the prices imply, net
benefit, and choosing a policy by its loss.

Conventions: ``y`` is True for reference AF. A recording is reviewed when its
score is at or above the cutoff. Prices are in reviews: one review costs 1. An
undefined rate is NaN, never 0. Among tied minimum-loss cutoffs the one that
reviews fewest recordings is chosen.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve

DEFAULT_COSTS = {"TP": 1.0, "FP": 1.0, "FN": 5.0, "TN": 0.0}
SEED = 20261106
RESAMPLES = 2000
OUTCOMES = ("TP", "FP", "FN", "TN")


def threshold_probability(costs):
    """The AF probability at which review and no review have equal expected loss.

    p_t = harm / (harm + benefit), with harm = C_FP - C_TN and benefit = C_FN - C_TP.
    """
    harm, benefit = costs["FP"] - costs["TN"], costs["FN"] - costs["TP"]
    if harm <= 0 or benefit <= 0:
        raise ValueError("The threshold rule needs C_FP > C_TN and C_FN > C_TP; compare the "
                         "expected losses directly.")
    return harm / (harm + benefit)


def expected_loss(probability, action, costs):
    """Expected loss of one action for a recording with this AF probability."""
    if action == "review":
        return probability * costs["TP"] + (1 - probability) * costs["FP"]
    if action == "no_review":
        return probability * costs["FN"] + (1 - probability) * costs["TN"]
    raise ValueError("Action must be 'review' or 'no_review'.")


def counts(y, review):
    """TP, FP, FN and TN. ``confusion_matrix(...).ravel()`` returns them as tn, fp, fn, tp."""
    tn, fp, fn, tp = confusion_matrix(np.asarray(y, dtype=bool), np.asarray(review, dtype=bool),
                                      labels=[False, True]).ravel()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def operating_summary(y, review):
    """Reviews, AF detected and missed, and the four rates of one set of decisions."""
    y, review = np.asarray(y, dtype=bool), np.asarray(review, dtype=bool)
    c = counts(y, review)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        rates = {
            "sensitivity": recall_score(y, review, zero_division=np.nan),
            "specificity": recall_score(y, review, pos_label=False, zero_division=np.nan),
            "ppv": precision_score(y, review, zero_division=np.nan),
            "npv": precision_score(y, review, pos_label=False, zero_division=np.nan),
        }
    return {"reviews": c["TP"] + c["FP"], "af_detected": c["TP"], "af_missed": c["FN"],
            **{k: float(v) for k, v in rates.items()}}


def policy_cost(y, review, costs):
    """Total loss of a set of decisions: each count times its price."""
    c = counts(y, review)
    return float(sum(c[k] * costs[k] for k in OUTCOMES))


def operating_points(y, scores, costs):
    """Every distinct cutoff and its consequences, from ``roc_curve``.

    One row per cutoff, highest first: ``inf`` reviews nobody and the lowest
    score reviews everyone. Columns: threshold, TP, FP, FN, TN, reviews,
    af_missed, sensitivity, specificity, youden_j and loss.

    ``roc_curve`` needs both classes. With one class present the cutoffs are
    the distinct scores (plus ``inf``), counts come from ``confusion_matrix``,
    and the rates that need the missing class, and J, stay NaN.
    """
    y = np.asarray(y, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives, negatives = int(y.sum()), int((~y).sum())
    if positives and negatives:
        fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
        tp = np.rint(tpr * positives).astype(int)
        fp = np.rint(fpr * negatives).astype(int)
    else:
        thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
        tallies = [counts(y, scores >= cut) for cut in thresholds]
        tp = np.array([c["TP"] for c in tallies], dtype=int)
        fp = np.array([c["FP"] for c in tallies], dtype=int)
    table = pd.DataFrame({"threshold": thresholds, "TP": tp, "FP": fp,
                          "FN": positives - tp, "TN": negatives - fp})
    table["reviews"] = table["TP"] + table["FP"]
    table["af_missed"] = table["FN"]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["sensitivity"] = table["TP"] / positives if positives else np.nan
        table["specificity"] = table["TN"] / negatives if negatives else np.nan
    table["youden_j"] = table["sensitivity"] + table["specificity"] - 1
    table["loss"] = sum(table[k] * costs[k] for k in OUTCOMES).astype(float)
    return table


def cheapest(points):
    """The minimum-loss operating point; on ties, the one with fewest reviews."""
    return points.loc[points.sort_values(["loss", "reviews"], kind="stable").index[0]]


def per_1000(value, n):
    return 1000 * value / n


def net_benefit(y, review, p_t):
    """TP/n - FP/n * p_t / (1 - p_t): true-positive equivalents per recording."""
    if not 0 < p_t < 1:
        raise ValueError("The threshold probability must lie strictly between 0 and 1.")
    c = counts(y, review)
    n = sum(c.values())
    return (c["TP"] - c["FP"] * p_t / (1 - p_t)) / n if n else float("nan")


def decision_curve(frame, models, preferences, net_benefit_function=net_benefit):
    """Net benefit of reviewing when a model's score is at least p_t, for each p_t.

    Returns one row per preference with a column per model and the two
    reference strategies, review all and review none.
    """
    y = frame["af"].to_numpy()
    rows = []
    for p_t in preferences:
        row = {"preference": float(p_t)}
        for model in models:
            row[model] = net_benefit_function(y, frame[model].to_numpy() >= p_t, p_t)
        row["review all"] = net_benefit_function(y, np.ones(len(y), dtype=bool), p_t)
        row["review none"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(frame, models, costs):
    """Each model at its own minimum-loss cutoff under the same prices, per 1,000 recordings."""
    y, n, rows = frame["af"].to_numpy(), len(frame), []
    for model in models:
        best = cheapest(operating_points(y, frame[model], costs))
        rows.append({"model": model, "cutoff": float(best["threshold"]),
                     "reviews": per_1000(best["reviews"], n),
                     "af_detected": per_1000(best["TP"], n),
                     "af_missed": per_1000(best["FN"], n), "loss": float(best["loss"])})
    return pd.DataFrame(rows).set_index("model")


def miss_price_sensitivity(frame, models, base_costs, miss_prices):
    """Each model's cheapest policy at each price of a missed AF recording.

    One row per (miss price, model): cutoff, reviews and AF missed per 1,000,
    and the minimum loss under those prices.
    """
    y, n, rows = frame["af"].to_numpy(), len(frame), []
    points = {model: operating_points(y, frame[model], base_costs) for model in models}
    for miss in miss_prices:
        prices = {**base_costs, "FN": float(miss)}
        for model in models:
            table = points[model].assign(loss=sum(points[model][k] * prices[k] for k in OUTCOMES))
            best = cheapest(table)
            rows.append({"miss_price": miss, "model": model, "cutoff": float(best["threshold"]),
                         "reviews": per_1000(best["reviews"], n),
                         "af_missed": per_1000(best["FN"], n), "loss": float(best["loss"])})
    return pd.DataFrame(rows)


def first_flip(sensitivity, old, new):
    """First miss price at which ``new``'s minimum loss is no greater than ``old``'s, or None."""
    wide = sensitivity.pivot(index="miss_price", columns="model", values="loss")
    flips = wide.index[wide[new] <= wide[old]]
    return flips[0] if len(flips) else None


def paired_bootstrap(data, statistic, resamples=RESAMPLES, seed=SEED):
    """``scipy.stats.bootstrap`` with paired resampling and percentile intervals.

    With very few recordings some resamples have an undefined or constant
    statistic; scipy then warns, and the interval is degenerate or NaN. That is
    the honest answer for such data, so the warning is not passed on. With
    fewer than two recordings there is nothing to resample: the interval is
    returned as NaN (``intervals_available`` is False).
    """
    if len(data[0]) < 2:
        estimate = np.asarray(statistic(*data), dtype=float)
        nan = np.full_like(estimate, np.nan)
        return SimpleNamespace(confidence_interval=SimpleNamespace(low=nan, high=nan),
                               intervals_available=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.DegenerateDataWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return stats.bootstrap(data, statistic, paired=True, vectorized=True,
                               n_resamples=resamples, method="percentile",
                               rng=np.random.default_rng(seed))


REPORT_ROWS = ("reviews per 1,000", "AF detected per 1,000", "AF missed per 1,000",
               "sensitivity", "PPV", "net benefit")


def _report_statistic(p_t):
    weight = p_t / (1 - p_t)

    def statistic(y, review, axis=-1):
        tp = np.sum(y & review, axis=axis)
        fp = np.sum(~y & review, axis=axis)
        fn = np.sum(y & ~review, axis=axis)
        n = y.shape[axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.stack([1000 * (tp + fp) / n, 1000 * tp / n, 1000 * fn / n,
                             tp / (tp + fn), tp / (tp + fp), (tp - fp * weight) / n])
    return statistic


def evaluation_report(y, review, p_t, resamples=RESAMPLES, seed=SEED):
    """A frozen policy's consequences with 95% paired percentile bootstrap intervals.

    Rows: reviews, AF detected and AF missed per 1,000 recordings, sensitivity,
    PPV and net benefit at ``p_t``. Columns: estimate, low, high. Recordings are
    resampled with their decisions, so each interval reflects which recordings
    happened to be in this split; it leaves out how the policy was chosen.
    With fewer than two recordings the intervals are NaN.
    """
    y, review = np.asarray(y, dtype=bool), np.asarray(review, dtype=bool)
    statistic = _report_statistic(p_t)
    estimate = statistic(y, review)
    result = paired_bootstrap((y, review), statistic, resamples, seed)
    return pd.DataFrame({"estimate": estimate, "low": result.confidence_interval.low,
                         "high": result.confidence_interval.high}, index=list(REPORT_ROWS))
