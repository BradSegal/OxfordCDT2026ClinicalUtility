"""Comparing two models on the same recordings: ranking, reclassification and net benefit.

Conventions: ``y`` is 1 for reference AF and 0 otherwise. A recording is
reviewed when its probability is >= the threshold, matching
``reference.should_review``. "Up" means the new model places a recording in a
higher-risk category than the old model. Undefined quantities (no AF or no
non-AF recordings) are NaN, never 0. Non-finite scores are refused, as in
``reference``.
"""

import numpy as np

from .reference import net_benefit


def _arrays(*probabilities, y):
    y = np.asarray(y, dtype=float)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Reference labels must be 0 or 1.")
    arrays = [np.asarray(p, dtype=float) for p in probabilities]
    if any(len(a) != len(y) for a in arrays):
        raise ValueError("Every input needs one value per recording.")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Scores must be finite; NaN or infinite scores have no place in a ranking.")
    return (*arrays, y.astype(bool))


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def auc(scores, y):
    """Area under the ROC curve = P(score of an AF recording > score of a non-AF one).

    Ties count one half (the Mann-Whitney statistic). NaN without both classes.
    """
    scores, y = _arrays(scores, y=y)
    positives, negatives = scores[y], scores[~y]
    if not len(positives) or not len(negatives):
        return float("nan")
    order = np.sort(negatives)
    below = np.searchsorted(order, positives, side="left")
    ties = np.searchsorted(order, positives, side="right") - below
    return float((below + .5 * ties).sum() / (len(positives) * len(negatives)))


def risk_category(probabilities, cuts):
    """Category index: 0 below cuts[0], i below cuts[i] and >= cuts[i-1], len(cuts) above."""
    return np.searchsorted(np.asarray(cuts, dtype=float), np.asarray(probabilities, dtype=float),
                           side="right")


def reclassification_table(old_probs, new_probs, y, cuts):
    """Cross-tabulate old (rows) against new (columns) risk categories, separately by outcome.

    Returns {"labels", "events", "non_events"}; each table is a list of rows of counts.
    """
    old, new, y = _arrays(old_probs, new_probs, y=y)
    cuts = [float(c) for c in cuts]
    edges = [0.0, *cuts, 1.0]
    labels = [f"[{lo:g}, {hi:g})" for lo, hi in zip(edges[:-1], edges[1:])]
    labels[-1] = labels[-1][:-1] + "]"
    old_cat, new_cat = risk_category(old, cuts), risk_category(new, cuts)
    k = len(cuts) + 1

    def table(mask):
        counts = np.zeros((k, k), dtype=int)
        np.add.at(counts, (old_cat[mask], new_cat[mask]), 1)
        return counts.tolist()

    return {"labels": labels, "events": table(y), "non_events": table(~y)}


def _nri(up, down, y):
    event = _mean(up[y]) - _mean(down[y])
    non_event = _mean(down[~y]) - _mean(up[~y])
    return {"event": event, "non_event": non_event, "total": event + non_event}


def categorical_nri(old_probs, new_probs, y, cuts):
    """Net reclassification improvement across risk categories.

    event NRI     = P(up | AF) - P(down | AF)
    non-event NRI = P(down | no AF) - P(up | no AF)
    total         = event + non-event.
    The total adds two proportions with equal weight, whatever the prevalence
    or the relative cost of the errors; it is not a measure of net benefit.
    """
    old, new, y = _arrays(old_probs, new_probs, y=y)
    old_cat, new_cat = risk_category(old, cuts), risk_category(new, cuts)
    return _nri(new_cat > old_cat, new_cat < old_cat, y)


def continuous_nri(old_probs, new_probs, y):
    """Category-free NRI: any increase counts as up and any decrease as down; ties are neither."""
    old, new, y = _arrays(old_probs, new_probs, y=y)
    return _nri(new > old, new < old, y)


def idi(old_probs, new_probs, y):
    """Integrated discrimination improvement.

    event     = mean(new - old) among AF recordings
    non_event = mean(old - new) among non-AF recordings
    total     = event + non_event (the change in discrimination slope).
    """
    old, new, y = _arrays(old_probs, new_probs, y=y)
    event = _mean(new[y] - old[y])
    non_event = _mean(old[~y] - new[~y])
    return {"event": event, "non_event": non_event, "total": event + non_event}


def counts_at_threshold(probabilities, y, threshold):
    """TP, FP, FN and TN when recordings with probability >= threshold are reviewed."""
    p, y = _arrays(probabilities, y=y)
    review = p >= threshold
    return {"TP": int(np.sum(review & y)), "FP": int(np.sum(review & ~y)),
            "FN": int(np.sum(~review & y)), "TN": int(np.sum(~review & ~y))}


def policy_net_benefit(probabilities, y, threshold_probability):
    """NB = TP/n - FP/n * p_t / (1 - p_t), reviewing when probability >= p_t."""
    return net_benefit(counts_at_threshold(probabilities, y, threshold_probability),
                       threshold_probability)


def delta_net_benefit_bootstrap(old_probs, new_probs, y, threshold_probability,
                                resamples=1000, seed=20261106, level=.95):
    """NB(new) - NB(old) at one threshold, with a recording-level percentile bootstrap CI.

    Each resample draws n recordings with replacement and recomputes both net
    benefits on the same draw, so the interval reflects the paired comparison.
    Returns {"estimate", "low", "high", "resamples", "seed", "level"}.
    """
    old, new, y = _arrays(old_probs, new_probs, y=y)
    p = threshold_probability
    if not 0 < p < 1:
        raise ValueError("Threshold probability must be strictly between 0 and 1.")
    weight = p / (1 - p)
    # Per-recording contribution to NB: +1 for a true positive, -weight for a false positive.
    contribution = ((new >= p).astype(float) - (old >= p)) * np.where(y, 1.0, -weight)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(y), size=(resamples, len(y)))
    boot = contribution[draws].mean(axis=1)
    tail = (1 - level) / 2
    return {"estimate": float(contribution.mean()),
            "low": float(np.quantile(boot, tail)), "high": float(np.quantile(boot, 1 - tail)),
            "resamples": resamples, "seed": seed, "level": level}


def minimum_losses(scores, y, costs, miss_prices):
    """Each miss price's minimum total loss over every cutoff, other prices fixed.

    Cutoffs are every distinct score plus review-all and review-none, as in
    ``reference.sweep_thresholds``. ``costs`` supplies C_TP, C_FP and C_TN;
    C_FN takes each value in ``miss_prices`` in turn.
    """
    scores, y = _arrays(scores, y=y)
    order = np.argsort(scores, kind="stable")
    ordered, truth = scores[order], y[order]
    below = np.r_[0, np.searchsorted(ordered, np.unique(ordered), side="left"), len(ordered)]
    fn = np.r_[0, np.cumsum(truth)][below]
    tn = below - fn
    tp, fp = int(truth.sum()) - fn, len(truth) - int(truth.sum()) - tn
    fixed = tp * costs["TP"] + fp * costs["FP"] + tn * costs["TN"]
    return np.array([np.min(fixed + miss * fn) for miss in miss_prices])


def flip_price(old_scores, new_scores, y, costs, miss_prices):
    """First miss price at which the new model's minimum loss is no greater than the old's.

    Returns None when the new model never catches up within ``miss_prices``.
    """
    old = minimum_losses(old_scores, y, costs, miss_prices)
    new = minimum_losses(new_scores, y, costs, miss_prices)
    for miss, a, b in zip(miss_prices, old, new):
        if b <= a:
            return miss
    return None


def bootstrap_flip_prices(old_scores, new_scores, y, costs, miss_prices, resamples=200,
                          seed=20261106):
    """``flip_price`` on paired recording-level bootstrap resamples of one cohort.

    Each resample draws n recordings with replacement and keeps both models'
    scores and the label together. The spread shows how far the flip price
    could move on another sample of the same size; it says nothing about a
    different population.
    """
    old, new, y = _arrays(old_scores, new_scores, y=y)
    rng = np.random.default_rng(seed)
    flips = []
    for _ in range(resamples):
        rows = rng.integers(0, len(y), len(y))
        flips.append(flip_price(old[rows], new[rows], y[rows], costs, miss_prices))
    return flips
