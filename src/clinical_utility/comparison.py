"""Comparing two models on the same recordings: reclassification and net-benefit differences.

Conventions: ``y`` is True for reference AF. A recording is reviewed when its
score is at or above the cut. "Up" means the new model reviews a recording
that the old one did not. Net reclassification improvement (NRI) and the
integrated discrimination improvement (IDI) have no standard Python
implementation, so they are written out here; the bootstrap comes from
``scipy.stats.bootstrap``.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .reference import OUTCOMES, RESAMPLES, SEED, operating_points, paired_bootstrap


def _arrays(old, new, y):
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    y = np.asarray(y, dtype=bool)
    if not (len(old) == len(new) == len(y)):
        raise ValueError("Every input needs one value per recording.")
    if not (np.isfinite(old).all() and np.isfinite(new).all()):
        raise ValueError("Scores must be finite.")
    return old, new, y


def _nri(up, down, y):
    event = up[y].mean() - down[y].mean()
    non_event = down[~y].mean() - up[~y].mean()
    return {"event": float(event), "non_event": float(non_event), "total": float(event + non_event)}


def categorical_nri(old, new, y, cut):
    """NRI at one cut: event = P(up | AF) - P(down | AF); non-event = P(down | no AF) - P(up | no AF)."""
    old, new, y = _arrays(old, new, y)
    before, after = old >= cut, new >= cut
    return _nri(after & ~before, before & ~after, y)


def continuous_nri(old, new, y):
    """Category-free NRI: any rise in score counts as up and any fall as down; ties count neither."""
    old, new, y = _arrays(old, new, y)
    return _nri(new > old, new < old, y)


def idi(old, new, y):
    """IDI: mean rise in score among AF recordings plus mean fall among the rest."""
    old, new, y = _arrays(old, new, y)
    event, non_event = (new[y] - old[y]).mean(), (old[~y] - new[~y]).mean()
    return {"event": float(event), "non_event": float(non_event), "total": float(event + non_event)}


def reclassification(frame, old, new, cut):
    """Counts of old action (rows) against new action (columns), separately by reference AF."""
    return pd.crosstab([frame["af"].rename("AF"), (frame[old] >= cut).rename(f"{old} review")],
                       (frame[new] >= cut).rename(f"{new} review"))


def _delta_statistic(p_t):
    weight = p_t / (1 - p_t)

    def statistic(y, old_review, new_review, axis=-1):
        value = np.where(y, 1.0, -weight)
        return np.mean((new_review.astype(float) - old_review) * value, axis=axis)
    return statistic


def delta_net_benefit(y, old_review, new_review, p_t, resamples=RESAMPLES, seed=SEED):
    """NB(new) - NB(old) with a 95% paired percentile bootstrap interval.

    Each recording contributes +1 if it is AF and moves into review, -1 if it
    is AF and moves out, -p_t/(1-p_t) if it is not AF and moves in, and the
    opposite if it moves out; the mean of these contributions is the difference.
    """
    y = np.asarray(y, dtype=bool)
    old_review, new_review = np.asarray(old_review, dtype=bool), np.asarray(new_review, dtype=bool)
    statistic = _delta_statistic(p_t)
    result = paired_bootstrap((y, old_review, new_review), statistic, resamples, seed)
    return {"estimate": float(statistic(y, old_review, new_review)),
            "low": float(result.confidence_interval.low),
            "high": float(result.confidence_interval.high)}


def changed_decisions(frame, old, new, cut, p_t):
    """The recordings whose review decision changes between the two models, in input order.

    Columns: recording, reference_label, both scores, direction ("up" or "down")
    and contribution, the change in that recording's net-benefit value.
    """
    before, after = frame[old] >= cut, frame[new] >= cut
    changed = frame[before != after]
    value = np.where(changed["af"], 1.0, -p_t / (1 - p_t))
    moved_up = after[before != after]
    return pd.DataFrame({"recording": changed["recording"],
                         "reference_label": changed["reference_label"],
                         old: changed[old], new: changed[new],
                         "direction": np.where(moved_up, "up", "down"),
                         "contribution": np.where(moved_up, value, -value)}).reset_index(drop=True)


def compare_net_benefit(frame, cutoffs, p_t):
    """Net benefit of each model's decisions at its own cutoff, all valued at the same p_t."""
    from .reference import net_benefit
    y = frame["af"].to_numpy()
    return {model: net_benefit(y, frame[model].to_numpy() >= cut, p_t)
            for model, cut in cutoffs.items()}


NO_FLIP = np.inf


def _flip(y, old_scores, new_scores, base_costs, prices):
    minima = []
    for scores in (old_scores, new_scores):
        points = operating_points(y, scores, base_costs)
        fixed = sum(points[k].to_numpy() * base_costs[k] for k in OUTCOMES if k != "FN")
        minima.append((fixed[None, :] + prices[:, None] * points["FN"].to_numpy()).min(axis=1))
    caught = np.flatnonzero(minima[1] <= minima[0])
    return float(prices[caught[0]]) if len(caught) else NO_FLIP


def bootstrap_flip_prices(frame, old, new, base_costs, miss_prices, resamples=200, seed=SEED):
    """The flip price on paired bootstrap resamples of the recordings.

    ``scipy.stats.bootstrap`` resamples the recordings with both models'
    scores and the label kept together; in each resample the statistic is the
    first miss price at which the new model's minimum loss is no greater than
    the old model's, or ``NO_FLIP`` (infinity) when there is none in
    ``miss_prices``. Returns ``{"flips", "no_flip"}``: the flip prices found
    and the number of resamples without one.
    """
    prices = np.asarray(list(miss_prices), dtype=float)
    data_ = (frame["af"].to_numpy(dtype=bool), frame[old].to_numpy(dtype=float),
             frame[new].to_numpy(dtype=float))

    def statistic(y, old_scores, new_scores):
        return _flip(y, old_scores, new_scores, base_costs, prices)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the percentile interval of a draw with NO_FLIP is unused
        result = stats.bootstrap(data_, statistic, paired=True, vectorized=False,
                                 n_resamples=resamples, method="percentile",
                                 rng=np.random.default_rng(seed))
    draws = result.bootstrap_distribution
    return {"flips": draws[np.isfinite(draws)].astype(int), "no_flip": int(np.sum(~np.isfinite(draws)))}


__all__ = ["categorical_nri", "continuous_nri", "idi", "reclassification", "delta_net_benefit",
           "changed_decisions", "compare_net_benefit", "bootstrap_flip_prices", "NO_FLIP"]
