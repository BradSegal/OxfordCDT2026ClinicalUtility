"""Running a policy in a service: fixed review capacity, daily variation and repeat recordings.

Prices are illustrative common loss units, as in ``reference``. Counts are
per recording unless a function says otherwise.
"""

import math

import numpy as np


def top_k_policy(scores, k):
    """Review the k highest-scoring recordings; return a boolean array in input order.

    Ties at the boundary are broken by input order (earlier recordings first),
    so exactly min(k, n) recordings are reviewed.
    """
    scores = np.asarray(scores, dtype=float)
    if k < 0 or int(k) != k:
        raise ValueError("k must be a non-negative whole number.")
    review = np.zeros(len(scores), dtype=bool)
    review[np.argsort(-scores, kind="stable")[:int(k)]] = True
    return review


def overflow_probability(expected_reviews_per_day, capacity):
    """P(reviews > capacity) when daily reviews are Poisson with the given mean.

    P(X > c) = 1 - sum_{i=0..c} exp(-m) m^i / i!. If arrivals are Poisson(lambda)
    and each is reviewed independently with probability r, reviews are Poisson(lambda r).
    """
    m, c = float(expected_reviews_per_day), int(capacity)
    if m < 0 or c < 0:
        raise ValueError("Use a non-negative mean and capacity.")
    if m == 0:
        return 0.0
    log_terms = [-m + i * math.log(m) - math.lgamma(i + 1) for i in range(c + 1)]
    return max(0.0, 1 - math.fsum(math.exp(t) for t in log_terms))


def daily_reviews(recordings_per_day, review_rate, days=365, seed=20261106):
    """Simulated review demand on each day: Poisson arrivals, then binomial reviews.

    All arrivals are drawn first and then all reviews, so a given seed gives the
    same days as ``simulate_daily_queue``.
    """
    if not 0 <= review_rate <= 1:
        raise ValueError("review_rate must lie between 0 and 1.")
    if recordings_per_day < 0 or days <= 0:
        raise ValueError("Use a non-negative arrival rate and a positive number of days.")
    rng = np.random.default_rng(seed)
    return rng.binomial(rng.poisson(recordings_per_day, size=days), review_rate)


def simulate_daily_queue(recordings_per_day, review_rate, capacity, days=365, seed=20261106):
    """Simulate daily review demand for a policy that reviews a fixed fraction of recordings.

    Each day: arrivals ~ Poisson(recordings_per_day); reviews ~ Binomial(arrivals,
    review_rate). Returns the fraction of days over capacity and summaries of
    daily reviews. A mean below capacity does not mean every day fits.
    """
    if capacity < 0:
        raise ValueError("Use a non-negative capacity.")
    reviews = daily_reviews(recordings_per_day, review_rate, days, seed)
    return {"fraction_over_capacity": float(np.mean(reviews > capacity)),
            "mean_reviews": float(reviews.mean()),
            "p95_reviews": float(np.quantile(reviews, .95)),
            "max_reviews": int(reviews.max()), "days": days, "seed": seed}


def repeat_pathway_expected_cost(af_probability, costs, repeat_cost, completion_probability,
                                 sensitivity, specificity, delay_cost=0.0, fallback="review"):
    """Expected loss per unreadable recording for three pathways.

    p = AF probability, q = probability the repeat recording is readable,
    Se/Sp = classifier sensitivity/specificity on readable repeats,
    C = the four-cell price matrix.

        unreadable recording
        |-- review now ............ p C_TP + (1-p) C_FP
        |-- no review now ......... p C_FN + (1-p) C_TN
        `-- repeat (repeat_cost + delay_cost, paid once)
            |-- readable (q) ...... p [Se C_TP + (1-Se) C_FN]
            |                       + (1-p) [(1-Sp) C_FP + Sp C_TN]
            `-- unreadable (1-q) .. fallback action, priced as "review now"
                                    or "no review now"

    Assumes AF status does not change between recordings. Returns the three
    expected losses and the name of the cheapest pathway ("best").
    """
    p, q = af_probability, completion_probability
    for name, value in (("af_probability", p), ("completion_probability", q),
                        ("sensitivity", sensitivity), ("specificity", specificity)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie between 0 and 1.")
    if fallback not in ("review", "no_review"):
        raise ValueError("fallback must be 'review' or 'no_review'.")
    review_now = p * costs["TP"] + (1 - p) * costs["FP"]
    no_review_now = p * costs["FN"] + (1 - p) * costs["TN"]
    readable = (p * (sensitivity * costs["TP"] + (1 - sensitivity) * costs["FN"])
                + (1 - p) * ((1 - specificity) * costs["FP"] + specificity * costs["TN"]))
    fallback_loss = review_now if fallback == "review" else no_review_now
    repeat = repeat_cost + delay_cost + q * readable + (1 - q) * fallback_loss
    losses = {"review_now": review_now, "no_review_now": no_review_now, "repeat": repeat}
    return {**losses, "readable_branch": readable, "best": min(losses, key=losses.get)}
