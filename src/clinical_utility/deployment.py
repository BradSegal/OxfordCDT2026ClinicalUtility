"""Running a policy in a service: prevalence, review capacity and repeat recordings.

Daily review demand is Poisson: if arrivals are Poisson and each is reviewed
independently, the reviews are Poisson too, so ``scipy.stats.poisson.sf``
gives the chance that a day's demand exceeds capacity. Projecting counts to
another prevalence and pricing a repeat recording are written out because
they are the decisions being taught.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .reference import OUTCOMES


def project(points, prevalence, costs, per=1000):
    """Expected consequences of every operating point at another prevalence, per ``per`` recordings.

    Sensitivity and specificity are held fixed: the AF group is
    ``per * prevalence`` and the rest is its complement.
    """
    af, non_af = per * prevalence, per * (1 - prevalence)
    table = pd.DataFrame({"threshold": points["threshold"],
                          "TP": af * points["sensitivity"], "FN": af * (1 - points["sensitivity"]),
                          "FP": non_af * (1 - points["specificity"]),
                          "TN": non_af * points["specificity"]})
    table["reviews"] = table["TP"] + table["FP"]
    table["af_missed"] = table["FN"]
    table["loss"] = sum(table[k] * costs[k] for k in OUTCOMES)
    return table


def overflow_probability(expected_reviews, capacity):
    """P(reviews > capacity) for Poisson daily demand with this mean."""
    return stats.poisson.sf(capacity, expected_reviews)


def capacity_policy(projected, capacity, tolerance):
    """The cheapest operating point whose chance of exceeding capacity on a day is within tolerance."""
    table = projected.assign(overflow=overflow_probability(projected["reviews"], capacity))
    feasible = table[table["overflow"] <= tolerance]
    return feasible.loc[feasible.sort_values(["loss", "reviews"], kind="stable").index[0]]


def highest_safe_prevalence(points, cutoff, costs, capacity, tolerance, prevalences):
    """The highest prevalence on the grid at which a fixed cutoff stays within the tolerance."""
    row = points[points["threshold"] == cutoff]
    safe = [p for p in prevalences
            if overflow_probability(project(row, p, costs)["reviews"].iloc[0], capacity) <= tolerance]
    return max(safe) if safe else None


def boundary_prevalence(points, cutoff, costs, capacity, tolerance):
    """The prevalence at which a fixed cutoff's chance of a day over capacity reaches the tolerance.

    Solved continuously with ``scipy.optimize.brentq`` between 0.1% and 50%;
    None if the tolerance is met or broken across that whole range.
    """
    from scipy import optimize
    row = points[points["threshold"] == cutoff]

    def excess(prevalence):
        demand = project(row, prevalence, costs)["reviews"].iloc[0]
        return overflow_probability(demand, capacity) - tolerance
    low, high = .001, .5
    if excess(low) * excess(high) > 0:
        return None
    return float(optimize.brentq(excess, low, high, xtol=1e-10))


def simulate_days(arrivals_per_day, review_rate, days=365, seed=20261106):
    """Each simulated day's review demand: Poisson arrivals, then binomial reviews."""
    rng = np.random.default_rng(seed)
    return rng.binomial(rng.poisson(arrivals_per_day, size=days), review_rate)


def top_k_days(af_scores, other_scores, prevalence, arrivals_per_day, capacity, cutoff, days=365,
               seed=20261106, p_t=.2):
    """Daily batches at the planning prevalence, each reviewed by a top-k cap and by a fixed cutoff.

    Each day draws Poisson arrivals, then a binomial number with AF, then
    scores for them by resampling the given development scores. A top-k
    policy reviews the ``capacity`` highest scores (ties by arrival order); the
    fixed cutoff reviews every score at or above ``cutoff``. Returns one row
    per day: arrivals; for the fixed cutoff, flagged, AF detected, AF missed
    and net benefit; for the cap, reviewed, AF detected, AF missed (below the
    cutoff or beyond the cap), net benefit, deferred (flagged but beyond the
    cap) and AF deferred. Net benefit is per recording at ``p_t``.
    """
    rng = np.random.default_rng(seed)
    af_scores, other_scores = np.asarray(af_scores, float), np.asarray(other_scores, float)
    rows = []
    for _ in range(days):
        arrivals = int(rng.poisson(arrivals_per_day))
        n_af = int(rng.binomial(arrivals, prevalence))
        scores = np.r_[rng.choice(af_scores, n_af), rng.choice(other_scores, arrivals - n_af)]
        is_af = np.r_[np.ones(n_af, bool), np.zeros(arrivals - n_af, bool)]
        order = rng.permutation(arrivals)
        scores, is_af = scores[order], is_af[order]
        capped = np.zeros(arrivals, dtype=bool)
        capped[np.argsort(-scores, kind="stable")[:capacity]] = True
        flagged = scores >= cutoff
        deferred = flagged & ~capped
        weight = p_t / (1 - p_t)

        def net_benefit(review):
            return (np.sum(review & is_af) - weight * np.sum(review & ~is_af)) / max(arrivals, 1)
        rows.append({"arrivals": arrivals, "flagged": int(flagged.sum()),
                     "cutoff_detected": int(np.sum(flagged & is_af)),
                     "cutoff_missed": int(np.sum(~flagged & is_af)),
                     "cutoff_net_benefit": net_benefit(flagged),
                     "reviewed": int(capped.sum()), "cap_detected": int(np.sum(capped & is_af)),
                     "cap_missed": int(np.sum(~capped & is_af)),
                     "cap_net_benefit": net_benefit(capped),
                     "deferred": int(deferred.sum()), "af_deferred": int(np.sum(deferred & is_af))})
    return pd.DataFrame(rows)


UNREADABLE_PATHWAYS = ("accept", "over_read", "repeat", "refer")


def unreadable_losses(af_probability, costs, assumptions):
    """Expected loss per unreadable recording for four ways of handling it.

    An unreadable recording cannot be reviewed in the ordinary way. The options:

    - ``accept``: no further action; any AF goes undetected.
    - ``over_read``: a clinician reads the noisy trace (one review). With
      probability ``over_read_success`` the read is conclusive and AF, if
      present, is established; otherwise the recording is unresolved and any
      AF goes undetected.
    - ``repeat``: ask for a repeat recording, paying ``repeat_cost`` and
      ``delay_cost`` once. With probability ``repeat_readable`` it is readable
      and the classifier decides, with ``sensitivity`` and ``specificity``; a
      flagged repeat is reviewed. An unreadable repeat is over-read.
    - ``refer``: send for an alternative assessment (a clinic ECG, for
      example) costing ``refer_cost``, assumed conclusive. AF is then referred
      for assessment, not yet established.

    Prices: a review costs ``costs["TP"]`` if it finds AF and ``costs["FP"]``
    otherwise; undetected AF costs ``costs["FN"]`` on top of any review it used.
    AF status is assumed not to change between recordings, and readability not
    to depend on AF. Returns the four expected losses and the cheapest as ``best``.
    """
    p, a = af_probability, assumptions
    for key in ("over_read_success", "repeat_readable", "sensitivity", "specificity"):
        if not 0 <= a[key] <= 1:
            raise ValueError(f"{key} must lie between 0 and 1.")
    s, read = a["over_read_success"], costs["FP"]
    over_read = p * (s * costs["TP"] + (1 - s) * (read + costs["FN"])) + (1 - p) * costs["FP"]
    se, sp, q = a["sensitivity"], a["specificity"], a["repeat_readable"]
    readable = (p * (se * costs["TP"] + (1 - se) * costs["FN"])
                + (1 - p) * ((1 - sp) * costs["FP"] + sp * costs["TN"]))
    losses = {"accept": p * costs["FN"] + (1 - p) * costs["TN"],
              "over_read": over_read,
              "repeat": a["repeat_cost"] + a["delay_cost"] + q * readable + (1 - q) * over_read,
              "refer": a["refer_cost"]}
    return {**losses, "best": min(losses, key=losses.get)}


def unreadable_consequences(af_probability, assumptions, per=1000):
    """Expected consequences per ``per`` unreadable recordings of each pathway.

    Columns: reviews (including over-reads), repeats, referrals, AF established
    (by a conclusive read), AF referred for assessment, and AF undetected
    (unresolved or missed).
    """
    p, a = af_probability, assumptions
    s, se, sp, q = (a[k] for k in ("over_read_success", "sensitivity", "specificity",
                                   "repeat_readable"))
    rows = {
        "accept": {"reviews": 0, "repeats": 0, "referrals": 0, "af_established": 0,
                   "af_referred": 0, "af_undetected": p},
        "over_read": {"reviews": 1, "repeats": 0, "referrals": 0, "af_established": p * s,
                      "af_referred": 0, "af_undetected": p * (1 - s)},
        "repeat": {"reviews": q * (p * se + (1 - p) * (1 - sp)) + (1 - q), "repeats": 1,
                   "referrals": 0, "af_established": q * p * se + (1 - q) * p * s,
                   "af_referred": 0, "af_undetected": q * p * (1 - se) + (1 - q) * p * (1 - s)},
        "refer": {"reviews": 0, "repeats": 0, "referrals": 1, "af_established": 0,
                  "af_referred": p, "af_undetected": 0},
    }
    return per * pd.DataFrame(rows).T[["reviews", "repeats", "referrals", "af_established",
                                       "af_referred", "af_undetected"]]


def best_pathway_by_probability(costs, assumptions, probabilities):
    """The cheapest pathway at each assumed AF probability among unreadable recordings."""
    return pd.DataFrame([{"af_probability": p, **unreadable_losses(p, costs, assumptions)}
                         for p in probabilities])
