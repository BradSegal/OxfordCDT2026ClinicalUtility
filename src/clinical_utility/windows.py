"""From window scores to a recording decision, on synthetic recordings with known AF intervals.

CinC 2017 gives one label per recording and no AF intervals, so it cannot
show how window-level rules behave. These recordings are entirely synthetic
but follow a simplified version of CinC's lengths (CinC recordings run from
about 9 to 61 seconds): most last 30 seconds and the rest up to 60. A
classifier scores 10-second windows taken every 5 seconds, so neighbouring
windows overlap by half and every second of a recording (except at the ends)
lies in two windows. AF episodes and artefacts are placed in seconds. A
recording's reference is "AF present" if any second is AF.

Window arrays are padded to the longest recording; ``valid`` marks the real
windows, and padding is never positive.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import expit

from .reference import OUTCOMES, counts

KINDS = {"no_af": .40, "artefact": .20, "paroxysmal": .20, "sustained": .20}
WINDOW_SECONDS, STRIDE_SECONDS = 10, 5


def covered_span(run_length, window=WINDOW_SECONDS, stride=STRIDE_SECONDS):
    """Seconds of recording covered by ``run_length`` consecutive windows."""
    return window + (run_length - 1) * stride


def synthetic_window_recordings(n=400, seed=20261106):
    """Deterministic synthetic recordings with AF intervals and artefacts in seconds.

    Lengths: 70% last 30 s, the rest a whole number of seconds from 31 to 60.
    Kinds (expected shares): no_af 40%; artefact 20% (no AF; one or two
    movement artefacts of 1 to 3 s); paroxysmal 20% (one AF episode of 5 to
    15 s); sustained 20% (AF for the whole recording). A window's score logit
    is N(-3, 1) plus 6 times the fraction of the window that is AF, plus
    N(4.5, 1) if it overlaps an artefact. Returns a dictionary of arrays:
    recording, kind, seconds, valid and af_windows (recordings × windows),
    window_scores, af_present, af_interval ([start, stop) in seconds; -1, -1
    without AF) and half ("choose" or "report", alternating within each kind).
    """
    rng = np.random.default_rng(seed)
    kinds = rng.choice(list(KINDS), size=n, p=list(KINDS.values()))
    seconds = np.where(rng.random(n) < .7, 30, rng.integers(31, 61, size=n))
    most = (seconds.max() - WINDOW_SECONDS) // STRIDE_SECONDS + 1
    valid = np.zeros((n, most), dtype=bool)
    af_fraction = np.zeros((n, most))
    artefact = np.zeros((n, most), dtype=bool)
    interval = np.full((n, 2), -1)
    for i, (kind, length) in enumerate(zip(kinds, seconds)):
        starts = np.arange(0, length - WINDOW_SECONDS + 1, STRIDE_SECONDS)
        valid[i, :len(starts)] = True
        if kind == "paroxysmal":
            duration = int(rng.integers(5, 16))
            begin = int(rng.integers(0, length - duration + 1))
            interval[i] = (begin, begin + duration)
        elif kind == "sustained":
            interval[i] = (0, length)
        if interval[i, 0] >= 0:
            overlap = (np.minimum(starts + WINDOW_SECONDS, interval[i, 1])
                       - np.maximum(starts, interval[i, 0]))
            af_fraction[i, :len(starts)] = np.clip(overlap, 0, None) / WINDOW_SECONDS
        if kind == "artefact":
            for _ in range(int(rng.integers(1, 3))):
                duration = int(rng.integers(1, 4))
                begin = int(rng.integers(0, length - duration + 1))
                hit = (starts < begin + duration) & (starts + WINDOW_SECONDS > begin)
                artefact[i, :len(starts)] |= hit
    logits = rng.normal(-3.0, 1.0, size=(n, most)) + 6 * af_fraction
    logits += np.where(artefact, rng.normal(4.5, 1.0, size=(n, most)), 0.0)
    scores = np.where(valid, expit(logits), 0.0)
    half = np.empty(n, dtype=object)
    for kind in KINDS:
        rows = np.flatnonzero(kinds == kind)
        half[rows] = np.where(np.arange(len(rows)) % 2 == 0, "choose", "report")
    return {"recording": np.array([f"W{i + 1:03d}" for i in range(n)]), "kind": kinds,
            "seconds": seconds, "valid": valid, "af_windows": af_fraction > 0,
            "window_scores": scores, "af_present": interval[:, 0] >= 0, "af_interval": interval,
            "half": half.astype(str)}


def aggregate(positive, valid, k):
    """Any-positive, strict-majority and at-least-k decisions, one per recording.

    Majority means more than half of that recording's own windows.
    """
    positive = np.asarray(positive, dtype=bool) & np.asarray(valid, dtype=bool)
    total = positive.sum(axis=1)
    return {"any": positive.any(axis=1), "majority": total > np.asarray(valid).sum(axis=1) / 2,
            "count": total >= k}


def persistent(positive, run_length):
    """True for a recording with at least ``run_length`` consecutive positive windows."""
    positive = np.asarray(positive, dtype=bool)
    return sliding_window_view(positive, run_length, axis=1).all(axis=-1).any(axis=1)


def candidate_rules(positive, valid):
    """The rules compared in Extension 5, each one decision per recording."""
    positive = np.asarray(positive, dtype=bool) & np.asarray(valid, dtype=bool)
    basic = aggregate(positive, valid, 2)
    return {"any window": basic["any"], "majority": basic["majority"],
            "at least 2 windows": basic["count"],
            "at least 3 windows": aggregate(positive, valid, 3)["count"],
            f"run of 2 ({covered_span(2)} s)": persistent(positive, 2),
            f"run of 3 ({covered_span(3)} s)": persistent(positive, 3)}


def choose_rule(table):
    """The rule with the lowest loss; on a tie, fewest reviews, then the earlier rule."""
    order = table.assign(position=range(len(table))).sort_values(
        ["loss", "reviews", "position"], kind="stable")
    return order.index[0]


def rule_table(decisions, af_present, kind, costs, per=1000):
    """Per rule: reviews, AF detected and missed, artefact reviews, brief AF missed, per ``per``, and loss."""
    af_present, kind = np.asarray(af_present, dtype=bool), np.asarray(kind)
    n, rows = len(af_present), []
    for name, review in decisions.items():
        review = np.asarray(review, dtype=bool)
        c = counts(af_present, review)
        rows.append({"rule": name, "reviews": per * (c["TP"] + c["FP"]) / n,
                     "af_detected": per * c["TP"] / n, "af_missed": per * c["FN"] / n,
                     "artefact_reviewed": per * np.sum(review & (kind == "artefact")) / n,
                     "brief_af_missed": per * np.sum(~review & (kind == "paroxysmal")) / n,
                     "loss": float(sum(c[k] * costs[k] for k in OUTCOMES))})
    return pd.DataFrame(rows).set_index("rule")


def recording_sensitivity(window_positive, af_present, rule):
    """Fraction of AF recordings that the rule sends for review."""
    from sklearn.metrics import recall_score
    return float(recall_score(np.asarray(af_present, dtype=bool), rule(window_positive)))
