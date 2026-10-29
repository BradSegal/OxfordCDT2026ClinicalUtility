"""From window scores to a recording decision, on synthetic recordings with known AF intervals.

CinC 2017 gives one label per recording and no AF intervals, so it cannot
show how window-level rules behave. These recordings are entirely synthetic:
each has 100 windows, a known set of AF windows and per-window scores. A
recording's reference is "AF present" if any window is AF.

Every rule takes a boolean array of positive windows, shape (recordings,
windows) or (windows,), and returns one decision per recording.
"""

import numpy as np

from .logistic import sigmoid
from .reference import metrics

KINDS = {"no_af": .40, "artefact": .20, "paroxysmal": .20, "sustained": .20}


def synthetic_window_recordings(n=400, seed=20261106, windows=100):
    """Deterministic synthetic recordings with known AF intervals and window scores.

    Kinds (expected shares): no_af 40%; artefact 20% (no AF, one to three
    isolated high-scoring artefact windows); paroxysmal 20% (one AF run of 3-10
    windows); sustained 20% (one AF run of 60-100 windows). Window score logits
    are N(-3, 1) for non-AF windows, N(2, 1.2) for AF windows and N(2.5, 0.8)
    for artefact spikes. Returns a dictionary of numpy arrays: recording, kind,
    af_windows (bool), window_scores, af_present, af_burden, and af_interval
    as [start, stop) window indices (-1, -1 when there is no AF).
    """
    rng = np.random.default_rng(seed)
    kinds = rng.choice(list(KINDS), size=n, p=list(KINDS.values()))
    af = np.zeros((n, windows), dtype=bool)
    spikes = np.zeros((n, windows), dtype=bool)
    interval = np.full((n, 2), -1)
    for i, kind in enumerate(kinds):
        if kind in ("paroxysmal", "sustained"):
            length = int(rng.integers(3, 11) if kind == "paroxysmal" else rng.integers(60, 101))
            start = int(rng.integers(0, windows - length + 1))
            af[i, start:start + length] = True
            interval[i] = (start, start + length)
        elif kind == "artefact":
            # Isolated spikes: chosen from every third window so no two are adjacent.
            count = int(rng.integers(1, 4))
            spikes[i, rng.choice(np.arange(0, windows, 3), size=count, replace=False)] = True
    logits = rng.normal(-3.0, 1.0, size=(n, windows))
    logits[af] = rng.normal(2.0, 1.2, size=int(af.sum()))
    logits[spikes] = rng.normal(2.5, 0.8, size=int(spikes.sum()))
    return {"recording": np.array([f"W{i + 1:03d}" for i in range(n)]), "kind": kinds,
            "af_windows": af, "window_scores": sigmoid(logits),
            "af_present": af.any(axis=1), "af_burden": af.mean(axis=1), "af_interval": interval}


def _per_recording(window_positive, decide):
    positive = np.asarray(window_positive, dtype=bool)
    rows = np.atleast_2d(positive)
    result = np.array([decide(row) for row in rows], dtype=bool)
    return result if positive.ndim > 1 else bool(result[0])


def longest_run(window_positive):
    """Length of the longest run of consecutive positive windows in each recording."""
    def run(row):
        best = current = 0
        for positive in row:
            current = current + 1 if positive else 0
            best = max(best, current)
        return best
    positive = np.asarray(window_positive, dtype=bool)
    lengths = np.array([run(row) for row in np.atleast_2d(positive)])
    return lengths if positive.ndim > 1 else int(lengths[0])


def any_positive(window_positive):
    """Flag the recording if at least one window is positive."""
    return _per_recording(window_positive, lambda row: row.any())


def majority(window_positive):
    """Flag the recording if more than half of its windows are positive."""
    return _per_recording(window_positive, lambda row: row.sum() > len(row) / 2)


def count_at_least(k):
    """Rule: flag the recording if at least k windows are positive (anywhere)."""
    def rule(window_positive):
        return _per_recording(window_positive, lambda row: row.sum() >= k)
    rule.__name__ = f"count_at_least_{k}"
    return rule


def persistence(run_length):
    """Rule: flag the recording if at least run_length consecutive windows are positive."""
    def rule(window_positive):
        return _per_recording(window_positive, lambda row: longest_run(row) >= run_length)
    rule.__name__ = f"persistence_{run_length}"
    return rule


def evaluate_window_rule(rule, window_scores, af_present, window_threshold=.5):
    """Apply a window threshold, then a rule, and score decisions against recording truth.

    A window is positive when its score >= window_threshold. Returns
    {"flagged": decision per recording, "counts": TP/FP/FN/TN, "metrics": ...}.
    """
    flagged = np.asarray(rule(np.asarray(window_scores) >= window_threshold), dtype=bool)
    truth = np.asarray(af_present, dtype=bool)
    counts = {"TP": int(np.sum(flagged & truth)), "FP": int(np.sum(flagged & ~truth)),
              "FN": int(np.sum(~flagged & truth)), "TN": int(np.sum(~flagged & ~truth))}
    return {"flagged": flagged, "counts": counts, "metrics": metrics(counts)}
