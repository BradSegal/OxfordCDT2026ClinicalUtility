"""Report a recommended policy on held-out patients, with uncertainty.

The counts come from your own ``outcome_counts``: the report resamples the
patients and asks your function to count each resample.
"""

import numpy as np
import pandas as pd

SEED = 20261106
RESAMPLES = 1000
REPORT_ROWS = ("flags per 1,000", "AF found per 1,000", "AF missed per 1,000")


def _per_1000(counts):
    counts = pd.Series(counts, dtype=float)
    patients = counts.sum()
    return np.array([counts["TP"] + counts["FP"], counts["TP"], counts["FN"]]) * 1000 / patients


def report(y, flagged, outcome_counts, resamples=RESAMPLES, seed=SEED):
    """A policy's consequences per 1,000 patients with 95% percentile bootstrap intervals.

    Rows: flags, AF found and AF missed per 1,000 patients. Columns: estimate,
    low, high. Patients are resampled together with their flags, so an interval
    shows how much the result depends on which patients were in this sample.
    """
    y, flagged = np.asarray(y, dtype=bool), np.asarray(flagged, dtype=bool)
    estimate = _per_1000(outcome_counts(y, flagged))
    generator = np.random.default_rng(seed)
    draws = np.array([_per_1000(outcome_counts(y[i], flagged[i]))
                      for i in generator.integers(0, len(y), size=(resamples, len(y)))])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return pd.DataFrame({"estimate": estimate, "low": low, "high": high}, index=list(REPORT_ROWS))
