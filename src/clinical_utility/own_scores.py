"""Read a learner's own classifier scores for Extension 6, and analyse whatever they allow.

The expected input is a CSV with a header row containing ``recording`` and
``score`` (other columns are ignored), one row per CinC 2017 recording. Reading
never stops the notebook: a missing file or unreadable text is reported, and
the simulated demonstration column is used instead, with a statement saying so.

A learner's upload can be small, have one outcome class, separate the classes
completely, or fall in only one split. The analysis helpers below therefore
check what each calculation needs, print one line explaining any result that
is undefined, and return None for it, so the defined calculations carry on.
"""

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import calibration, data, reference

DEMO_NOTE = ("Using the simulated demonstration column, not your classifier. It blends the "
             "course's two simulated models and includes a few deliberate problems so you can "
             "see what the join reports.")
FEW = 20


def parse(text):
    """A DataFrame of recording and score from CSV text; a score that is not a number becomes NaN.

    Every row must have as many fields as the header, and the header may name
    each column once (after trimming spaces and ignoring case), so a malformed
    file is reported instead of silently misread.
    """
    try:
        lines = list(csv.reader(io.StringIO(text.strip())))
    except csv.Error as error:  # for example, a field longer than the csv module's limit
        raise ValueError(f"The scores could not be parsed as CSV: {error}.") from None
    if not lines:
        raise ValueError("The scores are empty.")
    header = [name.strip().lower() for name in lines[0]]
    repeated = sorted({name for name in header if header.count(name) > 1})
    if repeated:
        raise ValueError(f"The header names {', '.join(map(repr, repeated))} more than once.")
    for number, row in enumerate(lines[1:], start=2):
        if row and len(row) != len(header):
            raise ValueError(f"Line {number} has {len(row)} fields but the header has {len(header)}.")
    frame = pd.read_csv(io.StringIO(text.strip()), dtype=str, index_col=False)
    columns = {name.strip().lower(): name for name in frame.columns}
    if "recording" not in columns or "score" not in columns:
        raise ValueError("The first line must be a header containing 'recording' and 'score'.")
    return pd.DataFrame({"recording": frame[columns["recording"]].fillna("").str.strip(),
                         "score": pd.to_numeric(frame[columns["score"]], errors="coerce")})


def demo_scores():
    """A simulated score for every development and evaluation recording, plus five problems.

    The score blends the two simulated models on the log-odds scale. The
    problems are two noise recordings, an unknown ID, a duplicated ID and a
    score above 1, so the join has something to report.
    """
    scores = data.load_scores()
    kept = scores[scores["split"].isin(["development", "evaluation"])]
    blended = expit((logit(kept["random_forest"]) + logit(kept["resnet_calibrated"])) / 2)
    frame = pd.DataFrame({"recording": kept["recording"].to_numpy(), "score": blended.to_numpy()})
    labels = data.reference_labels()
    noise = labels.loc[labels["reference_label"] == "~", "recording"].head(2)
    outside = scores.loc[scores["split"] == "calibration", "recording"].iloc[0]
    extra = pd.DataFrame({"recording": [*noise, "Z99999", frame["recording"].iloc[0], outside],
                          "score": [.5, .5, .3, frame["score"].iloc[0], 1.2]})
    return pd.concat([frame, extra], ignore_index=True)


def read_scores(path="", pasted=""):
    """(scores, note): the learner's scores from a file or pasted text, else the demonstration column."""
    try:
        if path:
            if not Path(path).exists():
                raise ValueError(f"No file called {path!r}. Upload it in Colab's Files pane first.")
            return parse(Path(path).read_text()), f"Read {path}."
        if pasted.strip():
            return parse(pasted), "Read the pasted scores."
    except (ValueError, OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        return demo_scores(), f"Could not read your scores ({error}). {DEMO_NOTE}"
    return demo_scores(), DEMO_NOTE


def join_scores(scores, labels):
    """The reference join: exclude duplicated, unknown, out-of-range and noise rows, in that order.

    Returns ``{"records", "duplicates", "unknown", "out_of_range", "noise"}``:
    the kept rows as a DataFrame of recording, score and af (True only for A),
    in input order; the excluded IDs as lists; and the number of noise rows.
    """
    duplicated = scores["recording"].duplicated(keep=False)
    rest = scores[~duplicated].merge(labels, on="recording", how="left", validate="one_to_one")
    unknown = rest["reference_label"].isna()
    out_of_range = ~unknown & ~rest["score"].between(0, 1)
    noise = ~unknown & ~out_of_range & (rest["reference_label"] == "~")
    kept = rest[~(unknown | out_of_range | noise)]
    return {"records": pd.DataFrame({"recording": kept["recording"], "score": kept["score"],
                                     "af": kept["reference_label"] == "A"}).reset_index(drop=True),
            "duplicates": list(dict.fromkeys(scores.loc[duplicated, "recording"])),
            "unknown": rest.loc[unknown, "recording"].tolist(),
            "out_of_range": rest.loc[out_of_range, "recording"].tolist(),
            "noise": int(noise.sum())}


def with_splits(records):
    """The joined records of each course split, as DataFrames with the same columns."""
    splits = data.load_scores()[["recording", "split"]]
    joined = records.merge(splits, on="recording", how="left")
    return {split: joined[joined["split"] == split].drop(columns="split").reset_index(drop=True)
            for split in data.SPLITS}


def describe_splits(own):
    """Print each split's size, the development AF count only, and what the data cannot support.

    Evaluation outcomes are not summarised here: they belong to the reveal.
    """
    development, evaluation = own["development"], own["evaluation"]
    n_af = int(development["af"].sum())
    print(f"development: {len(development)} recordings, {n_af} with AF")
    print(f"evaluation: {len(evaluation)} recordings (outcomes stay hidden until the reveal)")
    if len(own.get("calibration", [])):
        print(f"calibration: {len(own['calibration'])} recordings, not used here")
    if development.empty:
        print("No development recordings: there is nothing to choose a policy on, so the "
              "development analysis, the freeze and the reveal cannot run.")
    elif n_af in (0, len(development)):
        print(f"Every development recording {'has' if n_af else 'lacks'} AF: J and the calibration "
              "fit are undefined. The minimum-loss table still runs.")
    elif min(n_af, len(development) - n_af) < FEW:
        print(f"Fewer than {FEW} development recordings in one outcome group: every result below "
              "is very uncertain.")
    if evaluation.empty:
        print("No evaluation recordings: you can freeze a policy, but there is nothing to reveal "
              "it on.")


def max_j(points):
    """The maximum-J operating point (fewest reviews on ties), or None, with the reason, if J is undefined."""
    if not points["youden_j"].notna().any():
        print("J is undefined: it needs at least one recording with AF and one without.")
        return None
    best = points["youden_j"].max()
    tied = points[np.isclose(points["youden_j"], best, rtol=0, atol=1e-12)]
    return tied.loc[tied["reviews"].idxmin()]


def equivalent_interval(scores, threshold):
    """The cutoffs that review exactly the recordings ``threshold`` reviews.

    A policy reviews a recording when its score is at or above the cutoff, so
    the bounds come from the largest excluded score and the smallest reviewed
    one: every c with low < c <= high is equivalent. When every recording is
    reviewed, the interval is [0, smallest score], closed at 0; when none is,
    it runs from just above the largest score to infinity.
    Returns {"low", "high", "low_included"}.
    """
    scores = np.asarray(scores, dtype=float)
    excluded, reviewed = scores[scores < threshold], scores[scores >= threshold]
    high = float(reviewed.min()) if len(reviewed) else float("inf")
    if not len(excluded):
        return {"low": 0.0, "high": high, "low_included": True}
    return {"low": float(excluded.max()), "high": high, "low_included": False}


def interval_text(interval):
    """The interval in words, for example "above 0.300 and up to 0.400"."""
    high = "any higher value" if np.isinf(interval["high"]) else f"{interval['high']:.3f}"
    if interval["low_included"]:
        return f"from {interval['low']:.3f} up to {high}"
    return f"above {interval['low']:.3f} and up to {high}"


def stated_versus_optimum(records, costs, stated, best_threshold):
    """Apparent development results of a stated cutoff beside the development minimum-loss cutoff.

    Both are measured on the recordings the optimum was chosen on, so they are
    apparent results, not estimates for new recordings; the comparison that
    counts is made later on the evaluation split. Returns each policy's
    reviews, AF missed and loss, the interval of cutoffs equivalent to the
    optimum, and whether ``stated`` makes the same decisions.
    """
    y, scores = records["af"].to_numpy(dtype=bool), records["score"].to_numpy(dtype=float)

    def summary(cut):
        review = scores >= cut
        return {"reviews": int(review.sum()), "af_missed": int(np.sum(y & ~review)),
                "loss": reference.policy_cost(y, review, costs)}

    return {"stated": summary(stated), "best": summary(best_threshold),
            "interval": equivalent_interval(scores, best_threshold),
            "same_decisions": bool(np.all((scores >= stated) == (scores >= best_threshold)))}


def calibration_fit(records):
    """Calibration intercept and slope; each is None, with the reason printed, if it cannot be estimated.

    A score of exactly 0 or 1 has infinite log-odds, so neither can be
    estimated then; the Brier score and reliability points are still defined.
    """
    fit = calibration.intercept_slope(records["af"].to_numpy(dtype=bool),
                                      records["score"].to_numpy(dtype=float))
    for reason in fit["reasons"]:
        print(f"Calibration intercept or slope not available: {reason}.")
    return {"intercept": fit["intercept"], "slope": fit["slope"]}


def compare_frozen(evaluation, frozen, resamples=2000, seed=20261106):
    """The two frozen cutoffs on the evaluation split: each one's report and the paired NB difference.

    ``frozen`` holds ``stated`` and ``optimum`` cutoffs chosen on development
    and ``p_t``. Returns the two evaluation reports and NB(optimum) - NB(stated)
    with a 95% paired percentile bootstrap interval.
    """
    from .comparison import delta_net_benefit
    y, scores = evaluation["af"].to_numpy(dtype=bool), evaluation["score"].to_numpy(dtype=float)
    if len(y) < 2:
        print("Fewer than two evaluation recordings: the estimates are shown, but there is nothing "
              "to resample, so the intervals are unavailable.")
    stated, optimum = scores >= frozen["stated"], scores >= frozen["optimum"]
    print(f"Evaluation: {len(y)} recordings, {int(y.sum())} with AF; the two policies disagree on "
          f"{int(np.sum(stated != optimum))}.")
    return {"stated": reference.evaluation_report(y, stated, frozen["p_t"], resamples, seed),
            "optimum": reference.evaluation_report(y, optimum, frozen["p_t"], resamples, seed),
            "difference": delta_net_benefit(y, stated, optimum, frozen["p_t"], resamples, seed)}


def evaluation_label(untouched):
    """How to read the reveal, given the learner's statement about the evaluation recordings."""
    if untouched:
        return ("Held-out evaluation: you stated that no evaluation recording influenced your "
                "model, its preprocessing, its tuning or its calibration.")
    return ("Retrospective policy analysis: you have not stated that the evaluation recordings "
            "were untouched by your model's development, so this is not a held-out evaluation.")
