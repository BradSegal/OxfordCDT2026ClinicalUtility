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
import math
from pathlib import Path

import numpy as np

from . import calibration, data, reference
from .logistic import logit, sigmoid

DEMO_NOTE = ("Using the simulated demonstration column, not your classifier. It blends the "
             "course's two simulated models and includes a few deliberate problems so you can "
             "see what the join reports.")


def parse(text):
    """Rows of {"recording", "score"} from CSV text; a score that is not a number becomes NaN."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
    if "recording" not in fields or "score" not in fields:
        raise ValueError("The first line must be a header containing 'recording' and 'score'.")
    rows = []
    for row in reader:
        try:
            score = float(row[fields["score"]])
        except (TypeError, ValueError):
            score = float("nan")
        rows.append({"recording": (row[fields["recording"]] or "").strip(), "score": score})
    return rows


def demo_scores():
    """A simulated score for every development and evaluation recording, plus five problems.

    The score blends the two simulated models on the log-odds scale. The
    problems are two noise recordings, an unknown ID, a duplicated ID and a
    score above 1, so the join has something to report.
    """
    table = {split: data.load_arrays(split) for split in ("development", "evaluation")}
    rows = []
    for split in table.values():
        blended = sigmoid((logit(split["random_forest"]) + logit(split["resnet_calibrated"])) / 2)
        rows += [{"recording": str(r), "score": float(s)} for r, s in zip(split["recording"], blended)]
    noise = [r for r, label in data.reference_labels().items() if label == "~"][:2]
    rows += [{"recording": r, "score": .5} for r in noise]
    outside = str(data.load_arrays("calibration")["recording"][0])
    rows += [{"recording": "Z99999", "score": .3}, dict(rows[0]),
             {"recording": outside, "score": 1.2}]
    return rows


def read_scores(path="", pasted=""):
    """(rows, note): the learner's scores from a file or pasted text, else the demonstration column."""
    try:
        if path:
            if not Path(path).exists():
                raise ValueError(f"No file called {path!r}. Upload it in Colab's Files pane first.")
            return parse(Path(path).read_text()), f"Read {path}."
        if pasted.strip():
            return parse(pasted), "Read the pasted scores."
    except (ValueError, OSError, UnicodeDecodeError) as error:
        return demo_scores(), f"Could not read your scores ({error}). {DEMO_NOTE}"
    return demo_scores(), DEMO_NOTE


def with_splits(records):
    """Attach the course split to each joined record; the noise and unknown IDs are already gone."""
    splits = data.split_of()
    return {split: [r for r in records if splits.get(r["recording"]) == split]
            for split in data.SPLITS}


def score_array(records):
    """Scores and labels of joined records as numpy arrays, for the calibration summaries."""
    return (np.array([r["score"] for r in records], dtype=float),
            np.array([r["af_present"] for r in records], dtype=bool))


def join_scores(scores, labels):
    """The reference join: exclude duplicated, unknown, out-of-range and noise rows, in that order."""
    seen = {}
    for row in scores:
        seen[row["recording"]] = seen.get(row["recording"], 0) + 1
    result = {"records": [], "duplicates": [], "unknown": [], "out_of_range": [], "noise": 0}
    for row in scores:
        recording, score = row["recording"], row["score"]
        if seen[recording] > 1:
            if recording not in result["duplicates"]:
                result["duplicates"].append(recording)
        elif recording not in labels:
            result["unknown"].append(recording)
        elif not 0 <= score <= 1:
            result["out_of_range"].append(recording)
        elif labels[recording] == "~":
            result["noise"] += 1
        else:
            result["records"].append({"recording": recording, "score": score,
                                      "af_present": labels[recording] == "A"})
    return result


# Below this many recordings in either outcome group, say that results are very uncertain.
FEW = 20


def describe_splits(own):
    """Print each split's size, the development AF count only, and what the data cannot support.

    Evaluation outcomes are not summarised here: they belong to the reveal.
    """
    development, evaluation = own["development"], own["evaluation"]
    n_af = sum(r["af_present"] for r in development)
    print(f"development: {len(development)} recordings, {n_af} with AF")
    print(f"evaluation: {len(evaluation)} recordings (outcomes stay hidden until the reveal)")
    if own.get("calibration"):
        print(f"calibration: {len(own['calibration'])} recordings, not used here")
    if not development:
        print("No development recordings: there is nothing to choose a policy on, so the "
              "development analysis, the freeze and the reveal cannot run.")
    elif n_af in (0, len(development)):
        print(f"Every development recording {'has' if n_af else 'lacks'} AF: J and the calibration "
              "fit are undefined. The minimum-loss sweep still runs.")
    elif min(n_af, len(development) - n_af) < FEW:
        print(f"Fewer than {FEW} development recordings in one outcome group: every result below "
              "is very uncertain.")
    if not evaluation:
        print("No evaluation recordings: you can freeze a policy, but there is nothing to reveal "
              "it on.")


def max_j(sweep):
    """The maximum-J operating point (lowest cutoff on ties), or None, with the reason, if J is undefined."""
    if not any(math.isfinite(row["youden_j"]) for row in sweep):
        print("J is undefined: it needs at least one recording with AF and one without.")
        return None
    return reference.best_thresholds(sweep, "youden_j", minimise=False)[0]


def equivalent_interval(records, threshold):
    """The cutoffs that review exactly the recordings ``threshold`` reviews.

    A policy reviews a recording when its score is at or above the cutoff, so
    the bounds come from the largest excluded score and the smallest reviewed
    one: every c with low < c <= high is equivalent. When every recording is
    reviewed, the interval is [0, smallest score], closed at 0; when none is,
    it runs from just above the largest score to 1.01, the sweep's review-none
    cutoff. Returns {"low", "high", "low_included"}.
    """
    scores = [r["score"] for r in records]
    excluded = [s for s in scores if s < threshold]
    reviewed = [s for s in scores if s >= threshold]
    if not excluded:
        return {"low": 0.0, "high": min(reviewed) if reviewed else 1.01, "low_included": True}
    return {"low": max(excluded), "high": min(reviewed) if reviewed else 1.01,
            "low_included": False}


def interval_text(interval):
    """The interval in words, for example "above 0.300 and up to 0.400"."""
    if interval["low_included"]:
        return f"from {interval['low']:.3f} up to {interval['high']:.3f}"
    return f"above {interval['low']:.3f} and up to {interval['high']:.3f}"


def same_decisions(records, first, second):
    """True when two cutoffs review exactly the same recordings."""
    return all((r["score"] >= first) == (r["score"] >= second) for r in records)


def _losses(records, threshold, costs):
    return np.array([row["loss"] for row in reference.evaluate_policy(records, threshold, costs)])


def stated_versus_optimum(records, costs, stated, best, resamples=1000, seed=20261106):
    """Decisions and loss at a stated cutoff beside the minimum-loss cutoff ``best``.

    Returns the two policy summaries, the interval of cutoffs equivalent to
    ``best``, whether ``stated`` lies in it, the extra loss at ``stated``, and a
    95% percentile interval for that extra loss from a paired bootstrap of the
    recordings with both cutoffs held fixed. Because ``best`` was chosen on the
    same recordings, the interval leaves out the optimism of that choice.
    """
    difference = _losses(records, stated, costs) - _losses(records, best["threshold"], costs)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(resamples, len(difference)))
    totals = difference[draws].sum(axis=1)
    return {"stated": reference.policy_summary(records, stated, costs),
            "best": reference.policy_summary(records, best["threshold"], costs),
            "interval": equivalent_interval(records, best["threshold"]),
            "same_decisions": same_decisions(records, stated, best["threshold"]),
            "extra_loss": float(difference.sum()),
            "extra_loss_low": float(np.quantile(totals, .025)),
            "extra_loss_high": float(np.quantile(totals, .975))}


def calibration_fit(records):
    """Calibration intercept and slope; each is None, with the reason printed, if it cannot be estimated.

    Scores of exactly 0 or 1 have infinite log-odds, so they are moved 1e-6
    inside the unit interval for this fit only.
    """
    p, y = score_array(records)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if y.all() or not y.any():
        print("Calibration intercept and slope are undefined: these recordings have only one outcome.")
        return {"intercept": None, "slope": None}
    fit = {}
    for key, function in (("intercept", calibration.calibration_in_the_large),
                          ("slope", calibration.calibration_slope)):
        try:
            value = function(p, y)
        except ValueError:
            print(f"The calibration {key} cannot be estimated: your scores separate AF from non-AF "
                  "completely here, so the logistic fit has no finite answer. Read the reliability "
                  "summary instead.")
            value = None
        if value is not None and not math.isfinite(value):
            print(f"The calibration {key} is undefined: every score is the same.")
            value = None
        fit[key] = value
    return fit


def evaluation_label(untouched):
    """How to read the reveal, given the learner's statement about the evaluation recordings."""
    if untouched:
        return ("Held-out evaluation: you stated that no evaluation recording influenced your "
                "model, its preprocessing, its tuning or its calibration.")
    return ("Retrospective policy analysis: you have not stated that the evaluation recordings "
            "were untouched by your model's development, so this is not a held-out evaluation.")
