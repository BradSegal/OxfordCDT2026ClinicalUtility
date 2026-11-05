"""Load the vendored CinC 2017 labels and the course scores as pandas DataFrames."""

import io
import json
from functools import lru_cache
from importlib.resources import files

import pandas as pd

MODEL_NAMES = {
    "random_forest": "Random forest",
    "resnet": "1D ResNet",
}
SPLITS = ("development", "calibration", "evaluation")


def _asset(name):
    return files("clinical_utility").joinpath(f"assets/{name}").read_text()


@lru_cache(maxsize=1)
def provenance():
    return json.loads(_asset("provenance.json"))


@lru_cache(maxsize=1)
def _scores():
    # Read as text, then convert: every float was written with repr, and this keeps it exact.
    frame = pd.read_csv(io.StringIO(_asset("predictions.csv")), dtype=str)
    for model in ("random_forest", "resnet", "resnet_calibrated"):
        if model in frame:
            frame[model] = frame[model].astype(float)
    frame.insert(2, "af", frame["reference_label"] == "A")
    return frame


def load_scores():
    """Every assessable (A, N or O) recording, one row each.

    Columns: ``recording``, ``reference_label`` (A, N or O), ``af`` (True for A),
    ``split`` (development, calibration or evaluation) and one score column per
    model, ``random_forest`` and ``resnet``. Noisy recordings (``~``) are not
    here: they have no AF reference. Each call returns a fresh copy, so you can
    add columns.
    """
    return _scores().drop(columns="resnet_calibrated", errors="ignore")


def reference_labels():
    """Every CinC 2017 recording's reference label (A, N, O or ~) as a DataFrame."""
    return pd.read_csv(io.StringIO(_asset("REFERENCE-v3.csv")), header=None,
                       names=["recording", "reference_label"], dtype=str)
