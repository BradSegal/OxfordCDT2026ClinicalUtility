"""Load small vendored labels and simulated scores without network access."""

import csv
import json
from importlib.resources import files

import numpy as np

MODEL_NAMES = {
    "random_forest": "Random forest · ECG features",
    "resnet": "1D ResNet · waveform",
    "resnet_calibrated": "1D ResNet · recalibrated",
}


def teaching_cases():
    return json.loads(files("clinical_utility").joinpath("assets/teaching_cases.json").read_text())


def provenance():
    return json.loads(files("clinical_utility").joinpath("assets/provenance.json").read_text())


SPLITS = ("development", "calibration", "evaluation")


def _rows():
    text = files("clinical_utility").joinpath("assets/predictions.csv").read_text()
    return list(csv.DictReader(text.splitlines()))


def load_records(model="random_forest", split="development"):
    """Assessable A/N/O records only; original reference labels remain visible.

    Noise records are retained in REFERENCE-v3.csv and counted in provenance.
    They are not assigned fabricated binary AF truth.
    """
    if model not in MODEL_NAMES:
        raise ValueError(f"Choose a model from {list(MODEL_NAMES)}.")
    if split not in SPLITS:
        raise ValueError("Choose development, calibration or evaluation.")
    rows = []
    for row in _rows():
        if row["split"] == split:
            rows.append({"recording": row["recording"], "reference_label": row["reference_label"],
                         "af_present": row["reference_label"] == "A", "score": float(row[model]),
                         "model": model, "split": split})
    return rows


def load_arrays(split="development"):
    """One split as numpy arrays: recording, reference_label, af_present and every model column.

    Rows are in the same order as ``load_records`` for that split.
    """
    if split not in SPLITS:
        raise ValueError("Choose development, calibration or evaluation.")
    rows = [row for row in _rows() if row["split"] == split]
    table = {"recording": np.array([r["recording"] for r in rows]),
             "reference_label": np.array([r["reference_label"] for r in rows])}
    table["af_present"] = table["reference_label"] == "A"
    for model in MODEL_NAMES:
        table[model] = np.array([float(r[model]) for r in rows])
    return table


def reference_labels():
    """Every CinC 2017 recording's reference label, A, N, O or ~ (noise), by recording ID."""
    text = files("clinical_utility").joinpath("assets/REFERENCE-v3.csv").read_text()
    return {row[0]: row[1] for row in csv.reader(text.splitlines()) if row}


def split_of():
    """The teaching split of every assessable (A, N or O) recording, by recording ID."""
    return {row["recording"]: row["split"] for row in _rows()}
