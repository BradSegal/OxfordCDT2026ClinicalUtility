# From AF scores to clinical decisions

**A clinical-utility practical for the Oxford CDT module "Machine Learning for Time Series".** Friday 6 November 2026, 14:30–16:00.

---

Bradley Segal<br />
[bradley.segal@eng.ox.ac.uk](mailto:bradley.segal@eng.ox.ac.uk)<br />
Computational Health Informatics Lab<br />
Institute of Biomedical Engineering,<br />
Department of Engineering Science,<br />
University of Oxford

Organiser: Xiao Gu. Instructor: Bradley Segal. Demonstrators: Xiao Gu and Edward Phillips.

---

The text and exercises are licensed under [CC BY 4.0](LICENSE-content.md) and the code under the [MIT licence](LICENSE). You are free to reuse and adapt both with appropriate attribution.

---

## What this practical is

You have already trained ECG classifiers on the PhysioNet/CinC 2017 recordings. Each one outputs a score. A service cannot act on a score: it has to decide, recording by recording, whether to send the recording for AF review. This practical is about that decision.

You will price the four outcomes of the decision, derive a threshold from those prices, write the function that prices a policy, and use it to choose a threshold and a model. You will see why maximising Youden's J and minimising expected loss pick different cutoffs, why the preferred model can change with the price of a missed AF recording, and how net benefit and decision curves follow from the same priced matrix.

It is written for doctoral students who can read and write short Python functions with dictionaries and loops. No model is trained or run.

Everything runs in [Google Colab](https://colab.research.google.com), a free service that runs Python in your web browser. A CPU runtime is enough. Nothing is downloaded apart from the small course helper package; there are no model weights.

## How to start

1. Click the **Open in Colab** badge for the core practical in the table below.
2. Sign in with a Google account if Colab asks. A personal account is fine.
3. Click **File > Save a copy in Drive**. The copy opens in a new tab; work in that one. If you work in the original, your changes are not saved.
4. Run the first code cell, the setup cell. Colab warns that the notebook "was not authored by Google". Click **Run anyway**: the notebook comes from this course. The cell installs the course helpers in about 20 seconds and prints `Course helpers ready.`
5. Work down the notebook in order, running each cell with **Shift+Enter**.

The [lab sheet](Clinical_Utility_Lab_Sheet.pdf) gives the same steps, the formulas you will use and the route through the session on two pages.

## Learning path

| No. | Notebook | What you learn | Time | Open |
|-----|----------|----------------|------|------|
| Core | [From AF scores to clinical decisions](notebooks/01_clinical_utility.ipynb) | Price a decision matrix, choose a threshold and a model, read net benefit and decision curves, then freeze and defend a policy | about 80 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/notebooks/01_clinical_utility.ipynb) |
| Extension 1 | [Calibration and Platt scaling](extension/01_calibration_and_platt_scaling.ipynb) | Build a reliability summary, fit a recalibration on held-out recordings, catch a leaky assessment, and move a calibrated model to a new prevalence | about 40 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/01_calibration_and_platt_scaling.ipynb) |
| Extension 2 | [Comparing models properly](extension/02_comparing_models.ipynb) | Reclassification, NRI and IDI, a paired bootstrap of the net-benefit difference, and an audit of the recordings that change decision | about 45 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/02_comparing_models.ipynb) |
| Extension 3 | [Class weighting versus moving the threshold](extension/03_class_weighting_vs_thresholds.ipynb) | Fit weighted and unweighted logistic regressions and test whether weighting buys anything a different cutoff cannot | about 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/03_class_weighting_vs_thresholds.ipynb) |
| Extension 4 | [Capacity, prevalence and daily queues](extension/04_capacity_prevalence_queues.ipynb) | Project counts to a new prevalence, respect a review capacity, simulate days that overflow it, and find how far prevalence can drift | about 35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/04_capacity_prevalence_queues.ipynb) |
| Extension 5 | [Window-level calls and unreadable recordings](extension/05_windows_and_unreadable_recordings.ipynb) | Turn window scores into one recording-level call, and price a repeat recording for an unreadable one | about 40 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/05_windows_and_unreadable_recordings.ipynb) |
| Extension 6 | [Your own classifier's scores](extension/06_your_own_classifier.ipynb) | Join your own classifier's scores to the CinC labels, compare J with minimum loss, then freeze a policy and check it on evaluation recordings untouched by your model's development | about 20 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/06_your_own_classifier.ipynb) |

The core practical is the live session: problems in pairs, then a closing discussion. Do not worry if you do not finish it in the session. The extensions are self-paced follow-up work. Each reuses the core's priced matrix and runs on its own. If the decision-curve section was hard, take Extension 1; to compare models more carefully, Extension 2; for deployment, Extension 4; and if you have scores from your own classifier, Extension 6. Extensions 3 and 5 are optional. The times are rough guides for a first attempt, not measurements.

## How the problems work

Each problem is labelled **Predict** (work out the answer before running anything), **Make** (write a function or a value), **Modify** (change working code) or **Fix the bug**. Where you see `...`, replace it with your answer and run the cell. The checker in that cell tells you how you did:

- ✓ correct;
- **Not yet**, with a hint that names the likely mistake; or
- **Not attempted yet**, while the `...` is still there.

Run a cell as many times as you like. Nothing is collected or graded, and a checker never stops the notebook. Two core predictions, Problems 7 and 13, are hypotheses made before you have the evidence: the checker records your answer without marking it, and the next cell shows the result beside it. Folded explanations headed "Open after your own attempt" are best read once you have tried the problem.

**If you are stuck.** Give a problem about ten minutes, then move on.

- In the core notebook, the three functions that later cells need (Problems 4, 6 and 12) each have a commented `check.use_reference(...)` line below them. Delete the `#` and run it to carry on with the reference version.
- In the extensions, every function that a later cell uses has the same switch below it. Fix-the-bug functions have none, because the repair is the exercise.
- A later cell that needs an unfinished function prints `Waiting for ...`, naming the problem and how to unblock it. Once that problem is done or switched, click in the waiting cell, choose **Runtime > Run before** (it reruns every cell above, including any that waited for the same problem), then run the waiting cell.
- For any problem, the worked copy shows one answer (see below).

The core notebook has a few sliders. Each slider has an ordinary function call in the next cell that draws the same figure: change its numbers and rerun it if a slider does not appear.

## Worked solutions

Every notebook has a worked copy with the answers filled in: [`solutions/`](solutions/) for the core practical and [`extension/solutions/`](extension/solutions/) for the extensions. Each has its own Colab badge.

Make a real attempt and read the checker's hint first. If you are still stuck, look at that one problem in the worked copy, close it and write the answer yourself. There is often more than one good answer, and where several cutoffs tie, any of them is accepted.

## The data

- **Recording IDs and reference labels** are real. They come from the [PhysioNet/CinC Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/) database: Clifford GD, Liu C, Moody B, Lehman LH, Silva I, Li Q, Johnson AE, Mark RG. AF classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017. [doi:10.22489/CinC.2017.065-469](https://doi.org/10.22489/CinC.2017.065-469). The labels are used under the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/), which continues to apply to them.
- **Every course model score is simulated** for teaching. "Random forest" and "ResNet" name plausible approaches. No model was trained, and nothing here measures how such models perform.
- **The six teaching recordings**, the window-level recordings and the two rhythm features in Extension 3 are invented. In Extension 6 the scores are your own, or a simulated demonstration column if you have none.
- **The prices** are illustrative loss units, not measured health economics.

Full attribution and how the scores were generated are in [the data notes](src/clinical_utility/assets/SOURCES.md).

## Using the notebooks after the session

Your saved copies are in your Google Drive under **Colab Notebooks**. To start again, open a notebook from its badge here and save a new copy. The setup cell installs the helpers from this repository, so your copies keep working while the repository is online.

## Running on your own computer (optional)

Not needed for the session. With Python 3.10 or newer:

```bash
git clone https://github.com/BradSegal/OxfordCDT2026ClinicalUtility.git
cd OxfordCDT2026ClinicalUtility
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e . jupyterlab
jupyter lab
```

With [uv](https://docs.astral.sh/uv/) the last four lines become `uv run --with jupyterlab jupyter lab`.

## A note on safety

This is teaching code, written to be read and learnt from. It is not a validated tool and must not be used to make decisions about real patients. The course example scores are simulated, so no result from them says how well a real classifier or review pathway would work. Extension 6 can use your own classifier's scores; its reveal is a held-out evaluation only if no evaluation recording played any part in developing that model, and even then it describes these recordings and prices, not a validated pathway.

## Licence

Copyright 2026 Bradley Segal. The course text and exercises (notebooks, lab sheet and guides) are licensed under [CC BY 4.0](LICENSE-content.md); the code, including the `clinical_utility` package and the notebooks' code cells, under the [MIT licence](LICENSE). The CinC 2017 reference labels keep their Open Data Commons Attribution licence; see [the data notes](src/clinical_utility/assets/SOURCES.md).

Instructors and demonstrators: see [INSTRUCTORS.md](INSTRUCTORS.md) and [extension/FACILITATOR.md](extension/FACILITATOR.md).
