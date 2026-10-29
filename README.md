# From AF scores to clinical decisions

**A practical on clinical utility for Oxford's CDT module: Machine Learning for Time Series.** Friday 6 November 2026, 14:30–16:00.

---

Bradley Segal<br />
[bradley.segal@eng.ox.ac.uk](mailto:bradley.segal@eng.ox.ac.uk)<br />
Computational Health Informatics Lab<br />
Institute of Biomedical Engineering,<br />
Department of Engineering Science,<br />
University of Oxford

Organiser: Xiao Gu. Instructor: Bradley Segal. Demonstrators: Xiao Gu and Edward Phillips.

---

The practical's text and exercises are licensed under [CC BY 4.0](LICENSE-content.md). The code is licensed under the [MIT licence](LICENSE). Both may be freely reused and modified with appropriate attribution.

---

## What this practical is

You've already trained classifiers for ECG recordings from the PhysioNet/CinC 2017 dataset. Each one outputs a score for a given recording. The challenge is that a service can't act on a score: it needs to decide, recording by recording, whether to send the recording for AF review. This practical bridges the gap from a score to that decision.

You will price the four outcomes of the decision, convert these prices into a threshold, write the function that prices a set of decisions, and then use this to choose a threshold and model. You will see why maximising Youden's J and minimising expected loss result in different cutoffs, why the preferred model can change with the price of missing an AF recording, why you need to recalibrate a model's score before a stated threshold means anything, as well as how net benefit and decision curves follow from the same priced matrix. Finally, you will freeze a policy, read its evaluation report with uncertainty intervals, and write the decision record a service would file.

This practical is intended for PhD students who use pandas and scikit-learn. The notebooks leverage standard library functions (`confusion_matrix`, `roc_curve`, `calibration_curve`, `LogisticRegression`, `IsotonicRegression`, GLMs from statsmodels, `scipy.stats.bootstrap`) and only require you to write code where no library does the job. The practical does not train or run any ECG models.

Everything runs within [Google Colab](https://colab.research.google.com), a free service that enables you to run Python code from your browser. For this practical you only need a CPU runtime. The practical does not download anything except a small helper package for the course content; there are no model weights.

## How to start

1. Click on the **Open in Colab** badge for the core practical in the table below.
2. If prompted, sign in to your Google account to continue. This can be a personal account.
3. Click **File > Save a copy in Drive**. This will open a new tab with your own copy of the notebook. Ensure you work on this copy. If you try to edit the original, your changes will not be saved.
4. Run the first code cell to set up the notebook. Colab warns that the notebook "was not authored by Google"; when prompted, click **Run anyway** to confirm that you trust the code, which comes from this course. This code sets up the helper functions for the course and should take around 20 seconds to complete. You will see `Course helpers ready.` once finished.
5. Work through the cells sequentially, pressing **Shift+Enter** to run each cell.

The [lab sheet](Clinical_Utility_Lab_Sheet.pdf) includes the same setup instructions, the formulas you will use, and the route through the session on a two-page sheet.

## Learning path

| No. | Notebook | What you learn | Time | Open |
|-----|----------|----------------|------|------|
| Core | [From AF scores to clinical decisions](notebooks/01_clinical_utility.ipynb) | Price a decision matrix, choose a threshold and a model, recalibrate a score, read net benefit and decision curves, then freeze, evaluate and record a policy | about 75 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/notebooks/01_clinical_utility.ipynb) |
| Extension 1 | [Calibration: Platt, isotonic and prior shift](extension/01_calibration_and_platt_scaling.ipynb) | Build a calibration report, fit Platt and isotonic maps on the calibration split, repair a leaky comparison, and move a calibrated model to a new prevalence | about 35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/01_calibration_and_platt_scaling.ipynb) |
| Extension 2 | [Comparing models: reclassification and net-benefit differences](extension/02_comparing_models.ipynb) | Reclassification, NRI and IDI, a paired bootstrap of the net-benefit difference, an audit of the recordings that change decision, and a comparison valued at the wrong preference | about 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/02_comparing_models.ipynb) |
| Extension 3 | [Class weighting versus moving the threshold](extension/03_class_weighting_vs_thresholds.ipynb) | Fit weighted and unweighted logistic regressions, repair an evaluation that used `predict()`, and tune a threshold to the prices with scikit-learn | about 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/03_class_weighting_vs_thresholds.ipynb) |
| Extension 4 | [Capacity, prevalence and daily queues](extension/04_capacity_prevalence_queues.ipynb) | Project counts to a planning prevalence, choose a cutoff under a daily overflow tolerance, compare a top-k cap, and find how far prevalence can drift | about 30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/04_capacity_prevalence_queues.ipynb) |
| Extension 5 | [Window-level calls and unreadable recordings](extension/05_windows_and_unreadable_recordings.ipynb) | Turn overlapping window scores into one recording-level call, chosen on one half and reported on the other, and price the options for an unreadable recording | about 35 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/05_windows_and_unreadable_recordings.ipynb) |
| Extension 6 | [Your own classifier's scores](extension/06_your_own_classifier.ipynb) | Join your own classifier's scores to the CinC labels, analyse them on development, then freeze two policies and compare them on evaluation recordings | about 25 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/extension/06_your_own_classifier.ipynb) |

The core practical is the live session. This is worked through independently, with a short discussion facilitated by the instructor at the end. Don't stress if you don't complete it during the session. The extensions are self-paced follow-up work. Each is standalone and reuses the priced matrix from the core practical. If you struggled with the decision-curve section, try Extension 1; if you want to compare models more carefully, try Extension 2; if you want to think through deployment, try Extension 4; or if you have scores from your own classifier, try Extension 6. Extensions 3 & 5 are optional. The indicated times are rough estimates for a first attempt, not measurements.

## How the problems work

Problems are flagged as either **Make** (construct a function or value) or **Fix the bug** (resolve an issue with code that runs but produces an incorrect result, as can occur in real-world evaluation code). Where you see `...`, substitute it with your answer and execute the cell. The automatic checker in that cell will indicate how you did:

- ✓ for correct;
- **Not yet**, with a hint on what you likely did wrong; or
- **Not attempted yet**, with the `...` still up.

You can run a cell as many times as you like. Nothing is collected or marked, and the checker will never block notebook execution. Cells without a problem will run the relevant library code for you and are followed by a note on what to look for in the output. Additional explanations are included as folded content and are best read after attempting the problem.

**If you are stuck.** If you've been stuck on a problem for about ten minutes, please move on.

- Within the core notebook, there are four answers (Problems 1, 3, 5 and 6) that later cells need. These are indicated by a commented `check.use_reference(...)` line below them. To proceed using the reference version, delete the `#` and run the code block.
- Within the extensions, the answers that are required for the subsequent code blocks to run are similarly indicated with the same switch below them. Note that the fix-the-bug problems do not have this switch, as the goal of these problems is to repair the error.
- If a subsequent code block needs an unfinished function, it will print `Waiting for ...`, notifying you which problem needs to be completed or 'uncommented' to proceed. Once this has been completed, select the code block that is waiting and choose **Runtime > Run before**. This will re-run all prior code blocks (including any others that were waiting for the same problem), and then you can run the waiting block.
- For any of the problems, one worked answer is shown in the worked copy (see below).

The core notebook includes a few interactive elements using sliders. Each has an ordinary function call in the next cell that produces the same figure: if the slider does not load, change its numbers and re-run it.

## Worked solutions

Each notebook has a paired worked copy that includes the answers to each problem. The worked copy for the core practical is in [`solutions/`](solutions/), and those for the extensions are in [`extension/solutions/`](extension/solutions/). Each worked copy has its own Colab badge.

Please attempt to solve the problems yourself prior to looking at the worked copies. If you're stuck, first check the hint from the automatic checker; if you're still stuck, you can look at that one problem in the worked copy, then close it and write the answer yourself. Please note that there is often more than one good answer. Where several cutoffs tie, the automatic checker will accept any of them.

## The data

- **Recording IDs and reference labels** are from a real-world dataset, namely the [PhysioNet/CinC Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/) database: Clifford GD, Liu C, Moody B, Lehman LH, Silva I, Li Q, Johnson AE, Mark RG. AF classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017. [doi:10.22489/CinC.2017.065-469](https://doi.org/10.22489/CinC.2017.065-469). These labels are used in accordance with the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/), which continues to apply to them.
- **The course model scores are simulated for teaching**, from stated mixtures conditional on each recording's real label. The terms "random forest" and "ResNet" are used to indicate the type of model that could be trained; however, no model was trained, and as such nothing here is indicative of how such a model would perform. The window recordings in Extension 5 and the two rhythm features in Extension 3 are synthetic, and Extension 6 leverages scores from either your own model or a demonstration column.
- **The prices** are in units of reviews, where a review has a price of one and a missed AF recording has a price of five reviews. They are not an empiric determination of health economics but are instead set by the service.

Please see [the data notes](src/clinical_utility/assets/SOURCES.md) for full attribution details as well as how the scores were generated.

## Using the notebooks after the session

Any copies you've saved are in your Google Drive under the **Colab Notebooks** folder. If you wish to restart, you can open the notebook using its badge above and save a new copy to your Drive. The setup cell for the notebook installs the necessary helper code from this repo, so your copies keep working as long as the repo is online.

## Running on your own computer (optional)

This is not required for the session. If you wish to set up the practical locally, you'll need a Python environment (3.10 or newer):

```bash
git clone https://github.com/BradSegal/OxfordCDT2026ClinicalUtility.git
cd OxfordCDT2026ClinicalUtility
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e . jupyterlab
jupyter lab
```

If you're using [uv](https://docs.astral.sh/uv/), you can simplify the last four lines to just `uv run --with jupyterlab jupyter lab`.

## A note on safety

The code within this repository is teaching code, written to be read and learnt from. It is not a validated tool and must not be used to make decisions about the care of real patients. While the practical can be used with your own classifier's scores (Extension 6), please be aware that the reveal step is only a valid held-out evaluation if no evaluation recording was involved in any way in developing that model. Even if this is true, it only reflects performance on these recordings with the specific prices defined, and is not a validated pathway.

## Licence

Copyright 2026 Bradley Segal. The course text and exercises (notebooks, lab sheet, guides) are licensed under [CC BY 4.0](LICENSE-content.md). The code (the `clinical_utility` helper package & the notebooks' code cells) is licensed under the [MIT licence](LICENSE). The reference labels from the CinC 2017 dataset keep their Open Data Commons Attribution licence; see [the data notes](src/clinical_utility/assets/SOURCES.md) for details.

For instructor and demonstrator notes, please see [INSTRUCTORS.md](INSTRUCTORS.md) & [extension/FACILITATOR.md](extension/FACILITATOR.md).
