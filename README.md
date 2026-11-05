# How to make decisions from model outputs

Practical for the Oxford CDT in Healthcare Data Science module: Machine Learning for Time Series.

Bradley Segal<br />
[bradley.segal@eng.ox.ac.uk](mailto:bradley.segal@eng.ox.ac.uk)<br />
Computational Health Informatics Lab, Institute of Biomedical Engineering, Department of Engineering Science, University of Oxford

Organiser: Xiao Gu, Instructor: Bradley Segal, Demonstrators: Xiao Gu and Edward Phillips.

## What this practical is

This week you've built models to score ECG data for atrial fibrillation (AF).
A model score by itself does not change anything: someone needs to decide who is sent to a cardiologist for review.
This practical is a guide to making that decision.
You will learn how to put a value on the decision, understand what different cutoffs mean on the ROC and precision-recall curves, find the most cost-effective cutoff, translate the costs into a threshold probability, compare two models with decision curve analysis, recalibrate a model, and then make a policy recommendation.

| Notebook | Open |
|----------|------|
| [How to make decisions from model outputs](notebooks/clinical_utility.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BradSegal/OxfordCDT2026ClinicalUtility/blob/main/notebooks/clinical_utility.ipynb) |

## How to start

1. Click the **Open in Colab** button above. You will need to sign in with a Google account (a personal account is fine).
2. Make a copy of the notebook by going to **File > Save a copy in Drive**, and work in that copy.
3. Run the first setup cell to install the course helpers (this takes about 20 seconds).

This notebook is designed to run on a CPU-only runtime.
You do not need to upload any data; it is installed with the course helpers.

## How the tasks work

Within the notebook you will find tasks to complete, each marked by a `...` where you need to insert your own code.
Once you complete a task, run the cell: the checker under it tells you whether you've completed the task correctly or need to keep working on it.
You will see ✓ if you are correct, or **Not yet** with a tip if you need to keep working.
**Not attempted yet** means you still have a `...` that needs to be replaced with code.
If a cell relies on a previous task, it will print **Waiting for** and the name of that task.

Please note that demonstrators will be marking the practical based on your work during the session, so please complete the tasks in your own copy of the notebook.
At the end of the notebook is a progress cell that summarises your progress.
A worked solution will be uploaded to this repo after the session.

## The data

- **Recording IDs and reference labels** come from the [PhysioNet/CinC Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/) database (Clifford GD, Liu C, Moody B, Lehman LH, Silva I, Li Q, Johnson AE, Mark RG. AF classification from a short single lead ECG recording: the PhysioNet/Computing in Cardiology Challenge 2017. [doi:10.22489/CinC.2017.065-469](https://doi.org/10.22489/CinC.2017.065-469)), used under the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/), which continues to apply to them. Each recording is treated as an individual patient.
- **The model scores have been simulated** for educational purposes, based on the real label of each recording. They were not produced by an actual model and so do not reflect the performance of any model architecture (e.g. a random forest or a ResNet).
- **The dollar values** used in the notebook are round numbers for the exercise, built from published figures that the notebook cites. They do not reflect the true costs of any particular ward.

## After the session

Your copy of the notebook will be saved in your Google Drive, under the **Colab Notebooks** folder.
Its setup cell installs the course helpers from this repository, so it will continue to work as long as the repository remains online.

If you would like to run the practical locally (requires Python 3.10+):

```bash
git clone https://github.com/BradSegal/OxfordCDT2026ClinicalUtility.git
cd OxfordCDT2026ClinicalUtility
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e . jupyterlab
jupyter lab
```

## Licence

Copyright 2026 Bradley Segal.
The text and exercises are released under [CC BY 4.0](LICENSE-content.md) and the code under the [MIT licence](LICENSE).
The CinC 2017 reference labels remain under their original Open Data Commons Attribution licence.
