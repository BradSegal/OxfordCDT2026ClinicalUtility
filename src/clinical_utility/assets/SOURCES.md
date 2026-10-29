# Teaching data and attribution

`REFERENCE-v3.csv` is the unchanged recording reference file from the
[PhysioNet/CinC 2017 dataset](https://physionet.org/content/challenge-2017/1.0.0/),
acquired 28 October 2026. The IDs and N/A/O/~ labels in `predictions.csv` derive
from that file. Attribution: Clifford GD, Liu C, Moody B, Lehman LH, Silva I,
Li Q, Johnson AE, Mark RG. AF classification from a short single lead ECG
recording: The PhysioNet/Computing in Cardiology Challenge 2017.
DOI: [10.22489/CinC.2017.065-469](https://doi.org/10.22489/CinC.2017.065-469).

PhysioNet platform citation: Pollard T, Moody BE, Lehman LH, et al. (2026).
PhysioNet as a global platform for biomedical research. Nature Health 1, 792–795.
DOI: [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).

The reference database is licensed under the
[Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).
These database attribution terms continue to apply to the included labels.
The course code licence does not replace them.

The three score columns (`random_forest`, `resnet`, `resnet_calibrated`) are
**simulated predictions**, drawn conditional on each recording's reference label
from stated Gaussian mixtures for a decision-theory exercise; `provenance.json`
records the mixtures, seed and recalibration parameters. They are not outputs
of trained classifiers, evidence of the performance of named architectures, or
calibrated probabilities by assertion. `resnet` is deliberately overconfident;
`resnet_calibrated` is its Platt recalibration (an unpenalised scikit-learn
logistic regression on logit(score)), fitted on the separate calibration
partition only.

The six hand-checkable recordings used in the core notebook are real
development recordings chosen by the rule recorded in `provenance.json`, which
also gives the generation and partition details.

The window-level recordings (`clinical_utility.windows`) and the two rhythm
features used for the class-weighting demonstration
(`clinical_utility.cost_sensitive`) are entirely synthetic and generated in code
with fixed seeds. They do not derive from CinC data.
