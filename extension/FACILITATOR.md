# Facilitator notes: the extensions

The extension notebooks are not part of the live session; rather, they are self-guided explorations that each student works through alone after the core practical. These notebooks are self-contained: they load the reference functions from the course, and so a student does not need a working version of their own core notebook to work through them. The time estimates at the top of each notebook are intended to be rough guidelines for how long the first attempt should take to complete, from about <!-- number:extension.minutes.own|.0f -->25<!-- /number --> minutes for Extension 6 to about <!-- number:extension.minutes.calibration|.0f -->35<!-- /number --> for Extensions 1 and 5. These are not empirically determined.

| Notebook | Question | Time | Uses from the core |
|----------|----------|------|--------------------|
| [Extension 1: Calibration: Platt, isotonic and prior shift](01_calibration_and_platt_scaling.ipynb) | Can a score be read as an AF probability, and which recalibration should you trust? | about <!-- number:extension.minutes.calibration|.0f -->35<!-- /number --> min | Threshold probability; the Platt fit |
| [Extension 2: Comparing models: reclassification and net-benefit differences](02_comparing_models.ipynb) | Which recordings change decision, and is the change worth making? | about <!-- number:extension.minutes.comparison|.0f -->30<!-- /number --> min | Model choice by price; net benefit; resampling |
| [Extension 3: Class weighting versus moving the threshold](03_class_weighting_vs_thresholds.ipynb) | Does a weighted training loss buy a better review policy? | about <!-- number:extension.minutes.weighting|.0f -->30<!-- /number --> min | Prices to threshold; operating points |
| [Extension 4: Capacity, prevalence and daily queues](04_capacity_prevalence_queues.ipynb) | Can a policy that fits average capacity still overwhelm a service? | about <!-- number:extension.minutes.queues|.0f -->30<!-- /number --> min | Denominators; choosing a cutoff by loss |
| [Extension 5: Window-level calls and unreadable recordings](05_windows_and_unreadable_recordings.ipynb) | What should make a whole recording enter AF review, and what happens to one nobody can read? | about <!-- number:extension.minutes.windows|.0f -->35<!-- /number --> min | Counts and denominators; pricing a decision |
| [Extension 6: Your own classifier's scores](06_your_own_classifier.ipynb) | What does the core analysis say about the classifier you trained this week? | about <!-- number:extension.minutes.own|.0f -->25<!-- /number --> min | The whole core analysis, on the student's own scores |

Worked copies of these notebooks are in [solutions/](solutions/), and all use the same prices as the core practical, in reviews: $C_{TP}$ = <!-- number:extension.price.TP|.0f -->1<!-- /number -->, $C_{FP}$ = <!-- number:extension.price.FP|.0f -->1<!-- /number -->, $C_{FN}$ = <!-- number:extension.price.FN|.0f -->5<!-- /number -->, $C_{TN}$ = <!-- number:extension.price.TN|.0f -->0<!-- /number -->. These notebooks make use of ordinary function calls without any widgets. Each of the problems is labelled either Make or Fix the bug, and all use the same checkers as the core practical, which print ✓, "Not yet" with a hint, or "Not attempted yet". The worked copies are run with strict checks and so will raise an error if a check fails. A checker will report a value of the wrong kind, or a key or column missing from a dictionary or table, before checking any calculation.

**When a student is stuck.** Each answer that is required by a later cell has a commented-out stuck switch below it, for example `# project = check.use_reference("project")`; removing the `#` will swap in the reference version and will indicate this to the student. Fix-the-bug problems do not have a stuck switch, as the purpose is for the student to fix the code. Cells that need an unfinished answer will print a single `Waiting for ...` line naming the problem that needs to be solved or switched to continue. To proceed, once the problem is done or switched, the student will need to click in the waiting cell, select **Runtime > Run before**, and then run the waiting cell.

## Which extension to suggest

The final cell of the core notebook provides students with the same guidance:

- **Extension 1** for students that struggled with the concepts of calibration or understanding the decision curves.
- **Extension 2** for students that want to engage more deeply with model comparisons, including the reclassification indices typically requested by peer reviewers.
- **Extension 4** for students interested in the downstream implications of model deployment, such as capacity constraints, prevalence shift or overflow days.
- **Extension 6** for students that have scores from a classifier they've trained this week.
- **Extensions 3 and 5** are optional, for students interested in concepts such as class weighting or windowed modelling and unreadable recordings.

After the session, suggest a single extension to work through per sitting.

## Extension 1: Calibration: Platt, isotonic and prior shift

**Purpose.** Setting a threshold from the prices assumes that the score is a probability. This notebook provides a guided process for building a calibration report from library calls, fitting both Platt and isotonic maps on the calibration split, discovering that a comparison of the maps fitted on the evaluation split chose the wrong map, and finally, applying a prior shift to move a calibrated model to a new prevalence.

**Prerequisites.** The threshold probability and the Platt fit from the core practical. The logit and sigmoid functions are provided within the notebook.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Make), `calibration_report` | `brier_score_loss`; `calibration_curve(y, p, n_bins=10, strategy="quantile")`, which returns `(prob_true, prob_pred)`; calibration-in-the-large from a binomial GLM with a column of ones and `offset=logit(p)`; the slope from a binomial GLM on `add_constant(logit(p))` | The two `calibration_curve` arrays swapped; the intercept taken from the free-slope fit; a slope fitted without a constant |
| B (Make), `platt_map` | The core's pipeline, fitted on the calibration split | The default penalty; a fit on the raw score; a fit on development or evaluation |
| C (Make), `isotonic_map` | `IsotonicRegression(out_of_bounds="clip")` fitted on the calibration split | NaN for evaluation scores outside the calibration range (no clipping); the wrong split |
| D (Fix the bug), `compare_recalibrations` | Fit both maps on the calibration split and compare them on development, where the map is chosen; evaluation then reports the frozen choice | Both maps fitted and judged on evaluation; the choice made on evaluation; fitted on development |
| E (Make), `shift_prior` | Add logit(new) − logit(old) to logit(p), then `expit`; the old prevalence is the calibration split's AF fraction | Multiplying probabilities by the prevalence ratio; the shift the wrong way round |

**Numbers to expect.** On development, the raw ResNet has a calibration intercept of <!-- number:dev.resnet_raw.calibration_intercept|.2f -->-2.70<!-- /number --> and a slope of <!-- number:dev.resnet_raw.calibration_slope|.2f -->0.74<!-- /number -->. The top tenth of its scores has an average score of <!-- number:dev.resnet_raw.top_decile.mean_score|.3f -->0.944<!-- /number -->, but only <!-- number:dev.resnet_raw.top_decile.observed_af|.1% -->48.7%<!-- /number --> have AF. The Brier scores for the Platt and isotonic maps on development are <!-- number:cal.dev.platt.brier|.4f -->0.0559<!-- /number --> and <!-- number:cal.dev.isotonic.brier|.4f -->0.0565<!-- /number -->, respectively. The difference between the two is <!-- number:cal.dev.isotonic_minus_platt.estimate|.5f -->0.00057<!-- /number -->, with a paired interval of <!-- number:cal.dev.isotonic_minus_platt.low|.5f -->-0.00024<!-- /number --> to <!-- number:cal.dev.isotonic_minus_platt.high|.5f -->0.00146<!-- /number -->. This indicates that the two maps are close, and so the Platt map is frozen. If we were to fit and judge the maps on the evaluation split, the isotonic map's Brier score would fall to <!-- number:cal.leak.isotonic.brier|.4f -->0.0559<!-- /number --> and it would appear to be better than the Platt map. This is the leak. The Brier score on evaluation for the frozen Platt map is <!-- number:cal.eval.platt.brier|.4f -->0.0575<!-- /number -->. Following the prior shift to <!-- number:prior_shift.target_prevalence|.0% -->2%<!-- /number -->, the mean probability shifts from <!-- number:prior_shift.eval_resampled.mean_predicted_unshifted|.4f -->0.0699<!-- /number --> to <!-- number:prior_shift.eval_resampled.mean_predicted_shifted|.4f -->0.0201<!-- /number -->, compared to an observed value of <!-- number:prior_shift.eval_resampled.observed_prevalence|.4f -->0.0202<!-- /number -->.

**Sticking points.**

- The calibration-in-the-large intercept value is the value of the intercept if the slope is fixed at one, not the value of the intercept if both are free.
- Isotonic regression produces exactly zero (and potentially one) for extreme scores, which makes it impossible to estimate the calibration intercept and slope. The notebook mentions this and still produces the Brier score and the decisions. The piecewise nature of the map (flat blocks joined by linear interpolation in scikit-learn's `predict`) means that it collapses large ranges of raw scores into one value, which can result in the removal of achievable cutoffs.
- While the impact of the data leak is minimal on the two-parameter Platt scaling model, it flatters the more flexible isotonic regression model. This demonstrates that the comparison, and by extension which map should be used, is altered by the presence of the leak.
- Following the prior shift, almost no recordings are above the threshold at these prices. This is not an error but rather a natural consequence of the increased rarity of the condition, given this model.

**Question to consider.** If the calibration slope were also to change in a new service, what evidence would you need to make you comfortable using an intercept-only update?

## Extension 2: Comparing models: reclassification and net-benefit differences

**Purpose.** This notebook aims to familiarise students with the mechanics of calculating the reclassification indices that are typically reported in papers, here for the forest-to-ResNet comparison. It demonstrates how a positive NRI can sit beside a net benefit that is actually worse. The notebook then walks through putting an interval on the net-benefit difference with a paired bootstrap from `scipy.stats.bootstrap`. It then demonstrates how to audit which recordings change decision, and how to repair a comparison that valued each model at a different preference.

**Prerequisites.** Model choice by price and net benefit from the core practical.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Make), `categorical_nri` | Event NRI = P(up given AF) − P(down given AF); non-event NRI = P(down given no AF) − P(up given no AF); a score equal to the cut is reviewed | Dividing by the whole cohort; a strict comparison at the cut |
| B (Make), `delta_net_benefit` | An axis-aware statistic of the mean contribution, passed to `stats.bootstrap(..., paired=True, vectorized=True, method="percentile")` | Weighting false referrals by `p_t` rather than its odds; unpaired resampling, which widens the interval |
| C (Make), `changed_decisions` | A DataFrame of the recordings whose decision changes, in input order, with direction and contribution | Every recording whose score moved; the wrong direction or sign |
| D (Fix the bug), `compare_net_benefit` | Weight false referrals by the service's `p_t` for both models, with each model's decisions at its own cutoff | Each model's cutoff used as `p_t` |

**Numbers to expect.** At a cut of <!-- number:nri.models.cut|.1f -->0.2<!-- /number --> on development, <!-- number:nri.dev.events.up|.0f -->74<!-- /number --> AF recordings are reclassified upwards and <!-- number:nri.dev.events.down|.0f -->9<!-- /number --> downwards, out of a total of <!-- number:nri.dev.events.total|.0f -->341<!-- /number -->; <!-- number:nri.dev.non_events.up|.0f -->639<!-- /number --> non-AF recordings are reclassified upwards. Despite a categorical NRI total of <!-- number:nri.dev.total|+.3f -->+0.034<!-- /number -->, a continuous NRI of <!-- number:nri.dev.continuous.total|+.3f -->+0.985<!-- /number --> and an IDI of <!-- number:nri.dev.idi.total|+.3f -->+0.197<!-- /number -->, the net-benefit difference is <!-- number:nri.dev.delta_nb|.4f -->-0.0181<!-- /number --> (<!-- number:nri.dev.delta_nb_ci_low|.4f -->-0.0241<!-- /number --> to <!-- number:nri.dev.delta_nb_ci_high|.4f -->-0.0124<!-- /number -->). With the wrong preference, the ResNet has a net benefit of <!-- number:nri.preference_bug.resnet_raw.wrong|.3f -->-0.337<!-- /number --> instead of <!-- number:nri.preference_bug.resnet_raw.right|.3f -->0.053<!-- /number -->.

**Sticking points.**

- NRI is the addition of two within-group fractions, which treats both as equally valuable. The continuous variant of the NRI and the IDI have the additional concern that they reward movement of scores even if it does not change a decision. They are descriptions, not judgements.
- The bootstrapped intervals are constructed by resampling the set of recordings given both policies fixed. It does not take into account uncertainty from fitting the models or choosing the prices. If several recordings are from the same patient, the bootstrap would need to be constructed by sampling patients and not individual recordings.
- The cutoff determines who gets reviewed; the `p_t` value is determined by the prices. These are only the same if the model is calibrated.

**Question to consider.** If the service had a much higher concern for misses, what would you want to recalculate to determine which model to recommend?

## Extension 3: Class weighting versus moving the threshold

**Purpose.** The goal of this notebook is to have students compare the impact of using class weights within unpenalised logistic regressions on two rhythm features. The notebook is optional and demonstrates that most of the impact of using class weights is simply moving the model's intercept. The notebook includes a Fix-the-bug problem where students have to fix an evaluation that uses `predict()`. The notebook then moves to demonstrating how to use `TunedThresholdClassifierCV` to tune a threshold to the prices and then how to recover probabilities from the weighted model.

**Prerequisites.** Prices to threshold and `policy_cost` from the core practical.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Make), `fit_models` | `LogisticRegression(C=np.inf, solver="newton-cholesky")` without and with `class_weight="balanced"` on the training rows | The default penalty (`C=1`), under which weighting also changes the shrinkage and the intercept-only correction is approximate; no weights on the weighted model |
| B (Fix the bug), `test_report` | Each model's cutoff chosen on validation by loss, applied to `predict_proba` on the test rows | `predict()`, which applies <!-- number:core.cutoffs.half|.1f -->0.5<!-- /number --> to both models |
| C (Make), `tuned` | `TunedThresholdClassifierCV` with `make_scorer(policy_cost, greater_is_better=False, costs=costs)` and `cv=5`, fitted on the training rows | A scorer that maximises the loss; a scorer that is not the priced loss |
| D (Make), `undo_weighting` | `expit(logit(q) - log(w))` | Dividing the probability by the weight; the correction the wrong way round |

**Numbers to expect.** The intercept moves by <!-- number:weighting.intercept_shift|.2f -->2.06<!-- /number --> (close to the log weight ratio of <!-- number:weighting.log_ratio|.2f -->2.20<!-- /number -->), and the slope vectors have a cosine similarity of <!-- number:weighting.coefficient_cosine|.3f -->0.995<!-- /number -->. Evaluating with `predict()` suggests the weighted model improves sensitivity from <!-- number:weighting.unweighted.predict.sensitivity|.2f -->0.30<!-- /number --> to <!-- number:weighting.weighted.predict.sensitivity|.2f -->0.80<!-- /number -->; using the cutoffs chosen on validation, the two test losses are <!-- number:weighting.unweighted.validation_cut.loss|.0f -->162<!-- /number --> and <!-- number:weighting.weighted.validation_cut.loss|.0f -->160<!-- /number -->. The threshold tuner chooses <!-- number:weighting.tuned.threshold|.3f -->0.232<!-- /number -->. With the log-weight correction, the weighted model's Brier score on the test rows reduces from <!-- number:weighting.weighted.test_brier|.4f -->0.1344<!-- /number --> to <!-- number:weighting.corrected.test_brier|.4f -->0.0599<!-- /number -->.

**Sticking points.**

- `cv=5` is the tuner's internal selection of the threshold based on the training rows. This is separate to the validation rows in Problem B. The fitted tuner should be evaluated through `.predict()`.
- Note that with only <!-- number:weighting.test.af|.0f -->50<!-- /number --> AF recordings in the test rows, differences of a handful of recordings are within sampling noise.

**Question to consider.** If the log-odds were misspecified, or the model was a flexible model of limited capacity, how might using class weights change which patterns are learned? What experiment could you run to determine if this was occurring or if it was just a shift in intercept?

## Extension 4: Capacity, prevalence and daily queues

**Purpose.** The goal for this notebook is to map the forest model's development operating points to a planning prevalence. Students are required to select a cutoff based on a daily overflow tolerance using `scipy.stats.poisson.sf`. The notebook then compares an alternative strategy, a top-k cap on the number of daily reviews, on simulated daily batches of recordings. Finally, students determine how far the prevalence can drift before the chosen cutoff results in an overflow rate that exceeds the tolerance.

**Prerequisites.** Denominators and choosing a cutoff by loss from the core practical.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Make), `project` | Apply sensitivity to per × prevalence and the false-positive rate to the rest, vectorised over the table; price all four counts | The false-positive rate applied to every recording |
| B (Make), `capacity_policy` | Add `overflow = poisson.sf(capacity, reviews)`, keep rows within the tolerance, return the cheapest row | Mean reviews within capacity as the filter; returning only the cutoff |
| C (Make), `highest_safe_prevalence` | The highest prevalence on the grid at which the Problem B cutoff stays within the tolerance, as a proportion | A percentage; the first failing prevalence |

**Numbers to expect.** For a <!-- number:capacity.prevalence|.0% -->5%<!-- /number --> prevalence and a capacity of <!-- number:capacity.per_1000|.0f -->50<!-- /number --> reviews per day, the cheapest policy requests <!-- number:capacity.unconstrained.reviews|.1f -->72.5<!-- /number --> reviews. The cheapest policy that on average fits within the capacity requests <!-- number:capacity.mean_feasible.reviews|.1f -->49.7<!-- /number --> reviews, but exceeds capacity on <!-- number:capacity.mean_feasible.overflow|.0% -->44%<!-- /number --> of days based on the Poisson calculation. The cutoff that fits within a <!-- number:capacity.tolerance|.0% -->10%<!-- /number --> tolerance is <!-- number:capacity.tolerance.threshold|.3f -->0.543<!-- /number --> and results in <!-- number:capacity.tolerance.reviews|.1f -->41.4<!-- /number --> reviews and <!-- number:capacity.tolerance.af_missed|.1f -->22.7<!-- /number --> AF recordings missed per day. Using a top-k cap at the cheapest cutoff defers <!-- number:capacity.top_k.mean_deferred|.1f -->22.5<!-- /number --> recordings per day on average, <!-- number:capacity.top_k.mean_af_deferred|.1f -->6.7<!-- /number --> of which have AF. The cap results in <!-- number:capacity.top_k.mean_cap_missed|.1f -->20.2<!-- /number --> AF recordings missed per day, compared to <!-- number:capacity.top_k.mean_cutoff_missed|.1f -->13.5<!-- /number --> for the fixed cutoff. The tolerance cutoff is close to the edge: the highest safe prevalence on the grid is <!-- number:capacity.tolerance.highest_safe_prevalence|.1% -->5.0%<!-- /number -->, and the continuous boundary is about <!-- number:capacity.tolerance.boundary_prevalence|.2% -->5.14%<!-- /number -->.

**Sticking points.**

- A capacity is a constraint, not a change in the price of a review.
- Top-k changes the effective cutoff day-to-day, so fixed-cutoff sensitivity and specificity do not transfer. The notebook simulates daily batches and evaluates performance assuming the planning prevalence.
- The model assumes independent arrivals with no carry-over of backlog. Clustering and backlog modify the frequency, volume, and latency of overflow. Ask students to separate these out.
- The tolerance is a service preference, not a statistical result.

**Question to consider.** If demand for the service is 'bursty', i.e., it clusters and the queue carries over from the previous day, which assumptions break down? What additional data from the service would you want to collect before choosing a new policy?

## Extension 5: Window-level calls and unreadable recordings

**Purpose.** The CinC dataset provides a single label per recording and so it is not possible to validate window-level localisation on it. Students instead use a set of recordings with a simplified version of the CinC dataset recording lengths (<!-- number:windows.seconds.min|.0f -->30<!-- /number --> to <!-- number:windows.seconds.max|.0f -->60<!-- /number --> seconds with <!-- number:windows.window_seconds|.0f -->10<!-- /number -->-second windows every <!-- number:windows.stride_seconds|.0f -->5<!-- /number --> seconds) and known AF intervals. The student is tasked with creating rules for converting window calls to a recording decision, selecting one based on one half of the data, and then reporting it on the other half. The student then has to fix a sensitivity calculation that uses the incorrect granularity. Finally, the student needs to price the various options for managing a recording that is unreadable. This notebook is optional.

**Prerequisites.** Counts, denominators and pricing a decision from the core practical. Additional requirement: Boolean numpy arrays.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Make), `aggregate` | Any, strict majority of the recording's own windows, and at least k positive | Half treated as a majority; the padded width instead of the recording's own windows |
| B (Make), `persistent` | `sliding_window_view(positive, k, axis=1).all(axis=-1).any(axis=1)` | Counting positives anywhere |
| C (Make), `rule_table` | Per rule: reviews, AF detected and missed, artefact reviews and brief AF missed per 1,000, and the priced loss | Wrong counts or a loss without all four prices |
| D (Fix the bug), `recording_sensitivity` | Apply the rule, then `recall_score` against the recording labels | A window-level recall reported as recording sensitivity |
| E (Make), `unreadable_losses` | The four pathways priced as in the notebook's table, with unresolved over-reads leaving AF undetected | An over-read priced as an ordinary review |

**Numbers to expect.** On the choose half, the cheapest rule is *<!-- number:windows.chosen_name|s -->any window<!-- /number -->*. On the report half, it reviews <!-- number:windows.report.any.reviews|.0f -->598<!-- /number --> per 1,000 recordings, misses <!-- number:windows.report.any.af_missed|.1f -->10.1<!-- /number --> AF per 1,000 and reviews <!-- number:windows.report.any.artefact_reviewed|.0f -->206<!-- /number --> artefact recordings per 1,000. For unreadable recordings, accepting is optimal for an assumed AF probability of <!-- number:extension.preference.low|.1f -->0.1<!-- /number -->, and repeating from about <!-- number:unreadable.repeat_from|.2f -->0.26<!-- /number -->.

**Sticking points.**

- The rule is chosen on the choose half and reported on the report half, irrespective of the rule. In this case, a rule that asks for a second positive window results in far more missed brief episodes (<!-- number:windows.paroxysmal.two_or_more|.0f -->49<!-- /number --> out of <!-- number:windows.kind.paroxysmal|.0f -->79<!-- /number --> light up two or more windows), and so at these prices it is inferior to the rule that reviews on any window, at the expense of reviewing nearly every artefact recording.
- The duration of time for a run of k windows would be `window + (k - 1) * stride` seconds long. This is the period of time the windows cover, not an established duration of AF.
- Unreadable recordings cannot be given the ordinary review benefit. The pathways differentiate between AF established on a recording that has a conclusive read and AF referred for assessment. The probability of AF within this group is an assumption the service would need to measure.

**Question to consider.** If the rate of unreadable recordings was higher during AF, how would you modify the pathways, and what data would you need to determine their probabilities?

## Extension 6: Your own classifier's scores

**Purpose.** This notebook is intended to have a student take the scores from the classifier they trained this week and run them through the core analysis. The notebook demonstrates how to join the scores to the CinC labels using pandas without making assumptions. It then requires the student to analyse the development recordings and then freeze two policies (the threshold implied by the prices and the optimal cutoff on development). The student then needs to compare the two policies on the evaluation recordings by calculating the paired difference in net benefit.

**Prerequisites.** The core notebook, and a `recording,score` CSV file with their classifier's AF scores. (If one is not available, the notebook uses a demo column that includes two noise recordings, an unknown ID, a duplicated ID and a score above one, to ensure that every branch of the join is exercised.)

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Make), `join_scores` | Duplicated IDs (every copy), unknown IDs, scores outside zero to one or NaN, and noise recordings excluded and reported, in that order; the rest kept in input order with `af` true only for A. It has a stuck switch | Noise kept as AF-negative; one copy of a duplicate kept; scores outside the unit interval kept; O treated as AF |
| B (Make), `frozen` | Both cutoffs, the prices and the `p_t` they imply, set before the reveal | `p_t` not following from the prices; a missing cutoff. The checker never reads the evaluation split |

**What to stress.** The reveal is only a held-out evaluation if no evaluation recording was ever used for any part of the model. This includes training, preprocessing or feature selection, hyperparameter tuning, early stopping, checkpoint or model selection, recalibration, or even an earlier choice of threshold. Unless the student sets `EVALUATION_UNTOUCHED = True`, the reveal is by default labelled as a retrospective policy analysis. The development results for the optimum are apparent; the key comparison is on evaluation.

**Sticking points.**

- A minimum-loss cutoff that does not align with the stated threshold is not necessarily an indicator of poor calibration. This is due to the presence of regions where the same decisions would be made across a whole interval of cutoffs. The notebook outputs that interval and leaves the determination of calibration to the reliability plot and, if it is possible to estimate them, the calibration intercept and slope.
- The notebook has a variety of edge cases based on the uploaded data. If students do not upload any development recordings, it will be impossible to freeze a policy. If they do not upload any evaluation recordings, there will be nothing to reveal. If they only have a single outcome class, it will not be possible to determine J or fit the calibration model. If their scores contain values of exactly zero or one, it will not be possible to determine the calibration intercept and slope. If their scores are perfectly separated, only the slope will be excluded. Each case will result in a single line printed in the notebook to this effect, with the calculations that are defined executing as normal.
- If students have very few AF recordings, the bootstrap intervals on the evaluation set can be unreliable, can collapse (zero width if both policies make the same decisions), or may be unavailable. Please make sure they report the number of AF recordings and the number of disagreements (printed with the reveal) and not over-interpret a single result.

**Question to consider.** Given this classifier, what would you report in addition to the AUC? What would a service need to know prior to acting on it?
