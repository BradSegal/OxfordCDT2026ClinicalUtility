# Facilitator notes: the extensions

The extensions are self-paced follow-up to the core practical. They are not part of the live session. Each notebook runs on its own: it loads the reference functions, so a student does not need their own core answers. Each opens with a rough time, from about <!-- number:extension.minutes.own|.0f -->20<!-- /number --> minutes for Extension 6 to about <!-- number:extension.minutes.comparison|.0f -->45<!-- /number --> for Extension 2. These are planning estimates for a first attempt, not times measured with students; some will take much longer.

| Notebook | Question | Time | Uses from the core |
|----------|----------|------|--------------------|
| [Extension 1: Calibration and Platt scaling](01_calibration_and_platt_scaling.ipynb) | Can a score be read as an AF probability? | about <!-- number:extension.minutes.calibration|.0f -->40<!-- /number --> min | Threshold probability; net benefit |
| [Extension 2: Comparing models properly](02_comparing_models.ipynb) | Which recordings change decision, and is the change worth making? | about <!-- number:extension.minutes.comparison|.0f -->45<!-- /number --> min | Model choice by price; net benefit; the flip-price resampling |
| [Extension 3: Class weighting versus moving the threshold](03_class_weighting_vs_thresholds.ipynb) | Does a weighted training loss buy a better review policy? | about <!-- number:extension.minutes.weighting|.0f -->30<!-- /number --> min | Prices to threshold; exact cutoff search |
| [Extension 4: Capacity, prevalence and daily queues](04_capacity_prevalence_queues.ipynb) | Can a policy that fits average capacity still overwhelm a service? | about <!-- number:extension.minutes.queues|.0f -->35<!-- /number --> min | Denominators; choosing a cutoff by loss |
| [Extension 5: Window-level calls and unreadable recordings](05_windows_and_unreadable_recordings.ipynb) | What should make a whole recording enter AF review? | about <!-- number:extension.minutes.windows|.0f -->40<!-- /number --> min | Counts and denominators; pricing a decision |
| [Extension 6: Your own classifier's scores](06_your_own_classifier.ipynb) | What does today's analysis say about the classifier you trained this week? | about <!-- number:extension.minutes.own|.0f -->20<!-- /number --> min | The whole core analysis, on the student's own scores |

Worked copies are in [solutions/](solutions/). All share the core's prices: $C_{TP}$ = <!-- number:extension.price.TP|.0f -->1<!-- /number -->, $C_{FP}$ = <!-- number:extension.price.FP|.0f -->1<!-- /number -->, $C_{FN}$ = <!-- number:extension.price.FN|.0f -->5<!-- /number -->, $C_{TN}$ = <!-- number:extension.price.TN|.0f -->0<!-- /number -->, in illustrative loss units. The CinC labels are real and the course scores simulated. The two rhythm features in Extension 3 and the window recordings in Extension 5 are entirely synthetic. Extension 6 uses the student's own scores, or a simulated demonstration column if they have none.

Every extension uses ordinary function calls; none needs a widget. Problems are labelled Predict, Make, Modify or Fix the bug, and the checkers print ✓, "Not yet" with a hint, or "Not attempted yet", as in the core. A checker tests the code or value, and reports a value of the wrong kind or a dictionary without the expected keys before it compares any calculation; each starter that returns a structure shows its keys. The free-text cells (`reason`, `leak`, `explanation`, `overflow_plan`, `missing`, `why_it_reverses`) need a person to read them. In the worked copies, run with strict checks, a failed check raises.

**When a student is stuck.** As in the core, every function that a later cell uses has a commented stuck switch below it, for example `# expected_counts = check.use_reference("expected_counts")`; deleting the `#` swaps in the reference version and says so. Fix-the-bug functions have no switch, because the repair is the exercise. A cell that needs an unfinished function prints one `Waiting for ...` line naming the problem and how to unblock it; no combination of finished and unfinished problems ends in a traceback. Once the problem is done or switched, the student clicks in the waiting cell, chooses **Runtime > Run before**, then runs the waiting cell, as in the core.

## Which extension to suggest

The core notebook's last cell gives students the same advice:

- **Extension 1** for anyone who found *From prices to decision curves* hard. It builds the recalibration and the reliability plot that the core only mentions.
- **Extension 2** for anyone who wants to compare models more carefully.
- **Extension 4** for anyone interested in deployment: capacity, prevalence drift and overflow days.
- **Extension 6** for anyone with scores from a classifier they trained this week. It is the shortest and the most direct follow-up to the session.
- **Extensions 3 and 5** are optional, for students curious about class weighting or about window-level models and unreadable recordings.

After the session, suggest one extension per sitting. As a follow-up seminar, pick one extension, give pairs the full time and use its closing prompt for a ten-minute discussion. The same rule as the core applies to solutions: a real attempt and a hint first, then one problem in the worked copy, then close it and write the answer.

## Extension 1: Calibration and Platt scaling

**Purpose.** A threshold derived from prices assumes the score is a probability. Students build a reliability summary, fit a Platt recalibration on the calibration split, repair a leaky assessment and transfer the map to a new prevalence. Better probabilities can change decisions at a stated preference without changing the ranking. This extension carries the reliability plot that left the core.

**Prerequisites.** From the core: the threshold probability, net benefit and the remark on calibration in *From prices to decision curves*. Basic numpy indexing. The notebook defines the logit and the sigmoid itself, with one worked value, so students need not have met them.

**Time.** Allow about <!-- number:extension.minutes.calibration|.0f -->40<!-- /number --> minutes.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Make), `probability_summary` | Sort scores and labels together with `np.argsort`, split into equal-count groups with `np.array_split`, return the mean score, observed AF fraction and size of each group, and the Brier score over every recording | Sorting the labels separately; squaring the mean residual; averaging bin errors instead of recording errors |
| B (Predict), `ranking_prediction` | `"same"`, with a sentence in `reason`: a positive-slope Platt map keeps the order and ties, so the exact minimum loss is unchanged | Mistaking probability repair for a ranking improvement |
| C (Modify), `my_logit` and `fit_platt` | The log-odds, then Newton steps for logistic regression on `logit(score)` with an intercept column: gradient `X.T @ (y - p)`, information `X.T @ ((p*(1-p))[:, None] * X)` | Regressing on the probability (a legitimate calibration model, but not the logit-linear specification asked for); `log(p)` in place of the log-odds |
| D (Fix the bug), `calibration_brier` | Fit the map on the calibration split and compute the Brier score on the evaluation split; `leak` says the original fitted to the evaluation outcomes it was judged on | Fitting on evaluation; assessing on calibration; fitting on development; dropping the map |
| E (Make), `shift_prior` | Add the new-minus-old prior log-odds to the calibrated logit, then apply the sigmoid; the old prevalence is the calibration split's AF fraction | Multiplying probabilities by the prevalence ratio; reversing the shift |

**Numbers to expect.** In the top tenth of raw ResNet evaluation scores the mean score is <!-- number:eval.resnet_raw.top_decile.mean_score|.3f -->0.944<!-- /number --> but only <!-- number:eval.resnet_raw.top_decile.observed_af|.1% -->49.8%<!-- /number --> have AF. Raw and recalibrated ResNet have the same development minimum loss, <!-- number:dev.resnet_raw.min_loss.total_loss|.0f -->922<!-- /number --> and <!-- number:dev.resnet_cal.min_loss.total_loss|.0f -->922<!-- /number -->. Recalibration raises net benefit by <!-- number:nb.dev.p0_1.resnet_cal_minus_raw|.4f -->0.0054<!-- /number -->, <!-- number:nb.dev.p0_2.resnet_cal_minus_raw|.4f -->0.0102<!-- /number --> and <!-- number:nb.dev.p0_4.resnet_cal_minus_raw|.4f -->0.0321<!-- /number --> per recording at preferences <!-- number:extension.preference.low|.1f -->0.1<!-- /number -->, <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number --> and <!-- number:extension.preference.high|.1f -->0.4<!-- /number -->; only the middle one follows the default matrix. After the prior shift to <!-- number:prior_shift.target_prevalence|.0% -->2%<!-- /number -->, the mean predicted probability moves from <!-- number:prior_shift.eval_resampled.mean_predicted_unshifted|.4f -->0.0699<!-- /number --> to <!-- number:prior_shift.eval_resampled.mean_predicted_shifted|.4f -->0.0201<!-- /number -->, against an observed AF fraction of <!-- number:prior_shift.eval_resampled.observed_prevalence|.4f -->0.0202<!-- /number -->.

**Sticking points.**

- Calibration-in-the-large is the intercept with the slope fixed at one. It is not the free intercept from the slope fit.
- The Brier score mixes calibration and resolution. A lower Brier score does not by itself mean better calibration, and it does not choose a clinical preference.
- Isotonic regression (pool-adjacent-violators) pools neighbouring blocks whose AF fractions decrease, weighted by block size. Its flat steps create ties, so unlike the positive-slope Platt map it can remove achievable thresholds.
- The prior shift assumes the score distributions within AF and within non-AF are stable. It does not assume PPV is stable, and the agreement checks this scenario only.

**Discussion.** If the calibration slope also changes in a new service, what evidence would you need before trusting an intercept-only update? Ask which device or patient-mix changes would break the within-class assumption.

## Extension 2: Comparing models properly

**Purpose.** Students compute reclassification measures for the forest-to-ResNet comparison and find that a positive NRI sits beside a worse net benefit. The lesson is that movement indices describe change; only a priced comparison says whether the change helps. A paired bootstrap puts an interval on the difference, extending the core's flip-price resampling.

**Prerequisites.** From the core: model choice by price and net benefit. Extension 1 helps with the remark on miscalibration.

**Time.** Allow about <!-- number:extension.minutes.comparison|.0f -->45<!-- /number --> minutes.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Predict), `denominator` | `"within_outcome"`, predicted before the formula is shown; the answer and the formulas are folded below the problem: each component divides by its own outcome group | Dividing both components by the whole cohort |
| B (Make), `categorical_nri` | Event NRI = P(up given AF) − P(down given AF); non-event NRI = P(down given no AF) − P(up given no AF); add them. A score equal to the cut is in the upper category | Wrong denominators; excluding equality at the cut |
| C (Modify), `movement_indices` | Continuous NRI counts strict increases and decreases only; IDI is the mean score difference within each group | Counting ties; carrying categories into continuous NRI |
| D (Make), `paired_bootstrap` | Mean per-recording difference in net-benefit contribution, new minus old, with a <!-- number:extension.interval.level|.0% -->95%<!-- /number --> percentile interval from resampling recording indices, the same draw for both models and the labels. Other valid index layouts are accepted | Resampling the two models independently; a narrower interval; weighting by $p_t$ rather than its odds |
| E (Make), `changed_recordings` | IDs, direction and signed contribution for the recordings whose review decision changes, in input order | Including every recording whose score moved; strict comparison at the cut; pricing a non-AF change at one |
| F (Fix the bug), `recommend` | Choose by net benefit at the stated preference, returning `{"winner": "random_forest", "basis": "net_benefit", "preference": ...}`, with a sentence in `explanation` separating the description (NRI) from the judgement (net benefit) | Treating positive NRI as utility; hard-coding a winner |

**Numbers to expect.** At a cut of <!-- number:nri.pair.cut|.1f -->0.2<!-- /number --> on the development split, <!-- number:nri.dev.events.up|.0f -->74<!-- /number --> AF recordings move up and <!-- number:nri.dev.events.down|.0f -->9<!-- /number --> down, out of <!-- number:nri.dev.events.total|.0f -->341<!-- /number -->. Event NRI is <!-- number:nri.dev.event|+.3f -->+0.191<!-- /number -->, non-event NRI <!-- number:nri.dev.non_event|+.3f -->-0.157<!-- /number --> and the total <!-- number:nri.dev.total|+.3f -->+0.034<!-- /number -->. The net-benefit difference is <!-- number:nri.dev.delta_nb|.3f -->-0.018<!-- /number -->, with a <!-- number:nri.bootstrap.resamples|,.0f -->1,000<!-- /number -->-resample percentile interval from <!-- number:nri.dev.delta_nb_ci_low|.3f -->-0.024<!-- /number --> to <!-- number:nri.dev.delta_nb_ci_high|.3f -->-0.012<!-- /number -->.

**Sticking points.**

- Before the bootstrap, a table and a five-recording example show what each changed decision contributes at $p_t$ = <!-- number:nri.pair.cut|.1f -->0.2<!-- /number -->: <!-- number:extension.bridge.af_up|+.0f -->+1<!-- /number --> or <!-- number:extension.bridge.af_down|+.0f -->-1<!-- /number --> for an AF recording, <!-- number:extension.bridge.non_af_up|+.2f -->-0.25<!-- /number --> or <!-- number:extension.bridge.non_af_down|+.2f -->+0.25<!-- /number --> for one without AF. Students who cannot see why the mean of these is a difference in net benefit should work that example by hand.
- NRI adds two within-group proportions as if they were equally valuable. It applies no prevalence weight and no prices, so it can reward a miscalibrated model.
- The bootstrap resamples recordings with the scores held fixed. It says nothing about uncertainty from fitting the models or choosing the prices, and it does not validate a service. Several recordings from one patient would need the patient as the sampling unit.
- The audit names one AF and one non-AF recording, gives both scores and both actions, and states the changed consequence. It cannot give a physiological reason without more evidence.

**Discussion.** If a service cared much more about misses, which calculation would you redo before recommending a model?

## Extension 3: Class weighting versus moving the threshold

**Purpose.** Students fit weighted and unweighted logistic regressions on two synthetic rhythm features and find that, in this correctly specified example, weighting mostly moves the intercept. Moving the threshold on the unweighted model reaches almost the same priced minimum. This extension is optional.

**Prerequisites.** From the core: prices to threshold and the exact cutoff search. The Newton loop is supplied and tested, so the exercise is about where the weights enter. The logit and sigmoid are defined in the notebook, and the weight-ratio example is worked on the log-odds scale.

**Time.** Allow about <!-- number:extension.minutes.weighting|.0f -->30<!-- /number --> minutes.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Predict), `weight_prediction` | `"intercept"`, predicted before the derivation | Expecting a different ranking from a different objective |
| B (Make), `balanced_weights` | Each training row gets `n/(2*n_class)` for its class, in the original order | Equal row weights rather than equal class totals; the right ratio at the wrong scale |
| C (Modify), `fit_weighted` | Put the row weights into both the gradient and the information lines of the supplied Newton fit | Weights missing from the gradient, which is the unweighted fit |
| D (Make), `minimum_loss` | Sweep every exact score cut plus review-all and review-none, count all four outcomes and return the minimum priced loss | Leaving out the price of correct reviews |
| E (Fix the bug), `undo_weighting` | Subtract the log weight ratio on the log-odds scale, then apply the sigmoid | Dividing probabilities by the weight; subtracting on the probability scale; the wrong direction |

**Numbers to expect.** The two slope vectors have cosine similarity <!-- number:weighting.coefficient_cosine|.3f -->0.997<!-- /number -->, and test AUC is <!-- number:weighting.unweighted.auc|.3f -->0.909<!-- /number --> unweighted against <!-- number:weighting.weighted.auc|.3f -->0.907<!-- /number --> weighted. The intercept moves by <!-- number:weighting.intercept_shift|.2f -->2.27<!-- /number -->, close to the log weight ratio of <!-- number:extension.weight.log_ratio|.2f -->2.20<!-- /number -->. At a cutoff of <!-- number:extension.half|.1f -->0.5<!-- /number --> the unweighted model reviews <!-- number:weighting.unweighted.reviews_at_0_5|.0f -->49<!-- /number --> recordings and the weighted one <!-- number:weighting.weighted.reviews_at_0_5|.0f -->233<!-- /number -->, but the exact minimum losses are <!-- number:weighting.unweighted.min_loss|.0f -->299<!-- /number --> and <!-- number:weighting.weighted.min_loss|.0f -->303<!-- /number -->.

**Sticking points.**

- The derivation is that the weighted odds are the weight ratio times the true odds. With linear log-odds this is an intercept shift at the population optimum; finite-sample slopes still move a little.
- Weighting only the gradient still reaches the weighted optimum. The notebook asks for both lines so that each step is a true Newton step.
- Compare priced minima, not review counts at <!-- number:extension.half|.1f -->0.5<!-- /number -->. A cutoff of <!-- number:extension.half|.1f -->0.5<!-- /number --> has no special meaning under these prices.
- The minima are chosen on the test half in hindsight. They describe what was attainable, not what a deployed model would achieve.

**Discussion.** With misspecified log-odds, or a flexible model with limited capacity, how might weighting change which patterns are learned? What experiment would separate that from an intercept shift?

## Extension 4: Capacity, prevalence and daily queues

**Purpose.** Students project counts to a lower prevalence, choose the best policy that fits a review capacity on average, simulate days to find that it still overflows often, choose under an explicit overflow tolerance, and find how far prevalence can drift before a fixed cutoff breaks that tolerance.

**Prerequisites.** From the core: denominators, prevalence and choosing a cutoff by priced loss.

**Time.** Allow about <!-- number:extension.minutes.queues|.0f -->35<!-- /number --> minutes.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Predict), `overflow_prediction` | `"sometimes"`, predicted before the explanation | Treating the expected workload as a daily bound |
| B (Make), `expected_counts` | Apply sensitivity to population × prevalence and the false-positive rate to the rest. The worked example (sensitivity <!-- number:extension.example.se|.0% -->80%<!-- /number -->, specificity <!-- number:extension.example.sp|.0% -->90%<!-- /number -->, prevalence <!-- number:extension.example.prevalence|.0% -->5%<!-- /number -->, <!-- number:extension.example.population|,.0f -->1,000<!-- /number --> recordings) gives TP <!-- number:extension.example.TP|.0f -->40<!-- /number -->, FN <!-- number:extension.example.FN|.0f -->10<!-- /number -->, FP <!-- number:extension.example.FP|.0f -->95<!-- /number -->, TN <!-- number:extension.example.TN|.0f -->855<!-- /number --> | Applying the false-positive rate to every recording |
| C (Modify), `select_mean_feasible` | Keep the rows whose expected reviews fit capacity, then return the threshold of the cheapest | Picking the unconstrained optimum |
| D (Make), `fraction_over_capacity` | Poisson arrivals each day, then binomial reviews; the fraction of days strictly above capacity. A day-by-day loop is accepted | Counting a day exactly at capacity as overflow |
| E (Fix the bug), `select_policy` | Filter on the stated overflow tolerance, then minimise priced loss; `overflow_plan` says what happens to recordings on an overflow day | Mean feasibility as daily service; minimising loss without the filter |
| F (Modify), `highest_safe_prevalence` | It uses `mean_cut` and `project`, so it needs Problems B and C or their switches. The highest prevalence on the grid from <!-- number:extension.queue.grid.low|.0% -->1%<!-- /number --> to <!-- number:extension.queue.grid.high|.0% -->10%<!-- /number --> at which the Problem C cutoff still meets the tolerance, as a proportion | Answering the planning prevalence; a percentage; the first failing prevalence |

**Numbers to expect.** At a prevalence of <!-- number:capacity.prevalence|.0% -->5%<!-- /number --> and a capacity of <!-- number:capacity.per_1000|.0f -->50<!-- /number --> reviews per <!-- number:capacity.queue.recordings_per_day|,.0f -->1,000<!-- /number --> recordings, the unconstrained optimum requests <!-- number:capacity.unconstrained.reviews|.1f -->72.5<!-- /number --> reviews. The best policy within capacity on average requests <!-- number:capacity.feasible.reviews|.1f -->49.7<!-- /number -->, at a higher loss (<!-- number:capacity.feasible.loss|.1f -->148.6<!-- /number --> against <!-- number:capacity.unconstrained.loss|.1f -->139.2<!-- /number --> per <!-- number:capacity.queue.recordings_per_day|,.0f -->1,000<!-- /number -->). In the seeded simulation that policy exceeds capacity on <!-- number:capacity.queue.feasible.fraction_over_capacity|.0% -->47%<!-- /number --> of days. Problems E and F use an overflow tolerance of <!-- number:extension.queue.tolerance|.0% -->10%<!-- /number -->.

**Sticking points.**

- A capacity is a constraint, not a higher price for a review. Raising the FP price is a different question.
- Top-k reviews a fixed number from each day's batch, so its implied cutoff changes from day to day. Fixed-threshold sensitivity and specificity do not carry over to it. The demonstration compares the cap with the Problem C cutoff.
- The simulation assumes independent arrivals and no queue carried to the next day. Clustering, weekday effects and a carried-over backlog break that model, but they need not make overflow more frequent: a daily rate that varies around the same mean can overflow on fewer days, by more on the busy ones, and a backlog lengthens waits. Ask students to separate how often overflow happens, how large it is, and how long recordings wait.
- The tolerance is a service preference supplied for the exercise, not a statistical result.

**Discussion.** If demand is clustered and yesterday's queue carries over, which assumptions fail, and what service data would you collect before choosing a new policy?

## Extension 5: Window-level calls and unreadable recordings

**Purpose.** CinC gives one label per recording, so window-level localisation cannot be validated on it. On synthetic recordings with known AF windows, students compare rules that turn window scores into a recording-level call, then price a repeat recording for an unreadable one. This extension is optional.

**Prerequisites.** From the core: counts, denominators and pricing a decision. Boolean numpy arrays and loops.

**Time.** Allow about <!-- number:extension.minutes.windows|.0f -->40<!-- /number --> minutes.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Predict), `localisation_validated` | `"no"`, with `missing` naming interval truth: a recording label says AF is present somewhere, not when | Copying a recording label onto every window |
| B (Make), `aggregate` | Any-positive, strict majority and count-at-least-k decisions, each reducing over the windows of one recording | Treating half as a majority; excluding equality at k |
| C (Modify), `persistent` | Walk each recording's windows and reset the run at every negative window | Counting separated spikes as persistence; forgetting a run that ends the recording |
| D (Make), `recording_counts` | Recording-level TP, FP, FN and TN from one decision and one label per recording, checked on unequal counts | Swapping FP and FN; swapping FN and TN |
| E (Fix the bug), `recording_sensitivity` | Apply the rule passed in to get one decision per recording, then take the fraction of AF recordings reviewed | Counting positive windows over AF recordings; a window-level rate; ignoring the rule passed in |
| F (Make), `repeat_losses` | Price review now, accept now and repeat, with the repeat and delay costs paid once and the readable and unreadable branches weighted by `q` and `1 - q`; `why_it_reverses` explains the change when only `q` falls | Leaving out the unreadable branch; a no-review fallback; charging the repeat twice |

**Numbers to expect.** Any-positive gives <!-- number:windows.any_positive.FP|.0f -->108<!-- /number --> false-positive reviews, including all <!-- number:windows.kind.artefact|.0f -->86<!-- /number --> artefact recordings. Majority misses all <!-- number:windows.kind.paroxysmal|.0f -->79<!-- /number --> brief paroxysms. Persistence with a run of <!-- number:extension.window.count|.0f -->3<!-- /number --> has <!-- number:windows.persistence_3.FP|.0f -->0<!-- /number --> false positives and flags <!-- number:windows.persistence_3.flagged.paroxysmal|.0f -->76<!-- /number --> of the <!-- number:windows.kind.paroxysmal|.0f -->79<!-- /number --> paroxysms. For an unreadable recording with an assumed AF probability of <!-- number:extension.repeat.probability|.0% -->30%<!-- /number -->, a repeat costs <!-- number:repeat.p0_3.repeat|.3f -->0.893<!-- /number -->, reviewing now <!-- number:repeat.p0_3.review_now|.3f -->1.000<!-- /number --> and accepting without review <!-- number:repeat.p0_3.no_review_now|.3f -->1.500<!-- /number -->. If only <!-- number:extension.repeat.low_completion|.0% -->20%<!-- /number --> of repeats are readable, instead of <!-- number:repeat.input.completion_probability|.0% -->80%<!-- /number -->, the repeat costs <!-- number:extension.repeat.low_completion.repeat|.3f -->1.123<!-- /number --> and reviewing now is preferred.

**Sticking points.**

- The CinC noise label does not give an AF probability. The probability in the repeat tree is an assumption.
- The tree assumes AF status does not change between recordings, readability does not depend on AF, and the readable repeat performs like the original. A delay cost can reverse the choice.
- Count-at-least-k and persistence disagree on artefacts because a count ignores whether positive windows are adjacent. The run length was set for the example, not chosen by optimisation.
- The synthetic windows have no stated length or overlap, so a run is counted in window indices. A folded question asks how overlap changes the rule: a run of `k` windows of length `L` taken every `s` seconds covers `L + (k - 1) * s` seconds, so overlapping windows make the same `k` a weaker requirement. That is the span the windows cover, not an established duration of continuous AF: each window is called on its own.
- Before Problem F, a worked example with other inputs prices each branch of the repeat tree; point students back to it if they charge the repeat twice or forget the unreadable branch.

**Discussion.** If unreadability is more common during AF, how would you change the repeat tree, and what data would identify its branch probabilities?

## Extension 6: Your own classifier's scores

**Purpose.** Students run the core analysis on scores from the classifier they trained this week: join the scores to the CinC labels without guessing, compare J with minimum loss on their own development recordings, ask what acting on a stated <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number --> would cost, then freeze a policy and predict its evaluation loss before the reveal.

**Prerequisites.** The core notebook, and a `recording,score` CSV of AF scores. Without one, the notebook says so and uses a simulated demonstration column that blends the two course models and includes two noise recordings, an unknown ID, a duplicated ID and a score above one, so every branch of the join is exercised.

**Time.** Allow about <!-- number:extension.minutes.own|.0f -->20<!-- /number --> minutes.

| Problem | Expected answer | Checker catches |
|---------|-----------------|-----------------|
| A (Make), `join_scores` | Duplicated IDs (every copy), unknown IDs, scores outside zero to one or NaN, and noise recordings are excluded and reported, checked in that order; the rest are kept in input order with `af_present` true only for `A`. It has a stuck switch | Keeping noise as AF-negative; keeping one copy of a duplicate; keeping scores outside the unit interval; treating O as AF; a result without the stated keys |
| B (Predict), `j_prediction` | Whether J's cutoff is lower or higher than the minimum-loss cutoff on the student's own development recordings, or the same. The checker computes the answer from their data and accepts any relation that a pair of tied optima supports. If J is undefined (one outcome class) it says so instead of marking. The worked copy's `"lower"` is the demonstration column's answer | A prediction carried over from the core without looking at their own case mix |
| C (Predict), `final_threshold` and `predicted_evaluation_loss` | A cutoff frozen on development and a predicted total loss on their evaluation recordings, scaled by the number of recordings, before the reveal. Before the freeze the notebook shows only how many evaluation recordings there are. A correctly scaled zero is accepted | Predicting the unscaled development total; an incomplete policy |

**What to stress.** Learners upload the CSV in Colab's Files pane and set `SCORES_FILE`, or paste its lines into `PASTED_SCORES`. Not having trained on a recording is not enough to make it a fair test. The reveal is a held-out evaluation only if no evaluation recording influenced the model at all: not in training, preprocessing or feature selection, hyperparameter tuning, early stopping, checkpoint or model selection, recalibration, or an earlier choice of threshold. A validation set used for early stopping fails that test. Unless the student sets `EVALUATION_UNTOUCHED = True`, the reveal labels itself a retrospective policy analysis.

**Sticking points.**

- A minimum-loss cutoff far from <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number --> does not by itself show that the scores are miscalibrated. A whole interval of cutoffs makes the same decisions, so a distant optimum can make exactly the decisions <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number --> makes, and sampling variation moves an empirical optimum. The notebook therefore compares the decisions and the loss at a stated <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number --> with the optimum, gives the interval of equivalent cutoffs and a bootstrap interval for the extra loss, and leaves the calibration judgement to the reliability summary and, where they can be estimated, the intercept and slope. A large, stable extra loss is a reason to investigate calibration; Extension 1 shows the repair.
- Uploads vary. The notebook checks split sizes and class counts first. With no development recordings it cannot choose a policy; with no evaluation recordings it has nothing to reveal; with one outcome class J and the calibration fit are undefined; with perfectly separated scores only the free intercept-and-slope fit fails, so the slope has no finite estimate, while calibration-in-the-large, which fixes the slope and fits only the intercept, is still estimated and reported. Each case prints one line saying so, and the calculations that are defined carry on.
- With few AF recordings in their development set, the comparison of J and minimum loss, and the evaluation loss, are noisy. Say so rather than over-reading one result.

**Discussion.** What would you report beside AUC for this classifier, and what would a service need to know before acting on it?
