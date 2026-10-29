# Notes for instructors and demonstrators

These notes are intended for the session lead and the demonstrators. Students start from the [README.md](README.md) file, which links to all the content they need. A [lab sheet](Clinical_Utility_Lab_Sheet.pdf) is provided to guide them through the session. The extensions have their own facilitator notes in [extension/FACILITATOR.md](extension/FACILITATOR.md).

The session is on Friday 6 November 2026 from 14:30 to 16:00.

Facilitator: Bradley Segal (session lead)

Demonstrators: Xiao Gu (organiser), Edward Phillips

## What the session teaches

Students have trained ECG classifiers on the PhysioNet/CinC 2017 dataset. The intent of this session is to move students from a score to a justified review policy. This leverages the libraries they already know (pandas, scikit-learn, statsmodels, scipy), and students write code only where no library does the job. Students are expected to work individually within the session, and should leave able to explain the prices and denominators behind their numbers rather than just pointing to the lowest point on a curve.

The main concepts that will be covered, in order, are:

1. A decision matrix with a price for each outcome gives the implied threshold probability, and so a threshold.
2. `confusion_matrix(...).ravel()` returns `tn, fp, fn, tp`; sensitivity, specificity, PPV and NPV all have different denominators.
3. Maximising Youden's J and minimising loss with pricing choose different cutoffs, because J prices an extra miss by the case mix.
4. Two models must be compared under identical prices, each at its own cutoff; the preferred model can change based on the price of a miss, despite the AUC.
5. A recalibration is fit on a separate split, and calibration matters when a policy acts on a stated probability; net benefit is a scaled version of the priced matrix.
6. A policy is frozen before it is evaluated, with uncertainty intervals included in the evaluation results and the decision documented.

The notebook includes a notation table at the onset that details the definitions of the symbols (review, the four counts and prices, loss, score, cutoff, $p_t$, operating point, per 1,000). Please point students to it if they have questions about the meaning of any symbol.

## The session shape

It is difficult to know how long a particular room will take. This is a rough guide and should not be interpreted as a rigid timetable. The notebook's roadmap indicates to students that each section should take "about N minutes", with a total of <!-- derived:core.total -->75<!-- /derived --> minutes of content, which leaves the first ten minutes or so for initial setup and a brief wrap-up at the end.

| Roughly | What happens | Sections | Listen for |
|---------|--------------|----------|------------|
| 14:30 | Arrive, open the core notebook from its badge, save a copy, run the setup cell | – | Every tab title starts "Copy of". |
| 14:40 | Price the decision and check counts by hand | *Price the decision* (about <!-- derived:core.minutes.1 -->8<!-- /derived --> min); *Six development recordings* (about <!-- derived:core.minutes.2 -->10<!-- /derived -->); *The cost of a policy* (about <!-- derived:core.minutes.3 -->7<!-- /derived -->) | A correct review still costs a review. The ravel order is tn, fp, fn, tp. |
| 15:05 | Choose a cutoff and a model | *Every operating point, and J* (about <!-- derived:core.minutes.4 -->12<!-- /derived -->); *Which model? It depends on the price of a miss* (about <!-- derived:core.minutes.5 -->12<!-- /derived -->) | Each model at its own cutoff under the same prices. AUC does not settle the choice, and one split does not fix the flip price. |
| 15:25 | Recalibrate and value | *Calibration and net benefit* (about <!-- derived:core.minutes.6 -->14<!-- /derived -->) | The map is fitted on the calibration split. Fix the preference first, then compare. |
| 15:40 | Freeze, evaluate and record | *Freeze, evaluate and record* (about <!-- derived:core.minutes.7 -->12<!-- /derived -->) | The policy is frozen before the reveal; development results are apparent. |
| 15:52 | Wrap-up | – | See "The closing wrap-up" below. |

The session has not been piloted with students to determine time estimates.

### If the room is behind

The facilitator should take stock of where most students are at about 15:05. The notebook roadmap calls out which sections can be skipped and returned to at a later point. This should also be communicated to students aloud:

- the visualisation of the six recordings in *The cost of a policy*; the non-interactive call below it draws the same view;
- the bootstrapped resampling cell in *Which model? It depends on the price of a miss*; state that the flip price could move;
- the visualisation of the decision curve in *Calibration and net benefit*; the non-interactive figure is sufficient.

The *Freeze, evaluate and record* section should not be skipped. If a student is stuck on one of the answers that later cells need, they should be directed to the switch below it (see "The I'm-stuck switches") to prevent them falling further behind. If students are having issues with their browser or are attempting to install anything locally, these should not be resolved during the session. The ordinary function calls under each of the sliders do the same job.

### The I'm-stuck switches

The notebook includes instructions for students to move on if they have been stuck on a particular problem for about <!-- derived:core.stuck -->10<!-- /derived --> minutes. For the four answers that later cells depend on, a commented-out line is included below the answer:

| Answer | The commented line |
|----------|--------------------|
| `threshold_probability` (Problem 1) | `# threshold_probability = check.use_reference("threshold_probability")` |
| `policy_cost` (Problem 3) | `# policy_cost = check.use_reference("policy_cost")` |
| `recalibration` (Problem 5) | `# recalibration = check.use_reference("recalibration")` |
| `net_benefit` (Problem 6) | `# net_benefit = check.use_reference("net_benefit")` |

Removing the `#` and running the cell will replace the student's answer with the reference version and print a line saying that this has occurred; the student's own attempt will not be overwritten unless this is actioned. The notebook is designed such that any cell that needs an unfinished answer prints one `Waiting for ...` line naming the problem and how to unblock it. This ensures that students will not receive a traceback if they attempt to proceed without completing a prior problem. The line always ends with the recovery: once the problem is done or switched, click in the waiting cell, choose Runtime > Run before, then run the waiting cell. Note that the two Fix-the-bug problems, Problems 2 and 4, do not have a switch, as the intent of the problem is for the student to make the repair. Later cells use the reference versions, so students can proceed even if they have not fixed them. Problem 7 is the final problem in the notebook.

The same rule is implemented for the extensions. Every answer that a later cell uses has a stuck switch below it, with the exception of the Fix-the-bug problems. Please see [extension/FACILITATOR.md](extension/FACILITATOR.md) for more details.

### The extensions

The extensions are included as self-directed follow-up work and are not intended to be completed within the session. The final cell of the notebook highlights them to students, and they should be briefly mentioned by the facilitator in the last minute of the session: Extension 1 (if students found calibration hard), Extension 2 (to compare models more carefully), Extension 4 (if interested in thinking through deployment), and Extension 6 (if they have scores from their own classifier). Extensions 3 and 5 are optional. The time estimate for each extension is roughly <!-- number:extension.minutes.weighting|.0f -->30<!-- /number --> to <!-- number:extension.minutes.calibration|.0f -->35<!-- /number --> minutes, except for Extension 6 at about <!-- number:extension.minutes.own|.0f -->25<!-- /number -->. The plan, answer key and sticking points for each are in [extension/FACILITATOR.md](extension/FACILITATOR.md).

## Before the session

- Open each student notebook from the README badge, and each worked copy from the badge on top, with a new CPU runtime: **Runtime > Disconnect and delete runtime**, then **Runtime > Run all**. Student versions should run without error, and show "Not attempted yet" for all Make problems and "Not yet" for all Fix-the-bug problems (where the starter is the buggy code), and later cells should show "Waiting for ...". Worked copies should show ✓ for all.
- Confirm that the setup cell does not install any additional packages other than the course package. Colab already has pandas, scikit-learn, scipy, statsmodels, plotly and ipywidgets pre-installed.
- In the core notebook, confirm that the three sliders (expected loss, six-recording policy, decision curve) render and update. If they do not, confirm that the standard plot call in the cell below each one draws its figure.
- For Extension 6, upload a small `recording,score` CSV file to the Files tab in Colab, set the `SCORES_FILE` variable to the name of the file, and confirm that it reports which entries were excluded. If there is no file, it should report that and use a demonstration column instead.
- Ensure the internet connection of the room is not blocking Google sign-in. If a student is unable to sign in, have a demonstrator's laptop available to share.
- Print / share the [lab sheet](Clinical_Utility_Lab_Sheet.pdf) with the students.

## Facilitator tips

- **Make the denominator verbal.** "Of the AF recordings, how many were reviewed?" is sensitivity. "Of the reviewed recordings, how many had AF?" is PPV. Ask for the sentence describing a fraction before the fraction.
- **Send students to the checker's hint first.** The "Not yet" responses include a reason for why it may be incorrect. Refer to this prior to explaining.
- **Ask what the output says.** Following each worked cell, ask the student to read one value aloud and interpret it in its unit: reviews, AF detected or AF missed per 1,000, or net benefit.
- **Do not take the keyboard.** Ask students questions that lead them to the solution and let them type the response.
- **Keep the meanings apart.** Reference AF is not the same as a model score, a review, or a clinical diagnosis. O recordings are AF-negative here but are not 'healthy'. Recordings that are noisy do not have a binary reference and are excluded rather than relabelled.
- **Accept tied optima.** It is possible to have multiple cutoffs that produce the exact same calls. The answer checker will accept any of these as valid, and so should you.
- **On solutions:** they are provided within `solutions/`. Suggest a student attempts the problem and, if they are stuck, takes a look at the hint. If they are still stuck, they can open the solution, look at the specific problem they are stuck on, close it, and then attempt to implement the solution themselves.

## Common problems and fixes

| What the student sees | Cause | Fix |
|-----------------------|-------|-----|
| Colab asks them to sign in, or will not save | Colab needs a Google account | Any Google account works; a personal one is simplest. If the network blocks Google sign-in, try another network or a personal device. |
| "Warning: This notebook was not authored by Google" | Colab shows this for every notebook opened from GitHub | Choose **Run anyway**. |
| Their work disappears when they close the tab | They worked in the original, not a copy | Lost work cannot be recovered. Reopen from the badge, choose **File > Save a copy in Drive** first and redo the work. If the tab is still open, saving a copy now keeps what they have. |
| `ModuleNotFoundError: No module named 'clinical_utility'` | The setup cell was not run, or the runtime restarted | Run the setup cell at the top, then carry on. |
| A slider or explorer shows nothing, or only text | Colab's widget frontend did not load | Use the ordinary call in the next cell, which draws the same figure; change its numbers and rerun. |
| A slider ignores a function they have just fixed | The explorer was built with the old function | Rerun the explorer cell so it receives the new function. |
| `NameError: name '...' is not defined` | Cells were run out of order or skipped, or the runtime restarted | **Runtime > Run before** (runs every cell above this one), or **Runtime > Restart session and run all**. |
| A later cell prints "Waiting for ..." | It builds on an answer that has not passed its checker yet; the line names the problem | Go back to that problem and finish it, or use its I'm-stuck switch where the line offers one. Then click in the waiting cell, choose **Runtime > Run before**, and run the waiting cell. |
| The checker shows "Not attempted yet" after they wrote an answer | A `...` is still in the cell, or the answer is in a different cell | Replace every `...` in the same cell as the checker and rerun it. |
| `ValueError: not enough values to unpack` from `confusion_matrix(...).ravel()` | Only one class is present, so the matrix is one cell | Pass `labels=[False, True]`. |
| The Platt checker says the penalty is on | `LogisticRegression()` shrinks coefficients by default | Use `LogisticRegression(C=np.inf, solver="newton-cholesky")`. |

## Answer key: core notebook

The expected values below assume the default prices $C_{TP}$ = <!-- number:core.prices.TP|.0f -->1<!-- /number -->, $C_{FP}$ = <!-- number:core.prices.FP|.0f -->1<!-- /number -->, $C_{FN}$ = <!-- number:core.prices.FN|.0f -->5<!-- /number --> and $C_{TN}$ = <!-- number:core.prices.TN|.0f -->0<!-- /number -->, in reviews, with TP, FP, FN and TN being the counts. The worked copy in [solutions/01_clinical_utility.ipynb](solutions/01_clinical_utility.ipynb) includes the full code. Each entry names the variable or function that students are expected to write. *Checker catches* refers to the mistakes that the checker names in its "Not yet" hint.

**Problem 1 (Make), `threshold_probability`.** `harm / (harm + benefit)` with harm = C_FP − C_TN and benefit = C_FN − C_TP, with a check to raise `ValueError` if either value isn't positive. It produces <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number --> given the default prices, or <!-- number:core.threshold.fn_10|.1f -->0.1<!-- /number --> if a miss costs ten reviews. *Checker catches:* forgetting to subtract the price of a correct review (C_TP); taking the complement (the probability of no AF); the odds (harm / benefit); a percentage; not raising an error. The worked table below it shows how increasing the price of a miss reduces the threshold.

**Problem 2 (Fix the bug), `six_counts`.** Unpacks `confusion_matrix(y, review, labels=[False, True]).ravel()` as `tn, fp, fn, tp`. Given the six recordings and a cutoff of <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number -->, this should produce TP <!-- number:six.t_pt.TP|.0f -->2<!-- /number -->, FP <!-- number:six.t_pt.FP|.0f -->2<!-- /number -->, FN <!-- number:six.t_pt.FN|.0f -->1<!-- /number -->, TN <!-- number:six.t_pt.TN|.0f -->1<!-- /number -->. *Checker catches:* using the colleague's function with the TP and TN values swapped; using the textbook order for the confusion matrix; passing the reference labels as the second argument (swaps the FP and FN values); a single-class input without `labels=`. On the development split, the colleague's swap would make the cheapest-looking policy for the random forest review <!-- number:bug.ravel.reviews|,.0f -->3,686<!-- /number --> of the <!-- number:bug.ravel.records|,.0f -->3,711<!-- /number --> recordings.

**Problem 3 (Make), `policy_cost`.** Produces the sum of each count from the confusion matrix times its price from `costs`. It should produce <!-- number:six.t_pt.loss|.0f -->9<!-- /number --> for the six recordings at a cutoff of <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number -->, and <!-- number:six.t0_5.loss|.0f -->12<!-- /number --> at a cutoff of <!-- number:core.cutoffs.half|.1f -->0.5<!-- /number -->. *Checker catches:* using the incorrect order for `ravel()`; passing arguments in the incorrect order; a free correct review; hard-coding prices instead of reading them from `costs` (a matrix in which no two prices are equal); forgetting to include `labels=`. This problem has an I'm-stuck switch. The six recordings are used as a sanity check that students' arithmetic is correct; no cutoff is chosen on them.

**Problem 4 (Fix the bug), `compare_models`.** Replaces the shared cutoff of <!-- number:core.cutoffs.half|.1f -->0.5<!-- /number --> with each model's cutoff from the row with the lowest loss in `reference.operating_points` (if there is a tie, the cutoff with the fewest reviews). At <!-- number:core.cutoffs.half|.1f -->0.5<!-- /number -->, the loss for the raw ResNet is <!-- number:bug.shared_cut.resnet_raw.loss|.0f -->971<!-- /number --> compared to <!-- number:bug.shared_cut.rf.loss|.0f -->992<!-- /number --> for the random forest. At their own cutoffs, the random forest has the lower loss, <!-- number:dev.rf.min_loss.loss|.0f -->815<!-- /number --> compared to <!-- number:dev.resnet_raw.min_loss.loss|.0f -->922<!-- /number --> for the raw ResNet. *Checker catches:* both models still using the <!-- number:core.cutoffs.half|.1f -->0.5<!-- /number --> cutoff; a table not indexed by model. Following this is a worked sensitivity table, which finds that the preferred model flips at a miss price of <!-- number:price.dev.first_whole_flip|.0f -->8<!-- /number -->; in <!-- number:price.dev.flip.bootstrap.coverage|.0% -->90%<!-- /number --> of <!-- number:price.dev.flip.bootstrap.resamples|.0f -->200<!-- /number --> resamples, the flip lay between <!-- number:price.dev.flip.bootstrap.low|.0f -->6<!-- /number --> and <!-- number:price.dev.flip.bootstrap.high|.0f -->12<!-- /number -->. The key misconception this aims to address is that the model with the higher AUC (<!-- number:dev.resnet_raw.auc|.3f -->0.934<!-- /number --> vs <!-- number:dev.rf.auc|.3f -->0.926<!-- /number -->) settles the choice.

**Problem 5 (Make), `recalibration`.** Produces a `make_pipeline(FunctionTransformer(logit), LogisticRegression(C=np.inf, solver="newton-cholesky"))` model fitted on `calibration_split[["resnet"]]` and its `af`. This problem has an I'm-stuck switch. *Checker catches:* not fitting the pipeline; fitting on the development or evaluation split; using the default penalty for the logistic regression (`C=1`); regressing on the raw score instead of its logit. A worked cell follows, showing that the AUC does not change; however, the Brier score improves from <!-- number:dev.resnet_raw.brier|.3f -->0.118<!-- /number --> to <!-- number:dev.resnet_cal.brier|.3f -->0.056<!-- /number -->, and the number of reviews at a stated <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number --> decreases from <!-- number:dev.resnet_raw.at_0_2.per_1000.reviews|.1f -->283.2<!-- /number --> to <!-- number:dev.resnet_cal.at_0_2.per_1000.reviews|.1f -->199.1<!-- /number --> per 1,000.

**Problem 6 (Make), `net_benefit`.** Produces `TP/n - FP/n * p_t/(1 - p_t)`, where `n` is the total of all four counts. The net benefit for the six recordings at <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number --> should be <!-- number:six.t_pt.net_benefit|.3f -->0.250<!-- /number -->. *Checker catches:* not dividing by `n`; weighting false positives by `p_t` instead of its odds; inverting the odds; charging misses instead of false positives; dividing by the number of reviewed recordings. This problem has an I'm-stuck switch. At <!-- number:prices.default.indifference_probability|.1f -->0.2<!-- /number -->, the random forest has a net benefit of <!-- number:nb.dev.p0_2.rf|.3f -->0.059<!-- /number -->, followed by the recalibrated ResNet (<!-- number:nb.dev.p0_2.resnet_cal|.3f -->0.051<!-- /number -->), the raw ResNet (<!-- number:nb.dev.p0_2.resnet_raw|.3f -->0.041<!-- /number -->), and finally the review-everyone strategy (<!-- number:nb.dev.p0_2.review_all|.3f -->-0.135<!-- /number -->). The key misconception is that students may look at a decision curve and read off its highest point, irrespective of preference.

**Problem 7 (Make), `frozen`.** Produces a dictionary containing the model, the selected cutoff, the prices used, the `p_t` those prices imply, and a note of how the cutoff was chosen. The worked answer selects the random forest at the cutoff that minimises the loss on the development split (<!-- number:dev.rf.min_loss.threshold|.3f -->0.239<!-- /number -->); however, a cutoff derived from the prices, or another defended choice, is equally acceptable. The checker never reads the evaluation split; it just checks that the dictionary is complete and that `p_t` follows from the prices. On evaluation, the worked policy reviews <!-- number:core.frozen.eval.reviews.estimate|.1f -->117.3<!-- /number --> (<!-- number:core.frozen.eval.reviews.low|.1f -->105.9<!-- /number --> to <!-- number:core.frozen.eval.reviews.high|.1f -->128.8<!-- /number -->) recordings per 1,000 and misses <!-- number:core.frozen.eval.missed.estimate|.1f -->22.2<!-- /number --> (<!-- number:core.frozen.eval.missed.low|.1f -->16.6<!-- /number --> to <!-- number:core.frozen.eval.missed.high|.1f -->27.7<!-- /number -->) AF per 1,000. This compares to an apparent <!-- number:core.frozen.dev.reviews|.1f -->110.5<!-- /number --> reviews and <!-- number:core.frozen.dev.missed|.1f -->21.8<!-- /number --> AF missed per 1,000 on development. The second part of this problem, `decision_record`, holds the three sentences that students write; the checker adds the frozen policy and the reports and prints the whole record, and it cannot judge the valuation. The key misconception is rerunning the reveal after changing the choice and claiming that the new choice has been validated.

## The closing wrap-up

This is intended to be a short facilitator-led discussion with the entire class, of about five minutes, and is the only point in the session that is not intended to be individual work. The facilitator should ask two or three students to read out the decision record they've produced. They should state the action, the prices they've selected and why, which model and cutoff they've selected, the evaluation result with its interval, and what would make them reopen the decision. There is no 'correct' price for a missed AF recording, so facilitators should listen for a valuation that is stated and defended rather than for a particular number. The facilitator should then take the record's four prompts with the whole room, as time allows:

1. **The pathway.** Who will receive flagged recordings? How quickly? What will they do next? Who is accountable for recordings that were never reviewed?
2. **Drift and prevalence.** What metrics in the report would they track post-deployment? Extensions 1 and 4 cover recalibration and prevalence.
3. **What J assumed.** In our case, J assumes that a miss is worth <!-- number:dev.j.incremental_miss_penalty|.2f -->9.88<!-- /number --> additional false referrals, i.e., the class ratio rather than the matrix value of <!-- number:core.prices.review_benefit|.0f -->4<!-- /number -->. With all other prices fixed, this is equivalent to a miss being priced at about <!-- number:dev.j.implied_fn_price|.2f -->10.88<!-- /number --> reviews. Try and push them to explain why this would change with prevalence even if no one's clinical judgement has changed.
4. **Evidence before use.** Which of the following does this analysis give us today: prospective evidence on local data; post-deployment monitoring; clinical safety management; a regulatory assessment?

For the fourth question, care should be taken to ensure that the following governance facts are stated precisely:

- In England, the standards for clinical risk management are DCB0129 (for the manufacturer of a health IT system) and DCB0160 (for the organisation deploying / using it). These are not UK-wide.
- The classification of software as a medical device is based on the intended purpose as defined by the manufacturer. The MHRA is responsible for medical device regulation in the UK. Do not tell students that it is or isn't definitively a medical device; say it would require a regulatory assessment to determine this.
- If you were to use identifiable / pseudonymised patient data, this would invoke UK GDPR. Whether this would require a data protection impact assessment depends on the exact nature of the planned processing and its risk.

None of these are addressed by the analysis students have performed today, which has been on understanding the relationship between prices, thresholds and model choice. The final question for students is what they would report for their own classifier in addition to the AUC. Extension 6 performs today's analysis on their classifier's scores.

## Reporting a problem

If you are having issues with a notebook, if a checker is rejecting what you believe is a correct answer, or if you find an issue with the text, please open an issue in this repository or let a member of the course team know. Please include the name of the notebook and the problem number, what you tried to do and what you expected to happen, as well as the full error message copied as text. Please ensure you do not include anything that could identify a student.
