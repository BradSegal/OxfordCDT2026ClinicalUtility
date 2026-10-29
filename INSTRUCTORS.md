# Notes for instructors and demonstrators

This guide is for the session lead and the demonstrators. Students start from [README.md](README.md), which links to everything they need, and the [lab sheet](Clinical_Utility_Lab_Sheet.pdf) gives them the route through the session. The extensions have their own guide in [extension/FACILITATOR.md](extension/FACILITATOR.md).

The session runs on Friday 6 November 2026 from 14:30 to 16:00. Bradley Segal leads; Xiao Gu (organiser) and Edward Phillips demonstrate.

## What the session teaches

Students have trained ECG classifiers on PhysioNet/CinC 2017. The practical moves them from a score to a justified review policy. They should leave able to explain the prices and the denominators behind their numbers, not just point at the minimum of a curve. The core ideas, in order:

1. A decision matrix with a price for each outcome gives a threshold probability, and so a threshold.
2. Sensitivity, specificity, PPV and NPV divide by different groups.
3. Maximising Youden's J and minimising priced loss choose different cutoffs, because J prices a miss by the case mix.
4. The preferred model can change with the price of a miss, whatever the AUC says, and the price at which it changes is itself uncertain.
5. A scorer can run and return plausible numbers while optimising the wrong objective.
6. Net benefit is the priced matrix rescaled, and calibration matters when a policy acts on a stated probability.
7. A policy is frozen before it is evaluated, and the decision is stated in words.

All course model scores are simulated; the recording labels are real. State this once at the start and do not let the session drift into whether the "ResNet" is really better. It is a constructed example.

The notebook opens with a notation table (review, the four counts and prices, loss, score, cutoff, $p$, $p_t$ and the three models). Point students to it when a symbol is unfamiliar.

## The session shape

Nobody can say reliably how long a room will take, so treat this as a rough shape, not a timetable. The notebook's roadmap gives each section "about N minutes", <!-- derived:core.total -->80<!-- /derived --> minutes in all, which leaves the first ten minutes or so for setting up.

| Roughly | What happens | Sections | Listen for |
|---------|--------------|----------|------------|
| 14:30 | Arrive, open the core notebook from its badge, save a copy, run the setup cell, pair up | – | Every tab title starts "Copy of". |
| 14:40 | Price a decision | *One recording, two actions* (about <!-- derived:core.minutes.1 -->10<!-- /derived --> min); *Six teaching recordings* (about <!-- derived:core.minutes.2 -->10<!-- /derived -->); *Price one decision* (about <!-- derived:core.minutes.3 -->5<!-- /derived -->) | A correct review still costs a review. Students say the denominator aloud before computing a rate. |
| 15:05 | Choose a policy and a model | *A policy and two objectives* (about <!-- derived:core.minutes.4 -->15<!-- /derived -->); *Which model? It depends on the prices* (about <!-- derived:core.minutes.5 -->10<!-- /derived -->) | The same recordings and objective for each comparison. AUC does not settle the choice, and one split does not fix the flip price. |
| 15:30 | Test the scorer, then net benefit | *Fix the bug* (about <!-- derived:core.minutes.6 -->5<!-- /derived -->); *From prices to decision curves* (about <!-- derived:core.minutes.7 -->10<!-- /derived -->) | One correct review exposes the missing price. Fix the preference first, then compare. |
| 15:45 | Freeze a policy | *Freeze and defend a policy* (about <!-- derived:core.minutes.8 -->5<!-- /derived -->) | The policy and the predicted loss are written down before the evaluation split is opened. |
| 15:50 | Close | *Closing discussion* (about <!-- derived:core.minutes.9 -->10<!-- /derived -->) | A spoken decision statement; the pathway, drift, what J assumed, and the evidence needed before use. |

No timed pilot with students has been run.

### If the room is behind

Check where most pairs are at about 15:05. The notebook's roadmap tells students what they may skip and come back to later, so say the same thing aloud:

- Problem 3: read its folded answer instead of computing it;
- the six-recording explorer in *Price one decision*; the ordinary call below it draws the same view;
- Problem 9 and the resampling cell after it in *Which model? It depends on the prices*; state the conclusion that the flip price could move;
- the decision-curve explorer in *From prices to decision curves*; its plain figure is enough.

Do not skip *Freeze and defend a policy* or the *Closing discussion*. They carry the main lesson, and the notebook tells students the same. If a pair is stuck on one of the three functions that later cells need, point them to the switch below it (see "The I'm-stuck switches") rather than letting them fall further behind.

Do not spend the session repairing a browser or installing anything locally. The ordinary function calls under each slider do the same job.

### The I'm-stuck switches

The notebook tells students that after about <!-- derived:core.stuck -->10<!-- /derived --> minutes on a problem they should move on. Below the three functions that later cells depend on is a commented line:

| Function | The commented line |
|----------|--------------------|
| `decision_cost` (Problem 4) | `# decision_cost = check.use_reference("decision_cost")` |
| `youden_j` (Problem 6) | `# youden_j = check.use_reference("youden_j")` |
| `net_benefit` (Problem 12) | `# net_benefit = check.use_reference("net_benefit")` |

Deleting the `#` and running the cell replaces the student's function with the reference version and prints a line saying so. Their own attempt is untouched until then, so they can return to it. A cell that needs an unfinished function prints one `Waiting for ...` line naming the problem and how to unblock it, so nothing fails with a traceback. The line ends with the recovery: once the problem is done or switched, click in the waiting cell, choose **Runtime > Run before**, then run the waiting cell. Rerunning only the waiting cell is not always enough, because an earlier cell may have skipped its work for the same reason; Run before reruns those cells too. The only other function a later cell calls is `repaired_colleague_cost` (Problem 11), whose before-and-after cell waits for the repair; it has no switch, because the repair is the exercise. Every other problem is a value that nothing later depends on, and a stuck pair can read the worked copy for that one problem and carry on.

The extensions follow the same rule: every function that a later cell uses has a stuck switch below it, except the Fix-the-bug functions. See [extension/FACILITATOR.md](extension/FACILITATOR.md).

### The extensions

The extensions are self-paced follow-up work, not part of the live session. Mention them in the last minute, as the notebook's final cell does: Extension 1 if *From prices to decision curves* was hard, Extension 2 to compare models more carefully, Extension 4 for deployment, and Extension 6 for anyone with scores from their own classifier. Extensions 3 and 5 are optional. Each takes roughly <!-- number:extension.minutes.own|.0f -->20<!-- /number --> to <!-- number:extension.minutes.comparison|.0f -->45<!-- /number --> minutes. The plan, answer key and sticking points for each are in [extension/FACILITATOR.md](extension/FACILITATOR.md).

## Before the session

- Open every student notebook from its badge in the README, and every worked copy from the badge at its top, on a fresh CPU runtime: **Runtime > Disconnect and delete runtime**, then **Runtime > Run all**. The student copies should finish without errors, with problems showing "Not attempted yet" and later cells "Waiting for ...". The worked copies should show ✓ throughout.
- In the core notebook, confirm that the three sliders (expected cost, policy explorer, decision curve) render and respond. If they do not, check that the ordinary call in the cell below each one draws its figure.
- In Extension 6, upload a small `recording,score` CSV in Colab's Files pane, set `SCORES_FILE` to its name and confirm the join reports what it excluded. Without a file, the notebook says so and uses a simulated demonstration column. A file with recordings from one split only, or with one outcome class, should run to the end and say which results are undefined.
- Check the room's network allows Google sign-in. Have a demonstrator's laptop ready to share if a student cannot sign in.
- Plan pairs. Put someone comfortable with Python beside someone who can talk through the clinical consequence, and ask them to swap who types at each section.
- Print or share the [lab sheet](Clinical_Utility_Lab_Sheet.pdf).

## Facilitator tips

- **Ask for the prediction first.** Every Predict problem should be answered on paper before anything runs. A wrong prediction is the useful moment.
- **Make the denominator verbal.** "Of the reviewed recordings, how many had AF?" is PPV. "Of the AF recordings, how many were reviewed?" is sensitivity. Ask for the sentence before the fraction.
- **Read the checker together.** Each "Not yet" names the likely mistake. Point students at the hint before you explain anything.
- **Do not take the keyboard.** Ask the question that leads to the fix and let them type it.
- **Keep the meanings apart.** Reference AF, a model score, a review and a clinical diagnosis are four different things. O recordings are AF-negative here but are not healthy. Noisy recordings have no binary reference and are excluded, not relabelled.
- **Accept tied optima.** Several cutoffs can make identical calls. The checkers accept any of them, and so should you.
- **On solutions:** they are in `solutions/`. Suggest a student opens one after a real attempt and a hint, reads that one problem, closes it and writes the answer.

## Common problems and fixes

| What the student sees | Cause | Fix |
|-----------------------|-------|-----|
| Colab asks them to sign in, or will not save | Colab needs a Google account | Any Google account works; a personal one is simplest. If the network blocks Google sign-in, try another network or a personal device. |
| "Warning: This notebook was not authored by Google" | Colab shows this for every notebook opened from GitHub | Choose **Run anyway**. |
| Their work disappears when they close the tab | They worked in the original, not a copy | Lost work cannot be recovered. Reopen from the badge, choose **File > Save a copy in Drive** first and redo the work. If the tab is still open, saving a copy now keeps what they have. |
| `ModuleNotFoundError: No module named 'clinical_utility'` | The setup cell was not run, or the runtime restarted | Run the setup cell at the top, then carry on. |
| A slider or explorer shows nothing, or only text | Colab's widget frontend did not load | Use the ordinary call in the next cell, which draws the same figure; change its numbers and rerun. Do not spend time repairing the browser. |
| A slider ignores a function they have just fixed | The explorer was built with the old function | Rerun the explorer cell so it receives the new function. |
| `NameError: name '...' is not defined` | Cells were run out of order or skipped, or the runtime restarted | **Runtime > Run before** (runs every cell above this one), or **Runtime > Restart session and run all**. |
| "Runtime disconnected" | Colab stops idle sessions | Reconnect, then **Runtime > Run before**. The text and code are kept; variables are not. |
| A later cell prints "Waiting for ..." | It builds on a function that has not passed its checker yet; the line names the problem | Go back to that problem and finish it, or use its I'm-stuck switch where the line offers one. Then click in the waiting cell, choose **Runtime > Run before**, and run the waiting cell. |
| The checker shows "Not attempted yet" after they wrote an answer | A `...` is still in the cell, or the answer is in a different cell | Replace every `...` in the same cell as the checker and rerun it. |
| `decision_cost` passes but totals look wrong | A price is typed as a number instead of read from `costs` | Read every price from the `costs` argument. The checker also tries a matrix where no two prices are equal. |

## Answer key: core notebook

Expected values use the default prices $C_{TP}$ = <!-- number:core.prices.TP|.0f -->1<!-- /number -->, $C_{FP}$ = <!-- number:core.prices.FP|.0f -->1<!-- /number -->, $C_{FN}$ = <!-- number:core.prices.FN|.0f -->5<!-- /number --> and $C_{TN}$ = <!-- number:core.prices.TN|.0f -->0<!-- /number --> unless stated; TP, FP, FN and TN are counts. The worked copy in [solutions/01_clinical_utility.ipynb](solutions/01_clinical_utility.ipynb) has full code. Each entry names the variable or function the student writes, so the key can be matched to the notebook if problems are renumbered. *Checker catches* lists the mistakes the checker names in its "Not yet" hint.

**Problem 1 (Predict), `threshold_if_miss_costs_10`.** <!-- number:core.predict.indifference_probability|g -->0.1<!-- /number -->. With $C_{FN}$ = <!-- number:core.predict.fn|.0f -->10<!-- /number -->, review still costs <!-- number:core.expected.low.review|.0f -->1<!-- /number --> at every probability and no review costs $C_{FN} p$, so the crossing is at <!-- number:core.predict.indifference_probability|g -->0.1<!-- /number -->. *Checker catches:* the unchanged threshold of <!-- number:prices.default.indifference_probability|g -->0.2<!-- /number -->; a free correct review, which gives <!-- number:core.predict.tp_free_probability|.3f -->0.091<!-- /number -->; the probability of no AF instead of AF; a threshold that moved up rather than down; a percentage. The key misconception is direction: a dearer miss means weaker evidence justifies a review.

**Problem 2 (Predict), `predicted_reviews`.** <!-- number:cases.t0_5.reviews|.0f -->3<!-- /number --> reviews (Teaching A, B and C) and <!-- number:cases.t0_5.missed_af|.0f -->1<!-- /number --> missed AF recording (Teaching E), stored with `predicted_misses`. *Checker catches:* counting only the AF recordings that were reviewed (Teaching B has no AF but meets the cutoff); counting no misses; counting every unreviewed recording as a miss.

**Problem 3 (Predict), `npv_if_nobody_reviewed`.** NPV = <!-- number:core.dev.non_af|,.0f -->3,370<!-- /number -->/<!-- number:dev.records|,.0f -->3,711<!-- /number --> = <!-- number:core.dev.review_none.npv|.3f -->0.908<!-- /number --> when nobody is reviewed. *Checker catches:* misses divided by true negatives; the AF fraction instead of the AF-free fraction; specificity in place of NPV; a percentage; NaN, which is right for PPV but not for NPV. The key misconception is reading a high NPV as good performance: this policy misses all <!-- number:dev.af|.0f -->341<!-- /number --> AF recordings, and NPV is high because AF is uncommon (<!-- number:dev.prevalence|.1% -->9.2%<!-- /number --> of the development split).

**Problem 4 (Make), `decision_cost`.** Each branch returns its price: `costs["TP"]`, `costs["FP"]`, `costs["FN"]`, `costs["TN"]`. *Checker catches:* returning `0` for a true positive (the review is done whether or not AF is found); returning `0` for a true negative when the matrix says otherwise; typing the default prices instead of reading `costs`, caught by a matrix in which no two prices are equal; swapping FP and FN. It has an I'm-stuck switch.

**Problem 5 (Make), `evaluate_policy`.** A loop that reviews when `record["score"] >= threshold`, prices the decision with `decision_cost` and appends `{"recording", "review", "loss"}` for each record. The ledger at a cutoff of <!-- number:core.cases.cutoffs.t0_5|g -->0.5<!-- /number --> totals <!-- number:cases.t0_5.total_loss|.0f -->8<!-- /number -->. *Checker catches:* `return` inside the loop, which returns one row; `>` in place of `>=`, caught at Teaching E's score; returning `None`; missing keys; reordered rows; prices not passed through from the `costs` argument.

**Problem 6 (Make), `youden_j`.** `J = TP/(TP + FN) + TN/(TN + FP) - 1`; on the six recordings at <!-- number:core.cases.cutoffs.t0_5|g -->0.5<!-- /number --> it is <!-- number:cases.t0_5.youden_j|.3f -->0.333<!-- /number -->. *Checker catches:* forgetting to subtract one; PPV in place of specificity; the two predictive values; the false-positive rate in place of specificity. It has an I'm-stuck switch.

**Problem 7 (Predict), `j_cutoff_compared_with_loss`.** `"lower"`. Students have not yet seen how J values a miss, so this is a recorded hypothesis: the checker records any valid answer without marking it, and the reveal cell prints the result beside it with J's implied miss price. Ask pairs whose hypothesis differed what they assumed about J. On the development split maximum J picks cutoff <!-- number:dev.rf.max_j.threshold|.3f -->0.102<!-- /number -->: <!-- number:dev.rf.max_j.reviews|.0f -->613<!-- /number --> reviews, <!-- number:dev.rf.max_j.missed_af|.0f -->52<!-- /number --> missed AF, loss <!-- number:dev.rf.max_j.total_loss|.0f -->873<!-- /number -->. Minimum loss picks <!-- number:dev.rf.min_loss.threshold|.3f -->0.239<!-- /number -->: <!-- number:dev.rf.min_loss.reviews|.0f -->410<!-- /number --> reviews, <!-- number:dev.rf.min_loss.missed_af|.0f -->81<!-- /number --> missed, loss <!-- number:dev.rf.min_loss.total_loss|.0f -->815<!-- /number -->. The key misconception is that J is neutral. It prices a miss at $n_{neg}/n_{pos}$ = <!-- number:core.dev.j_implied_miss_ratio|.2f -->9.88<!-- /number --> false referrals, set by the case mix, where the matrix says <!-- number:core.prices.review_benefit|.0f -->4<!-- /number -->.

**Problem 8 (Make), `flip`.** <!-- number:core.price.dev.first_whole_flip|.0f -->8<!-- /number -->. Loop over whole miss prices from <!-- number:core.price.dev.flip_range.low|.0f -->2<!-- /number --> to <!-- number:core.price.dev.flip_range.high|.0f -->30<!-- /number --> and stop at the first price where the raw ResNet's minimum loss is no greater than the forest's. At $C_{FN}$ = <!-- number:core.price.dev.first_whole_flip|.0f -->8<!-- /number --> both are <!-- number:core.price.dev.flip.rf.total_loss|,.0f -->1,023<!-- /number -->. At $C_{FN}$ = <!-- number:core.price.compare.low|.0f -->5<!-- /number --> the forest is cheaper (<!-- number:core.price.dev.fn_5.rf.total_loss|.0f -->815<!-- /number --> against <!-- number:core.price.dev.fn_5.resnet_raw.total_loss|.0f -->922<!-- /number -->); at $C_{FN}$ = <!-- number:core.price.compare.high|.0f -->15<!-- /number --> the ResNet is (<!-- number:core.price.dev.fn_15.resnet_raw.total_loss|,.0f -->1,122<!-- /number --> against <!-- number:core.price.dev.fn_15.rf.total_loss|,.0f -->1,393<!-- /number -->). *Checker catches:* `<` instead of `<=`, which gives the next price up; the price below the flip; a loop that stops on its first price or never breaks; a loop that never sets `flip`. The key misconception is that the higher AUC (<!-- number:dev.resnet_raw.auc|.3f -->0.934<!-- /number --> against <!-- number:dev.rf.auc|.3f -->0.926<!-- /number -->) settles the choice.

**Problem 9 (Predict), `flip_on_another_sample`.** `"could_move"`. In <!-- number:core.flip.bootstrap.coverage|.0% -->90%<!-- /number --> of <!-- number:core.flip.bootstrap.resamples|.0f -->200<!-- /number --> resamples the flip price lay between <!-- number:core.flip.bootstrap.low|.0f -->6<!-- /number --> and <!-- number:core.flip.bootstrap.high|.0f -->12<!-- /number -->. *Checker catches:* `"same"`. The key misconception is treating one split's flip price as exact: one split supports "the forest when misses are cheap, the ResNet when they are dear", not "the flip is at exactly <!-- number:core.price.dev.first_whole_flip|.0f -->8<!-- /number -->".

**Problem 10 (Predict), `colleague_cutoff_compared_with_yours`.** `"lower"`. Students read `original_colleague_cost`, which stays unedited for the whole section. It charges nothing for a correct review, so reviews look cheaper and the cheapest cutoff falls. *Checker catches:* `"higher"` (cheaper reviews pull the cutoff down, not up) and `"equal"`.

**Problem 11 (Fix the bug), `repaired_colleague_cost`.** Students repair a copy of the original: add the two missing branches so that a true positive returns `costs["TP"]` and a true negative returns `costs["TN"]`. The before-and-after cell then prices each scorer's choice with the full matrix. The original scorer picks cutoff <!-- number:faulty.dev.rf.threshold|.3f -->0.164<!-- /number --> and <!-- number:faulty.dev.rf.reviews|.0f -->498<!-- /number --> reviews; it reports a loss of <!-- number:faulty.dev.rf.reported_loss|.0f -->553<!-- /number -->, but the full matrix prices that policy at <!-- number:faulty.dev.rf.true_loss|.0f -->828<!-- /number -->, against <!-- number:dev.rf.min_loss.total_loss|.0f -->815<!-- /number --> for the correct cutoff. The <!-- number:core.faulty.dev.rf.extra_reviews|.0f -->88<!-- /number --> extra reviews find <!-- number:core.faulty.dev.rf.extra_af_found|.0f -->15<!-- /number --> more AF recordings and include <!-- number:core.faulty.dev.rf.extra_false_positives|.0f -->73<!-- /number --> false positives; the original scorer reports the change as <!-- number:core.faulty.dev.rf.reported_change|+.0f -->-2<!-- /number --> units when it is really <!-- number:core.faulty.dev.rf.true_change|+.0f -->+13<!-- /number -->. A free true positive lowers the threshold probability from <!-- number:prices.default.indifference_probability|.0% -->20%<!-- /number --> to <!-- number:prices.faulty.indifference_probability|.1% -->16.7%<!-- /number -->. *Checker catches:* any cell not read from `costs`, on a matrix in which every price differs from the default, so `return 0` fails. The key misconception is "it runs and charges for both errors, so it is right". The repaired scorer chooses <!-- number:dev.rf.min_loss.threshold|.3f -->0.239<!-- /number --> again. How far the policy moves depends on how many recordings sit near the decision point. The reported loss is wrong regardless for every policy that makes a priced correct decision, here any policy that reviews an AF recording; reviewing nobody is priced the same by both scorers.

**Problem 12 (Make), `net_benefit`.** `TP/n - FP/n * p_t/(1 - p_t)` with `n` the total of all four counts. The worked example on the six recordings gives <!-- number:core.nb.cases.net_benefit|.3f -->0.417<!-- /number -->. *Checker catches:* not dividing by `n`; weighting false positives by `p_t` rather than its odds; inverted odds; charging misses instead of false positives; dividing by the reviewed recordings instead of all of them. It has an I'm-stuck switch.

**Problem 13 (Predict), `highest_net_benefit_at_0_2`.** `"random_forest"`. A recorded hypothesis, like Problem 7: what each model's stated <!-- number:prices.default.indifference_probability|g -->0.2<!-- /number --> means is shown only in the cell that follows, which prints the result beside the student's answer and, for a different answer, the reason. At a preference of <!-- number:prices.default.indifference_probability|g -->0.2<!-- /number -->: forest <!-- number:nb.dev.p0_2.rf|.3f -->0.059<!-- /number -->, recalibrated ResNet <!-- number:nb.dev.p0_2.resnet_cal|.3f -->0.051<!-- /number -->, raw ResNet <!-- number:nb.dev.p0_2.resnet_raw|.3f -->0.041<!-- /number -->, review everyone <!-- number:nb.dev.p0_2.review_all|.3f -->-0.135<!-- /number -->. *Reveal explains:* `"resnet"` (its raw scores are too high, so a stated <!-- number:prices.default.indifference_probability|g -->0.2<!-- /number --> sends <!-- number:core.at_0_2.dev.resnet_raw.reviews|,.0f -->1,051<!-- /number --> recordings for review, against <!-- number:core.at_0_2.dev.resnet_cal.reviews|.0f -->739<!-- /number --> after recalibration) and `"resnet_calibrated"` (recalibration helps, but the forest's best policy was already cheaper at $C_{FN}$ = <!-- number:core.prices.FN|.0f -->5<!-- /number -->). The raw and recalibrated ResNet rank recordings identically and share a minimum loss of <!-- number:dev.resnet_raw.min_loss.total_loss|.0f -->922<!-- /number -->; the Brier score falls from <!-- number:dev.resnet_raw.brier|.3f -->0.118<!-- /number --> to <!-- number:dev.resnet_cal.brier|.3f -->0.056<!-- /number -->, and the forest's is <!-- number:dev.rf.brier|.3f -->0.041<!-- /number -->. When students move on to the decision curves, the misconception to catch is reading a curve for its highest point across preferences: each preference is a different valuation, so fix it first.

**Problem 14 (Predict), `predicted_evaluation_loss`.** Students also set `final_model`, `final_threshold` and `final_costs`. The worked choice is the forest at its development minimum-loss cutoff, <!-- number:dev.rf.min_loss.threshold|.3f -->0.239<!-- /number -->: development's <!-- number:core.frozen.dev.total_loss|.0f -->815<!-- /number --> scaled to the evaluation size predicts <!-- number:core.frozen.scaled_anchor|.0f -->634<!-- /number -->. The reveal gives <!-- number:core.frozen.eval.reviews|.0f -->339<!-- /number --> reviews, <!-- number:core.frozen.eval.missed_af|.0f -->64<!-- /number --> missed AF and a loss of <!-- number:core.frozen.eval.total_loss|.0f -->659<!-- /number -->; the loss per recording rose from <!-- number:core.frozen.dev.mean_loss|.3f -->0.220<!-- /number --> to <!-- number:core.frozen.eval.mean_loss|.3f -->0.228<!-- /number -->. Another defensible policy is fine. *Checker catches:* an incomplete policy; the unscaled development total; a loss per recording instead of a total. A correctly scaled zero is accepted. It cannot judge the choice or the prediction. The key misconception is rerunning the reveal after changing the choice and calling the new choice validated. That is more exploration of the same data.

## The closing discussion

The notebook's last section runs the close in three steps. Keep them in this order.

**First, in pairs: the decision statement.** Each pair states its decision in four or five sentences: the action (send for AF review), their prices, their model and cutoff, the evidence from the frozen report, and one changed assumption that would make them reconsider, such as a higher miss price or a different prevalence. This replaces a written answer: say it aloud. The optional cell below the prompt records a statement of at least <!-- derived:checks.reflection_words -->40<!-- /derived --> words but cannot judge it. There is no correct price for a missed AF recording, so listen for a valuation that is stated and defended, not for a particular number.

**Then split the room.** Half the groups take the first two questions and the other half the last two. Each half reports back in about two minutes.

1. **The pathway.** Who receives a flagged recording, how quickly, and what do they do next? Who acts on a review that finds AF, and who answers for a recording that was never reviewed? A review that nobody has capacity to act on has no benefit, whatever the matrix says.
2. **Drift and prevalence.** The prices and cutoff were chosen on one cohort. What changes if AF becomes commoner or rarer, or a new recording device shifts the scores? Which numbers in the report would you monitor? Extensions 1 and 4 work through recalibration and prevalence.
3. **What J assumed.** Maximising J implied a miss price of <!-- number:core.dev.j_implied_miss_ratio|.2f -->9.88<!-- /number --> false referrals here, set by the case mix. Push for the reason it would change with prevalence: J keeps its cutoff when only the case mix changes, so the price it implies moves although nobody's clinical judgement has.
4. **Evidence before use.** Which of these does today's analysis address, and which does it not? A prospective evaluation on local data with the local case mix and pathway; monitoring of reviews, misses and calibration after go-live; clinical safety management; a regulatory assessment.

For the fourth question, keep the governance facts precise:

- In England, the clinical risk management standards are DCB0129, which applies to the manufacturer of a health IT system, and DCB0160, which applies to the organisation that deploys and uses it. They are not UK-wide.
- Whether software like this is a medical device depends on its intended purpose, as stated by the manufacturer. The MHRA regulates medical devices in the UK. Do not tell students it definitely is or is not one; say it needs a regulatory assessment.
- Using identifiable or pseudonymised patient data would bring in UK GDPR. Whether a data protection impact assessment is needed depends on the proposed processing and its risk; it is required where the processing is likely to result in a high risk to individuals.

Today's analysis addresses none of these. It shows how prices, thresholds and model choice connect.

**Finally, everyone.** Ask what each student will report beside AUC for their own classifier. Extension 6 runs today's analysis on its scores. End the session there.

## Reporting a problem

If a notebook fails, a checker rejects a correct answer, or something in the text is wrong, open an issue in this repository or tell the course team. Include the notebook name and the problem number, what you did and what you expected, and the full error message copied as text. Do not include anything that could identify a student.
