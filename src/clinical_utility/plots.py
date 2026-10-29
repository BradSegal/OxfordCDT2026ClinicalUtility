"""Linked, labelled displays for the learner's own decision-cost function."""

import html

import numpy as np
import plotly.graph_objects as go
from IPython.display import HTML, display
from plotly.subplots import make_subplots

from . import data, reference as ref

COLOURS = ["#167b76", "#b75b35", "#7663a8", "#306db1"]


def _finish(figure, title, height=400):
    figure.update_layout(
        template="plotly_white", title={"text": title, "font": {"size": 17}},
        font={"family": "Arial", "size": 12}, height=height,
        margin={"l": 60, "r": 25, "t": 65, "b": 130},
        legend={"orientation": "h", "y": -.3, "yanchor": "top", "x": 0},
    )
    return figure


def show_ledger(ledger, limit=12):
    columns = ["recording", "reference_label", "score", "review", "outcome", "loss"]
    def cell(row, key):
        value = row.get(key, "")
        return html.escape(f"{value:.3f}" if isinstance(value, float) else str(value))
    head = "".join(f"<th style='text-align:left;padding:8px'>{k}</th>" for k in columns)
    body = "".join("<tr>" + "".join(f"<td style='padding:8px;border-top:1px solid #d5dfe3'>{cell(r,k)}</td>"
                                   for k in columns) + "</tr>" for r in ledger[:limit])
    display(HTML(f"<div style='overflow:auto'><table>{head}{body}</table></div>"))
    if len(ledger) > limit:
        print(f"Showing {limit} of {len(ledger)} recording decisions; totals use every row.")


def expected_cost_figure(costs):
    probabilities = np.linspace(0, 1, 101)
    fig = go.Figure()
    for action, name, colour in (("review", "Review", COLOURS[0]),
                                  ("no_review", "No review", COLOURS[1])):
        fig.add_trace(go.Scatter(x=probabilities, y=[ref.expected_cost(p, action, costs)
                            for p in probabilities], name=name, line={"color": colour, "width": 3}))
    try:
        threshold = ref.probability_threshold(costs)
        fig.add_vline(x=threshold, line_dash="dash", line_color="#64748b",
                      annotation_text=f"Equal expected cost: {threshold:.1%}")
    except ValueError:
        fig.add_annotation(text="Compare actions directly: the usual threshold assumptions do not hold.",
                           x=.5, y=1.1, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(title="AF probability for this recording", tickformat=".0%")
    fig.update_yaxes(title="Expected loss · illustrative common units")
    return _finish(fig, "One recording, two actions · before the reference outcome is known")


def expected_cost_explorer():
    import ipywidgets as widgets
    missed = widgets.FloatSlider(value=5, min=1.1, max=20, step=.1,
                                  description="Missed AF:", continuous_update=False)
    extra = widgets.FloatSlider(value=0, min=0, max=8, step=.5,
                                 description="FP harm:", continuous_update=False)
    def update(missed_af, fp_extra):
        display(expected_cost_figure({"TP": 1, "FP": 1 + fp_extra, "FN": missed_af, "TN": 0}))
    out = widgets.interactive_output(update, {"missed_af": missed, "fp_extra": extra})
    display(widgets.VBox([missed, extra]), out)
    return {"missed_af": missed, "fp_extra": extra, "output": out}


def recording_figure(ledger, threshold):
    fig = go.Figure()
    for reviewed, colour, symbol, name in ((True, COLOURS[0], "circle", "Review"),
                                           (False, "#889aa6", "circle-open", "No review")):
        subset = [r for r in ledger if r["review"] == reviewed]
        fig.add_trace(go.Scatter(x=[r["score"] for r in subset],
                     y=[r["recording"] for r in subset], mode="markers", name=name,
                     marker={"color": colour, "size": 15, "symbol": symbol},
                     text=[f"Reference {r['reference_label']} · {r['outcome']} · loss {r['loss']:g}"
                           for r in subset], hovertemplate="%{y}<br>Score %{x:.3f}<br>%{text}<extra></extra>"))
    fig.add_vline(x=threshold, line_dash="dash", line_color=COLOURS[1])
    fig.update_xaxes(range=[0, 1.03], title="AF score · simulated")
    return _finish(fig, "The same recording can change action when the threshold changes", 340)


def population_figure(counts):
    m = ref.metrics(counts)
    values = [[counts["TP"], counts["FN"]], [counts["FP"], counts["TN"]]]
    fig = go.Figure(go.Heatmap(z=values, x=["Review", "No review"], y=["AF", "Non-AF"],
                              text=[[f"{v:.1f}" for v in row] for row in values],
                              texttemplate="%{text}", textfont={"size": 22},
                              colorscale=[[0, "#f1f5f8"], [1, "#79b4b0"]], showscale=False))
    fig.update_yaxes(autorange="reversed")
    ppv = "undefined" if not np.isfinite(m["ppv"]) else f"{m['ppv']:.1%}"
    npv = "undefined" if not np.isfinite(m["npv"]) else f"{m['npv']:.1%}"
    return _finish(fig, f"PPV {ppv} = TP/(TP+FP) · NPV {npv} = TN/(TN+FN)", 300)


def policy_figures(records, threshold, costs, cost_function=ref.decision_cost, prevalence=None):
    """Both the ledger and cost curve are computed from the supplied learner function."""
    ledger = ref.evaluate_policy(records, threshold, costs, cost_function)
    actual_counts = ref.confusion_counts(ledger)
    costs_by_cell = {key: cost_function(actual, reviewed, costs) for key, actual, reviewed in (
        ("TP", True, True), ("FP", False, True), ("FN", True, False), ("TN", False, False))}
    rows = ref.sweep_thresholds(records, costs, cost_function)
    if prevalence is not None:
        for row in rows:
            row["total_loss"] = ref.matrix_loss(ref.projected_counts(row, prevalence), costs_by_cell)
        counts = ref.projected_counts(actual_counts, prevalence)
        units = "per 1,000 · assumed prevalence"
    else:
        counts = actual_counts
        units = f"across {len(records):,} recordings"
    chosen_loss = ref.matrix_loss(counts, costs_by_cell)
    best = ref.best_thresholds(rows)
    best_j = ref.best_thresholds(rows, "youden_j", minimise=False)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Consequences at this cutoff", "Loss against cutoff"])
    for i, key in enumerate(("TP", "FP", "FN", "TN")):
        fig.add_trace(go.Bar(x=["Total loss"], y=[counts[key] * costs_by_cell[key]], name=key,
                            marker_color=COLOURS[i]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["threshold"] for r in rows], y=[r["total_loss"] for r in rows],
                            line={"color": COLOURS[0], "shape": "vh"}, name="Policy loss"), row=1, col=2)
    fig.add_vline(x=threshold, line_dash="dash", line_color=COLOURS[1], row=1, col=2)
    for selected, name, colour, symbol in ((best[0], "Minimum loss", COLOURS[0], "star"),
                                            (best_j[0], "Maximum J", COLOURS[2], "diamond")):
        fig.add_trace(go.Scatter(x=[selected["threshold"]], y=[selected["total_loss"]],
                                mode="markers", name=name, marker={"size": 11, "color": colour,
                                "symbol": symbol}), row=1, col=2)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Score cutoff; 1.01 represents review none", row=1, col=2)
    fig.update_yaxes(title_text="Illustrative loss units", row=1, col=1)
    _finish(fig, f"Simulated predictions · loss {chosen_loss:.1f} {units}", 420)
    return {"ledger": ledger, "counts": counts, "metrics": ref.metrics(counts),
            "loss": chosen_loss, "best": best, "best_j": best_j,
            "figure": fig, "matrix": population_figure(counts), "rows": rows}


def show_policy(records, threshold, costs, cost_function=ref.decision_cost, prevalence=None):
    result = policy_figures(records, threshold, costs, cost_function, prevalence)
    m, c = result["metrics"], result["counts"]
    print(f"Sensitivity {m['sensitivity']:.1%}; specificity {m['specificity']:.1%}.")
    print(f"PPV denominator: {c['TP']:.1f}/{c['TP']+c['FP']:.1f}; "
          f"NPV denominator: {c['TN']:.1f}/{c['TN']+c['FN']:.1f}.")
    print(f"Reviews {m['reviews']:.1f}; missed AF {m['missed_af']:.1f}; total loss {result['loss']:.1f}.")
    if prevalence is not None:
        print("Expected counts per 1,000; fixed sensitivity/specificity. This is a prior-shift scenario.")
    print(f"Review-all loss {result['rows'][0]['total_loss']:.1f}; "
          f"review-none loss {result['rows'][-1]['total_loss']:.1f}.")
    if len(records) <= 12:
        display(recording_figure(result["ledger"], threshold))
        show_ledger(result["ledger"])
    display(result["matrix"], result["figure"])
    print(f"Minimum-loss grid cutoffs: {len(result['best'])}; "
          f"one is {result['best'][0]['threshold']:.4f}. Multiple cutoffs can induce identical calls.")
    return result


def policy_explorer(cost_function, records=None, allow_models=False, allow_prevalence=False):
    import ipywidgets as widgets
    source = data.teaching_cases() if records is None else records
    threshold = widgets.FloatSlider(value=.5, min=0, max=1.01, step=.01,
                                     description="Cutoff:", continuous_update=False)
    missed = widgets.FloatSlider(value=5, min=1.1, max=50, step=.1,
                                  description="Missed AF:", continuous_update=False)
    extra = widgets.FloatSlider(value=0, min=0, max=8, step=.5,
                                 description="FP harm:", continuous_update=False)
    model = widgets.Dropdown(options=[(v, k) for k, v in data.MODEL_NAMES.items()],
                             description="Model:", layout={"width": "450px"})
    prevalence = widgets.FloatSlider(value=.1, min=.01, max=.3, step=.01,
                                      description="AF fraction:", readout_format=".0%",
                                      continuous_update=False)
    out = widgets.Output()
    state = {}
    def redraw(_=None):
        with out:
            out.clear_output(wait=True)
            selected = data.load_records(model.value) if allow_models else source
            costs = {"TP": 1, "FP": 1 + extra.value, "FN": missed.value, "TN": 0}
            try:
                state["result"] = show_policy(selected, threshold.value, costs, cost_function,
                                               prevalence.value if allow_prevalence else None)
                state["settings"] = {"threshold": threshold.value, "costs": costs,
                                     "model": model.value if allow_models else "teaching cases"}
            except (ValueError, TypeError, KeyError) as error:
                print(f"Your function needs attention: {error}")
                print("Run its checker, correct it and rerun this explorer cell. No reference "
                      "function was substituted.")
    controls = ([model] if allow_models else []) + [threshold, missed, extra]
    if allow_prevalence:
        controls.append(prevalence)
    for control in controls:
        control.observe(redraw, names="value")
    display(widgets.VBox(controls), out)
    redraw()
    return {"threshold": threshold, "missed_af": missed, "fp_extra": extra,
            "model": model, "prevalence": prevalence, "state": state, "output": out}


def _counts_at(scores, truth, threshold):
    review = scores >= threshold
    return {"TP": int(np.sum(review & truth)), "FP": int(np.sum(review & ~truth)),
            "FN": int(np.sum(~review & truth)), "TN": int(np.sum(~review & ~truth))}


def decision_curves(split="development", preference=.2, models=None, net_benefit_function=None):
    """Net benefit of each model's 'review when p >= preference' policy, review all and none.

    ``net_benefit_function(counts, p)`` defaults to the reference; pass the learner's own.
    """
    models = list(data.MODEL_NAMES) if models is None else models
    net_benefit = ref.net_benefit if net_benefit_function is None else net_benefit_function
    pts = np.linspace(.01, .65, 100)
    fig = go.Figure()
    table = data.load_arrays(split)
    truth = table["af_present"]
    prevalence = float(np.mean(truth))
    for i, model in enumerate(models):
        values = [net_benefit(_counts_at(table[model], truth, p), p) for p in pts]
        fig.add_trace(go.Scatter(x=pts, y=[1000 * v for v in values],
                                name=data.MODEL_NAMES[model], line={"color": COLOURS[i]}))
    review_all = _counts_at(np.ones(len(truth)), truth, 0)
    fig.add_trace(go.Scatter(x=pts, y=[1000 * net_benefit(review_all, p) for p in pts],
                            name="Review all", line={"color": "#889aa6", "dash": "dash"}))
    fig.add_trace(go.Scatter(x=pts, y=np.zeros_like(pts), name="Review none",
                            line={"color": "#243e48", "dash": "dot"}))
    fig.add_vline(x=preference, line_color="#243e48", line_dash="dash")
    fig.update_xaxes(title="Threshold probability · clinical preference", tickformat=".0%")
    fig.update_yaxes(title="Net benefit × 1,000", range=[-100, 1000 * prevalence * 1.2])
    fig.add_annotation(text="Vertical view clipped below −100; zoom out to inspect lower values",
                       x=0, y=1.08, xref="paper", yref="paper", showarrow=False, font={"size": 10})
    return _finish(fig, f"Simulated predictions · {split} · compare at the same preference", 470)


def decision_curve_explorer(net_benefit_function=None):
    import ipywidgets as widgets
    preference = widgets.FloatSlider(value=.2, min=.01, max=.65, step=.01,
                                       description="Preference:", readout_format=".0%",
                                       continuous_update=False)
    def update(value):
        display(decision_curves(preference=value, models=["random_forest", "resnet", "resnet_calibrated"],
                                net_benefit_function=net_benefit_function))
        print(f"At {value:.0%}, one false-positive action is weighted {value/(1-value):.3f} "
              "relative to the benefit of one true-positive action.")
        print("For legibility the vertical axis hides net benefit below −100 per 1,000. "
              "Raw scores are displayed to reveal miscalibration, not endorsed as calibrated probabilities.")
    out = widgets.interactive_output(update, {"value": preference})
    display(preference, out)
    return {"preference": preference, "output": out}


def calibration_figure(split="evaluation"):
    fig = go.Figure()
    for i, model in enumerate(data.MODEL_NAMES):
        result = ref.calibration_summary(data.load_records(model, split))
        fig.add_trace(go.Scatter(x=[p["predicted"] for p in result["points"]],
                                y=[p["observed"] for p in result["points"]], mode="lines+markers",
                                name=f"{data.MODEL_NAMES[model]} · Brier {result['brier']:.3f}",
                                line={"color": COLOURS[i]},
                                text=[f"n={p['n']}" for p in result["points"]],
                                hovertemplate="Mean score %{x:.3f}<br>AF fraction %{y:.3f}<br>%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect agreement", line={"dash": "dot", "color": "#667788"}))
    fig.update_xaxes(title="Mean stated probability in a quantile bin", range=[0, 1])
    fig.update_yaxes(title="Observed AF reference fraction", range=[0, 1])
    return _finish(fig, f"Calibration on {split} records · simulated predictions", 480)


def objective_figure(rows, marks, title="Total loss of every distinct cutoff"):
    """Total loss against cutoff for one sweep, with labelled operating points.

    ``marks`` maps a label to a sweep row; hover text shows its reviews and misses.
    """
    fig = go.Figure(go.Scatter(x=[r["threshold"] for r in rows], y=[r["total_loss"] for r in rows],
                               name="Total loss", line={"color": "#889aa6", "shape": "vh"},
                               hovertemplate="Cutoff %{x:.3f}<br>Loss %{y:.0f}<extra></extra>"))
    symbols = ("star", "diamond", "square", "triangle-up")
    for i, (label, row) in enumerate(marks.items()):
        fig.add_trace(go.Scatter(
            x=[row["threshold"]], y=[row["total_loss"]], mode="markers", name=label,
            marker={"size": 13, "color": COLOURS[i % len(COLOURS)], "symbol": symbols[i % 4]},
            text=[f"{row['reviews']} reviews · {row['missed_af']} missed AF"],
            hovertemplate="Cutoff %{x:.3f}<br>Loss %{y:.0f}<br>%{text}<extra></extra>"))
    fig.update_xaxes(title="Score cutoff · review when score ≥ cutoff", range=[0, 1.02])
    fig.update_yaxes(title="Total loss · illustrative units")
    return _finish(fig, title, 420)


def model_choice_figure(rows_by_model, policy_loss, miss_prices=range(2, 31), base_costs=None):
    """Each model's minimum total loss as the price of a missed AF recording changes.

    ``rows_by_model`` maps a display name to an exact sweep; ``policy_loss(row, prices)``
    prices one operating point, so the learner's own pricing drives the curves.
    """
    base = dict(ref.DEFAULT_COSTS if base_costs is None else base_costs)
    prices = list(miss_prices)
    fig = go.Figure()
    for i, (name, rows) in enumerate(rows_by_model.items()):
        losses = [min(policy_loss(row, {**base, "FN": miss}) for row in rows) for miss in prices]
        fig.add_trace(go.Scatter(x=prices, y=losses, name=name, mode="lines+markers",
                                 line={"color": COLOURS[i % len(COLOURS)], "width": 3},
                                 hovertemplate="Miss price %{x}<br>Minimum loss %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title="Price of one missed AF recording · other prices fixed", dtick=2)
    fig.update_yaxes(title="Minimum total loss · each model at its own best cutoff")
    return _finish(fig, "Which model is cheaper depends on the price of a miss", 420)


def show_report(report, predicted_loss):
    """Development and evaluation results of one frozen policy, side by side, with the prediction.

    ``report`` maps "development" and "evaluation" to ``reference.policy_summary``
    results. The scaled anchor is the development loss per recording times the
    number of evaluation recordings.
    """
    development, evaluation = report["development"], report["evaluation"]
    rows = [("Recordings", "recordings", "{:,}"), ("Reviews", "reviews", "{:,}"),
            ("Missed AF", "missed_af", "{:,}"), ("Total loss", "total_loss", "{:,.0f}"),
            ("Loss per recording", "mean_loss", "{:.3f}")]
    print(f"{'':20}{'Development':>14}{'Evaluation':>14}")
    for label, key, form in rows:
        print(f"{label:20}{form.format(development[key]):>14}{form.format(evaluation[key]):>14}")
    scaled = development["mean_loss"] * evaluation["recordings"]
    print(f"\nYour predicted evaluation loss: {predicted_loss:,.0f}. Development loss scaled to "
          f"{evaluation['recordings']:,} recordings: {scaled:,.0f}.")
