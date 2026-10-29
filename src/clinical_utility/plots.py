"""Figures and explorers for the core notebook. Each one draws what the learner's functions return."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display

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


def expected_cost_figure(costs):
    probabilities = np.linspace(0, 1, 101)
    fig = go.Figure()
    for action, name, colour in (("review", "Review", COLOURS[0]),
                                  ("no_review", "No review", COLOURS[1])):
        fig.add_trace(go.Scatter(x=probabilities,
                                 y=[ref.expected_loss(p, action, costs) for p in probabilities],
                                 name=name, line={"color": colour, "width": 3}))
    try:
        threshold = ref.threshold_probability(costs)
        fig.add_vline(x=threshold, line_dash="dash", line_color="#64748b",
                      annotation_text=f"Equal expected loss: {threshold:.1%}")
    except ValueError:
        fig.add_annotation(text="Compare the actions directly: the threshold rule does not apply.",
                           x=.5, y=1.1, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(title="AF probability for this recording", tickformat=".0%")
    fig.update_yaxes(title="Expected loss, in reviews")
    return _finish(fig, "One recording, two actions, before the reference outcome is known")


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


def show_policy(frame, threshold, costs, policy_cost=ref.policy_cost, model="random_forest"):
    """Each recording's decision and outcome, and the policy's total from ``policy_cost``."""
    review = frame[model] >= threshold
    outcome = np.select([review & frame["af"], review & ~frame["af"], ~review & frame["af"]],
                        ["TP", "FP", "FN"], "TN")
    ledger = pd.DataFrame({"recording": frame["recording"], "reference": frame["reference_label"],
                           "score": frame[model].round(3), "review": review, "outcome": outcome,
                           "price": [costs[o] for o in outcome]})
    display(ledger)
    total = policy_cost(frame["af"], review, costs)
    print(f"Cutoff {threshold:.3f}: {int(review.sum())} reviews, "
          f"{int((frame['af'] & ~review).sum())} AF missed; your policy_cost gives {total:g} "
          f"(the prices above sum to {ledger['price'].sum():g}).")
    return ledger


def policy_explorer(policy_cost, frame=None):
    """Sliders for the cutoff and two prices; the total comes from the learner's ``policy_cost``."""
    import ipywidgets as widgets
    source = data.six_recordings() if frame is None else frame
    threshold = widgets.FloatSlider(value=.5, min=0, max=1.01, step=.01,
                                    description="Cutoff:", continuous_update=False)
    missed = widgets.FloatSlider(value=5, min=1.1, max=50, step=.1,
                                 description="Missed AF:", continuous_update=False)
    extra = widgets.FloatSlider(value=0, min=0, max=8, step=.5,
                                description="FP harm:", continuous_update=False)
    out = widgets.Output()

    def redraw(_=None):
        with out:
            out.clear_output(wait=True)
            costs = {"TP": 1, "FP": 1 + extra.value, "FN": missed.value, "TN": 0}
            try:
                show_policy(source, threshold.value, costs, policy_cost)
            except (ValueError, TypeError, KeyError) as error:
                print(f"Your function needs attention: {error}")
                print("Run its checker, correct it and rerun this explorer cell.")
    for control in (threshold, missed, extra):
        control.observe(redraw, names="value")
    display(widgets.VBox([threshold, missed, extra]), out)
    redraw()
    return {"threshold": threshold, "missed_af": missed, "fp_extra": extra, "output": out}


def objective_figure(points, marks, title="Loss of every operating point"):
    """Loss against cutoff for one operating-point table, with labelled rows.

    ``marks`` maps a label to a row; hover text shows its reviews and AF missed.
    The review-nobody row (cutoff infinity) is drawn at 1.0.
    """
    x = np.minimum(points["threshold"].to_numpy(dtype=float), 1.0)
    fig = go.Figure(go.Scatter(x=x, y=points["loss"], name="Loss",
                               line={"color": "#889aa6", "shape": "vh"},
                               hovertemplate="Cutoff %{x:.3f}<br>Loss %{y:.0f}<extra></extra>"))
    symbols = ("star", "diamond", "square", "triangle-up")
    for i, (label, row) in enumerate(marks.items()):
        fig.add_trace(go.Scatter(
            x=[min(float(row["threshold"]), 1.0)], y=[row["loss"]], mode="markers", name=label,
            marker={"size": 13, "color": COLOURS[i % len(COLOURS)], "symbol": symbols[i % 4]},
            text=[f"{int(row['reviews'])} reviews · {int(row['af_missed'])} AF missed"],
            hovertemplate="Cutoff %{x:.3f}<br>Loss %{y:.0f}<br>%{text}<extra></extra>"))
    fig.update_xaxes(title="Score cutoff · review when score ≥ cutoff", range=[0, 1.02])
    fig.update_yaxes(title="Loss, in reviews")
    return _finish(fig, title, 420)


def model_choice_figure(sensitivity):
    """Each model's minimum loss against the price of a missed AF recording.

    ``sensitivity`` is the table from ``miss_price_sensitivity``: one row per
    miss price and model.
    """
    fig = go.Figure()
    for i, (model, rows) in enumerate(sensitivity.groupby("model", sort=False)):
        fig.add_trace(go.Scatter(x=rows["miss_price"], y=rows["loss"], mode="lines+markers",
                                 name=data.MODEL_NAMES.get(model, model),
                                 line={"color": COLOURS[i % len(COLOURS)], "width": 3},
                                 customdata=np.c_[rows["reviews"], rows["af_missed"]],
                                 hovertemplate="Miss price %{x}<br>Minimum loss %{y:.0f}<br>"
                                               "%{customdata[0]:.1f} reviews and %{customdata[1]:.1f}"
                                               " AF missed per 1,000<extra></extra>"))
    fig.update_xaxes(title="Price of one missed AF recording, in reviews", dtick=2)
    fig.update_yaxes(title="Minimum loss · each model at its own cheapest cutoff")
    return _finish(fig, "Which model is cheaper depends on the price of a miss", 420)


def decision_curves(curve, preference=None):
    """Draw a decision-curve table: one row per preference, a column per strategy."""
    fig = go.Figure()
    strategies = [c for c in curve.columns if c != "preference"]
    for i, name in enumerate(strategies):
        style = ({"color": "#889aa6", "dash": "dash"} if name == "review all" else
                 {"color": "#243e48", "dash": "dot"} if name == "review none" else
                 {"color": COLOURS[i % len(COLOURS)]})
        fig.add_trace(go.Scatter(x=curve["preference"], y=1000 * curve[name],
                                 name=data.MODEL_NAMES.get(name, name), line=style))
    if preference is not None:
        fig.add_vline(x=preference, line_color="#243e48", line_dash="dash")
    models = [c for c in strategies if c not in ("review all", "review none")]
    top = 1000 * float(np.nanmax(curve[models].to_numpy())) if models else 100
    fig.update_xaxes(title="Threshold probability (preference)", tickformat=".0%")
    fig.update_yaxes(title="Net benefit per 1,000 recordings", range=[-100, 1.2 * top])
    fig.add_annotation(text="Values below −100 per 1,000 are clipped", x=0, y=1.08,
                       xref="paper", yref="paper", showarrow=False, font={"size": 10})
    return _finish(fig, "Compare strategies at one preference: read the curves vertically", 470)


def decision_curve_explorer(decision_curve_function, frame, models):
    """A preference slider over the curves from the learner's ``decision_curve``."""
    import ipywidgets as widgets
    curve = decision_curve_function(frame, models, np.round(np.arange(.01, .66, .01), 2))
    preference = widgets.FloatSlider(value=.2, min=.01, max=.65, step=.01,
                                     description="Preference:", readout_format=".0%",
                                     continuous_update=False)

    def update(value):
        display(decision_curves(curve, value))
        row = curve.iloc[int(np.argmin(np.abs(curve["preference"] - value)))]
        print(f"At {value:.0%} a false referral is weighted {value / (1 - value):.3f}, which "
              f"corresponds to a miss priced at {1 / value:.1f} reviews.")
        print("Net benefit per 1,000: " + ", ".join(
            f"{data.MODEL_NAMES.get(m, m)} {1000 * row[m]:.1f}" for m in models))
    out = widgets.interactive_output(update, {"value": preference})
    display(preference, out)
    return {"preference": preference, "output": out}


def show_report(reports):
    """Development and evaluation reports of one frozen policy, side by side."""
    table = pd.concat({split: report for split, report in reports.items()}, axis=1)
    display(table.round(3))
    return table
