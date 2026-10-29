"""Displays of learner-computed results; no exercise calculation is substituted."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOURS = ("#167b76", "#b75b35", "#7663a8")


def _finish(fig, title):
    fig.update_layout(template="plotly_white", title=title, height=430,
                      margin=dict(l=65, r=30, t=65, b=65),
                      font=dict(family="Arial", size=13))
    return fig


def table(columns, rows, title):
    """Rows are dictionaries computed in the notebook."""
    values = [[r[c] for r in rows] for c in columns]
    fig = go.Figure(go.Table(header=dict(values=columns), cells=dict(values=values)))
    return _finish(fig, title)


def reliability(summaries):
    """Each summary has predicted, observed and n arrays from the learner's bins."""
    fig = go.Figure()
    for i, (name, result) in enumerate(summaries.items()):
        fig.add_trace(go.Scatter(x=result["predicted"], y=result["observed"],
                                mode="markers+lines", name=name, customdata=result["n"],
                                line=dict(color=COLOURS[i % len(COLOURS)]),
                                hovertemplate="Predicted %{x:.3f}<br>Observed %{y:.3f}<br>n=%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Agreement", line=dict(dash="dot")))
    fig.update_xaxes(title="Mean probability in equal-count bin", range=[0, 1])
    fig.update_yaxes(title="Observed AF fraction", range=[0, 1])
    return _finish(fig, "Reliability: bin averages hide variation within bins")


def reclassification(events, non_events):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Reference AF", "Reference non-AF"))
    for col, values in enumerate((events, non_events), start=1):
        fig.add_trace(go.Heatmap(z=values, text=values, texttemplate="%{text}",
                                 x=["No review", "Review"], y=["No review", "Review"],
                                 showscale=False, colorscale="Teal"), row=1, col=col)
        fig.update_xaxes(title="New: raw ResNet action", row=1, col=col)
        fig.update_yaxes(title="Old: RF action", autorange="reversed", row=1, col=col)
    return _finish(fig, "Reclassification at the same cut: rows old, columns new")


def queue(daily_reviews, capacity):
    fig = go.Figure(go.Histogram(x=daily_reviews, xbins=dict(size=1), name="Days"))
    fig.add_vline(x=capacity + .5, line_dash="dash", annotation_text="Capacity boundary")
    fig.update_xaxes(title="Review demand in a day")
    fig.update_yaxes(title="Number of days")
    return _finish(fig, "Demand above capacity needs an explicit overflow pathway")


def windows(recording, scores, truth, threshold):
    fig = go.Figure()
    index = np.arange(len(scores))
    fig.add_trace(go.Scatter(x=index, y=scores, name="Window score", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=index, y=np.asarray(truth, int), name="Synthetic AF interval",
                            line=dict(dash="dot", shape="hv")))
    fig.add_hline(y=threshold, line_dash="dash", annotation_text="Window cut")
    fig.update_xaxes(title="Window index within this recording")
    fig.update_yaxes(title="Score / reference AF indicator", range=[-.05, 1.05])
    return _finish(fig, f"{recording}: inspect duration and isolated spikes")


def coefficients(fits, feature_names):
    fig = go.Figure()
    for name, beta in fits.items():
        fig.add_trace(go.Bar(x=["Intercept", *feature_names], y=beta, name=name))
    fig.update_yaxes(title="Log-odds coefficient")
    return _finish(fig, "Weighting changes the fitted log-odds scale")
