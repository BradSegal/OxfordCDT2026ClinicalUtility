"""Figures of learner-computed results for the extensions; nothing here substitutes a calculation."""

import numpy as np
import plotly.graph_objects as go

COLOURS = ("#167b76", "#b75b35", "#7663a8", "#306db1")


def _finish(fig, title):
    fig.update_layout(template="plotly_white", title=title, height=430,
                      margin=dict(l=65, r=30, t=65, b=65), font=dict(family="Arial", size=13))
    return fig


def reliability(curves):
    """``curves`` maps a name to ``(prob_true, prob_pred)`` from ``calibration_curve``."""
    fig = go.Figure()
    for i, (name, (observed, predicted)) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=predicted, y=observed, mode="markers+lines", name=name,
                                 line=dict(color=COLOURS[i % len(COLOURS)]),
                                 hovertemplate="Mean probability %{x:.3f}<br>"
                                               "AF fraction %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Agreement", line=dict(dash="dot")))
    fig.update_xaxes(title="Mean stated probability in an equal-count bin", range=[0, 1])
    fig.update_yaxes(title="Observed AF fraction", range=[0, 1])
    return _finish(fig, "Reliability: points below the diagonal mean the probabilities are too high")


def queue(daily_reviews, capacity):
    fig = go.Figure(go.Histogram(x=daily_reviews, xbins=dict(size=1), name="Days"))
    fig.add_vline(x=capacity + .5, line_dash="dash", annotation_text="Capacity")
    fig.update_xaxes(title="Review demand in a day")
    fig.update_yaxes(title="Number of days")
    return _finish(fig, "Days whose demand is above capacity need an overflow pathway")


def windows(recording, scores, truth, threshold):
    fig = go.Figure()
    index = np.arange(len(scores))
    fig.add_trace(go.Scatter(x=index, y=scores, name="Window score", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=index, y=np.asarray(truth, int), name="AF window (reference)",
                             line=dict(dash="dot", shape="hv")))
    fig.add_hline(y=threshold, line_dash="dash", annotation_text="Window cut")
    fig.update_xaxes(title="Window index within this recording")
    fig.update_yaxes(title="Score / reference AF indicator", range=[-.05, 1.05])
    return _finish(fig, f"{recording}: duration and isolated spikes")


def coefficients(models, feature_names):
    """``models`` maps a name to a fitted ``LogisticRegression``."""
    fig = go.Figure()
    for name, model in models.items():
        fig.add_trace(go.Bar(x=["Intercept", *feature_names],
                             y=[float(model.intercept_[0]), *model.coef_[0]], name=name))
    fig.update_yaxes(title="Log-odds coefficient")
    return _finish(fig, "Weighting mostly moves the intercept")
