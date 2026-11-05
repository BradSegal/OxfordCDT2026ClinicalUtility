"""Figures for the notebook. All but the schematic draw tables that your own functions returned."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.metrics import auc

NAMES = {"random_forest": "Random forest", "resnet": "ResNet",
         "resnet_recalibrated": "ResNet, recalibrated", "flag all": "Flag everyone",
         "flag none": "Flag no one"}
COLOURS = {"random_forest": "#167b76", "resnet": "#b75b35", "resnet_recalibrated": "#7663a8",
           "flag all": "#94a3b8", "flag none": "#334155"}


def _finish(figure, title, height=430):
    figure.update_layout(
        template="plotly_white", title={"text": title, "font": {"size": 16}},
        font={"family": "Arial", "size": 12}, height=height,
        margin={"l": 60, "r": 25, "t": 60, "b": 110},
        legend={"orientation": "h", "y": -.22, "yanchor": "top", "x": 0},
    )
    return figure


def _row_at(table, cutoff):
    """The row whose policy is 'flag when the score is at or above ``cutoff``'."""
    at_or_above = table[table["cutoff"] >= cutoff]
    return at_or_above.loc[at_or_above["cutoff"].idxmin()]


def costs_by_flags(tables):
    """Each model's cost per 1,000 patients against how many it flags, cheapest cutoff marked.

    ``tables`` maps a model name to the table ``price_cutoffs`` returned.
    """
    figure = go.Figure()
    for model, table in tables.items():
        table = table.sort_values("flags")
        colour = COLOURS.get(model, "#306db1")
        figure.add_trace(go.Scatter(
            x=table["flags"], y=table["cost"], mode="lines", name=NAMES.get(model, model),
            line={"color": colour, "width": 2}, customdata=table["cutoff"],
            hovertemplate="cutoff %{customdata:.3f}<br>%{x:.1f} flags<br>$%{y:,.0f}<extra></extra>"))
        best = table.loc[table["cost"].idxmin()]
        half = _row_at(table, 0.5)
        for row, label, symbol in ((best, "cheapest", "star"), (half, "cutoff 0.5", "circle-open")):
            figure.add_trace(go.Scatter(
                x=[row["flags"]], y=[row["cost"]], mode="markers", showlegend=False,
                marker={"size": 13, "color": colour, "symbol": symbol},
                hovertemplate=(f"{NAMES.get(model, model)}, {label}<br>cutoff {row['cutoff']:.3f}"
                               "<br>%{x:.1f} flags<br>$%{y:,.0f}<extra></extra>")))
    figure.update_xaxes(title="Patients flagged per 1,000", range=[0, 500])
    figure.update_yaxes(title="Cost per 1,000 patients ($)")
    return _finish(figure, "Cost of every cutoff (stars: cheapest; open circles: 0.5)")


def decision_curves(curves):
    """Net benefit against the threshold probability.

    ``curves`` has one row per threshold and one column per strategy.
    """
    figure = go.Figure()
    for strategy in curves.columns:
        figure.add_trace(go.Scatter(
            x=curves.index, y=curves[strategy], mode="lines", name=NAMES.get(strategy, strategy),
            line={"color": COLOURS.get(strategy, "#306db1"), "width": 2.5,
                  "dash": "dot" if strategy.startswith("flag") else "solid"},
            hovertemplate="threshold %{x:.2f}<br>net benefit %{y:.4f}<extra></extra>"))
    top = float(np.nanmax(curves.to_numpy()))
    figure.update_xaxes(title="Threshold probability")
    figure.update_yaxes(title="Net benefit (AF found per patient)", range=[-0.01, top * 1.1])
    return _finish(figure, "Decision curves")


def calibration(curves):
    """Observed share with AF against the predicted probability, for each score.

    ``curves`` maps a name to the (observed, predicted) pair from ``calibration_curve``.
    """
    figure = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                  line={"color": "#94a3b8", "dash": "dot"}))
    for name, (observed, predicted) in curves.items():
        figure.add_trace(go.Scatter(x=predicted, y=observed, mode="lines+markers",
                                    name=NAMES.get(name, name),
                                    line={"color": COLOURS.get(name, "#306db1"), "width": 2.5}))
    figure.update_xaxes(title="Predicted probability of AF", range=[0, 1])
    figure.update_yaxes(title="Observed share with AF", range=[0, 1])
    return _finish(figure, "Calibration on the development patients")


def roc_demo():
    """A schematic ROC curve with three labelled cutoffs, drawn from invented numbers."""
    figure, ax = plt.subplots(figsize=(5.5, 5.5))
    fpr = np.linspace(0.0005, 0.9995, 400)
    curve = norm.cdf(1.5 + norm.ppf(fpr))
    ax.plot(np.r_[0, fpr, 1], np.r_[0, curve, 1], color="#306db1", linewidth=2.5)
    ax.plot([0, 1], [0, 1], ":", color="grey")
    points = ((0.03, "High cutoff:\nfew flags, many AF missed", (45, -30)),
              (0.2, "Middle cutoff", (30, -22)),
              (0.65, "Low cutoff:\nmost patients flagged", (-55, -55)))
    for x, label, offset in points:
        y = norm.cdf(1.5 + norm.ppf(x))
        ax.plot(x, y, "o", markersize=9, color="#306db1")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=offset, fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": "#64748b"})
    ax.annotate("Flag no one", (0, 0), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.annotate("Flag everyone", (1, 1), textcoords="offset points", xytext=(-82, -14), fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Schematic: every point is one cutoff")
    return figure, ax


def roc_curves(tables):
    """Draw each model's ROC curve from the tables ``roc_points`` returned; return (figure, axes)."""
    figure, ax = plt.subplots(figsize=(5.5, 5.5))
    for model, table in tables.items():
        table = table.sort_values(["fpr", "tpr"])
        ax.plot(table["fpr"], table["tpr"], color=COLOURS.get(model, "#306db1"),
                label=f"{NAMES.get(model, model)} (AUC {auc(table['fpr'], table['tpr']):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    return figure, ax


def mark(ax, model, fpr, tpr, cutoff):
    """Mark one cutoff's rates on the ROC axes, or say what is still missing."""
    if fpr is Ellipsis or tpr is Ellipsis:
        print(f"Fill in fpr and tpr to mark the cutoff of {cutoff} for {NAMES.get(model, model)}.")
        return
    ax.plot(fpr, tpr, "o", markersize=9, color=COLOURS.get(model, "#306db1"))
    ax.annotate(f"{cutoff}", (fpr, tpr), textcoords="offset points", xytext=(6, -12))


def pr_curves(curves, prevalence):
    """Precision against recall for each model, with the 0.5 point and the share with AF."""
    figure = go.Figure()
    for model, curve in curves.items():
        colour = COLOURS.get(model, "#306db1")
        figure.add_trace(go.Scatter(
            x=curve["recall"], y=curve["precision"], mode="lines",
            name=f"{NAMES.get(model, model)} (average precision {curve['average precision']:.3f})",
            line={"color": colour, "width": 2}))
        recall, precision = curve["at 0.5"]
        figure.add_trace(go.Scatter(x=[recall], y=[precision], mode="markers", showlegend=False,
                                    marker={"size": 11, "color": colour, "symbol": "circle-open"},
                                    hovertemplate="cutoff 0.5<br>recall %{x:.2f}<br>precision %{y:.2f}<extra></extra>"))
    figure.add_hline(y=prevalence, line_dash="dot", line_color="#94a3b8",
                     annotation_text=f"share with AF ({prevalence:.1%})")
    figure.update_xaxes(title="True positive rate (recall)", range=[0, 1])
    figure.update_yaxes(title="Precision (share of flags with AF)", range=[0, 1.02])
    return _finish(figure, "Precision and recall (open circles: cutoff 0.5)")


def capacity_curves(costs):
    """Each model's lowest cost per 1,000 patients under each limit on reviews."""
    figure = go.Figure()
    for model in costs.columns:
        figure.add_trace(go.Scatter(
            x=costs.index, y=costs[model], mode="lines+markers", name=NAMES.get(model, model),
            line={"color": COLOURS.get(model, "#306db1"), "width": 2.5},
            hovertemplate="at most %{x} reviews<br>$%{y:,.0f}<extra></extra>"))
    figure.update_xaxes(title="Most patients the cardiologists can review, per 1,000")
    figure.update_yaxes(title="Lowest cost per 1,000 patients ($)")
    return _finish(figure, "The cheapest policy each model can offer within a limit on reviews")


def daily_flags(days, capacity):
    """Histograms of simulated daily flag counts for each policy, with the daily capacity."""
    figure = go.Figure()
    for (name, counts), colour in zip(days.items(), ["#167b76", "#b75b35", "#7663a8", "#306db1"]):
        figure.add_trace(go.Histogram(x=counts, name=name, opacity=0.55, marker_color=colour,
                                      xbins={"size": 1}))
    figure.add_vline(x=capacity + 0.5, line_dash="dash", line_color="#334155",
                     annotation_text=f"capacity: {capacity} a day")
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title="Patients flagged in a day")
    figure.update_yaxes(title="Days in a simulated year")
    return _finish(figure, "Daily flags for each policy over a simulated year")
