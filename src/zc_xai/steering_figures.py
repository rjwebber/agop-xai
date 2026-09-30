"""Plotting-only implementations for final ZC steering Figures 10--12.

This module depends only on the compact public bundle. It deliberately does
not import campaign orchestration, rescue policies, or adjoint solvers.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .steering_plot_bundle import SteeringPlotBundle

IBM_RED = "#DA1E28"
IBM_BLUE = "#0F62FE"
LIGHT_RED = "#FFD7D9"
LIGHT_BLUE = "#D0E2FF"
BLACK = "#000000"
MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)

FIGURE10_EVENT_ORDER = ("extreme_el_nino", "extreme_la_nina")
FIGURE10_EVENT_START_MONTH = {"extreme_el_nino": 8, "extreme_la_nina": 5}
FIGURE10_CALENDAR_TICK_OFFSETS = (1, 4, 7, 10, 13)
FIGURE10_NUDGE_SPAN_MONTHS = (-13.0, -10.0)

FIGURE11_Y_LIMITS = (-2.5, 5.0)
FIGURE11_COLD_START_MONTH = 5
FIGURE11_WARM_START_MONTH = 8
FIGURE11_COLD_NUDGE_SPAN = (1.0, 4.0)
FIGURE11_WARM_NUDGE_SPAN_CONTROL = (4.0, 7.0)
FIGURE11_WARM_NUDGE_SPAN_EVENT = (1.0, 4.0)
FIGURE11_NUDGED_LINE_STYLE = (0, (4.0, 2.2))

FIGURE12_EVENT_ORDER = ("extreme_el_nino", "extreme_la_nina")
FIGURE12_METHOD_ORDER = ("AGOP", "GRAD", "IG", "GradientSHAP", "Composite")
FIGURE12_METHOD_KEYS = {
    "AGOP": "agop",
    "GRAD": "grad",
    "IG": "ig",
    "GradientSHAP": "gradientshap",
    "Composite": "composite",
}
FIGURE12_METHOD_LABELS = {
    "AGOP": "AGOP XAI",
    "GRAD": "GRAD",
    "IG": "IG",
    "GradientSHAP": "GradientSHAP",
    "Composite": "10% Composite",
}
FIGURE12_METHOD_STYLES: dict[str, dict[str, Any]] = {
    "AGOP": {"color": "#648FFF", "linestyle": "-", "linewidth": 2.7},
    "GRAD": {
        "color": "#DC267F",
        "linestyle": (0, (1.2, 1.4)),
        "linewidth": 2.3,
    },
    "IG": {
        "color": "#FFB000",
        "linestyle": (0, (5.0, 1.7, 1.2, 1.7)),
        "linewidth": 2.3,
    },
    "GradientSHAP": {
        "color": "#FE6100",
        "linestyle": (0, (5.0, 2.0)),
        "linewidth": 2.3,
    },
    "Composite": {
        "color": "#785EF0",
        "linestyle": (0, (7.0, 2.1)),
        "linewidth": 2.4,
    },
}
FIGURE12_DISPLAY_STEP_OFFSETS = np.arange(-42, 1, dtype=np.int64)
FIGURE12_MONTHS_RELATIVE_TO_TARGET = FIGURE12_DISPLAY_STEP_OFFSETS / 3.0
FIGURE12_NUDGE_SPAN_MONTHS = (-13.0, -10.0)


def _tinted_color(base: str, magnitude: float) -> tuple[float, float, float]:
    absolute = abs(float(magnitude))
    if not math.isfinite(absolute) or absolute < 0.1 or absolute > 1.0:
        raise ValueError("coefficient magnitude must lie in [0.1, 1.0]")
    scaled = (absolute - 0.1) / 0.9
    weight = 0.82 - 0.55 * scaled
    rgb = np.asarray(mcolors.to_rgb(base), dtype=np.float64)
    return tuple((1.0 - weight) + weight * rgb)


def _figure10_ticks(event: str) -> tuple[np.ndarray, list[str]]:
    offsets = np.asarray(FIGURE10_CALENDAR_TICK_OFFSETS, dtype=np.float64)
    ticks = -14.0 + offsets
    start_month = FIGURE10_EVENT_START_MONTH[event]
    labels = [MONTH_NAMES[(start_month + int(offset)) % 12] for offset in offsets]
    return ticks, labels


def _figure10_events(bundle: SteeringPlotBundle) -> tuple[SimpleNamespace, ...]:
    events = []
    for event in FIGURE10_EVENT_ORDER:
        prefix = f"figure10__{event}"
        coefficients = tuple(float(v) for v in bundle.array(f"{prefix}__coefficients"))
        nudged = bundle.array(f"{prefix}__nudged_nino3_c")
        events.append(
            SimpleNamespace(
                event_label=event,
                months_relative_to_target=bundle.array(
                    f"{prefix}__months_relative_to_target"
                ),
                nudge_months=bundle.array(f"{prefix}__nudge_months"),
                baseline_nino3_c=bundle.array(f"{prefix}__baseline_nino3_c"),
                coefficients=coefficients,
                trajectories={
                    coefficient: nudged[position]
                    for position, coefficient in enumerate(coefficients)
                },
            )
        )
    return tuple(events)


def build_figure10(bundle: SteeringPlotBundle) -> Figure:
    """Build Figure 10 exactly from compact bundled trajectories."""

    warm, cold = _figure10_events(bundle)
    if not np.array_equal(
        warm.months_relative_to_target, cold.months_relative_to_target
    ) or not np.array_equal(warm.nudge_months, cold.nudge_months):
        raise ValueError("Figure 10 event schedules differ")
    values = [warm.baseline_nino3_c, cold.baseline_nino3_c]
    values.extend(warm.trajectories.values())
    values.extend(cold.trajectories.values())
    low = min(float(np.min(value)) for value in values)
    high = max(float(np.max(value)) for value in values)
    margin = max(0.25, 0.055 * (high - low))
    y_limits = (
        math.floor(2.0 * (low - margin)) / 2.0,
        math.ceil(2.0 * (high + margin)) / 2.0,
    )
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            1, 2, figsize=(10.0, 4.55), sharex=False, sharey=True
        )
        figure.subplots_adjust(
            left=0.078, right=0.992, bottom=0.14, top=0.97, wspace=0.065
        )
        for axis, event, color, light in (
            (axes[0], warm, IBM_RED, LIGHT_RED),
            (axes[1], cold, IBM_BLUE, LIGHT_BLUE),
        ):
            axis.axhline(0.0, color="0.78", linewidth=0.8, zorder=0)
            patch = axis.axvspan(
                *FIGURE10_NUDGE_SPAN_MONTHS,
                facecolor=light,
                edgecolor=color,
                alpha=0.36,
                linewidth=0.7,
                zorder=1,
            )
            patch.set_gid(f"{event.event_label}-nudging-period")
            original = axis.plot(
                event.months_relative_to_target,
                event.baseline_nino3_c,
                color=color,
                linewidth=3.1,
                zorder=10,
            )[0]
            original.set_gid(f"{event.event_label}-original")
            for coefficient in sorted(
                event.coefficients, key=lambda value: abs(value), reverse=True
            ):
                line = axis.plot(
                    event.months_relative_to_target,
                    event.trajectories[coefficient],
                    color=_tinted_color(color, coefficient),
                    linestyle=(0, (4.0, 2.2)),
                    linewidth=1.55,
                    zorder=3.0 - abs(coefficient),
                )[0]
                line.set_gid(f"{event.event_label}-coefficient-{coefficient:+.1f}")
            axis.set_xlim(-14.0, 0.0)
            axis.set_ylim(*y_limits)
            ticks, labels = _figure10_ticks(event.event_label)
            axis.set_xticks(ticks, labels)
            axis.set_xticks(np.arange(-14, 1, 1), minor=True)
            axis.grid(axis="y", color="0.90", linewidth=0.65, zorder=0)
            axis.tick_params(direction="out")
        axes[0].set_ylabel("Niño-3 index (°C)")
    return figure


def _month_ticks(
    start_month: int,
    duration_months: int,
    *,
    nudge_spans: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, list[str]]:
    boundaries = {
        int(boundary)
        for span in nudge_spans
        for boundary in span
        if float(boundary).is_integer()
    }
    anchor = min(boundaries)
    ticks = set(boundaries)
    ticks.update(range(anchor, duration_months + 1, 3))
    ordered = sorted(ticks)
    return np.asarray(ordered, dtype=np.float64), [
        MONTH_NAMES[(start_month + offset) % 12] for offset in ordered
    ]


def _format_figure11_axis(
    axis: plt.Axes,
    *,
    duration_months: int,
    start_month: int,
    nudge_spans: tuple[tuple[float, float], ...],
) -> None:
    axis.axhline(0.0, color="0.72", linewidth=0.85, zorder=0)
    ticks, labels = _month_ticks(start_month, duration_months, nudge_spans=nudge_spans)
    axis.set_xlim(0.0, float(duration_months))
    axis.set_ylim(*FIGURE11_Y_LIMITS)
    axis.set_xticks(ticks, labels)
    axis.set_xticks(np.arange(0, duration_months + 1, 1), minor=True)
    axis.set_yticks(np.arange(-2, 6, 1))
    axis.grid(axis="y", color="0.91", linewidth=0.6, zorder=0)
    axis.tick_params(direction="out")


def _shade(
    axis: plt.Axes,
    span: tuple[float, float],
    *,
    facecolor: str,
    edgecolor: str,
    gid: str,
) -> None:
    patch = axis.axvspan(
        *span,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=0.36,
        linewidth=0.7,
        zorder=1,
    )
    patch.set_gid(gid)


def build_figure11(bundle: SteeringPlotBundle) -> Figure:
    """Build Figure 11 exactly from compact bundled trajectories."""

    metadata = bundle.manifest["figures"]["figure11"]
    control_months = bundle.array("figure11__control_months")
    event_months = bundle.array("figure11__event_months")
    control = bundle.array("figure11__control_nino3_c")
    warm = bundle.array("figure11__warm_nudged_nino3_c")
    cold = bundle.array("figure11__cold_nudged_nino3_c")
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(10.0, 4.55),
            sharex=False,
            sharey=True,
            gridspec_kw={"width_ratios": (1.18, 1.0, 1.0)},
        )
        figure.subplots_adjust(
            left=0.067, right=0.992, bottom=0.14, top=0.91, wspace=0.075
        )
        _format_figure11_axis(
            axes[0],
            duration_months=17,
            start_month=FIGURE11_COLD_START_MONTH,
            nudge_spans=(
                FIGURE11_COLD_NUDGE_SPAN,
                FIGURE11_WARM_NUDGE_SPAN_CONTROL,
            ),
        )
        _format_figure11_axis(
            axes[1],
            duration_months=14,
            start_month=FIGURE11_WARM_START_MONTH,
            nudge_spans=(FIGURE11_WARM_NUDGE_SPAN_EVENT,),
        )
        _format_figure11_axis(
            axes[2],
            duration_months=14,
            start_month=FIGURE11_COLD_START_MONTH,
            nudge_spans=(FIGURE11_COLD_NUDGE_SPAN,),
        )
        _shade(
            axes[0],
            FIGURE11_COLD_NUDGE_SPAN,
            facecolor=LIGHT_BLUE,
            edgecolor=IBM_BLUE,
            gid="cold-nudging-period",
        )
        _shade(
            axes[0],
            FIGURE11_WARM_NUDGE_SPAN_CONTROL,
            facecolor=LIGHT_RED,
            edgecolor=IBM_RED,
            gid="warm-nudging-period",
        )
        _shade(
            axes[1],
            FIGURE11_WARM_NUDGE_SPAN_EVENT,
            facecolor=LIGHT_RED,
            edgecolor=IBM_RED,
            gid="warm-nudging-period",
        )
        _shade(
            axes[2],
            FIGURE11_COLD_NUDGE_SPAN,
            facecolor=LIGHT_BLUE,
            edgecolor=IBM_BLUE,
            gid="cold-nudging-period",
        )
        for member, values in zip(metadata["control_labels"], control, strict=True):
            line = axes[0].plot(
                control_months,
                values,
                color="#161616",
                alpha=0.60,
                linewidth=1.25,
                zorder=3,
            )[0]
            line.set_gid(f"continuous-control-{member}")
        for member, values in zip(metadata["warm_labels"], warm, strict=True):
            line = axes[1].plot(
                event_months,
                values,
                color=IBM_RED,
                alpha=0.76,
                linestyle=FIGURE11_NUDGED_LINE_STYLE,
                linewidth=1.4,
                zorder=3,
            )[0]
            line.set_gid(f"warm-agop-xai-{member}")
        for member, values in zip(metadata["cold_labels"], cold, strict=True):
            line = axes[2].plot(
                event_months,
                values,
                color=IBM_BLUE,
                alpha=0.76,
                linestyle=FIGURE11_NUDGED_LINE_STYLE,
                linewidth=1.4,
                zorder=3,
            )[0]
            line.set_gid(f"cold-agop-xai-{member}")
        axes[0].set_title("Unnudged sample", pad=9.0, fontweight="normal")
        axes[1].set_title("El Niño nudged", pad=9.0, fontweight="normal")
        axes[2].set_title("La Niña nudged", pad=9.0, fontweight="normal")
        axes[0].set_ylabel(r"Niño-3 index ($^\circ$C)")
    return figure


def _figure12_ticks(event: str) -> tuple[np.ndarray, list[str]]:
    offsets = np.asarray(FIGURE10_CALENDAR_TICK_OFFSETS, dtype=np.float64)
    ticks = float(FIGURE12_MONTHS_RELATIVE_TO_TARGET[0]) + offsets
    start_month = FIGURE10_EVENT_START_MONTH[event]
    labels = [MONTH_NAMES[(start_month + int(offset)) % 12] for offset in offsets]
    return ticks, labels


def build_figure12(bundle: SteeringPlotBundle) -> tuple[Figure, tuple[float, float]]:
    """Build Figure 12 exactly from compact bundled trajectories."""

    events = []
    for event in FIGURE12_EVENT_ORDER:
        prefix = f"figure12__{event}"
        events.append(
            SimpleNamespace(
                event=event,
                unnudged=bundle.array(f"{prefix}__unnudged_nino3_c"),
                plotted_means={
                    method: bundle.array(
                        f"{prefix}__{FIGURE12_METHOD_KEYS[method]}"
                        "__plotted_mean_nino3_c"
                    )
                    for method in FIGURE12_METHOD_ORDER
                },
            )
        )
    curves = [event.unnudged.mean(axis=0) for event in events]
    curves.extend(
        event.plotted_means[method]
        for event in events
        for method in FIGURE12_METHOD_ORDER
    )
    low = min(float(curve.min()) for curve in curves)
    high = max(float(curve.max()) for curve in curves)
    y_limits = (
        math.floor(2.0 * (low - 0.25)) / 2.0,
        math.ceil(2.0 * (high + 0.25)) / 2.0,
    )
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(
            1, 2, figsize=(7.25, 3.55), sharex=False, sharey=True
        )
        figure.subplots_adjust(
            left=0.09, right=0.99, bottom=0.17, top=0.82, wspace=0.14
        )
        for axis, event in zip(axes, events, strict=True):
            axis.axhline(0.0, color="0.68", linewidth=0.9, zorder=0)
            shade, edge = (
                (LIGHT_RED, IBM_RED)
                if event.event == "extreme_el_nino"
                else (LIGHT_BLUE, IBM_BLUE)
            )
            axis.axvspan(
                *FIGURE12_NUDGE_SPAN_MONTHS,
                facecolor=shade,
                edgecolor=edge,
                alpha=0.36,
                linewidth=0.7,
                zorder=1,
            )
            for method in reversed(FIGURE12_METHOD_ORDER):
                style = FIGURE12_METHOD_STYLES[method]
                axis.plot(
                    FIGURE12_MONTHS_RELATIVE_TO_TARGET,
                    event.plotted_means[method],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    zorder=5 if method != "AGOP" else 6,
                )
            axis.plot(
                FIGURE12_MONTHS_RELATIVE_TO_TARGET,
                event.unnudged.mean(axis=0),
                color=BLACK,
                linewidth=2.8,
                zorder=7,
            )
            axis.set_xlim(-14.0, 0.0)
            axis.set_ylim(*y_limits)
            ticks, labels = _figure12_ticks(event.event)
            axis.set_xticks(ticks, labels)
            axis.set_xticks(np.arange(-14, 1, 1), minor=True)
            axis.grid(axis="y", color="0.90", linewidth=0.65, zorder=0)
            axis.tick_params(direction="out")
        axes[0].set_ylabel(r"Niño-3 index ($^\circ$C)")
        handles = [Line2D([], [], color=BLACK, linewidth=2.8, label="Unnudged")]
        handles.extend(
            Line2D(
                [],
                [],
                color=FIGURE12_METHOD_STYLES[method]["color"],
                linestyle=FIGURE12_METHOD_STYLES[method]["linestyle"],
                linewidth=FIGURE12_METHOD_STYLES[method]["linewidth"],
                label=FIGURE12_METHOD_LABELS[method],
            )
            for method in FIGURE12_METHOD_ORDER
        )
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.91),
            ncol=6,
            frameon=False,
            fontsize=8.2,
            handlelength=2.0,
            columnspacing=0.75,
        )
    return figure, y_limits
