"""Contract checks for the target-locked multi-lead Figures 8-9 design."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from generate_fresh_figures8_9_data import (  # noqa: E402
    CASE_ORDER,
    COLD_CASES,
    LEADS,
    WARM_CASES,
    build_target_locked_events,
)
from make_fresh_figures8_9 import (  # noqa: E402
    COLORBAR_BOTTOM,
    FIGURE_HEIGHT_INCHES,
    GRID_BOTTOM,
    GRID_HSPACE,
    GRID_TOP,
    PANEL_B_COLORBAR_BOTTOM,
    ROWS,
    SST_COLOR_LEVELS,
    SST_COLORMAP,
    THERMOCLINE_COLOR_LEVELS,
    THERMOCLINE_COLORMAP,
    THREE_COLUMN_FIGURE_WIDTH_INCHES,
    THREE_COLUMN_GRID_LEFT,
    THREE_COLUMN_GRID_RIGHT,
    THREE_COLUMN_WSPACE,
    _add_colorbar,
    _colorbar_bottom,
    _display_row_labels,
    _lead_title,
    select_composite_peaks,
)


@dataclass(frozen=True)
class _FixedSplit:
    test_inputs: np.ndarray
    lead_steps: int


class _SyntheticData:
    steps_per_month = 3

    def __init__(self) -> None:
        self.target = np.zeros(400, dtype=np.float32)
        self.target[200] = 4.5
        self.target[300] = -2.25

    def fixed_supervised_split(self, lead_months: int) -> _FixedSplit:
        lead_steps = lead_months * self.steps_per_month
        return _FixedSplit(
            test_inputs=np.arange(100, 370, dtype=np.int64),
            lead_steps=lead_steps,
        )

    def load_targets(self, input_indices: np.ndarray, *, lead_steps: int) -> np.ndarray:
        return self.target[np.asarray(input_indices) + lead_steps]


class FreshFigures89TargetLockTests(unittest.TestCase):
    def test_colorbar_uses_field_name_below_raised_bar(self) -> None:
        import matplotlib.pyplot as plt

        figure = plt.figure()
        _add_colorbar(
            figure,
            (0.16, COLORBAR_BOTTOM, 0.21, 0.017),
            colormap=THERMOCLINE_COLORMAP,
            color_levels=THERMOCLINE_COLOR_LEVELS,
            field_label="Thermocline-depth anomaly",
        )
        colorbar_axis = figure.axes[0]
        self.assertGreater(COLORBAR_BOTTOM, 0.048)
        self.assertEqual(colorbar_axis.get_title(), "")
        self.assertEqual(colorbar_axis.get_xlabel(), "Thermocline-depth anomaly")
        self.assertEqual(colorbar_axis.xaxis.get_label_position(), "bottom")
        plt.close(figure)

    def test_panel_b_colorbars_are_visibly_higher_than_panel_a(self) -> None:
        self.assertEqual(_colorbar_bottom("a"), COLORBAR_BOTTOM)
        self.assertEqual(_colorbar_bottom("b"), PANEL_B_COLORBAR_BOTTOM)
        self.assertGreater(PANEL_B_COLORBAR_BOTTOM - COLORBAR_BOTTOM, 0.01)

    def test_realized_horizontal_geometry_matches_fresh_figure6(self) -> None:
        import matplotlib.pyplot as plt

        figure = plt.figure(
            figsize=(THREE_COLUMN_FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        grid = figure.add_gridspec(
            3,
            3,
            left=THREE_COLUMN_GRID_LEFT,
            right=THREE_COLUMN_GRID_RIGHT,
            bottom=GRID_BOTTOM,
            top=GRID_TOP,
            wspace=THREE_COLUMN_WSPACE,
            hspace=GRID_HSPACE,
        )
        axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
        for axis in axes:
            axis.set_box_aspect(2.0 / 3.0)
        figure.canvas.draw()
        bounds = [axis.get_position().bounds for axis in axes]
        expected_width = (THREE_COLUMN_GRID_RIGHT - THREE_COLUMN_GRID_LEFT) / (
            3.0 + 2.0 * THREE_COLUMN_WSPACE
        )
        expected_gap = THREE_COLUMN_WSPACE * expected_width
        for bound in bounds:
            self.assertAlmostEqual(bound[2], expected_width, places=12)
        for left, right in zip(bounds[:-1], bounds[1:], strict=True):
            realized_gap = right[0] - (left[0] + left[2])
            self.assertAlmostEqual(realized_gap, expected_gap, places=12)
        self.assertAlmostEqual(
            expected_width * THREE_COLUMN_FIGURE_WIDTH_INCHES,
            2.8548387096774195,
            places=12,
        )
        self.assertAlmostEqual(
            expected_gap * THREE_COLUMN_FIGURE_WIDTH_INCHES,
            0.14274193548387098,
            places=12,
        )
        plt.close(figure)

    def test_display_contract_uses_event_composite_agop_order(self) -> None:
        records = {
            "el_nino_10m": {"target_nino3_c": 4.44665, "lead_months": 10},
            "la_nina_10m": {"target_nino3_c": -2.144, "lead_months": 10},
        }
        self.assertEqual(ROWS, ("Input", "Composite", "AGOP"))
        self.assertEqual(
            _display_row_labels(("el_nino_10m",), records),
            ("El Niño (4.45 °C)", "10% Composite", "AGOP XAI"),
        )
        self.assertEqual(
            _display_row_labels(("la_nina_10m",), records),
            ("La Niña (−2.14 °C)", "10% Composite", "AGOP XAI"),
        )
        self.assertEqual(_lead_title(records["el_nino_10m"]), "10-month lead")

    def test_sst_palette_has_even_bins_and_no_exact_white(self) -> None:
        self.assertEqual(SST_COLORMAP.N, 18)
        self.assertEqual(SST_COLOR_LEVELS.size, 19)
        np.testing.assert_array_equal(SST_COLOR_LEVELS, THERMOCLINE_COLOR_LEVELS)
        self.assertEqual(SST_COLOR_LEVELS[9], 0.0)
        colors = np.asarray(SST_COLORMAP(np.arange(SST_COLORMAP.N)))[:, :3]
        self.assertFalse(np.any(np.all(colors == 1.0, axis=1)))

    def test_case_order_is_warm_then_cold_and_long_to_short_lead(self) -> None:
        self.assertEqual(LEADS, (10, 5, 1))
        self.assertEqual(
            WARM_CASES,
            ("el_nino_10m", "el_nino_5m", "el_nino_1m"),
        )
        self.assertEqual(
            COLD_CASES,
            ("la_nina_10m", "la_nina_5m", "la_nina_1m"),
        )
        self.assertEqual(CASE_ORDER, (*WARM_CASES, *COLD_CASES))

    def test_each_triptych_uses_one_lead_ten_extreme_target(self) -> None:
        data = _SyntheticData()
        events = build_target_locked_events(data)  # type: ignore[arg-type]
        expected = {
            "el_nino": (200, 4.5, (170, 185, 197)),
            "la_nina": (300, -2.25, (270, 285, 297)),
        }
        for phase, cases in (("el_nino", WARM_CASES), ("la_nina", COLD_CASES)):
            target_step, target_value, input_steps = expected[phase]
            for case_id, lead, input_step in zip(
                cases, LEADS, input_steps, strict=True
            ):
                event = events[case_id]
                self.assertEqual(event.lead_months, lead)
                self.assertEqual(event.input_step, input_step)
                self.assertEqual(event.target_step, target_step)
                self.assertEqual(event.target_nino3_c, target_value)
                self.assertEqual(
                    event.input_step + lead * data.steps_per_month,
                    event.target_step,
                )

    def test_cold_composite_selection_is_exact_warm_sign_reversal(self) -> None:
        steps = np.arange(100, 113, dtype=np.int64)
        values = np.asarray(
            [0.0, 1.2, 2.0, 0.0, -1.2, -2.0, 0.0, 1.1, 1.5, 0.0, -1.1, -1.7, 0.0],
            dtype=np.float64,
        )
        full = np.zeros(200, dtype=np.float64)
        full[steps] = values
        warm_peaks, warm_top = select_composite_peaks(
            values,
            steps,
            full,
            kind="warm",
            top_percent=50.0,
        )
        cold_peaks, cold_top = select_composite_peaks(
            values,
            steps,
            full,
            kind="cold",
            top_percent=50.0,
        )
        np.testing.assert_array_equal(warm_peaks, np.asarray([102, 108]))
        np.testing.assert_array_equal(cold_peaks, np.asarray([105, 111]))
        np.testing.assert_array_equal(warm_top, np.asarray([102]))
        np.testing.assert_array_equal(cold_top, np.asarray([105]))


if __name__ == "__main__":
    unittest.main()
