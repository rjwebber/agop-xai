"""Focused tests for Figure 5 repetition summaries and uncertainty rendering."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from matplotlib.axes import Axes

from scripts import make_figure5


def _write_plot_rows(
    path: Path,
    right_repetitions: int,
    *,
    left_repetitions: int = 1,
) -> None:
    rows: list[dict[str, object]] = []
    for architecture_index, architecture in enumerate(("mlp", "cnn", "vit")):
        for repetition in range(left_repetitions):
            rows.append(
                {
                    "panel": "training_size",
                    "architecture": architecture,
                    "lead_months": 10,
                    "train_years_requested": 10000.0,
                    "repetition": repetition,
                    "seed": 42 + repetition,
                    "test_r2": (
                        0.90 + architecture_index * 0.01 + repetition * 0.001
                    ),
                }
            )
        for years in (10000.0, 50.0):
            for lead in (1, 2):
                for repetition in range(right_repetitions):
                    rows.append(
                        {
                            "panel": "lead_time",
                            "architecture": architecture,
                            "lead_months": lead,
                            "train_years_requested": years,
                            "repetition": repetition,
                            "seed": 42 + repetition,
                            "test_r2": (
                                0.95
                                - lead * 0.02
                                - (0.20 if years == 50.0 else 0.0)
                                + repetition * 0.001
                            ),
                        }
                    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MakeFigure5Tests(unittest.TestCase):
    def _render_and_count_ribbons(
        self,
        right_repetitions: int,
        *,
        left_repetitions: int = 1,
    ) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results.csv"
            output = root / "figure5.pdf"
            _write_plot_rows(
                results,
                right_repetitions,
                left_repetitions=left_repetitions,
            )
            fill_between = Axes.fill_between
            ribbon_calls = 0

            def count_ribbon(axis: Axes, *args: object, **kwargs: object) -> object:
                nonlocal ribbon_calls
                ribbon_calls += 1
                return fill_between(axis, *args, **kwargs)

            arguments = [
                "make_figure5.py",
                "--results",
                str(results),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(Axes, "fill_between", new=count_ribbon),
            ):
                self.assertEqual(make_figure5.main(), 0)
            self.assertTrue(output.is_file())
            metadata = json.loads(output.with_suffix(".json").read_text())
            return ribbon_calls, metadata

    def test_singleton_right_panel_remains_without_uncertainty_ribbons(self) -> None:
        ribbon_calls, metadata = self._render_and_count_ribbons(1)
        self.assertEqual(ribbon_calls, 0)
        self.assertEqual(metadata["left_repetition_counts"], [1])
        self.assertEqual(metadata["right_repetition_counts"], [1])

    def test_five_seed_right_panel_has_no_uncertainty_ribbons(self) -> None:
        ribbon_calls, metadata = self._render_and_count_ribbons(5)
        self.assertEqual(ribbon_calls, 0)
        self.assertEqual(metadata["right_repetition_counts"], [5])
        self.assertEqual(metadata["right_uncertainty"], "not displayed")

    def test_five_seed_both_panels_have_no_uncertainty_or_regime_labels(self) -> None:
        ribbon_calls, metadata = self._render_and_count_ribbons(
            5,
            left_repetitions=5,
        )
        self.assertEqual(ribbon_calls, 0)
        self.assertEqual(metadata["left_repetition_counts"], [5])
        self.assertEqual(metadata["right_repetition_counts"], [5])
        self.assertEqual(metadata["left_uncertainty"], "not displayed")
        self.assertEqual(metadata["y_axis_label"], "$R^2$")
        self.assertEqual(metadata["panel_titles"], [])
        self.assertFalse(metadata["legend"]["numeric_training_years_shown"])
        self.assertEqual(metadata["legend"]["data_regime_labels"], [])


if __name__ == "__main__":
    unittest.main()
