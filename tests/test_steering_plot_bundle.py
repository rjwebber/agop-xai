"""Tests for the compact public steering bundle and plotting-only renderer."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from zc_xai.steering_figures import (
    FIGURE10_EVENT_ORDER,
    FIGURE12_EVENT_ORDER,
    FIGURE12_METHOD_ORDER,
    build_figure10,
    build_figure11,
    build_figure12,
)
from zc_xai.steering_plot_bundle import load_steering_plot_bundle

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "artifacts/zc-v3/manuscript/steering/final_plot_data"


class SteeringPlotBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (BUNDLE_DIR / "manifest.json").is_file():
            raise unittest.SkipTest("companion Zenodo steering bundle is not installed")
        cls.bundle = load_steering_plot_bundle(BUNDLE_DIR)

    def test_public_bundle_is_small_and_portable(self) -> None:
        total = sum(path.stat().st_size for path in BUNDLE_DIR.iterdir())
        self.assertLess(total, 20 * 1024 * 1024)
        manifest = self.bundle.manifest_path.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", manifest)
        self.assertNotIn("scratch/", manifest)
        self.assertEqual(self.bundle.manifest["accepted_reports"]["member_count"], 119)

    def test_final_member_counts_are_preserved(self) -> None:
        figure11 = self.bundle.manifest["figures"]["figure11"]
        self.assertEqual(figure11["selected_episode_count"], 10)
        self.assertEqual(figure11["warm_accepted_count"], 10)
        self.assertEqual(figure11["cold_accepted_count"], 9)
        expected = {
            "extreme_el_nino": {
                "AGOP": 10,
                "GRAD": 10,
                "IG": 10,
                "GradientSHAP": 10,
                "Composite": 10,
            },
            "extreme_la_nina": {
                "AGOP": 9,
                "GRAD": 8,
                "IG": 6,
                "GradientSHAP": 10,
                "Composite": 10,
            },
        }
        records = self.bundle.manifest["figures"]["figure12"]["events"]
        for event, methods in expected.items():
            self.assertEqual(
                {
                    method: records[event]["methods"][method]["completed_member_count"]
                    for method in FIGURE12_METHOD_ORDER
                },
                methods,
            )

    def test_plotting_only_figures_use_all_bundled_curves(self) -> None:
        figure10 = build_figure10(self.bundle)
        self.assertEqual(len(figure10.axes), 2)
        for axis, event in zip(figure10.axes, FIGURE10_EVENT_ORDER, strict=True):
            prefix = f"figure10__{event}"
            self.assertTrue(
                np.array_equal(
                    axis.lines[1].get_ydata(),
                    self.bundle.array(f"{prefix}__baseline_nino3_c"),
                )
            )
            self.assertEqual(len(axis.lines), 15)
        plt.close(figure10)

        figure11 = build_figure11(self.bundle)
        self.assertEqual([len(axis.lines) for axis in figure11.axes], [11, 11, 10])
        plt.close(figure11)

        figure12, limits = build_figure12(self.bundle)
        self.assertEqual(len(figure12.axes), 2)
        self.assertEqual([len(axis.lines) for axis in figure12.axes], [7, 7])
        self.assertEqual(limits, (-2.0, 5.0))
        for axis, event in zip(figure12.axes, FIGURE12_EVENT_ORDER, strict=True):
            prefix = f"figure12__{event}"
            self.assertTrue(
                np.array_equal(
                    axis.lines[-1].get_ydata(),
                    self.bundle.array(f"{prefix}__unnudged_nino3_c").mean(axis=0),
                )
            )
        plt.close(figure12)

if __name__ == "__main__":
    unittest.main()
