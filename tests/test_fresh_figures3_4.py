"""Fresh-data contracts for the domain map and aligned CNN forecast plot."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from tests.test_fresh_zc_pipeline import FreshZCDataTests
from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_script(module_name: str, filename: str):
    specification = importlib.util.spec_from_file_location(
        module_name,
        REPOSITORY_ROOT / "scripts" / filename,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not import {filename} for testing.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


FIGURE3 = load_script("fresh_make_figure3", "make_figure3.py")
FIGURE4 = load_script("fresh_make_figure4", "make_figure4.py")


class FreshFigure3And4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = FreshZCDataTests()
        self.fixture.setUp()

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def test_figure3_infers_outer_edges_from_fresh_grid_centers(self) -> None:
        regions = FIGURE3.load_region_metadata(self.fixture.directory)
        self.assertEqual(regions.metadata_schema, FRESH_SCHEMA_VERSION)
        self.assertEqual(regions.data_latitude_edges, (-20.0, 20.0))
        self.assertEqual(regions.data_longitude_edges, (126.5625, 278.4375))
        self.assertEqual(regions.nino3_latitude_bounds, (-5.0, 5.0))
        self.assertEqual(regions.nino3_longitude_bounds, (210.0, 270.0))

    def test_figure3_rejects_fresh_nino3_without_90w_center(self) -> None:
        metadata = self.fixture.metadata()
        metadata["nino3"]["longitude_degrees_east"] = metadata["nino3"][
            "longitude_degrees_east"
        ][:-1]
        self.fixture.write_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "include the 90 W center"):
            FIGURE3.load_region_metadata(self.fixture.directory)

    def test_figure4_uses_common_targets_inside_fixed_test_block(self) -> None:
        data = ZCData(self.fixture.directory, input_profile="core4")
        targets, common_block = FIGURE4.select_common_test_target_steps(
            data,
            lead_months=(1, 2),
            segment_start_year=0.0,
            segment_years=1.0 / 12.0,
        )
        self.assertEqual(common_block, (11, 12))
        np.testing.assert_array_equal(targets, np.asarray([11]))
        for lead in (1, 2):
            split = data.fixed_supervised_split(lead)
            input_step = targets[0] - split.lead_steps
            self.assertIn(input_step, split.test_inputs)

    def test_figure4_uses_figure5_ibm_palette(self) -> None:
        self.assertEqual(
            FIGURE4.COLORS,
            {
                "nino3": "#648fff",
                "forecast_5m": "#fe6100",
                "forecast_10m": "#ffb000",
            },
        )

    def test_figure4_legend_is_above_the_axes(self) -> None:
        self.assertEqual(FIGURE4.LEGEND_LOCATION, "outside upper center")
        self.assertEqual(FIGURE4.LEGEND_COLUMNS, 3)

    def test_figure4_rejects_segment_beyond_common_test_period(self) -> None:
        data = ZCData(self.fixture.directory, input_profile="core4")
        with self.assertRaisesRegex(ValueError, "outside the common held-out"):
            FIGURE4.select_common_test_target_steps(
                data,
                lead_months=(1, 2),
                segment_start_year=0.0,
                segment_years=2.0 / 12.0,
            )


if __name__ == "__main__":
    unittest.main()
