from __future__ import annotations

import unittest

import numpy as np

from zc_xai.composites import (
    apply_population_standardization,
    area_weighted_mean,
    bootstrap_mean_interval,
    fit_linear_residual,
    fit_population_standardization,
    geographic_mask,
    rounded_symmetric_limit,
    strongest_fraction,
    threshold_event_peaks,
)


class CompositeSelectionTests(unittest.TestCase):
    def test_complete_threshold_events_contribute_one_peak(self) -> None:
        steps = np.arange(100, 112)
        values = np.asarray(
            [2.0, 0.0, 1.2, 1.8, 1.1, 0.2, 1.4, 1.4, 0.0, 1.1, 0.0, 2.0]
        )
        # The first and last exceedances touch the held-out boundaries and are
        # omitted.  The plateau tie is resolved at its first maximum.
        np.testing.assert_array_equal(
            threshold_event_peaks(values, steps, threshold_c=1.0),
            np.asarray([103, 106, 109]),
        )

    def test_top_percentages_are_nested_and_ceil_counts(self) -> None:
        target = np.arange(200, dtype=np.float64)
        peaks = np.arange(10, 29, dtype=np.int64)
        top10 = strongest_fraction(peaks, target, percent=10.0)
        top20 = strongest_fraction(peaks, target, percent=20.0)
        self.assertEqual(top10.size, 2)
        self.assertEqual(top20.size, 4)
        np.testing.assert_array_equal(top10, top20[: top10.size])
        np.testing.assert_array_equal(top20, np.asarray([28, 27, 26, 25]))

    def test_bootstrap_interval_is_reproducible(self) -> None:
        values = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        first = bootstrap_mean_interval(values, resamples=100, seed=7)
        second = bootstrap_mean_interval(values, resamples=100, seed=7)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_symmetric_limit_rounds_up(self) -> None:
        self.assertEqual(rounded_symmetric_limit(np.asarray([-1.01, 0.2])), 1.5)

    def test_geographic_region_uses_inclusive_grid_centers(self) -> None:
        latitudes = np.asarray([-5.0, 0.0, 5.0])
        longitudes = np.asarray([180.0, 210.0, 240.0])
        mask = geographic_mask(
            latitudes,
            longitudes,
            latitude_bounds=(-5.0, 0.0),
            longitude_bounds=(210.0, 240.0),
        )
        expected = np.asarray(
            [
                [False, True, True],
                [False, True, True],
                [False, False, False],
            ]
        )
        np.testing.assert_array_equal(mask, expected)

    def test_area_weighted_mean_preserves_leading_axes(self) -> None:
        latitudes = np.asarray([-60.0, 0.0])
        mask = np.ones((2, 2), dtype=bool)
        values = np.asarray(
            [
                [[2.0, 2.0], [4.0, 4.0]],
                [[4.0, 4.0], [8.0, 8.0]],
            ]
        )
        np.testing.assert_allclose(
            area_weighted_mean(values, mask, latitudes),
            np.asarray([10.0 / 3.0, 20.0 / 3.0]),
        )

    def test_scalar_population_standardization_uses_training_fit(self) -> None:
        mean, scale = fit_population_standardization(np.asarray([0.0, 2.0]))
        self.assertEqual(mean, 1.0)
        self.assertEqual(scale, 1.0)
        np.testing.assert_array_equal(
            apply_population_standardization(
                np.asarray([-1.0, 3.0]), mean=mean, scale=scale
            ),
            np.asarray([-2.0, 2.0]),
        )

    def test_linear_residual_removes_training_predictor(self) -> None:
        predictor = np.arange(6, dtype=np.float64)
        response = 2.0 * predictor + np.asarray([1.0, -1.0] * 3)
        fit = fit_linear_residual(response, predictor)
        residual = fit.transform(response, predictor)
        self.assertAlmostEqual(float(residual.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(residual.std(ddof=0)), 1.0, places=12)
        self.assertAlmostEqual(
            float(np.dot(residual, predictor - predictor.mean())),
            0.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
