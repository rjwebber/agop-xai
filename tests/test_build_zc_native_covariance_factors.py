"""Publication-gate tests for native ZC covariance-factor construction."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from scripts.build_zc_native_covariance_factors import _validate_annual_capture
from zc_xai.io import sha256_array


def _annual_capture(phases: np.ndarray) -> dict[str, object]:
    bases = np.arange(0, 360_000, 36, dtype=np.int64)
    phase_indices = bases[None, :] + phases[:, None]
    return {
        "schema_version": 1,
        "status": "complete",
        "sample_count": 10_000,
        "state_size": 28_591,
        "dtype": "<f4",
        "training_interval": [0, 360_000],
        "base_input_indices_sha256": sha256_array(bases),
        "phase_input_indices_sha256": sha256_array(phase_indices),
        "schedule": {
            "source_checkpoint_count": 1_000,
            "source_checkpoint_stride_steps": 360,
            "annual_base_stride_steps": 36,
            "years_per_source_checkpoint": 10,
            "maximum_replay_offset_steps": 9 * 36 + int(phases[-1]),
        },
        "overlap_audit": {
            "status": "passed_full_bitwise_overlap",
            "rows_per_phase": 1_000,
            "total_values_compared": int(phases.size * 1_000 * 28_591),
        },
        "history_sha256": "a" * 64,
        "production_report_sha256": "b" * 64,
        "phase_replay_executable_sha256": "c" * 64,
        "state_manifest_sha256": "d" * 64,
        "data_metadata_sha256": "e" * 64,
    }


def _with_independent_overlap(
    capture: dict[str, object], phases: np.ndarray
) -> dict[str, object]:
    result = copy.deepcopy(capture)
    block_hashes = {str(index): f"{index:064x}" for index in range(1_000)}
    result["overlap_audit"] = {
        "schema_version": 1,
        "status": "passed_full_bitwise_independent_overlap_replay",
        "validation_mode": "independent_overlap_replay",
        "reference_cache_used": False,
        "phase_offsets": phases.tolist(),
        "rows_per_phase": 1_000,
        "total_rows_compared": int(phases.size * 1_000),
        "total_values_compared": int(phases.size * 1_000 * 28_591),
        "source_checkpoint_count": 1_000,
        "replay_relative_offsets_sha256": sha256_array(phases[None, :]),
        "replayed_block_sha256": block_hashes,
        "ordered_block_hashes_sha256": "f" * 64,
        "provenance": {
            "history_sha256": result["history_sha256"],
            "phase_replay_executable_sha256": result[
                "phase_replay_executable_sha256"
            ],
        },
        "storage": {
            "persistent_reference_cache_created": False,
            "persistent_reference_cache_bytes": 0,
        },
    }
    return result


class AnnualCovariancePublicationGateTests(unittest.TestCase):
    def test_all_36_annual_phases_are_accepted(self) -> None:
        phases = np.arange(36, dtype=np.int64)
        capture = _with_independent_overlap(_annual_capture(phases), phases)
        _validate_annual_capture(capture, phases, expected_samples=10_000)

    def test_warm_and_cold_nine_phase_captures_are_accepted(self) -> None:
        for phases in (
            np.arange(26, 35, dtype=np.int64),
            np.arange(17, 26, dtype=np.int64),
        ):
            with self.subTest(phases=phases.tolist()):
                _validate_annual_capture(
                    _annual_capture(phases), phases, expected_samples=10_000
                )

    def test_independent_overlap_replay_is_accepted_without_legacy_cache(self) -> None:
        phases = np.arange(17, 26, dtype=np.int64)
        capture = _with_independent_overlap(_annual_capture(phases), phases)
        _validate_annual_capture(capture, phases, expected_samples=10_000)

        overlap = capture["overlap_audit"]
        assert isinstance(overlap, dict)
        overlap["phase_offsets"] = np.arange(26, 35).tolist()
        with self.assertRaisesRegex(ValueError, "exhaustive decadal overlap"):
            _validate_annual_capture(capture, phases, expected_samples=10_000)

    def test_phase_geometry_must_be_nine_or_the_complete_annual_cycle(self) -> None:
        invalid = (
            np.arange(17, 25, dtype=np.int64),
            np.asarray((17, 18, 19, 20, 21, 22, 23, 24, 36)),
            np.asarray((-1, 0, 1, 2, 3, 4, 5, 6, 7)),
            np.asarray((17, 18, 19, 20, 21, 22, 23, 25, 24)),
            np.arange(1, 37, dtype=np.int64),
            np.delete(np.arange(37, dtype=np.int64), 18),
        )
        for phases in invalid:
            with self.subTest(phases=phases.tolist()), self.assertRaisesRegex(
                ValueError, "either exactly nine increasing phase offsets"
            ):
                _validate_annual_capture({}, phases, expected_samples=10_000)

    def test_replay_horizon_is_derived_from_last_requested_phase(self) -> None:
        phases = np.arange(17, 26, dtype=np.int64)
        capture = _annual_capture(phases)
        capture["schedule"]["maximum_replay_offset_steps"] = 358  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "maximum_replay_offset_steps"):
            _validate_annual_capture(capture, phases, expected_samples=10_000)

    def test_phase_index_hash_remains_phase_specific(self) -> None:
        cold = np.arange(17, 26, dtype=np.int64)
        warm = np.arange(26, 35, dtype=np.int64)
        capture = _annual_capture(cold)
        capture["phase_input_indices_sha256"] = _annual_capture(warm)[
            "phase_input_indices_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "phase_input_indices_sha256"):
            _validate_annual_capture(capture, cold, expected_samples=10_000)

    def test_source_and_overlap_publication_gates_are_unchanged(self) -> None:
        phases = np.arange(17, 26, dtype=np.int64)
        valid = _annual_capture(phases)
        corruptions = (
            ("history_sha256", "short", "valid history_sha256"),
            ("production_report_sha256", None, "valid production_report_sha256"),
            ("phase_replay_executable_sha256", "", "valid phase_replay"),
            ("state_manifest_sha256", "x" * 63, "valid state_manifest"),
            ("data_metadata_sha256", 64, "valid data_metadata"),
        )
        for key, value, message in corruptions:
            capture = copy.deepcopy(valid)
            capture[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                _validate_annual_capture(capture, phases, expected_samples=10_000)

        for key, value in (
            ("status", "not_checked"),
            ("rows_per_phase", 999),
            ("total_values_compared", 1),
        ):
            capture = copy.deepcopy(valid)
            capture["overlap_audit"][key] = value  # type: ignore[index]
            with self.subTest(overlap_key=key), self.assertRaisesRegex(
                ValueError, "exhaustive decadal overlap"
            ):
                _validate_annual_capture(capture, phases, expected_samples=10_000)


if __name__ == "__main__":
    unittest.main()
