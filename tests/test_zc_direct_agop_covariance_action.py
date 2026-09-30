"""Focused unit tests for the direct AGOP covariance-action driver."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import run_zc_direct_agop_covariance_action as direct  # noqa: E402

from adjoint.balanced_control_map import (  # noqa: E402
    ControlSegment,
    IndependentControlLayout,
)
from zc_xai.native_observation import PackedZCState  # noqa: E402
from zc_xai.nonlinear_covariance_control import NativeScalarSQPResult  # noqa: E402
from zc_xai.zc_controlled_bridge import (  # noqa: E402
    ControlledReplay,
    PrimalStepResult,
    ZeroControlPath,
)


def _layout() -> IndependentControlLayout:
    return IndependentControlLayout(
        packed_size=4,
        segments=(
            ControlSegment(
                name="kept",
                packed_start=1,
                packed_stop=3,
                compact_start=0,
                compact_stop=2,
                shape=(2,),
                order="F",
                role="test",
                restart_record=1,
            ),
        ),
        manifest_schema_version=2,
    )


def _state(real32: np.ndarray, boundary: int = 0) -> PackedZCState:
    return PackedZCState(
        real32=np.asarray(real32, dtype=np.float32),
        complex64=np.asarray((1.0 + 0.0j,), dtype=np.complex64),
        real64=np.asarray((2.0,), dtype=np.float64),
        integers=np.asarray((boundary,), dtype=np.int32),
        passive_time=np.asarray((boundary / 3.0,), dtype=np.float32),
    )


def _solver_settings() -> dict[str, float | int]:
    return {
        "maximum_iterations": 30,
        "scale_continuation_warm_start": False,
        "zero_dual_restart_after_warm_rejection": False,
        "known_active_boundary": True,
        "constraint_tolerance": 1e-5,
        "stationarity_tolerance": 1e-5,
        "complementarity_tolerance": 1e-5,
        "relative_stationarity_tolerance": 2e-2,
        "relative_complementarity_tolerance": 1e-4,
        "initial_trust_radius": 10.0,
        "maximum_trust_radius": 100.0,
        "covariance_block_rows": 1024,
    }


def _successful_result() -> NativeScalarSQPResult:
    return NativeScalarSQPResult(
        native_interventions=np.ones((1, 2), dtype=np.float64),
        dual_variables=np.ones((1, 2), dtype=np.float64),
        success=True,
        status="first_order_kkt_satisfied",
        objective_value=1.0,
        squared_action=2.0,
        constraint_value=0.0,
        primal_violation=0.0,
        native_constraint_gradient=np.ones((1, 2), dtype=np.float64),
        lagrange_multiplier=1.0,
        covariance_stationarity_norm=0.01,
        covariance_stationarity_relative=0.01,
        complementarity_absolute=0.0,
        complementarity_relative=0.0,
        stationarity_reference_norm=1.0,
        constraint_gradient_covariance_norm=1.0,
        iterations=(),
        value_evaluations=2,
        gradient_evaluations=2,
        covariance_passes=3,
        radial_restoration_applied=False,
        radial_restoration_reason=None,
        radial_restoration_scale=None,
        radial_restoration_value_evaluations=0,
        radial_restoration_gradient_evaluations=0,
        radial_restoration_covariance_passes=0,
    )


def _annual_dense_manifest() -> dict[str, object]:
    return {
        "schema_version": direct.DENSE_COVARIANCE_SCHEMA_VERSION,
        "status": direct.DENSE_COVARIANCE_STATUS,
        "dimensions": {
            "state_size": direct.ANNUAL_COVARIANCE_STATE_SIZE,
            "sample_count_per_phase": (direct.ANNUAL_COVARIANCE_SAMPLE_COUNT_PER_PHASE),
            "phase_count": len(direct.ANNUAL_PHASES),
            "phase_offsets": list(direct.ANNUAL_PHASES),
        },
        "common_covariance_policy": "equal-phase pooled covariance",
        "covariance": {
            "file": "covariance.npy",
            "sha256": "a" * 64,
            "size_bytes": 123,
            "shape": [
                direct.ANNUAL_COVARIANCE_STATE_SIZE,
                direct.ANNUAL_COVARIANCE_STATE_SIZE,
            ],
            "dtype": "<f8",
            "order": "F",
        },
        "source_capture": {
            "training_half_open_interval": list(
                direct.ANNUAL_COVARIANCE_TRAINING_INTERVAL
            ),
            "data_metadata_sha256": "b" * 64,
            "state_manifest_sha256": "c" * 64,
        },
    }


def _valid_case_report(
    case_dir: Path, identity: dict[str, object]
) -> dict[str, object]:
    result = _successful_result()
    artifact_path = case_dir / "result.npz"
    direct.write_npz(
        artifact_path,
        overwrite=False,
        native_interventions=result.native_interventions,
    )
    report: dict[str, object] = {
        "schema_version": direct.REPORT_SCHEMA_VERSION,
        "status": "complete",
        "case_identity": identity,
        "case_identity_sha256": direct.sha256_json(identity),
        "publication_gates": direct._solver_publication_gates(
            result,
            solver_settings=identity["solver_settings"],  # type: ignore[arg-type]
            replay_constraint=0.0,
        ),
        "solver": direct._solver_report(
            result,
            covariance_case=str(identity["covariance_case"]),
            initial_projection_gap=1.0,
            stationarity_reference_norm=1.0,
            constraint_gradient_covariance_norm=1.0,
            reporting_covariance_operator_calls=0,
        ),
        "release": {
            "constraint_value": 0.0,
            "realized_agop_projection": 1.0,
            "cnn_forecast_c": 2.0,
        },
        "outcome_after_control_frozen": {
            "terminal_nino3_c": 3.0,
            "paired_terminal_nino3_change_c": 2.5,
        },
        "canonical_o3_frozen_control_transfer_audit": {
            "used_by_optimizer": False,
            "all_values_finite": True,
            "canonical_o3": {
                "release_agop_projection": 1.0,
                "release_constraint_value": 0.0,
                "terminal_nino3_c": 3.0,
                "integer_and_passive_clocks_match_unforced_o3": True,
            },
            "matched_o0_minus_canonical_o3": {"terminal_nino3_c": 0.0},
        },
        "runtime": {
            "solver_total_wall_seconds": 2.0,
            "post_solver_replay_and_report_preparation_wall_seconds": 1.0,
            "total_wall_seconds": 3.0,
        },
        "artifact": {
            "file": artifact_path.name,
            "sha256": direct.sha256_file(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
        },
    }
    return report


class _IdentityRunner:
    def __init__(self) -> None:
        self.calls = 0

    def advance(self, state: PackedZCState) -> PrimalStepResult:
        self.calls += 1
        output = _state(state.real32, int(state.integers[0]) + 1)
        return PrimalStepResult(output, np.asarray((self.calls,), dtype=np.int32), 0.0)


class _TinyObservationChain:
    def forward(self, state_9: PackedZCState, state_10: PackedZCState) -> np.ndarray:
        del state_9
        return np.asarray((state_10.real32[0], 0.0, 0.0), dtype=np.float32)


class _ContinuationOracle:
    def __init__(self) -> None:
        self.forward_replays = 0
        self.reverse_sweeps = 0
        self.reverse_wall_seconds = 0.0

    def value(self, native: np.ndarray) -> float:
        return float(np.sum(native) - 10.0)

    def value_gradient(self, native: np.ndarray) -> tuple[float, np.ndarray]:
        return self.value(native), np.ones_like(native)


class _InitiallyFeasibleOracle:
    def __init__(self, target: float) -> None:
        self.target = float(target)
        self.forward_replays = 0
        self.reverse_sweeps = 0
        self.reverse_wall_seconds = 0.0

    def value(self, native: np.ndarray) -> float:
        return float(np.sum(native) - self.target)

    def value_gradient(self, native: np.ndarray) -> tuple[float, np.ndarray]:
        return self.value(native), np.ones_like(native)


class _IdentityCovariance:
    state_size = 2

    def covariance_apply(self, native_vector: np.ndarray) -> np.ndarray:
        return np.asarray(native_vector, dtype=np.float64).copy()

    def covariance_apply_matrix(
        self, native_vectors: np.ndarray, *, block_rows: int = 128
    ) -> np.ndarray:
        del block_rows
        return np.asarray(native_vectors, dtype=np.float64).copy()


def _tiny_paths(
    value: float, tape_value: int
) -> tuple[ControlledReplay, ZeroControlPath]:
    states = tuple(_state(np.asarray((value,)), boundary) for boundary in range(11))
    baseline_states = tuple(
        _state(np.asarray((0.0,)), boundary) for boundary in range(11)
    )
    tapes = tuple(np.asarray((tape_value,), dtype=np.int32) for _ in range(10))
    baseline_tapes = tuple(np.asarray((0,), dtype=np.int32) for _ in range(10))
    replay = ControlledReplay(
        states=states,
        step_inputs=states[:-1],
        tapes=tapes,
        applied_real32_increments=tuple(
            np.zeros(1, dtype=np.float64) for _ in range(10)
        ),
        wall_seconds=0.0,
    )
    baseline = ZeroControlPath(
        states=baseline_states,
        tapes=baseline_tapes,
        wall_seconds=0.0,
    )
    return replay, baseline


class DirectDriverTests(unittest.TestCase):
    def test_centered_cache_policy_fails_before_insufficient_allocation(self) -> None:
        covariance = Mock()
        covariance.estimated_centered_cache_bytes = 2 * 1024**3
        covariance.centered_cache_bytes = 0
        with self.assertRaisesRegex(MemoryError, "requires"):
            direct._configure_centered_covariance_cache(
                covariance,
                requested_gib=1.0,
            )
        covariance.with_contiguous_float64_cache.assert_not_called()

    def test_centered_cache_policy_records_exact_realization(self) -> None:
        covariance = Mock()
        covariance.estimated_centered_cache_bytes = 4096
        covariance.centered_cache_bytes = 0
        cached = Mock()
        cached.centered_cache_bytes = 4096
        covariance.with_contiguous_float64_cache.return_value = cached
        result, identity, runtime = direct._configure_centered_covariance_cache(
            covariance,
            requested_gib=1.0,
        )
        self.assertIs(result, cached)
        covariance.with_contiguous_float64_cache.assert_called_once_with(
            maximum_bytes=1024**3
        )
        self.assertTrue(identity["enabled"])
        self.assertEqual(identity["realized_exact_bytes"], 4096)
        self.assertFalse(identity["operator_approximation"])
        self.assertGreaterEqual(runtime["setup_wall_seconds_this_process"], 0.0)

    def test_centered_cache_policy_zero_budget_is_streamed(self) -> None:
        covariance = Mock()
        covariance.estimated_centered_cache_bytes = 4096
        covariance.centered_cache_bytes = 0
        result, identity, runtime = direct._configure_centered_covariance_cache(
            covariance,
            requested_gib=0.0,
        )
        self.assertIs(result, covariance)
        covariance.with_contiguous_float64_cache.assert_not_called()
        self.assertFalse(identity["enabled"])
        self.assertEqual(identity["status"], "disabled_streaming_verified_sources")
        self.assertEqual(runtime["setup_wall_seconds_this_process"], 0.0)

    def test_annual_dense_manifest_requires_full_population_and_provenance(
        self,
    ) -> None:
        manifest = _annual_dense_manifest()
        direct._validate_annual_dense_covariance_manifest(
            manifest,
            data_metadata_sha256="b" * 64,
            state_manifest_sha256="c" * 64,
            state_size=direct.ANNUAL_COVARIANCE_STATE_SIZE,
        )
        mutations = (
            ("dimensions", "phase_offsets", list(range(35))),
            ("dimensions", "sample_count_per_phase", 9_999),
            ("source_capture", "training_half_open_interval", [0, 90_000]),
            ("source_capture", "data_metadata_sha256", "d" * 64),
            ("source_capture", "state_manifest_sha256", "e" * 64),
            ("covariance", "shape", [1, 1]),
            ("covariance", "order", "C"),
        )
        for section, key, value in mutations:
            changed = copy.deepcopy(manifest)
            changed[section][key] = value  # type: ignore[index]
            with (
                self.subTest(section=section, key=key),
                self.assertRaises(ValueError),
            ):
                direct._validate_annual_dense_covariance_manifest(
                    changed,
                    data_metadata_sha256="b" * 64,
                    state_manifest_sha256="c" * 64,
                    state_size=direct.ANNUAL_COVARIANCE_STATE_SIZE,
                )

    def test_dense_covariance_runtime_ignores_legacy_cache_budget(self) -> None:
        identity, runtime = direct._dense_covariance_runtime(
            _annual_dense_manifest(),
            requested_centered_cache_gib=24.0,
        )
        self.assertFalse(identity["enabled"])
        self.assertEqual(
            identity["status"], "not_applicable_precompiled_dense_covariance"
        )
        self.assertFalse(identity["centered_sample_cache_request_applied"])
        self.assertFalse(identity["operator_approximation"])
        self.assertEqual(runtime["setup_wall_seconds_this_process"], 0.0)

    def test_continuation_cli_is_explicit_and_exact_target_ending(self) -> None:
        defaults = direct.parser().parse_args([])
        self.assertEqual(defaults.continuation_fractions, (1.0,))
        self.assertFalse(defaults.scale_continuation_warm_start)
        self.assertEqual(defaults.initial_dual_scale, 1.0)
        self.assertFalse(defaults.adaptive_continuation)
        self.assertFalse(defaults.zero_dual_restart_after_warm_rejection)
        self.assertFalse(defaults.direct_exact_target_first)
        self.assertTrue(
            direct.parser()
            .parse_args(["--direct-exact-target-first"])
            .direct_exact_target_first
        )
        self.assertTrue(defaults.resume_continuation)
        self.assertFalse(
            direct.parser().parse_args(["--no-resume-continuation"]).resume_continuation
        )
        self.assertEqual(
            defaults.adaptive_minimum_fraction_step,
            direct.DEFAULT_ADAPTIVE_MINIMUM_FRACTION_STEP,
        )
        self.assertEqual(defaults.selection_seed, 42)
        self.assertEqual(
            defaults.covariance_policy,
            direct.ANNUAL_SHARED_COVARIANCE_POLICY,
        )
        self.assertEqual(defaults.case, "pooled")
        self.assertEqual(defaults.covariance_centered_cache_gib, 0.0)
        cached = direct.parser().parse_args(["--covariance-centered-cache-gib", "24"])
        self.assertEqual(cached.covariance_centered_cache_gib, 24.0)
        with self.assertRaises(SystemExit):
            direct.parser().parse_args(["--covariance-centered-cache-gib", "-0.1"])
        scaled = direct.parser().parse_args(["--scale-continuation-warm-start"])
        self.assertTrue(scaled.scale_continuation_warm_start)
        self.assertEqual(
            direct.parser()
            .parse_args(["--initial-dual-scale", "0.6666666666666666"])
            .initial_dual_scale,
            2.0 / 3.0,
        )
        with self.assertRaises(SystemExit):
            direct.parser().parse_args(["--initial-dual-scale", "0"])
        self.assertEqual(
            direct.continuation_fractions("0.25,0.5,0.75,1.0"),
            (0.25, 0.5, 0.75, 1.0),
        )
        for invalid in ("", "0", "0.5,0.5,1", "0.5,0.9", "0.5,1.1"):
            with (
                self.subTest(invalid=invalid),
                self.assertRaises(direct.argparse.ArgumentTypeError),
            ):
                direct.continuation_fractions(invalid)

    def test_direct_first_exact_target_success_skips_fallback(self) -> None:
        result = _successful_result()
        settings = _solver_settings()
        settings.update(
            direct_exact_target_first=True,
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            return_value=result,
        ) as solve:
            actual, records = direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
        self.assertIs(actual, result)
        self.assertEqual(solve.call_count, 1)
        self.assertIsNone(solve.call_args.kwargs["initial_dual_variables"])
        self.assertEqual([record["fraction"] for record in records], [1.0])
        self.assertTrue(records[0]["direct_exact_target_trial"])
        self.assertTrue(records[0]["publication_gates"]["all_passed"])
        self.assertFalse(records[0]["direct_fallback_triggered"])

    def test_external_initial_dual_records_raw_scale_and_scaled_hashes(self) -> None:
        raw = np.asarray([[3.0, 6.0]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.npz"
            direct.write_npz(source, overwrite=True, dual_variables=raw)
            scaled, provenance = direct._load_scaled_initial_dual(
                source,
                scale=2.0 / 3.0,
            )
            source_sha256 = direct.sha256_file(source)
        np.testing.assert_array_equal(scaled, np.asarray([[2.0, 4.0]]))
        self.assertEqual(provenance["initial_dual_npz_sha256"], source_sha256)
        self.assertEqual(
            provenance["initial_dual_raw_variables_sha256"],
            direct.sha256_array(raw),
        )
        self.assertEqual(provenance["initial_dual_scale"], 2.0 / 3.0)
        self.assertEqual(
            provenance["initial_dual_scaled_variables_sha256"],
            direct.sha256_array(scaled),
        )

    def test_scaled_external_dual_is_identity_bound_and_restart_validated(
        self,
    ) -> None:
        raw = np.asarray([[3.0, 6.0]], dtype=np.float64)
        scale = 2.0 / 3.0
        scaled = scale * raw
        settings = _solver_settings()
        run = {
            "model_checkpoint_sha256": "a" * 64,
            "initial_dual_npz_sha256": "b" * 64,
            "initial_dual_raw_variables_sha256": direct.sha256_array(raw),
            "initial_dual_scale": scale,
            "initial_dual_scaled_variables_sha256": direct.sha256_array(scaled),
            "initial_dual_variables_sha256": direct.sha256_array(scaled),
            "solver_settings": settings,
        }
        result = _successful_result()
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
            patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=result,
            ) as solve,
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(
                root,
                run_identity=run,
                fractions=(1.0,),
                solver_settings=settings,
            )
            actual, records = direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(1.0,),
                solver_settings=settings,
                initial_dual_variables=scaled,
                initial_dual_scale=scale,
                initial_dual_raw_sha256=direct.sha256_array(raw),
                continuation_journal=journal,
            )
            self.assertIs(actual, result)
            np.testing.assert_array_equal(
                solve.call_args.kwargs["initial_dual_variables"], scaled
            )
            self.assertEqual(records[0]["warm_start_scale"], scale)
            self.assertEqual(
                records[0]["previous_result_dual_sha256"],
                direct.sha256_array(raw),
            )
            self.assertEqual(
                records[0]["initial_dual_sha256"],
                direct.sha256_array(scaled),
            )
            resumed = self._continuation_journal(
                root,
                run_identity=run,
                fractions=(1.0,),
                solver_settings=settings,
            ).load()
            self.assertIsNotNone(resumed)

            changed = dict(run)
            changed["initial_dual_scale"] = 0.5
            with self.assertRaisesRegex(ValueError, "another scientific"):
                self._continuation_journal(
                    root,
                    run_identity=changed,
                    fractions=(1.0,),
                    solver_settings=settings,
                ).load()

    def test_direct_first_finite_rejection_falls_back_and_deduplicates_zero(
        self,
    ) -> None:
        direct_rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        half = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        warm_rejected = replace(direct_rejected, dual_variables=np.full((1, 2), 88.0))
        midpoint = replace(_successful_result(), dual_variables=np.full((1, 2), 3.0))
        final = replace(_successful_result(), dual_variables=np.full((1, 2), 4.0))
        settings = _solver_settings()
        settings.update(
            direct_exact_target_first=True,
            scale_continuation_warm_start=True,
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(direct_rejected, half, warm_rejected, midpoint, final),
        ) as solve:
            actual, records = direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
        self.assertIs(actual, final)
        self.assertEqual(
            [record["fraction"] for record in records],
            [1.0, 0.5, 1.0, 0.75, 1.0],
        )
        self.assertTrue(records[0]["direct_fallback_triggered"])
        self.assertEqual(records[0]["direct_fallback_eligibility"]["eligible"], True)
        self.assertTrue(records[2]["zero_dual_restart_skipped_prior_rejection"])
        self.assertEqual(records[2]["prior_rejected_zero_dual_attempt"]["stage"], 1)
        self.assertEqual(
            records[2]["candidate_zero_dual_solve_identity_sha256"],
            records[0]["zero_dual_solve_identity_sha256"],
        )
        self.assertFalse(
            any(record["zero_dual_restart_triggered"] for record in records)
        )
        self.assertEqual(solve.call_count, 5)
        self.assertIsNone(solve.call_args_list[0].kwargs["initial_dual_variables"])
        self.assertIsNone(solve.call_args_list[1].kwargs["initial_dual_variables"])
        for call in solve.call_args_list[2:]:
            initial = call.kwargs["initial_dual_variables"]
            self.assertIsNotNone(initial)
            self.assertFalse(np.array_equal(initial, direct_rejected.dual_variables))
            self.assertFalse(np.array_equal(initial, warm_rejected.dual_variables))

    def test_direct_first_nonfinite_or_infrastructure_failure_does_not_fallback(
        self,
    ) -> None:
        settings = _solver_settings()
        settings.update(
            direct_exact_target_first=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        nonfinite = replace(_successful_result(), objective_value=float("nan"))
        with (
            patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=nonfinite,
            ) as solve,
            self.assertRaisesRegex(
                direct.ProjectionContinuationFailure, "nonfinite metrics or arrays"
            ),
        ):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
        self.assertEqual(solve.call_count, 1)

        with (
            patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                side_effect=RuntimeError("synthetic infrastructure failure"),
            ) as solve,
            self.assertRaisesRegex(RuntimeError, "infrastructure failure"),
        ):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
        self.assertEqual(solve.call_count, 1)

    def test_fixed_batch_rescue_attempt_budget_is_eight_additional_solves(
        self,
    ) -> None:
        rejected_duals = [
            np.full((1, 2), value, dtype=np.float64)
            for value in (91.0, 92.0, 93.0, 94.0)
        ]

        def rejected(position: int) -> NativeScalarSQPResult:
            return replace(
                _successful_result(),
                success=False,
                status="maximum_iterations_reached",
                constraint_value=-1.0,
                primal_violation=1.0,
                covariance_stationarity_relative=0.5,
                complementarity_relative=0.5,
                dual_variables=rejected_duals[position],
            )

        half = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        bridge_1 = replace(_successful_result(), dual_variables=np.full((1, 2), 3.0))
        bridge_2 = replace(_successful_result(), dual_variables=np.full((1, 2), 4.0))
        bridge_3 = replace(_successful_result(), dual_variables=np.full((1, 2), 5.0))
        settings = _solver_settings()
        settings.update(
            maximum_iterations=55,
            direct_exact_target_first=False,
            scale_continuation_warm_start=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.00625,
            maximum_adaptive_subdivisions=3,
        )
        outcomes = (
            half,
            rejected(0),
            bridge_1,
            rejected(1),
            bridge_2,
            rejected(2),
            bridge_3,
            rejected(3),
        )
        with (
            patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                side_effect=outcomes,
            ) as solve,
            self.assertRaises(direct.ProjectionContinuationFailure) as caught,
        ):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
        self.assertEqual(solve.call_count, 8)
        records = caught.exception.stage_records
        np.testing.assert_allclose(
            [record["fraction"] for record in records],
            [0.5, 1.0, 0.75, 1.0, 0.875, 1.0, 0.9375, 1.0],
        )
        self.assertEqual(
            sum(record["adaptive_subdivision_triggered"] for record in records),
            3,
        )
        self.assertFalse(records[-1]["adaptive_subdivision_triggered"])
        for call in solve.call_args_list:
            initial = call.kwargs["initial_dual_variables"]
            if initial is not None:
                self.assertFalse(
                    any(np.array_equal(initial, values) for values in rejected_duals)
                )

    def test_fixed_batch_rescue_resume_preserves_consumed_subdivision_budget(
        self,
    ) -> None:
        rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        accepted = [
            replace(
                _successful_result(),
                dual_variables=np.full((1, 2), value, dtype=np.float64),
            )
            for value in (2.0, 3.0, 4.0, 5.0)
        ]
        settings = _solver_settings()
        settings.update(
            maximum_iterations=55,
            direct_exact_target_first=False,
            scale_continuation_warm_start=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.00625,
            maximum_adaptive_subdivisions=3,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            run = {
                "model_checkpoint_sha256": "a" * 64,
                "solver_settings": settings,
            }
            identity = direct._case_identity(
                run_identity_sha256=direct.sha256_json(run),
                release_index=123,
                initial_state_sha256="f" * 64,
                covariance_case="pooled",
                solver_settings=settings,
            )

            def journal() -> direct.DurableContinuationJournal:
                return direct.DurableContinuationJournal(
                    root,
                    run_identity=run,
                    case_identity=identity,
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    requested_fractions=(0.5, 1.0),
                    resume_enabled=True,
                )

            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    side_effect=(
                        accepted[0],
                        rejected,
                        accepted[1],
                        rejected,
                        accepted[2],
                        KeyboardInterrupt(),
                    ),
                ) as first_solve,
                self.assertRaises(KeyboardInterrupt),
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=journal(),
                )
            self.assertEqual(first_solve.call_count, 6)
            durable = journal().load()
            self.assertIsNotNone(durable)
            self.assertEqual(durable["adaptive_subdivision_count"], 2)  # type: ignore[index]

            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    side_effect=(rejected, accepted[3], rejected),
                ) as resumed_solve,
                self.assertRaises(direct.ProjectionContinuationFailure) as caught,
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=journal(),
                )
            self.assertEqual(resumed_solve.call_count, 3)
            records = caught.exception.stage_records
            self.assertEqual(len(records), 8)
            self.assertEqual(
                sum(record["adaptive_subdivision_triggered"] for record in records),
                3,
            )
            self.assertFalse(records[-1]["adaptive_subdivision_triggered"])

    def test_direct_first_requires_fallback_schedule_and_zero_dual(self) -> None:
        settings = _solver_settings()
        settings["direct_exact_target_first"] = True
        with self.assertRaisesRegex(ValueError, "fraction below 1"):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(1.0,),
                solver_settings=settings,
            )
        with self.assertRaisesRegex(ValueError, "canonical zero-dual start"):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=settings,
                initial_dual_variables=np.ones((1, 2)),
            )

    @staticmethod
    def _continuation_journal(
        root: Path,
        *,
        run_identity: dict[str, object] | None = None,
        resume_enabled: bool = True,
        fractions: tuple[float, ...] = (0.5, 1.0),
        solver_settings: dict[str, object] | None = None,
    ) -> direct.DurableContinuationJournal:
        settings = _solver_settings() if solver_settings is None else solver_settings
        run = (
            {
                "model_checkpoint_sha256": "a" * 64,
                "data_metadata_sha256": "b" * 64,
                "covariance_manifest_sha256": "c" * 64,
                "xai_direction_sha256": "d" * 64,
                "scientific_python_source_sha256": {"driver": "e" * 64},
                "solver_settings": settings,
            }
            if run_identity is None
            else run_identity
        )
        identity = direct._case_identity(
            run_identity_sha256=direct.sha256_json(run),
            release_index=123,
            initial_state_sha256="f" * 64,
            covariance_case="pooled",
            solver_settings=settings,
        )
        return direct.DurableContinuationJournal(
            root,
            run_identity=run,
            case_identity=identity,
            covariance_case="pooled",
            baseline_projection=0.0,
            target_projection=10.0,
            requested_fractions=fractions,
            resume_enabled=resume_enabled,
        )

    def test_direct_first_fallback_journal_resumes_and_deduplicates_zero(
        self,
    ) -> None:
        direct_rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        half = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        warm_rejected = replace(direct_rejected, dual_variables=np.full((1, 2), 88.0))
        midpoint = replace(_successful_result(), dual_variables=np.full((1, 2), 3.0))
        final = replace(
            _successful_result(),
            native_interventions=np.full((1, 2), 4.0),
            dual_variables=np.full((1, 2), 4.0),
            native_constraint_gradient=np.full((1, 2), 4.0),
        )
        settings = _solver_settings()
        settings.update(
            direct_exact_target_first=True,
            scale_continuation_warm_start=True,
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(
                root,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
            pending_commit = journal.commit_pending_transition

            def persist_direct_fallback_then_interrupt(**kwargs: object) -> None:
                pending_commit(**kwargs)  # type: ignore[arg-type]
                raise KeyboardInterrupt("synthetic interruption after direct fallback")

            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    return_value=direct_rejected,
                ) as interrupted_solve,
                patch.object(
                    journal,
                    "commit_pending_transition",
                    side_effect=persist_direct_fallback_then_interrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=journal,
                )
            self.assertEqual(interrupted_solve.call_count, 1)
            pending_path = root / "continuation_journal" / "pending_retry.json"
            pending = direct.load_json(pending_path)
            self.assertEqual(
                pending["scheduler"]["direct_exact_target_status"], "rejected"
            )
            self.assertEqual(pending["scheduler"]["next_position"], 0)
            self.assertEqual(pending["scheduler"]["planned_fractions"], [0.5, 1.0])
            self.assertEqual(
                len(pending["scheduler"]["rejected_zero_dual_attempts"]), 1
            )
            self.assertNotIn("arrays", pending)

            tampered = copy.deepcopy(pending)
            tampered["stage_records"][0]["zero_dual_solve_identity_sha256"] = "0" * 64
            tampered["scheduler"]["rejected_zero_dual_attempts"] = (
                direct._rejected_zero_dual_ledger(tampered["stage_records"])
            )
            tampered["stage_records_sha256"] = direct.sha256_json(
                {"stage_records": tampered["stage_records"]}
            )
            tampered["scheduler_sha256"] = direct.sha256_json(
                {"scheduler": tampered["scheduler"]}
            )
            direct.write_json(pending_path, tampered, overwrite=True)
            with self.assertRaisesRegex(ValueError, "warm-start linkage"):
                self._continuation_journal(
                    root,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                ).load()
            direct.write_json(pending_path, pending, overwrite=True)

            resumed_journal = self._continuation_journal(
                root,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
            with patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                side_effect=(half, warm_rejected, midpoint, final),
            ) as resumed_solve:
                actual, records = direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=resumed_journal,
                )
            self.assertIs(actual, final)
            self.assertEqual(resumed_solve.call_count, 4)
            self.assertIsNone(
                resumed_solve.call_args_list[0].kwargs["initial_dual_variables"]
            )
            for call in resumed_solve.call_args_list[1:]:
                self.assertIsNotNone(call.kwargs["initial_dual_variables"])
            self.assertEqual(
                [record["fraction"] for record in records],
                [1.0, 0.5, 1.0, 0.75, 1.0],
            )
            self.assertTrue(records[2]["zero_dual_restart_skipped_prior_rejection"])
            self.assertFalse(pending_path.exists())

            final_journal = self._continuation_journal(
                root,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
            with patch.object(
                direct, "solve_minimum_native_covariance_action_sqp"
            ) as forbidden_solve:
                restored, restored_records = direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=final_journal,
                )
            forbidden_solve.assert_not_called()
            np.testing.assert_array_equal(restored.dual_variables, final.dual_variables)
            self.assertEqual(restored_records, records)

    def test_direct_first_accepted_checkpoint_resumes_without_solver_call(
        self,
    ) -> None:
        result = _successful_result()
        settings = _solver_settings()
        settings.update(
            direct_exact_target_first=True,
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(
                root,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
            with patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=result,
            ) as solve:
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=journal,
                )
            self.assertEqual(solve.call_count, 1)
            checkpoint = self._continuation_journal(
                root,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
            with patch.object(
                direct, "solve_minimum_native_covariance_action_sqp"
            ) as forbidden_solve:
                restored, records = direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=checkpoint,
                )
            forbidden_solve.assert_not_called()
            np.testing.assert_array_equal(
                restored.dual_variables, result.dual_variables
            )
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["direct_exact_target_trial"])
            self.assertEqual(
                checkpoint.audit_record()["direct_exact_target_status"],  # type: ignore[index]
                "accepted",
            )

    def test_durable_continuation_resumes_after_interruption_equivalently(
        self,
    ) -> None:
        first = replace(
            _successful_result(),
            native_interventions=np.full((1, 2), 2.0),
            dual_variables=np.full((1, 2), 2.0),
            native_constraint_gradient=np.full((1, 2), 2.0),
        )
        final = replace(
            _successful_result(),
            native_interventions=np.full((1, 2), 3.0),
            dual_variables=np.full((1, 2), 3.0),
            native_constraint_gradient=np.full((1, 2), 3.0),
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(root)
            commit = journal.commit

            def commit_then_interrupt(**kwargs: object) -> dict[str, object]:
                commit(**kwargs)  # type: ignore[arg-type]
                raise KeyboardInterrupt("synthetic interruption after fsync")

            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    return_value=first,
                ) as interrupted_solve,
                patch.object(journal, "commit", side_effect=commit_then_interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=_solver_settings(),
                    continuation_journal=journal,
                )
            self.assertEqual(interrupted_solve.call_count, 1)
            self.assertFalse((root / "report.json").exists())
            checkpoint = root / "continuation_journal" / "accepted-0001.npz"
            self.assertTrue(checkpoint.is_file())
            with np.load(checkpoint, allow_pickle=False) as archive:
                document = direct.DurableContinuationJournal._decode_document(
                    checkpoint, archive
                )
            self.assertEqual(document["status"], "accepted_stage_nonpublishable")
            self.assertFalse(document["scientific_result_published"])

            resumed_journal = self._continuation_journal(root)
            with patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=final,
            ) as resumed_solve:
                resumed_result, resumed_records = direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=_solver_settings(),
                    continuation_journal=resumed_journal,
                )
            self.assertEqual(resumed_solve.call_count, 1)
            np.testing.assert_array_equal(
                resumed_solve.call_args.kwargs["initial_dual_variables"],
                first.dual_variables,
            )
            np.testing.assert_array_equal(
                resumed_result.dual_variables, final.dual_variables
            )
            self.assertEqual(
                [record["fraction"] for record in resumed_records], [0.5, 1.0]
            )
            self.assertTrue(resumed_records[-1]["publication_gates"]["all_passed"])

            # A second restart occurs after the exact-target solve but before
            # publication.  It reconstructs the final accepted result and does
            # not call the nonlinear solver at all.
            final_journal = self._continuation_journal(root)
            with patch.object(
                direct, "solve_minimum_native_covariance_action_sqp"
            ) as forbidden_solve:
                restored_result, restored_records = (
                    direct._solve_projection_continuation(
                        _ContinuationOracle(),  # type: ignore[arg-type]
                        object(),
                        covariance_case="pooled",
                        baseline_projection=0.0,
                        target_projection=10.0,
                        fractions=(0.5, 1.0),
                        solver_settings=_solver_settings(),
                        continuation_journal=final_journal,
                    )
                )
            forbidden_solve.assert_not_called()
            np.testing.assert_array_equal(
                restored_result.native_interventions, final.native_interventions
            )
            np.testing.assert_array_equal(
                restored_result.native_constraint_gradient,
                final.native_constraint_gradient,
            )
            self.assertEqual(restored_records, resumed_records)
            audit = final_journal.audit_record()
            self.assertIsNotNone(audit)
            self.assertTrue(audit["resumed_this_invocation"])  # type: ignore[index]

            with patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                side_effect=(first, final),
            ):
                uninterrupted_result, uninterrupted_records = (
                    direct._solve_projection_continuation(
                        _ContinuationOracle(),  # type: ignore[arg-type]
                        object(),
                        covariance_case="pooled",
                        baseline_projection=0.0,
                        target_projection=10.0,
                        fractions=(0.5, 1.0),
                        solver_settings=_solver_settings(),
                    )
                )
            np.testing.assert_array_equal(
                restored_result.dual_variables,
                uninterrupted_result.dual_variables,
            )
            self.assertEqual(
                [item["target_projection"] for item in restored_records],
                [item["target_projection"] for item in uninterrupted_records],
            )

    def test_zero_dual_pending_retry_resumes_without_repeating_warm_attempt(
        self,
    ) -> None:
        first = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        warm_rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        final = replace(
            _successful_result(),
            native_interventions=np.full((1, 2), 3.0),
            dual_variables=np.full((1, 2), 3.0),
            native_constraint_gradient=np.full((1, 2), 3.0),
        )
        settings = _solver_settings()
        settings.update(
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(root, solver_settings=settings)
            pending_commit = journal.commit_pending_transition

            def persist_then_interrupt(**kwargs: object) -> None:
                pending_commit(**kwargs)  # type: ignore[arg-type]
                raise KeyboardInterrupt("synthetic interruption after pending fsync")

            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    side_effect=(first, warm_rejected),
                ) as interrupted_solve,
                patch.object(
                    journal,
                    "commit_pending_transition",
                    side_effect=persist_then_interrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=journal,
                )
            self.assertEqual(interrupted_solve.call_count, 2)
            pending_path = root / "continuation_journal" / "pending_retry.json"
            self.assertTrue(pending_path.is_file())
            pending = direct.load_json(pending_path)
            self.assertEqual(
                pending["scheduler"]["pending_zero_dual_retry_fraction"], 1.0
            )
            self.assertNotIn("arrays", pending)
            tampered = copy.deepcopy(pending)
            tampered["stage_records"][-1]["initial_dual_sha256"] = "0" * 64
            tampered["stage_records_sha256"] = direct.sha256_json(
                {"stage_records": tampered["stage_records"]}
            )
            direct.write_json(pending_path, tampered, overwrite=True)
            with self.assertRaisesRegex(ValueError, "warm-start linkage"):
                self._continuation_journal(root, solver_settings=settings).load()
            direct.write_json(pending_path, pending, overwrite=True)

            resumed_journal = self._continuation_journal(root, solver_settings=settings)
            with patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=final,
            ) as resumed_solve:
                resumed_result, records = direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=resumed_journal,
                )
            self.assertEqual(resumed_solve.call_count, 1)
            self.assertIsNone(resumed_solve.call_args.kwargs["initial_dual_variables"])
            self.assertEqual(
                [record["fraction"] for record in records], [0.5, 1.0, 1.0]
            )
            self.assertTrue(records[2]["zero_dual_restart_attempt"])
            np.testing.assert_array_equal(
                resumed_result.dual_variables, final.dual_variables
            )
            self.assertFalse(pending_path.exists())

    def test_resume_does_not_repeat_zero_dual_retry_for_revisited_fraction(
        self,
    ) -> None:
        first = replace(_successful_result(), dual_variables=np.full((1, 2), 1.0))
        rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        zero_rejected = replace(rejected, dual_variables=np.full((1, 2), 88.0))
        midpoint = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        revisit_rejected = replace(rejected, dual_variables=np.full((1, 2), 77.0))
        settings = _solver_settings()
        settings.update(
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.1,
            maximum_adaptive_subdivisions=4,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(
                root,
                fractions=(0.25, 0.5, 1.0),
                solver_settings=settings,
            )
            original_commit = journal.commit
            commit_count = 0

            def interrupt_after_midpoint(**kwargs: object) -> dict[str, object]:
                nonlocal commit_count
                output = original_commit(**kwargs)  # type: ignore[arg-type]
                commit_count += 1
                if commit_count == 2:
                    raise KeyboardInterrupt("interrupt after accepted midpoint")
                return output

            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    side_effect=(first, rejected, zero_rejected, midpoint),
                ),
                patch.object(journal, "commit", side_effect=interrupt_after_midpoint),
                self.assertRaises(KeyboardInterrupt),
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.25, 0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=journal,
                )

            resumed_journal = self._continuation_journal(
                root,
                fractions=(0.25, 0.5, 1.0),
                solver_settings=settings,
            )
            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    return_value=revisit_rejected,
                ) as resumed_solve,
                self.assertRaises(direct.ProjectionContinuationFailure) as caught,
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.25, 0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=resumed_journal,
                )
            self.assertEqual(resumed_solve.call_count, 1)
            np.testing.assert_array_equal(
                resumed_solve.call_args.kwargs["initial_dual_variables"],
                midpoint.dual_variables,
            )
            records = caught.exception.stage_records
            self.assertEqual(
                sum(record["zero_dual_restart_triggered"] for record in records), 1
            )
            self.assertFalse(records[-1]["zero_dual_restart_triggered"])
            self.assertTrue(records[-1]["zero_dual_restart_skipped_prior_rejection"])
            self.assertEqual(
                records[-1]["prior_rejected_zero_dual_attempt"]["stage"], 3
            )

    def test_resume_after_zero_rejection_starts_at_persisted_midpoint(self) -> None:
        first = replace(_successful_result(), dual_variables=np.full((1, 2), 1.0))
        rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        zero_rejected = replace(rejected, dual_variables=np.full((1, 2), 88.0))
        midpoint = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        final = replace(
            _successful_result(),
            native_interventions=np.full((1, 2), 3.0),
            dual_variables=np.full((1, 2), 3.0),
            native_constraint_gradient=np.full((1, 2), 3.0),
        )
        settings = _solver_settings()
        settings.update(
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(root, solver_settings=settings)
            pending_commit = journal.commit_pending_transition
            pending_count = 0

            def interrupt_after_bisection(**kwargs: object) -> dict[str, object]:
                nonlocal pending_count
                output = pending_commit(**kwargs)  # type: ignore[arg-type]
                pending_count += 1
                if pending_count == 2:
                    raise KeyboardInterrupt("interrupt after bisection fsync")
                return output

            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    side_effect=(first, rejected, zero_rejected),
                ),
                patch.object(
                    journal,
                    "commit_pending_transition",
                    side_effect=interrupt_after_bisection,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=journal,
                )

            resumed_journal = self._continuation_journal(root, solver_settings=settings)
            with patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                side_effect=(midpoint, final),
            ) as solve:
                result, records = direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=settings,
                    continuation_journal=resumed_journal,
                )
            self.assertIs(result, final)
            self.assertEqual(
                [record["fraction"] for record in records],
                [0.5, 1.0, 1.0, 0.75, 1.0],
            )
            self.assertEqual(solve.call_count, 2)
            np.testing.assert_array_equal(
                solve.call_args_list[0].kwargs["initial_dual_variables"],
                first.dual_variables,
            )
            np.testing.assert_array_equal(
                solve.call_args_list[1].kwargs["initial_dual_variables"],
                midpoint.dual_variables,
            )
            self.assertEqual(
                sum(record["zero_dual_restart_attempt"] for record in records), 1
            )

    def test_durable_continuation_fails_closed_on_corruption_and_identity_change(
        self,
    ) -> None:
        result = _successful_result()
        record = {
            "stage": 1,
            "fraction": 1.0,
            "fraction_in_original_requested_schedule": True,
            "accepted_for_continuation": True,
            "acceptance_basis": "original_publication_gates",
            "result_dual_sha256": direct.sha256_array(result.dual_variables),
            "scaled_warm_start_enabled": False,
            "warm_started_from_external_exact_target_checkpoint": False,
            "warm_started_from_previous_stage": False,
            "last_accepted_fraction_before_attempt": 0.0,
            "warm_start_scale": None,
            "previous_result_dual_sha256": None,
            "initial_dual_sha256": None,
            "rejected_iterate_used_as_warm_start": False,
            "zero_dual_restart_fallback_enabled": False,
            "zero_dual_restart_attempt": False,
            "zero_dual_restart_triggered": False,
            "zero_dual_retry_fraction": None,
            "adaptive_continuation_enabled": False,
            "adaptive_subdivision_count_before_attempt": 0,
            "adaptive_subdivision_triggered": False,
            "adaptive_retry_fraction": None,
            "publication_gates": direct._solver_publication_gates(
                result, solver_settings=_solver_settings()
            ),
            "intermediate_continuation_gate": {},
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(root, fractions=(1.0,))
            self.assertIsNone(journal.load())
            zero_identity = direct._zero_dual_solve_identity_sha256(
                continuation_context_sha256=journal.identity_sha256,
                fraction=1.0,
                target_projection=10.0,
            )
            record.update(
                attempt_role="continuation_stage",
                target_projection=10.0,
                initialization_mode="zero_dual",
                candidate_zero_dual_solve_identity_sha256=zero_identity,
                zero_dual_solve_identity_sha256=zero_identity,
                zero_dual_restart_skipped_prior_rejection=False,
                prior_rejected_zero_dual_attempt=None,
                direct_exact_target_first_enabled=False,
                direct_exact_target_trial=False,
                direct_fallback_eligibility=None,
                direct_fallback_triggered=False,
            )
            journal.commit(
                result=result,
                stage_records=(record,),
                planned_fractions=(1.0,),
                next_position=1,
                adaptive_subdivision_count=0,
            )
            with self.assertRaisesRegex(ValueError, "resume is disabled"):
                self._continuation_journal(
                    root, resume_enabled=False, fractions=(1.0,)
                ).load()

            changed_run = dict(journal.identity["run_identity"])
            changed_run["model_checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "another scientific"):
                self._continuation_journal(
                    root,
                    run_identity=changed_run,
                    fractions=(1.0,),
                ).load()

            checkpoint = root / "continuation_journal" / "accepted-0001.npz"
            with np.load(checkpoint, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            arrays["dual_variables"][0, 0] += 1.0
            direct.write_npz(checkpoint, overwrite=True, **arrays)
            with self.assertRaisesRegex(ValueError, "array is corrupted"):
                self._continuation_journal(root, fractions=(1.0,)).load()

    def test_durable_journal_never_saves_a_rejected_dual_as_warm_start(self) -> None:
        accepted = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        midpoint = replace(_successful_result(), dual_variables=np.full((1, 2), 3.0))
        settings = _solver_settings()
        settings.update(
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            run = {
                "model_checkpoint_sha256": "a" * 64,
                "solver_settings": settings,
            }
            identity = direct._case_identity(
                run_identity_sha256=direct.sha256_json(run),
                release_index=123,
                initial_state_sha256="f" * 64,
                covariance_case="pooled",
                solver_settings=settings,
            )
            journal = direct.DurableContinuationJournal(
                root,
                run_identity=run,
                case_identity=identity,
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                requested_fractions=(0.1, 0.2, 1.0),
                resume_enabled=True,
            )
            original_commit = journal.commit
            commits = 0

            def interrupt_after_midpoint(**kwargs: object) -> dict[str, object]:
                nonlocal commits
                output = original_commit(**kwargs)  # type: ignore[arg-type]
                commits += 1
                if commits == 2:
                    raise KeyboardInterrupt
                return output

            with (
                patch.object(
                    direct,
                    "solve_minimum_native_covariance_action_sqp",
                    side_effect=(accepted, rejected, midpoint),
                ),
                patch.object(journal, "commit", side_effect=interrupt_after_midpoint),
                self.assertRaises(KeyboardInterrupt),
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.1, 0.2, 1.0),
                    solver_settings=settings,
                    continuation_journal=journal,
                )
            latest = direct.DurableContinuationJournal(
                root,
                run_identity=run,
                case_identity=identity,
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                requested_fractions=(0.1, 0.2, 1.0),
                resume_enabled=True,
            ).load()
            self.assertIsNotNone(latest)
            np.testing.assert_array_equal(
                latest["dual_variables"],
                midpoint.dual_variables,  # type: ignore[index]
            )
            self.assertNotEqual(
                latest["document"]["accepted_dual_sha256"],  # type: ignore[index]
                direct.sha256_array(rejected.dual_variables),
            )
            validator = direct.DurableContinuationJournal(
                root,
                run_identity=run,
                case_identity=identity,
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                requested_fractions=(0.1, 0.2, 1.0),
                resume_enabled=True,
            )
            first_path = root / "continuation_journal" / "accepted-0001.npz"
            second_path = root / "continuation_journal" / "accepted-0002.npz"
            first_checkpoint = validator._load_one(first_path, expected_sequence=1)
            second_checkpoint = validator._load_one(second_path, expected_sequence=2)
            for field, bad_value in (
                ("previous_result_dual_sha256", "0" * 64),
                ("initial_dual_sha256", "1" * 64),
                ("warm_start_scale", 99.0),
            ):
                with self.subTest(rejected_link_field=field):
                    tampered = copy.deepcopy(second_checkpoint)
                    # Record 2 is the rejected 0.2 attempt; record 3 is the
                    # accepted midpoint. Neither may start from record 2's dual.
                    tampered["document"]["stage_records"][1][field] = bad_value
                    with self.assertRaisesRegex(ValueError, "warm-start linkage"):
                        validator._validate_attempt_linkage(tampered, first_checkpoint)

    def test_durable_continuation_rejects_a_missing_checkpoint_gap(self) -> None:
        first = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        final = replace(_successful_result(), dual_variables=np.full((1, 2), 3.0))
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(direct, "CONTROL_STEPS", 1),
        ):
            root = Path(temporary) / "case"
            journal = self._continuation_journal(root)
            with patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                side_effect=(first, final),
            ):
                direct._solve_projection_continuation(
                    _ContinuationOracle(),  # type: ignore[arg-type]
                    object(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=10.0,
                    fractions=(0.5, 1.0),
                    solver_settings=_solver_settings(),
                    continuation_journal=journal,
                )
            first_checkpoint = root / "continuation_journal" / "accepted-0001.npz"
            first_checkpoint.unlink()
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                self._continuation_journal(root).load()

    def test_direct_cli_does_not_accept_abbreviated_options(self) -> None:
        self.assertFalse(direct.parser().allow_abbrev)
        with self.assertRaises(SystemExit):
            direct.parser().parse_args(["--sel=7"])

    def test_selection_seed_cli_accepts_zero_and_custom_nonnegative_seed(self) -> None:
        self.assertEqual(
            direct.parser().parse_args(["--selection-seed", "0"]).selection_seed,
            0,
        )
        self.assertEqual(
            direct.parser().parse_args(["--selection-seed", "46"]).selection_seed,
            46,
        )
        with self.assertRaises(SystemExit):
            direct.parser().parse_args(["--selection-seed", "-1"])

    def test_projection_continuation_shifts_values_and_carries_duals(self) -> None:
        oracle = _ContinuationOracle()
        first = replace(
            _successful_result(),
            dual_variables=np.full((1, 2), 2.0),
        )
        final = replace(
            _successful_result(),
            dual_variables=np.full((1, 2), 3.0),
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(first, final),
        ) as solve:
            result, records = direct._solve_projection_continuation(
                oracle,  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=_solver_settings(),
            )
        self.assertIs(result, final)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["target_projection"], 5.0)
        self.assertEqual(records[1]["target_projection"], 10.0)
        self.assertFalse(records[0]["warm_started_from_previous_stage"])
        self.assertTrue(records[1]["warm_started_from_previous_stage"])
        self.assertIsNone(records[0]["warm_start_scale"])
        self.assertEqual(records[1]["warm_start_scale"], 1.0)
        self.assertFalse(records[1]["scaled_warm_start_enabled"])
        first_value = solve.call_args_list[0].args[0]
        final_value = solve.call_args_list[1].args[0]
        self.assertEqual(first_value(np.zeros((1, 2))), -5.0)
        self.assertEqual(final_value(np.zeros((1, 2))), -10.0)
        self.assertIsNone(solve.call_args_list[0].kwargs["initial_dual_variables"])
        self.assertIs(solve.call_args_list[0].kwargs["known_active_boundary"], True)
        self.assertIs(solve.call_args_list[1].kwargs["known_active_boundary"], True)
        np.testing.assert_array_equal(
            solve.call_args_list[1].kwargs["initial_dual_variables"],
            first.dual_variables,
        )

    def test_projection_continuation_can_scale_warm_start_by_fraction_ratio(
        self,
    ) -> None:
        oracle = _ContinuationOracle()
        first = replace(
            _successful_result(),
            dual_variables=np.asarray([[2.0, 4.0]]),
        )
        second = replace(
            _successful_result(),
            dual_variables=np.asarray([[3.0, 6.0]]),
        )
        final = replace(
            _successful_result(),
            dual_variables=np.asarray([[5.0, 10.0]]),
        )
        settings = _solver_settings()
        settings["scale_continuation_warm_start"] = True
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(first, second, final),
        ) as solve:
            result, records = direct._solve_projection_continuation(
                oracle,  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.25, 0.5, 1.0),
                solver_settings=settings,
            )
        self.assertIs(result, final)
        self.assertEqual(
            [record["warm_start_scale"] for record in records],
            [None, 2.0, 2.0],
        )
        self.assertTrue(all(record["scaled_warm_start_enabled"] for record in records))
        np.testing.assert_array_equal(
            solve.call_args_list[1].kwargs["initial_dual_variables"],
            2.0 * first.dual_variables,
        )
        np.testing.assert_array_equal(
            solve.call_args_list[2].kwargs["initial_dual_variables"],
            2.0 * second.dual_variables,
        )
        self.assertEqual(
            records[1]["previous_result_dual_sha256"],
            direct.sha256_array(first.dual_variables),
        )
        self.assertEqual(
            records[1]["initial_dual_sha256"],
            direct.sha256_array(2.0 * first.dual_variables),
        )

    def test_zero_dual_restart_accepts_same_intermediate_target_before_bisection(
        self,
    ) -> None:
        first = replace(_successful_result(), dual_variables=np.full((1, 2), 1.0))
        rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        zero_accepted = replace(
            _successful_result(), dual_variables=np.full((1, 2), 2.0)
        )
        final = replace(_successful_result(), dual_variables=np.full((1, 2), 3.0))
        settings = _solver_settings()
        settings.update(
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=4,
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(first, rejected, zero_accepted, final),
        ) as solve:
            result, records = direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.25, 0.5, 1.0),
                solver_settings=settings,
            )
        self.assertIs(result, final)
        self.assertEqual(
            [record["fraction"] for record in records], [0.25, 0.5, 0.5, 1.0]
        )
        self.assertTrue(records[1]["zero_dual_restart_triggered"])
        self.assertFalse(records[1]["adaptive_subdivision_triggered"])
        self.assertTrue(records[2]["zero_dual_restart_attempt"])
        self.assertTrue(records[2]["accepted_for_continuation"])
        np.testing.assert_array_equal(
            solve.call_args_list[1].kwargs["initial_dual_variables"],
            first.dual_variables,
        )
        self.assertIsNone(solve.call_args_list[2].kwargs["initial_dual_variables"])
        self.assertNotEqual(
            records[2]["previous_result_dual_sha256"],
            direct.sha256_array(rejected.dual_variables),
        )

    def test_zero_dual_rejection_bisects_once_and_never_forwards_rejected_dual(
        self,
    ) -> None:
        first = replace(_successful_result(), dual_variables=np.full((1, 2), 1.0))
        warm_rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        zero_rejected = replace(warm_rejected, dual_variables=np.full((1, 2), 88.0))
        midpoint = replace(_successful_result(), dual_variables=np.full((1, 2), 2.0))
        # Rejection when revisiting 0.5 must bisect immediately: that exact
        # fraction has already consumed its one zero-dual retry.
        revisit_rejected = replace(warm_rejected, dual_variables=np.full((1, 2), 77.0))
        quarter_midpoint = replace(
            _successful_result(), dual_variables=np.full((1, 2), 3.0)
        )
        revisited = replace(_successful_result(), dual_variables=np.full((1, 2), 4.0))
        final = replace(_successful_result(), dual_variables=np.full((1, 2), 5.0))
        settings = _solver_settings()
        settings.update(
            zero_dual_restart_after_warm_rejection=True,
            adaptive_continuation=True,
            adaptive_minimum_fraction_step=0.01,
            maximum_adaptive_subdivisions=8,
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(
                first,
                warm_rejected,
                zero_rejected,
                midpoint,
                revisit_rejected,
                quarter_midpoint,
                revisited,
                final,
            ),
        ) as solve:
            result, records = direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.25, 0.5, 1.0),
                solver_settings=settings,
            )
        self.assertIs(result, final)
        np.testing.assert_allclose(
            [record["fraction"] for record in records],
            [0.25, 0.5, 0.5, 0.375, 0.5, 0.4375, 0.5, 1.0],
        )
        self.assertEqual(
            sum(
                record["zero_dual_restart_triggered"] and record["fraction"] == 0.5
                for record in records
            ),
            1,
        )
        self.assertTrue(records[2]["adaptive_subdivision_triggered"])
        self.assertTrue(records[4]["adaptive_subdivision_triggered"])
        self.assertTrue(records[4]["zero_dual_restart_skipped_prior_rejection"])
        self.assertEqual(records[4]["prior_rejected_zero_dual_attempt"]["stage"], 3)
        self.assertEqual(
            records[4]["candidate_zero_dual_solve_identity_sha256"],
            records[2]["zero_dual_solve_identity_sha256"],
        )
        forwarded = [
            call.kwargs["initial_dual_variables"] for call in solve.call_args_list
        ]
        self.assertIsNone(forwarded[2])
        for initial in forwarded[3:]:
            if initial is not None:
                self.assertFalse(np.array_equal(initial, warm_rejected.dual_variables))
                self.assertFalse(np.array_equal(initial, zero_rejected.dual_variables))
                self.assertFalse(
                    np.array_equal(initial, revisit_rejected.dual_variables)
                )

    def test_final_intermediate_only_gate_rejection_gets_zero_dual_retry(self) -> None:
        first = replace(_successful_result(), dual_variables=np.full((1, 2), 1.0))
        intermediate_only = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=5.0e-6,
            covariance_stationarity_relative=0.0201,
            dual_variables=np.full((1, 2), 99.0),
        )
        final = replace(_successful_result(), dual_variables=np.full((1, 2), 3.0))
        settings = _solver_settings()
        settings["zero_dual_restart_after_warm_rejection"] = True
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(first, intermediate_only, final),
        ) as solve:
            result, records = direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
        self.assertIs(result, final)
        self.assertTrue(records[1]["intermediate_continuation_gate"]["all_passed"])
        self.assertFalse(records[1]["intermediate_gate_applicable"])
        self.assertFalse(records[1]["accepted_for_continuation"])
        self.assertTrue(records[1]["zero_dual_restart_triggered"])
        self.assertTrue(records[2]["publication_gates"]["all_passed"])
        self.assertIsNone(solve.call_args_list[2].kwargs["initial_dual_variables"])

    def test_zero_dual_restart_is_not_applied_to_default_zero_initial_solve(
        self,
    ) -> None:
        rejected = replace(
            _successful_result(),
            success=False,
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
        )
        settings = _solver_settings()
        settings["zero_dual_restart_after_warm_rejection"] = True
        with (
            patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=rejected,
            ) as solve,
            self.assertRaises(direct.ProjectionContinuationFailure),
        ):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(1.0,),
                solver_settings=settings,
            )
        self.assertEqual(solve.call_count, 1)

    def test_adaptive_continuation_bisects_and_discards_rejected_dual(self) -> None:
        oracle = _ContinuationOracle()
        first = replace(
            _successful_result(),
            dual_variables=np.full((1, 2), 1.0),
        )
        rejected = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
            dual_variables=np.full((1, 2), 99.0),
        )
        midpoint = replace(
            _successful_result(),
            dual_variables=np.full((1, 2), 2.0),
        )
        retried = replace(
            _successful_result(),
            dual_variables=np.full((1, 2), 3.0),
        )
        final = replace(
            _successful_result(),
            dual_variables=np.full((1, 2), 4.0),
        )
        settings = _solver_settings()
        settings.update(
            {
                "scale_continuation_warm_start": True,
                "adaptive_continuation": True,
                "adaptive_minimum_fraction_step": 0.01,
                "maximum_adaptive_subdivisions": 4,
            }
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(first, rejected, midpoint, retried, final),
        ) as solve:
            result, records = direct._solve_projection_continuation(
                oracle,  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.1, 0.2, 1.0),
                solver_settings=settings,
            )
        self.assertIs(result, final)
        np.testing.assert_allclose(
            [record["fraction"] for record in records],
            [0.1, 0.2, 0.15, 0.2, 1.0],
        )
        self.assertFalse(records[1]["accepted_for_continuation"])
        self.assertTrue(records[1]["adaptive_subdivision_triggered"])
        self.assertAlmostEqual(records[1]["adaptive_retry_fraction"], 0.15)
        self.assertFalse(records[1]["rejected_iterate_used_as_warm_start"])
        self.assertFalse(records[2]["fraction_in_original_requested_schedule"])
        # The midpoint starts from the last accepted 0.1 solution, scaled by
        # 0.15 / 0.1.  The rejected all-99 dual is never reused.
        np.testing.assert_allclose(
            solve.call_args_list[2].kwargs["initial_dual_variables"],
            1.5 * first.dual_variables,
        )
        np.testing.assert_allclose(
            solve.call_args_list[3].kwargs["initial_dual_variables"],
            (0.2 / 0.15) * midpoint.dual_variables,
        )
        self.assertTrue(records[-1]["publication_gates"]["all_passed"])

    def test_adaptive_continuation_stops_fail_closed_at_minimum_step(self) -> None:
        invalid = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
        )
        settings = _solver_settings()
        settings.update(
            {
                "adaptive_continuation": True,
                "adaptive_minimum_fraction_step": 0.3,
                "maximum_adaptive_subdivisions": 4,
            }
        )
        with (
            patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=invalid,
            ),
            self.assertRaises(direct.ProjectionContinuationFailure) as caught,
        ):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=settings,
            )
        self.assertEqual(len(caught.exception.stage_records), 1)
        self.assertFalse(
            caught.exception.stage_records[0]["adaptive_subdivision_triggered"]
        )

    def test_adaptive_final_rejection_subdivides_without_relaxing_final_gate(
        self,
    ) -> None:
        motivating = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=5.0e-6,
            covariance_stationarity_relative=0.0201,
        )
        midpoint = replace(
            _successful_result(),
            dual_variables=np.full((1, 2), 2.0),
        )
        final = replace(
            _successful_result(),
            dual_variables=np.full((1, 2), 3.0),
        )
        settings = _solver_settings()
        settings.update(
            {
                "adaptive_continuation": True,
                "adaptive_minimum_fraction_step": 0.01,
                "maximum_adaptive_subdivisions": 4,
            }
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(motivating, midpoint, final),
        ):
            result, records = direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(1.0,),
                solver_settings=settings,
            )
        self.assertIs(result, final)
        self.assertTrue(records[0]["intermediate_continuation_gate"]["all_passed"])
        self.assertFalse(records[0]["intermediate_gate_applicable"])
        self.assertFalse(records[0]["accepted_for_continuation"])
        self.assertTrue(records[0]["adaptive_subdivision_triggered"])
        np.testing.assert_allclose(
            [record["fraction"] for record in records], [1.0, 0.5, 1.0]
        )
        self.assertTrue(records[-1]["publication_gates"]["all_passed"])

    def test_small_infeasible_near_stationary_intermediate_can_warm_start(
        self,
    ) -> None:
        oracle = _ContinuationOracle()
        motivating = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-4.30241e-5,
            primal_violation=4.30241e-5,
            covariance_stationarity_relative=0.02505379,
            complementarity_relative=1.99684e-6,
            dual_variables=np.asarray([[2.0, 4.0]]),
        )
        final = replace(
            _successful_result(),
            dual_variables=np.asarray([[3.0, 6.0]]),
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(motivating, final),
        ) as solve:
            result, records = direct._solve_projection_continuation(
                oracle,  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.65, 1.0),
                solver_settings=_solver_settings(),
            )
        self.assertIs(result, final)
        self.assertFalse(records[0]["publication_gates"]["all_passed"])
        self.assertTrue(records[0]["intermediate_continuation_gate"]["all_passed"])
        self.assertEqual(
            records[0]["intermediate_continuation_gate"][
                "original_publication_gate_failures"
            ],
            [
                "absolute_primal_violation_within_configured_tolerance",
                "solver_constraint_is_literally_nonnegative",
                "solver_declared_success",
                "stationarity_within_configured_tolerance",
            ],
        )
        self.assertEqual(
            records[0]["intermediate_continuation_gate"]["constraint"][
                "intermediate_warm_start_multiplier"
            ],
            5.0,
        )
        self.assertTrue(records[0]["intermediate_gate_used_to_continue"])
        self.assertEqual(
            records[0]["acceptance_basis"],
            "fail_closed_intermediate_warm_start_gate",
        )
        np.testing.assert_array_equal(
            solve.call_args_list[1].kwargs["initial_dual_variables"],
            motivating.dual_variables,
        )
        self.assertFalse(records[1]["intermediate_gate_used_to_continue"])
        self.assertEqual(records[1]["acceptance_basis"], "original_publication_gates")

    def test_originally_publishable_intermediate_does_not_need_warm_gate(
        self,
    ) -> None:
        oracle = _ContinuationOracle()
        publishable = replace(
            _successful_result(),
            constraint_value=2.0e-4,
            primal_violation=0.0,
            complementarity_relative=1.0e-5,
            dual_variables=np.asarray([[2.0, 4.0]]),
        )
        final = replace(
            _successful_result(),
            dual_variables=np.asarray([[3.0, 6.0]]),
        )
        with patch.object(
            direct,
            "solve_minimum_native_covariance_action_sqp",
            side_effect=(publishable, final),
        ) as solve:
            result, records = direct._solve_projection_continuation(
                oracle,  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=_solver_settings(),
            )
        self.assertIs(result, final)
        self.assertTrue(records[0]["publication_gates"]["all_passed"])
        self.assertFalse(records[0]["intermediate_continuation_gate"]["all_passed"])
        self.assertTrue(records[0]["accepted_for_continuation"])
        self.assertFalse(records[0]["intermediate_gate_used_to_continue"])
        self.assertEqual(records[0]["acceptance_basis"], "original_publication_gates")
        np.testing.assert_array_equal(
            solve.call_args_list[1].kwargs["initial_dual_variables"],
            publishable.dual_variables,
        )

    def test_final_stage_still_requires_original_publication_gates(self) -> None:
        motivating = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=5.16e-6,
            primal_violation=0.0,
            covariance_stationarity_relative=0.02020875,
            complementarity_relative=3.59e-7,
        )
        with (
            patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=motivating,
            ),
            self.assertRaises(direct.ProjectionContinuationFailure) as caught,
        ):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(1.0,),
                solver_settings=_solver_settings(),
            )
        record = caught.exception.stage_records[0]
        self.assertTrue(record["intermediate_continuation_gate"]["all_passed"])
        self.assertFalse(record["intermediate_gate_applicable"])
        self.assertFalse(record["intermediate_gate_used_to_continue"])
        self.assertFalse(record["accepted_for_continuation"])
        self.assertEqual(record["acceptance_basis"], "rejected")

    def test_intermediate_gate_rejects_unsafe_iterates(self) -> None:
        settings = _solver_settings()
        valid = _successful_result()
        unsafe = {
            "more_than_twice_stationarity": replace(
                valid,
                success=False,
                covariance_stationarity_relative=0.04000001,
            ),
            "more_than_five_times_constraint_tolerance": replace(
                valid,
                success=False,
                constraint_value=-5.00001e-5,
                primal_violation=0.0,
            ),
            "more_than_five_times_primal_violation_tolerance": replace(
                valid,
                success=False,
                constraint_value=0.0,
                primal_violation=5.00001e-5,
            ),
            "nonfinite_metric": replace(
                valid,
                success=False,
                covariance_stationarity_relative=float("nan"),
            ),
            "nonfinite_array": replace(
                valid,
                success=False,
                dual_variables=np.asarray([[np.nan, 1.0]]),
            ),
        }
        for label, result in unsafe.items():
            with self.subTest(label=label):
                gate = direct._intermediate_continuation_gate(
                    result,
                    solver_settings=settings,
                )
                self.assertFalse(gate["all_passed"])

    def test_intermediate_gate_accepts_exact_policy_boundaries(self) -> None:
        settings = _solver_settings()
        boundary = replace(
            _successful_result(),
            success=False,
            constraint_value=-5.0e-5,
            primal_violation=5.0e-5,
            covariance_stationarity_relative=4.0e-2,
            complementarity_relative=1.0e-4,
        )
        gate = direct._intermediate_continuation_gate(
            boundary,
            solver_settings=settings,
        )
        self.assertTrue(gate["all_passed"])
        for field in (
            "objective_value",
            "constraint_value",
            "covariance_stationarity_relative",
            "complementarity_relative",
        ):
            with self.subTest(nonfinite=field):
                candidate = replace(boundary, **{field: float("nan")})
                self.assertFalse(
                    direct._intermediate_continuation_gate(
                        candidate,
                        solver_settings=settings,
                    )["all_passed"]
                )

    def test_intermediate_gate_uses_absolute_stationarity_when_configured(
        self,
    ) -> None:
        settings = _solver_settings()
        settings["relative_stationarity_tolerance"] = None
        candidate = replace(
            _successful_result(),
            success=False,
            covariance_stationarity_norm=1.5e-5,
            covariance_stationarity_relative=1.5,
        )
        gate = direct._intermediate_continuation_gate(
            candidate,
            solver_settings=settings,
        )
        self.assertTrue(gate["all_passed"])
        self.assertEqual(
            gate["stationarity"]["metric"], "absolute_covariance_metric_norm"
        )
        self.assertEqual(
            gate["stationarity"]["intermediate_warm_start_tolerance"], 2e-5
        )

    def test_projection_continuation_publishes_feasible_zero_action(self) -> None:
        for target in (0.0, -1.0):
            with self.subTest(target=target):
                result, stages = direct._solve_projection_continuation(
                    _InitiallyFeasibleOracle(target),  # type: ignore[arg-type]
                    _IdentityCovariance(),
                    covariance_case="pooled",
                    baseline_projection=0.0,
                    target_projection=target,
                    fractions=(1.0,),
                    solver_settings=_solver_settings(),
                )
            self.assertEqual(result.squared_action, 0.0)
            self.assertTrue(stages[0]["baseline_already_feasible_zero_action_case"])
            self.assertTrue(stages[0]["publication_gates"]["all_passed"])

    def test_projection_continuation_scaling_config_requires_a_bool(self) -> None:
        settings = _solver_settings()
        settings["scale_continuation_warm_start"] = 1
        with self.assertRaisesRegex(
            ValueError, "scale_continuation_warm_start must be a bool"
        ):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(1.0,),
                solver_settings=settings,
            )

    def test_failed_continuation_stage_carries_nonpublishable_iterate(self) -> None:
        invalid = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
        )
        with (
            patch.object(
                direct,
                "solve_minimum_native_covariance_action_sqp",
                return_value=invalid,
            ),
            self.assertRaises(direct.ProjectionContinuationFailure) as caught,
        ):
            direct._solve_projection_continuation(
                _ContinuationOracle(),  # type: ignore[arg-type]
                object(),
                covariance_case="pooled",
                baseline_projection=0.0,
                target_projection=10.0,
                fractions=(0.5, 1.0),
                solver_settings=_solver_settings(),
            )
        self.assertIs(caught.exception.result, invalid)
        self.assertEqual(caught.exception.stage_fraction, 0.5)
        self.assertEqual(len(caught.exception.stage_records), 1)
        self.assertFalse(
            caught.exception.stage_records[0]["publication_gates"]["all_passed"]
        )

    def test_fixed_cohort_is_uniform_rng_draw_then_sorted(self) -> None:
        test_inputs = np.arange(396_000, 431_970, dtype=np.int64)
        selected, eligible = direct.select_uniform_release_indices(
            test_inputs,
            test_interval=(396_000, 432_000),
            event_index=421_199,
            steps_per_year=36,
        )
        np.testing.assert_array_equal(selected, direct.EXPECTED_RELEASE_INDICES)
        self.assertEqual(eligible.size, 999)
        self.assertEqual(int(eligible[0]), 396_035)
        self.assertEqual(int(eligible[-1]), 431_963)
        self.assertTrue(np.all(selected + 30 < 432_000))

    def test_fixed_cold_cohort_uses_extreme_la_nina_phase(self) -> None:
        test_inputs = np.arange(396_000, 431_970, dtype=np.int64)
        selected, eligible = direct.select_uniform_release_indices(
            test_inputs,
            test_interval=(396_000, 432_000),
            event_index=414_278,
            steps_per_year=36,
        )
        np.testing.assert_array_equal(selected, direct.EXPECTED_COLD_RELEASE_INDICES)
        self.assertEqual(eligible.size, 999)
        self.assertEqual(int(eligible[0]), 396_026)
        self.assertEqual(int(eligible[-1]), 431_954)
        self.assertTrue(np.all(selected % 36 == 26))

    def test_exact_release_sample_regression_is_seed_42_only(self) -> None:
        warm = direct.EXPECTED_WARM_RELEASE_INDICES.copy()
        cold = direct.EXPECTED_COLD_RELEASE_INDICES.copy()
        direct._validate_seed_42_release_sample(
            warm,
            target_event="extreme_el_nino",
            seed=42,
        )
        direct._validate_seed_42_release_sample(
            cold,
            target_event="extreme_la_nina",
            seed=42,
        )
        changed = warm.copy()
        changed[0] += 36
        with self.assertRaisesRegex(RuntimeError, "fixed seed-42"):
            direct._validate_seed_42_release_sample(
                changed,
                target_event="extreme_el_nino",
                seed=42,
            )
        direct._validate_seed_42_release_sample(
            changed,
            target_event="extreme_el_nino",
            seed=43,
        )

    def test_default_covariance_is_shared_annual_and_legacy_is_explicit(self) -> None:
        warm = direct.control_phases_for_release_phase(35, steps_per_year=36)
        cold = direct.control_phases_for_release_phase(26, steps_per_year=36)
        self.assertEqual(warm, tuple(range(26, 35)))
        self.assertEqual(cold, tuple(range(17, 26)))
        self.assertEqual(
            direct.default_covariance_manifest(warm),
            direct.DEFAULT_ANNUAL_DENSE_COVARIANCE_MANIFEST,
        )
        self.assertEqual(
            direct.default_covariance_manifest(cold),
            direct.DEFAULT_ANNUAL_DENSE_COVARIANCE_MANIFEST,
        )
        self.assertEqual(
            direct.default_covariance_manifest(
                warm,
                covariance_policy=direct.LEGACY_EVENT_WINDOW_COVARIANCE_POLICY,
            ),
            Path(
                "outputs/zc_native_covariance/"
                "training-years10000-phases26-34/manifest.json"
            ),
        )
        self.assertEqual(
            direct.default_covariance_manifest(
                cold,
                covariance_policy=direct.LEGACY_EVENT_WINDOW_COVARIANCE_POLICY,
            ),
            Path(
                "outputs/zc_native_covariance/"
                "training-years10000-phases17-25/manifest.json"
            ),
        )

    def test_annual_policy_is_pooled_only_and_legacy_arm_is_labelled(self) -> None:
        self.assertEqual(
            direct.covariance_cases(
                "pooled",
                covariance_policy=direct.ANNUAL_SHARED_COVARIANCE_POLICY,
            ),
            ("pooled",),
        )
        for case in ("both", "phase-specific"):
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(ValueError, "legacy"),
            ):
                direct.covariance_cases(
                    case,
                    covariance_policy=direct.ANNUAL_SHARED_COVARIANCE_POLICY,
                )
        self.assertEqual(
            direct.covariance_cases(
                "both",
                covariance_policy=direct.LEGACY_EVENT_WINDOW_COVARIANCE_POLICY,
            ),
            direct.CASE_NAMES,
        )

    def test_native_replay_uses_layout_instead_of_prefix_slice(self) -> None:
        layout = _layout()
        controls = np.zeros((direct.CONTROL_STEPS, layout.compact_size))
        controls[:, 0] = np.arange(1, direct.CONTROL_STEPS + 1)
        controls[:, 1] = -np.arange(1, direct.CONTROL_STEPS + 1)
        initial = _state(np.asarray((10.0, 20.0, 30.0, 40.0)))
        runner = _IdentityRunner()
        replay = direct.replay_native_interventions_with_runner(
            initial,
            controls,
            layout,
            runner,
            total_steps=10,
        )
        self.assertEqual(runner.calls, 10)
        cumulative = 0
        for step in range(direct.CONTROL_STEPS):
            cumulative += step + 1
            expected = np.asarray(
                (10.0, 20.0 + cumulative, 30.0 - cumulative, 40.0),
                dtype=np.float32,
            )
            np.testing.assert_array_equal(replay.step_inputs[step].real32, expected)
        np.testing.assert_array_equal(
            replay.step_inputs[direct.CONTROL_STEPS].real32,
            replay.states[direct.CONTROL_STEPS].real32,
        )

    def test_selection_applies_both_trajectory_boundaries(self) -> None:
        test_inputs = np.arange(100, 1_000, dtype=np.int64)
        selected, eligible = direct.select_uniform_release_indices(
            test_inputs,
            test_interval=(109, 970),
            event_index=35,
            steps_per_year=36,
            count=3,
        )
        self.assertEqual(selected.size, 3)
        self.assertTrue(np.all(eligible - direct.CONTROL_STEPS >= 109))
        self.assertTrue(np.all(eligible + 30 < 970))

    def test_help_formats_literal_percentages(self) -> None:
        help_text = direct.parser().format_help()
        self.assertIn("2% default", help_text)
        self.assertIn("0.01%", help_text)
        self.assertIn("--target-event", help_text)

    def test_active_cases_supports_pooled_only_and_legacy_manifests(self) -> None:
        self.assertEqual(direct._active_cases({}), direct.CASE_NAMES)
        self.assertEqual(
            direct._active_cases({"active_cases": ["pooled"]}), ("pooled",)
        )
        for invalid in ([], ["unknown"], ["pooled", "pooled"], "pooled"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                direct._active_cases({"active_cases": invalid})

    def test_scientific_source_provenance_covers_solver_and_bridges(self) -> None:
        provenance = direct._scientific_source_provenance()
        required = {
            "src/zc_xai/nonlinear_covariance_control.py",
            "src/zc_xai/native_covariance.py",
            "src/zc_xai/native_observation.py",
            "src/zc_xai/zc_controlled_adjoint_bridge.py",
            "src/zc_xai/zc_controlled_bridge.py",
        }
        self.assertTrue(required.issubset(provenance))
        self.assertTrue(all(len(digest) == 64 for digest in provenance.values()))

    def test_publication_gates_reject_original_solver_failures(self) -> None:
        settings = _solver_settings()
        valid = _successful_result()
        self.assertTrue(
            direct._require_publishable_solver_result(
                valid, solver_settings=settings, replay_constraint=0.0
            )["all_passed"]
        )
        failures = (
            replace(valid, success=False),
            replace(valid, constraint_value=-1e-8, primal_violation=1e-8),
            replace(valid, covariance_stationarity_relative=0.021),
            replace(valid, complementarity_relative=1.01e-4),
        )
        for invalid in failures:
            with (
                self.subTest(
                    status=invalid.status,
                    constraint=invalid.constraint_value,
                ),
                self.assertRaisesRegex(RuntimeError, "refusing to publish"),
            ):
                direct._require_publishable_solver_result(
                    invalid,
                    solver_settings=settings,
                    replay_constraint=invalid.constraint_value,
                )

    def test_solver_report_names_covariance_work_and_radial_restoration(self) -> None:
        result = replace(
            _successful_result(),
            radial_restoration_applied=True,
            radial_restoration_reason="finite_precision_boundary_repair",
            radial_restoration_scale=1.001,
            radial_restoration_value_evaluations=4,
            radial_restoration_gradient_evaluations=1,
            radial_restoration_covariance_passes=1,
        )
        report = direct._solver_report(
            result,
            covariance_case="pooled",
            initial_projection_gap=1.0,
            stationarity_reference_norm=1.0,
            constraint_gradient_covariance_norm=1.0,
            reporting_covariance_operator_calls=0,
        )
        self.assertEqual(
            report["covariance_work"]["solver_covariance_operator_apply_calls"],
            3,
        )
        self.assertEqual(
            report["covariance_work"]["solver_phase_factor_full_sample_passes"],
            27,
        )
        self.assertTrue(report["radial_restoration"]["applied"])
        self.assertEqual(
            report["radial_restoration"]["phase_factor_full_sample_passes"], 9
        )

        dense_report = direct._solver_report(
            result,
            covariance_case="pooled",
            initial_projection_gap=1.0,
            stationarity_reference_norm=1.0,
            constraint_gradient_covariance_norm=1.0,
            reporting_covariance_operator_calls=2,
            covariance_backend="precompiled_dense_covariance",
            covariance_source_phase_count=36,
        )
        self.assertEqual(
            dense_report["covariance_work"]["solver_phase_factor_full_sample_passes"],
            0,
        )
        self.assertEqual(
            dense_report["covariance_work"][
                "solver_dense_covariance_matrix_apply_calls"
            ],
            3,
        )
        self.assertEqual(
            dense_report["covariance_work"][
                "reporting_dense_covariance_matrix_apply_calls"
            ],
            2,
        )
        self.assertEqual(
            dense_report["radial_restoration"]["dense_covariance_matrix_apply_calls"],
            1,
        )

    def test_continuation_runtime_totals_sum_resumed_attempts_once(self) -> None:
        records = (
            {
                "runtime": {
                    "solver_wall_seconds": 2.5,
                    "release_only_forward_replays": 3,
                    "release_only_reverse_sweeps": 2,
                    "reverse_fortran_wall_seconds": 1.25,
                }
            },
            {
                "runtime": {
                    "solver_wall_seconds": 4.0,
                    "release_only_forward_replays": 5,
                    "release_only_reverse_sweeps": 4,
                    "reverse_fortran_wall_seconds": 2.75,
                }
            },
        )
        self.assertEqual(
            direct._continuation_runtime_totals(records),
            {
                "solver_wall_seconds": 6.5,
                "release_only_forward_replays": 8,
                "release_only_reverse_sweeps": 6,
                "reverse_fortran_wall_seconds": 4.0,
            },
        )

    def test_completed_cache_rejects_failed_solver_report(self) -> None:
        identity: dict[str, object] = {
            "run_identity_sha256": "a" * 64,
            "release_input_index": 399_095,
            "initial_state_sha256": "b" * 64,
            "covariance_case": "pooled",
            "solver_settings": _solver_settings(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "pooled"
            case_dir.mkdir()
            report = _valid_case_report(case_dir, identity)
            direct.write_json(case_dir / "report.json", report, overwrite=False)
            self.assertIsNotNone(direct._load_completed_case(case_dir, identity))
            report["solver"]["success"] = False  # type: ignore[index]
            direct.write_json(case_dir / "report.json", report, overwrite=True)
            with self.assertRaisesRegex(ValueError, "publication gates"):
                direct._load_completed_case(case_dir, identity)

    def test_completed_cache_requires_consistent_final_runtime(self) -> None:
        identity: dict[str, object] = {
            "run_identity_sha256": "a" * 64,
            "release_input_index": 399_095,
            "initial_state_sha256": "b" * 64,
            "covariance_case": "pooled",
            "solver_settings": _solver_settings(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "pooled"
            case_dir.mkdir()
            report = _valid_case_report(case_dir, identity)
            report["runtime"]["total_wall_seconds"] = 99.0  # type: ignore[index]
            direct.write_json(case_dir / "report.json", report, overwrite=False)
            with self.assertRaisesRegex(ValueError, "publication gates"):
                direct._load_completed_case(case_dir, identity)

    def test_canonical_o3_frozen_control_transfer_is_reported_not_gated(self) -> None:
        canonical, canonical_baseline = _tiny_paths(2.0, 1)
        differentiable, differentiable_baseline = _tiny_paths(3.0, 2)
        with patch.object(
            direct,
            "nino3_from_real_state",
            side_effect=lambda values: np.float32(values[0]),
        ):
            report, release, nino3 = direct._compiler_transfer_audit(
                canonical_replay=canonical,
                differentiable_replay=differentiable,
                canonical_baseline=canonical_baseline,
                differentiable_baseline=differentiable_baseline,
                observation_chain=_TinyObservationChain(),  # type: ignore[arg-type]
                direction=np.asarray((1.0,)),
                target_projection=1.0,
                canonical_baseline_final_nino3=0.0,
                differentiable_baseline_final_nino3=0.0,
            )
        self.assertFalse(report["used_by_optimizer"])
        self.assertFalse(report["numerical_differences_used_as_solver_gate"])
        self.assertEqual(report["canonical_o3"]["release_constraint_value"], 1.0)
        self.assertEqual(
            report["matched_o0_minus_canonical_o3"]["terminal_nino3_c"], 1.0
        )
        self.assertEqual(
            report["matched_o0_minus_canonical_o3"][
                "branch_tape_entries_different_between_compilers"
            ],
            10,
        )
        np.testing.assert_array_equal(release, np.asarray((2.0, 0.0, 0.0)))
        self.assertEqual(nino3[-1], 2.0)

    def test_case_failure_is_retryable_and_preserves_attempt_history(self) -> None:
        identity = {
            "run_identity_sha256": "a" * 64,
            "release_input_index": 399_095,
            "initial_state_sha256": "b" * 64,
            "covariance_case": "phase-specific",
            "solver_settings": _solver_settings(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "phase-specific"
            for attempt in (1, 2):
                try:
                    raise RuntimeError(
                        "tangent primal replay exceeds final3 tolerances at step 5"
                    )
                except RuntimeError as error:
                    report = direct._publish_case_failure(
                        case_dir,
                        identity=identity,
                        stage="nonlinear_covariance_action_sqp",
                        error=error,
                        runtime={"release_only_reverse_sweeps": attempt},
                    )
                self.assertEqual(report["attempt"], attempt)
                self.assertFalse(report["scientific_result_published"])
            self.assertTrue((case_dir / "failure.json").is_file())
            self.assertTrue((case_dir / "failures" / "attempt-0001.json").is_file())
            self.assertTrue((case_dir / "failures" / "attempt-0002.json").is_file())
            direct._clear_active_case_failure(case_dir)
            self.assertFalse((case_dir / "failure.json").exists())
            self.assertTrue((case_dir / "failures" / "attempt-0001.json").is_file())

    def test_failed_solver_checkpoint_is_diagnostic_not_canonical_result(self) -> None:
        identity = {
            "run_identity_sha256": "a" * 64,
            "release_input_index": 399_095,
            "initial_state_sha256": "b" * 64,
            "covariance_case": "pooled",
            "solver_settings": _solver_settings(),
        }
        invalid = replace(
            _successful_result(),
            success=False,
            status="maximum_iterations_reached",
            constraint_value=-1.0,
            primal_violation=1.0,
            covariance_stationarity_relative=0.5,
            complementarity_relative=0.5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "pooled"
            failure = direct._publish_case_failure(
                case_dir,
                identity=identity,
                stage="projection_continuation_stage_1_solver_publication_gates",
                error=RuntimeError("original gates failed"),
                runtime={},
                solver_result=invalid,
                solver_settings=_solver_settings(),
                covariance_case="pooled",
                initial_projection_gap=5.0,
                continuation_stages=({"fraction": 0.5},),
            )
            diagnostic = failure["nonpublishable_solver_diagnostic"]
            self.assertIsInstance(diagnostic, dict)
            self.assertFalse(diagnostic["scientific_result_published"])
            checkpoint = case_dir / "failures" / diagnostic["checkpoint"]["file"]
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(
                diagnostic["checkpoint"]["sha256"], direct.sha256_file(checkpoint)
            )
            with np.load(checkpoint, allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["dual_variables"], invalid.dual_variables
                )
                np.testing.assert_array_equal(
                    archive["native_interventions"], invalid.native_interventions
                )
            self.assertFalse((case_dir / "result.npz").exists())
            self.assertFalse((case_dir / "report.json").exists())

    def test_summary_counts_active_failure_without_treating_it_as_result(self) -> None:
        run_sha = "c" * 64
        identity = {
            "run_identity_sha256": run_sha,
            "release_input_index": int(direct.EXPECTED_RELEASE_INDICES[0]),
            "initial_state_sha256": "d" * 64,
            "covariance_case": "pooled",
            "solver_settings": _solver_settings(),
        }
        manifest = {
            "run_identity_sha256": run_sha,
            "run_identity": {"solver_settings": _solver_settings()},
            "selection": {"selected_indices": direct.EXPECTED_RELEASE_INDICES.tolist()},
            "agop_target": {
                "target_projection": 1.0,
                "extreme_target_nino3_c": 4.45,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            member = output / "members" / "member_01_i399095"
            direct.write_json(
                member / "baseline_report.json",
                {"release_agop_projection": 0.0, "terminal_nino3_c": 0.0},
                overwrite=False,
            )
            try:
                raise RuntimeError("invalid exact adjoint replay")
            except RuntimeError as error:
                direct._publish_case_failure(
                    member / "pooled",
                    identity=identity,
                    stage="nonlinear_covariance_action_sqp",
                    error=error,
                    runtime={},
                )
            summary = direct._finalize(output, manifest)
            self.assertEqual(summary["status"], "partial_with_failures")
            self.assertEqual(summary["completed_case_count"], 0)
            self.assertEqual(summary["failed_case_count"], 1)
            self.assertEqual(summary["pending_case_count"], 19)
            self.assertEqual(
                summary["failures"][0]["error_message"],
                "invalid exact adjoint replay",
            )
            finalized = direct.load_json(output / "run_manifest.json")
            self.assertEqual(finalized["status"], "partial_with_failures")
            self.assertEqual(finalized["finalization"]["failed_case_count"], 1)

    def test_summary_rejects_invalid_complete_report(self) -> None:
        run_sha = "e" * 64
        release_index = int(direct.EXPECTED_RELEASE_INDICES[0])
        identity: dict[str, object] = {
            "run_identity_sha256": run_sha,
            "release_input_index": release_index,
            "initial_state_sha256": "f" * 64,
            "covariance_case": "pooled",
            "solver_settings": _solver_settings(),
        }
        manifest = {
            "run_identity_sha256": run_sha,
            "run_identity": {"solver_settings": _solver_settings()},
            "selection": {"selected_indices": direct.EXPECTED_RELEASE_INDICES.tolist()},
            "agop_target": {
                "target_projection": 1.0,
                "extreme_target_nino3_c": 4.45,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            member = output / "members" / f"member_01_i{release_index}"
            direct.write_json(
                member / "baseline_report.json",
                {"release_agop_projection": 0.0, "terminal_nino3_c": 0.5},
                overwrite=False,
            )
            direct.write_json(
                member / "initial_state_report.json",
                {"packed_state": {"sha256": "f" * 64}},
                overwrite=False,
            )
            case_dir = member / "pooled"
            case_dir.mkdir()
            report = _valid_case_report(case_dir, identity)
            report["solver"]["success"] = False  # type: ignore[index]
            direct.write_json(case_dir / "report.json", report, overwrite=False)
            summary = direct._finalize(output, manifest)
            self.assertEqual(summary["completed_case_count"], 0)
            self.assertEqual(summary["failed_case_count"], 1)
            self.assertEqual(summary["pending_case_count"], 19)
            self.assertEqual(summary["status"], "partial_with_failures")
            self.assertEqual(
                summary["failures"][0]["error_type"],
                "InvalidCompletedCaseReport",
            )

    def test_pooled_only_run_finalizes_as_complete(self) -> None:
        run_sha = "9" * 64
        release_index = int(direct.EXPECTED_RELEASE_INDICES[0])
        settings = _solver_settings()
        identity: dict[str, object] = {
            "run_identity_sha256": run_sha,
            "release_input_index": release_index,
            "initial_state_sha256": "8" * 64,
            "covariance_case": "pooled",
            "solver_settings": settings,
        }
        manifest = {
            "run_identity_sha256": run_sha,
            "run_identity": {"solver_settings": settings},
            "active_cases": ["pooled"],
            "selection": {"selected_indices": [release_index]},
            "agop_target": {
                "target_projection": 1.0,
                "extreme_target_nino3_c": -2.14,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            member = output / "members" / f"member_01_i{release_index}"
            direct.write_json(
                member / "baseline_report.json",
                {"release_agop_projection": 0.0, "terminal_nino3_c": 0.5},
                overwrite=False,
            )
            direct.write_json(
                member / "initial_state_report.json",
                {"packed_state": {"sha256": "8" * 64}},
                overwrite=False,
            )
            case_dir = member / "pooled"
            case_dir.mkdir()
            direct.write_json(
                case_dir / "report.json",
                _valid_case_report(case_dir, identity),
                overwrite=False,
            )
            summary = direct._finalize(output, manifest)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["active_cases"], ["pooled"])
            self.assertEqual(summary["completed_case_count"], 1)
            self.assertEqual(summary["expected_case_count"], 1)
            self.assertEqual(summary["pending_case_count"], 0)


if __name__ == "__main__":
    unittest.main()
