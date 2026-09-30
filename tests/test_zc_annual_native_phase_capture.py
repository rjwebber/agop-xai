"""Focused tests for the annual native ZC phase-cache capture."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.capture_zc_annual_native_phases import (
    CaptureSchedule,
    _audit_independent_overlap_replay,
    _audit_reference_overlap,
    _block_hash,
    _canonical_block,
    _load_progress,
    _progress_document,
    build_capture_schedule,
    parser,
)
from zc_xai.io import sha256_array, sha256_file, write_json


class AnnualCaptureScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = np.arange(4, dtype=np.int64)
        self.clocks = np.asarray((360, 720, 1080, 1440), dtype=np.int64)
        self.phases = np.arange(26, 35, dtype=np.int64)

    def test_decadal_tiling_covers_every_training_year_once(self) -> None:
        schedule = build_capture_schedule(
            self.chunks,
            self.clocks,
            spinup_steps=720,
            source_checkpoint_stride_steps=360,
            training_interval=(0, 720),
            phase_offsets=self.phases,
        )
        np.testing.assert_array_equal(schedule.source_chunk_indices, (1, 2))
        np.testing.assert_array_equal(
            schedule.source_base_input_indices, (0, 360)
        )
        np.testing.assert_array_equal(
            schedule.annual_base_input_indices, np.arange(0, 720, 36)
        )
        self.assertEqual(schedule.sample_count, 20)
        self.assertEqual(schedule.maximum_replay_offset_steps, 358)
        self.assertEqual(schedule.phase_input_indices.shape, (9, 20))
        np.testing.assert_array_equal(schedule.phase_input_indices[:, 0], self.phases)
        np.testing.assert_array_equal(
            schedule.phase_input_indices[:, -1], 684 + self.phases
        )

    def test_training_boundary_may_not_cut_through_a_source_block(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source checkpoint"):
            build_capture_schedule(
                self.chunks,
                self.clocks,
                spinup_steps=720,
                source_checkpoint_stride_steps=360,
                training_interval=(0, 700),
                phase_offsets=self.phases,
            )


class AnnualCaptureResumeTests(unittest.TestCase):
    def test_resume_rehashes_every_committed_block(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = [root / "a.npy", root / "b.npy"]
            values = []
            for position, path in enumerate(paths):
                array = np.arange(18, dtype=np.float32).reshape(6, 3) + position
                np.save(path, array, allow_pickle=False)
                values.append(np.load(path, mmap_mode="r+", allow_pickle=False))
            binding = {
                "schema_version": 1,
                "status": "in_progress",
                "schedule": {"source_checkpoint_count": 3},
            }
            completed = {0: _block_hash(_canonical_block(values, 0, 2))}
            runtime = {0: 0.25}
            progress = root / "capture_progress.json"
            write_json(
                progress,
                _progress_document(
                    binding, completed=completed, model_runtime=runtime
                ),
                overwrite=False,
            )
            loaded_completed, loaded_runtime = _load_progress(
                progress,
                expected_binding=binding,
                arrays=values,
                years_per_block=2,
            )
            self.assertEqual(loaded_completed, completed)
            self.assertEqual(loaded_runtime, runtime)
            values[0][0, 0] += 1.0
            values[0].flush()
            with self.assertRaisesRegex(ValueError, "fails SHA-256"):
                _load_progress(
                    progress,
                    expected_binding=binding,
                    arrays=values,
                    years_per_block=2,
                )


class AnnualCaptureOverlapTests(unittest.TestCase):
    def test_every_decadal_overlap_row_is_compared_bitwise(self) -> None:
        phases = np.asarray((26, 27), dtype=np.int64)
        source_bases = np.arange(0, 360, 36, dtype=np.int64)
        years_per_source = 2
        sample_count = source_bases.size * years_per_source
        state_size = 3
        rng = np.random.default_rng(4401)
        annual = [
            rng.normal(size=(sample_count, state_size)).astype(np.float32)
            for _ in phases
        ]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            records = []
            overlap_rows = np.arange(source_bases.size) * years_per_source
            for phase, values in zip(phases, annual, strict=True):
                path = root / f"native_phase_{int(phase):02d}.npy"
                np.save(path, values[overlap_rows], allow_pickle=False)
                records.append(
                    {
                        "file": path.name,
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            manifest = {
                "phase_offsets": phases.tolist(),
                "checkpoint_count": int(source_bases.size),
                "history_sha256": "h" * 64,
                "phase_replay_executable_sha256": "e" * 64,
                "base_input_indices_sha256": sha256_array(source_bases),
                "cache_files": records,
            }
            write_json(
                root / "capture_complete.json", manifest, overwrite=False
            )
            audit = _audit_reference_overlap(
                annual,
                phases=phases,
                years_per_source_checkpoint=years_per_source,
                source_base_input_indices=source_bases,
                reference_cache_dir=root,
                expected_history_sha256="h" * 64,
                expected_executable_sha256="e" * 64,
                state_size=state_size,
            )
            self.assertEqual(audit["status"], "passed_full_bitwise_overlap")
            self.assertEqual(
                audit["total_values_compared"],
                phases.size * source_bases.size * state_size,
            )
            annual[1][overlap_rows[-1], -1] += 1.0
            with self.assertRaisesRegex(RuntimeError, "full phase-27 overlap"):
                _audit_reference_overlap(
                    annual,
                    phases=phases,
                    years_per_source_checkpoint=years_per_source,
                    source_base_input_indices=source_bases,
                    reference_cache_dir=root,
                    expected_history_sha256="h" * 64,
                    expected_executable_sha256="e" * 64,
                    state_size=state_size,
                )

    def test_independent_replay_compares_only_first_year_of_each_block(self) -> None:
        phases = np.asarray((4, 7), dtype=np.int64)
        state_size = 3
        years_per_source = 2
        source_count = 3
        rng = np.random.default_rng(8821)
        annual = [
            rng.normal(size=(source_count * years_per_source, state_size)).astype(
                np.float32
            )
            for _ in phases
        ]
        schedule = CaptureSchedule(
            source_chunk_indices=np.asarray((2, 3, 4), dtype=np.int64),
            source_base_input_indices=np.asarray((0, 72, 144), dtype=np.int64),
            annual_base_input_indices=np.arange(0, 216, 36, dtype=np.int64),
            phase_input_indices=np.empty((2, 6), dtype=np.int64),
            relative_offsets=np.asarray(((4, 7), (40, 43)), dtype=np.int64),
            source_checkpoint_stride_steps=72,
            annual_base_stride_steps=36,
            years_per_source_checkpoint=years_per_source,
        )

        def fake_replay(**kwargs: object) -> tuple[int, np.ndarray, float]:
            block = int(kwargs["block_index"])
            row = block * years_per_source
            samples = np.stack(
                [values[row] for values in annual], axis=0
            )[:, None, :]
            return block, samples, 0.25 + block

        with patch(
            "scripts.capture_zc_annual_native_phases._replay_block",
            side_effect=fake_replay,
        ) as replay:
            audit = _audit_independent_overlap_replay(
                annual,
                phases=phases,
                schedule=schedule,
                history=Path("unused-history"),
                chunk_bytes=128,
                spinup_steps=360,
                layout=SimpleNamespace(compact_size=state_size),
                source_dir=Path("unused-source"),
                executable=Path("unused-executable"),
                jobs=2,
                expected_history_sha256="h" * 64,
                expected_executable_sha256="e" * 64,
            )
        self.assertEqual(
            audit["status"],
            "passed_full_bitwise_independent_overlap_replay",
        )
        self.assertEqual(audit["rows_per_phase"], source_count)
        self.assertEqual(audit["total_rows_compared"], source_count * phases.size)
        self.assertEqual(
            audit["total_values_compared"],
            source_count * phases.size * state_size,
        )
        self.assertFalse(audit["reference_cache_used"])
        self.assertEqual(audit["runtime"]["jobs"], 2)
        self.assertEqual(replay.call_count, source_count)
        for call in replay.call_args_list:
            np.testing.assert_array_equal(
                call.kwargs["relative_offsets"], phases[None, :]
            )

    def test_independent_replay_compares_float32_bit_patterns(self) -> None:
        phases = np.asarray((5,), dtype=np.int64)
        annual = [np.zeros((1, 1), dtype=np.float32)]
        schedule = CaptureSchedule(
            source_chunk_indices=np.asarray((0,), dtype=np.int64),
            source_base_input_indices=np.asarray((0,), dtype=np.int64),
            annual_base_input_indices=np.asarray((0,), dtype=np.int64),
            phase_input_indices=np.asarray(((5,),), dtype=np.int64),
            relative_offsets=np.asarray(((5,),), dtype=np.int64),
            source_checkpoint_stride_steps=36,
            annual_base_stride_steps=36,
            years_per_source_checkpoint=1,
        )
        replayed_negative_zero = np.asarray([[[-0.0]]], dtype=np.float32)
        with (
            patch(
                "scripts.capture_zc_annual_native_phases._replay_block",
                return_value=(0, replayed_negative_zero, 0.1),
            ),
            self.assertRaisesRegex(RuntimeError, "independent phase-5 overlap"),
        ):
            _audit_independent_overlap_replay(
                annual,
                phases=phases,
                schedule=schedule,
                history=Path("unused-history"),
                chunk_bytes=128,
                spinup_steps=360,
                layout=SimpleNamespace(compact_size=1),
                source_dir=Path("unused-source"),
                executable=Path("unused-executable"),
                jobs=1,
                expected_history_sha256="h" * 64,
                expected_executable_sha256="e" * 64,
            )


class AnnualCaptureParserTests(unittest.TestCase):
    def test_overlap_modes_are_mutually_exclusive(self) -> None:
        defaults = parser().parse_args([])
        self.assertFalse(defaults.independent_overlap_replay)
        self.assertEqual(
            defaults.reference_cache_dir.name,
            "native_cache",
        )
        independent = parser().parse_args(["--independent-overlap-replay"])
        self.assertTrue(independent.independent_overlap_replay)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(
                [
                    "--independent-overlap-replay",
                    "--reference-cache-dir",
                    "legacy",
                ]
            )


if __name__ == "__main__":
    unittest.main()
