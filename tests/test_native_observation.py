"""Tests for the time-aligned native-state to fresh-core4 map."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from zc_xai.data import Standardizer, ZCData
from zc_xai.native_observation import (
    CORE4_FEATURES,
    DEFAULT_STATE_MANIFEST,
    FrozenCore4ObservationChain,
    PackedZCState,
    annual_phase,
    canonical_model_time,
    core_field_from_real_state,
    load_native_state_layout,
    nino3_from_real_state,
    read_packed_zc_state,
    write_packed_zc_state,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "processed" / "zc-v3"
ARTIFACT_ROOT = ROOT / "artifacts" / "zc-v3"
NORMALIZATION_DIRECTORY = "models/core4/cnn/lead-10m/years-10000/seed-000042"
STATE_PAIR_ROOT = (
    ROOT / "outputs" / "zc_adjoint" / "kernel_replay_validation_run23_final3"
)


def synthetic_state(
    chain: FrozenCore4ObservationChain,
    *,
    nt: int,
    real32: np.ndarray | None = None,
) -> PackedZCState:
    layout = chain.layout
    integers = np.zeros(layout.integer_length, dtype=np.int32)
    integers[layout.integer_segments["NT"].start] = nt
    passive_time = np.zeros(layout.passive_time_length, dtype=np.float32)
    passive_time[layout.passive_time_segments["TD"].start] = canonical_model_time(nt)
    return PackedZCState(
        real32=(
            np.zeros(layout.real32_length, dtype=np.float32)
            if real32 is None
            else np.asarray(real32, dtype=np.float32)
        ),
        complex64=np.zeros(layout.complex64_length, dtype=np.complex64),
        real64=np.zeros(layout.real64_length, dtype=np.float64),
        integers=integers,
        passive_time=passive_time,
    )


def set_native_field(
    chain: FrozenCore4ObservationChain,
    real_state: np.ndarray,
    name: str,
    values: np.ndarray,
) -> None:
    segment = chain.layout.real_segments[name]
    native = real_state[segment.start : segment.stop].reshape(segment.shape, order="F")
    native[24:4:-1, 5:32] = values


class NativeObservationUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1287)
        self.standardizer = Standardizer(
            mean=rng.normal(size=CORE4_FEATURES).astype(np.float32),
            scale=rng.uniform(0.2, 3.0, size=CORE4_FEATURES).astype(np.float32),
            scale_floor=1.0e-6,
            count=100,
        )
        self.chain = FrozenCore4ObservationChain(self.standardizer)

    def test_raw_forward_uses_post_sst_pre_coarse_fields_and_phase(self) -> None:
        previous_real = np.zeros(self.chain.layout.real32_length, dtype=np.float32)
        post_real = np.zeros_like(previous_real)
        blocks = [
            np.arange(540, dtype=np.float32).reshape(20, 27) + offset
            for offset in (1_000.0, 2_000.0, 3_000.0, 4_000.0)
        ]
        set_native_field(self.chain, post_real, "TO", blocks[0])
        for name, block in zip(("H1", "U1", "V1"), blocks[1:], strict=True):
            set_native_field(self.chain, previous_real, name, block)

        # Deliberately conflicting fields prove the staggered boundary choice.
        set_native_field(self.chain, previous_real, "TO", -blocks[0])
        for name, block in zip(("H1", "U1", "V1"), blocks[1:], strict=True):
            set_native_field(self.chain, post_real, name, -block)

        previous = synthetic_state(self.chain, nt=42, real32=previous_real)
        post = synthetic_state(self.chain, nt=43, real32=post_real)
        raw = self.chain.raw_features(previous, post)
        spatial = raw[:-2].reshape(4, 20, 27)
        for actual, expected in zip(spatial, blocks, strict=True):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            raw[-2:], annual_phase(float(post.passive_time[0]))
        )

        expected_standardized = raw.copy()
        np.subtract(
            expected_standardized,
            self.standardizer.mean,
            out=expected_standardized,
        )
        np.divide(
            expected_standardized,
            self.standardizer.scale,
            out=expected_standardized,
        )
        np.testing.assert_array_equal(
            self.chain.forward(previous, post), expected_standardized
        )

    def test_tangent_and_transpose_satisfy_full_dot_product_identity(self) -> None:
        rng = np.random.default_rng(90125)
        previous_tangent = rng.normal(size=self.chain.layout.real32_length)
        post_tangent = rng.normal(size=self.chain.layout.real32_length)
        td_tangent = float(rng.normal())
        cotangent = rng.normal(size=CORE4_FEATURES)
        native_time = float(canonical_model_time(424_800))

        tangent = self.chain.tangent(
            previous_tangent,
            post_tangent,
            post_step_native_time_months=native_time,
            post_step_td_tangent=td_tangent,
        )
        transpose = self.chain.transpose(
            cotangent,
            post_step_native_time_months=native_time,
        )
        left = float(np.dot(tangent, cotangent))
        right = float(
            np.dot(previous_tangent, transpose.previous_real32)
            + np.dot(post_tangent, transpose.post_step_real32)
            + td_tangent * transpose.phase_td_cotangent
        )
        self.assertAlmostEqual(left, right, delta=2.0e-12 * max(1.0, abs(left)))
        self.assertEqual(np.count_nonzero(transpose.previous_passive_time), 0)
        self.assertEqual(np.count_nonzero(transpose.post_step_passive_time[1:]), 0)

    def test_default_scientific_tangent_and_transpose_hold_phase_fixed(self) -> None:
        zeros = np.zeros(self.chain.layout.real32_length, dtype=np.float64)
        tangent = self.chain.tangent(
            zeros,
            zeros,
            post_step_native_time_months=12.5,
        )
        np.testing.assert_array_equal(tangent, np.zeros(CORE4_FEATURES))

        cotangent = np.zeros(CORE4_FEATURES)
        cotangent[-2:] = (2.0, -3.0)
        transpose = self.chain.transpose(
            cotangent,
            post_step_native_time_months=12.5,
        )
        previous, post = transpose.scientific_control_pair()
        self.assertEqual(np.count_nonzero(previous), 0)
        self.assertEqual(np.count_nonzero(post), 0)
        self.assertNotEqual(transpose.phase_td_cotangent, 0.0)

    def test_reader_preserves_typed_stream_layout_and_rejects_short_state(self) -> None:
        layout = load_native_state_layout(DEFAULT_STATE_MANIFEST)
        rng = np.random.default_rng(778)
        state = PackedZCState(
            real32=rng.normal(size=layout.real32_length).astype(np.float32),
            complex64=(
                rng.normal(size=layout.complex64_length)
                + 1j * rng.normal(size=layout.complex64_length)
            ).astype(np.complex64),
            real64=rng.normal(size=layout.real64_length),
            integers=np.asarray([9, 10, 11, 1], dtype=np.int32),
            passive_time=np.asarray([3.5, 0.25], dtype=np.float32),
        )
        payload = b"".join(
            (
                state.real32.astype("<f4").tobytes(),
                state.complex64.astype("<c8").tobytes(),
                state.real64.astype("<f8").tobytes(),
                state.integers.astype("<i4").tobytes(),
                state.passive_time.astype("<f4").tobytes(),
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.bin"
            path.write_bytes(payload)
            loaded = read_packed_zc_state(path, layout=layout)
            for name in ("real32", "complex64", "real64", "integers", "passive_time"):
                np.testing.assert_array_equal(
                    getattr(loaded, name), getattr(state, name)
                )
            path.write_bytes(payload[:-1])
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                read_packed_zc_state(path, layout=layout)

    def test_atomic_writer_round_trips_and_protects_existing_output(self) -> None:
        state = synthetic_state(self.chain, nt=91)
        rng = np.random.default_rng(448)
        state.real32[:] = rng.normal(size=state.real32.size)
        state.complex64[:] = (
            rng.normal(size=state.complex64.size)
            + 1j * rng.normal(size=state.complex64.size)
        )
        state.real64[:] = 3.25
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.bin"
            written = write_packed_zc_state(path, state, layout=self.chain.layout)
            self.assertEqual(written, path.resolve())
            loaded = read_packed_zc_state(path, layout=self.chain.layout)
            for name in ("real32", "complex64", "real64", "integers", "passive_time"):
                np.testing.assert_array_equal(
                    getattr(loaded, name), getattr(state, name)
                )
            with self.assertRaises(FileExistsError):
                write_packed_zc_state(path, state, layout=self.chain.layout)

            replacement = synthetic_state(self.chain, nt=91)
            replacement.real32[0] = np.float32(17.0)
            self.chain.write_state(path, replacement, overwrite=True)
            reread = self.chain.read_state(path)
            self.assertEqual(float(reread.real32[0]), 17.0)
            self.assertFalse(any(Path(temporary).glob(".state.bin.*")))

    def test_public_core_selector_and_nino3_head_use_native_indices(self) -> None:
        real = np.zeros(self.chain.layout.real32_length, dtype=np.float32)
        field = np.arange(540, dtype=np.float32).reshape(20, 27)
        set_native_field(self.chain, real, "TO", field)
        np.testing.assert_array_equal(
            core_field_from_real_state(real, "TO", layout=self.chain.layout),
            field,
        )
        # The scalar head accumulates longitudes outside native latitudes.
        expected = 0.0
        native_rows = field[12:6:-1, 15:26]
        for column in range(native_rows.shape[1]):
            for row in range(native_rows.shape[0]):
                expected += float(native_rows[row, column])
        self.assertEqual(
            nino3_from_real_state(real, layout=self.chain.layout),
            np.float32(expected / 66.0),
        )

    def test_transition_validation_rejects_wrong_step_or_time(self) -> None:
        previous = synthetic_state(self.chain, nt=10)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.chain.raw_features(previous, synthetic_state(self.chain, nt=12))
        post = synthetic_state(self.chain, nt=11)
        post.passive_time[0] = np.float32(post.passive_time[0] + 1.0)
        with self.assertRaisesRegex(ValueError, "canonical fresh-run NT"):
            self.chain.raw_features(previous, post)


class ReleasedNativeObservationTests(unittest.TestCase):
    def test_both_event_state_pairs_match_released_raw_and_standardized_inputs_bitwise(
        self,
    ) -> None:
        completion = ARTIFACT_ROOT / NORMALIZATION_DIRECTORY / "completed.json"
        required = [
            DATA_ROOT / "metadata.json",
            completion,
            DEFAULT_STATE_MANIFEST,
        ]
        for label in ("extreme_el_nino", "extreme_la_nina"):
            state_dir = STATE_PAIR_ROOT / f"{label}_001step" / "kernel"
            required.extend(
                (
                    state_dir / "kernel_initial_state.bin",
                    state_dir / "kernel_final_state.bin",
                )
            )
        if any(not path.is_file() for path in required):
            self.skipTest(
                "released data, normalizer, or authentic event state pairs missing"
            )

        normalization_digest = json.loads(completion.read_text(encoding="utf-8"))[
            "files"
        ]["normalization.npz"]
        chain = FrozenCore4ObservationChain.from_recorded_artifact(
            DATA_ROOT,
            ARTIFACT_ROOT,
            NORMALIZATION_DIRECTORY,
            expected_normalization_sha256=normalization_digest,
        )
        data = ZCData(DATA_ROOT, input_profile="core4")
        metadata = json.loads((DATA_ROOT / "metadata.json").read_text(encoding="utf-8"))
        events = {item["label"]: item for item in metadata["event_restart_checkpoints"]}

        self.assertEqual(set(events), {"extreme_el_nino", "extreme_la_nina"})
        for label, event in events.items():
            with self.subTest(label=label):
                state_dir = STATE_PAIR_ROOT / f"{label}_001step" / "kernel"
                previous = chain.read_state(state_dir / "kernel_initial_state.bin")
                post = chain.read_state(state_dir / "kernel_final_state.bin")
                index = int(event["input_index"])
                expected_raw = data.load_inputs(np.asarray([index]))[0]
                expected_standardized = data.load_inputs(
                    np.asarray([index]), standardizer=chain.standardizer
                )[0]
                np.testing.assert_array_equal(
                    chain.raw_features(previous, post).view(np.uint32),
                    expected_raw.view(np.uint32),
                )
                np.testing.assert_array_equal(
                    chain.forward(previous, post).view(np.uint32),
                    expected_standardized.view(np.uint32),
                )
                self.assertEqual(
                    nino3_from_real_state(
                        post.real32, layout=chain.layout
                    ).view(np.uint32),
                    np.asarray(data.target[index], dtype=np.float32).view(np.uint32),
                )


if __name__ == "__main__":
    unittest.main()
