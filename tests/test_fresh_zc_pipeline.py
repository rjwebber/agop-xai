"""Focused tests for the interpretation-first ZC data release contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from zc_xai.data import (
    CANONICAL_FIELDS,
    FRESH_FIELD_IDENTITIES,
    FRESH_NINO3_DEFINITION,
    FRESH_NORMALIZATION_SEMANTICS,
    FRESH_PHASE_FORMULA,
    FRESH_PHASE_STORAGE,
    FRESH_PROCESS_FIELDS,
    FRESH_SPLIT_SEMANTICS,
    ZCData,
)
from zc_xai.io import sha256_file


class FreshZCDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        n_steps = 12
        field_files = {}
        for index, name in enumerate(CANONICAL_FIELDS + FRESH_PROCESS_FIELDS):
            values = np.full((n_steps, 20, 27), index, dtype="<f4")
            path = self.directory / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            field_files[name] = {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": "0" * 64,
            }
        phase = np.column_stack(
            (
                np.arange(n_steps, dtype=np.float32),
                -np.arange(n_steps, dtype=np.float32),
            )
        ).astype("<f4")
        np.save(self.directory / "annual_phase_sin_cos.npy", phase, allow_pickle=False)
        np.save(
            self.directory / "nino3_index.npy",
            np.arange(n_steps, dtype="<f4"),
            allow_pickle=False,
        )
        np.save(
            self.directory / "native_time_months.npy",
            np.arange(n_steps, dtype="<f8"),
            allow_pickle=False,
        )
        auxiliary_files = {}
        for filename in (
            "annual_phase_sin_cos.npy",
            "nino3_index.npy",
            "native_time_months.npy",
        ):
            path = self.directory / filename
            auxiliary_files[filename] = {
                "size_bytes": path.stat().st_size,
                "sha256": "0" * 64,
            }
        metadata = {
            "schema_version": "fresh-zc-interpretability-v1",
            "integration": {
                "retained_steps": n_steps,
                "steps_per_month": 1,
                "steps_per_year": 12,
                "native_time_file": "native_time_months.npy",
                "native_time_units": (
                    "months; 0.5 is mid-January of nominal year 1960"
                ),
            },
            "grid": {
                "shape": [20, 27],
                "latitude_degrees_north": np.arange(-19, 20, 2).tolist(),
                "longitude_degrees_east": (129.375 + 5.625 * np.arange(27)).tolist(),
                "active_fortran_indices_one_based": {
                    "ordinary_fields": {
                        "latitude": [25, 6, -1],
                        "longitude": [6, 32, 1],
                    },
                    "HTAU_latitude_mapping": "HTAU longitude J, latitude 31-I",
                },
            },
            "field_files": field_files,
            "fields": [
                {
                    "name": name,
                    "fortran": source,
                    "group": group,
                    "role": "test role",
                    "units": "test units",
                }
                for name, source, group in FRESH_FIELD_IDENTITIES
            ],
            "phase": {
                "file": "annual_phase_sin_cos.npy",
                "shape": [n_steps, 2],
                "formula": FRESH_PHASE_FORMULA,
                "storage": FRESH_PHASE_STORAGE,
            },
            "nino3": {
                "file": "nino3_index.npy",
                "definition": FRESH_NINO3_DEFINITION,
                "grid_point_count": 66,
                "includes_270E_90W_center": True,
                "latitude_degrees_north": [-5, -3, -1, 1, 3, 5],
                "longitude_degrees_east": (213.75 + 5.625 * np.arange(11)).tolist(),
            },
            "auxiliary_files": auxiliary_files,
            "chronological_split": {
                "train": [0, 6],
                "validation": [6, 9],
                "test": [9, 12],
                "semantics": FRESH_SPLIT_SEMANTICS,
                "normalization": FRESH_NORMALIZATION_SEMANTICS,
                "testing_fixture": True,
            },
        }
        self.metadata_path = self.directory / "metadata.json"
        self.metadata_path.write_text(json.dumps(metadata))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def metadata(self) -> dict[str, object]:
        return json.loads(self.metadata_path.read_text())

    def write_metadata(self, metadata: dict[str, object]) -> None:
        self.metadata_path.write_text(json.dumps(metadata))

    def test_core_profile_is_flat_spatial_prefix_plus_two_phase_coordinates(
        self,
    ) -> None:
        data = ZCData(self.directory, input_profile="core4")
        self.assertEqual(data.spatial_input_shape, (4, 20, 27))
        self.assertEqual(data.input_shape, (4 * 20 * 27 + 2,))
        self.assertEqual(data.n_phase_features, 2)
        values = data.load_inputs(np.asarray([3], dtype=np.int64))
        self.assertEqual(values.shape, (1, 2162))
        expected_spatial_constants = (0.0, 5.0, 6.0, 7.0)
        spatial = values[0, :-2].reshape(4, 20, 27)
        for index, expected in enumerate(expected_spatial_constants):
            self.assertTrue(np.all(spatial[index] == expected))
        np.testing.assert_array_equal(values[0, -2:], np.asarray([3.0, -3.0]))
        self.assertFalse(data.checksum_verified)

    def test_profiles_and_fixed_blocks_are_deterministic(self) -> None:
        expected_channels = {"core4": 4, "core4_wp": 5, "legacy10": 10, "all13": 13}
        for profile, channels in expected_channels.items():
            data = ZCData(self.directory, input_profile=profile)
            self.assertEqual(data.spatial_input_shape, (channels, 20, 27))
        split = ZCData(self.directory, input_profile="core4").fixed_supervised_split(1)
        np.testing.assert_array_equal(split.train_inputs, np.arange(5))
        np.testing.assert_array_equal(split.validation_inputs, np.arange(6, 8))
        np.testing.assert_array_equal(split.test_inputs, np.arange(9, 11))
        self.assertEqual(split.train_block, (0, 6))
        self.assertEqual(split.validation_block, (6, 9))
        self.assertEqual(split.test_block, (9, 12))

    def test_standardizer_treats_phase_as_two_coordinates(self) -> None:
        data = ZCData(self.directory, input_profile="core4")
        standardizer = data.compute_standardizer(np.arange(6), batch_size=2)
        self.assertEqual(standardizer.mean.shape, (2162,))
        self.assertEqual(standardizer.scale.shape, (2162,))
        transformed = data.load_inputs(np.asarray([0, 1]), standardizer=standardizer)
        self.assertEqual(transformed.shape, (2, 2162))
        self.assertTrue(np.isfinite(transformed).all())

    def test_checksum_verified_records_full_verification(self) -> None:
        metadata = self.metadata()
        for entry in metadata["field_files"].values():
            entry["sha256"] = sha256_file(self.directory / entry["file"])
        for filename, entry in metadata["auxiliary_files"].items():
            entry["sha256"] = sha256_file(self.directory / filename)
        self.write_metadata(metadata)
        data = ZCData(
            self.directory,
            input_profile="core4",
            verify_checksums=True,
        )
        self.assertTrue(data.checksum_verified)
        self.assertTrue(data.provenance()["checksum_verified"])

    def test_split_overlap_and_gap_are_rejected(self) -> None:
        baseline = self.metadata()
        invalid_validation_blocks = ([5, 9], [7, 9])
        for validation in invalid_validation_blocks:
            with self.subTest(validation=validation):
                metadata = json.loads(json.dumps(baseline))
                metadata["chronological_split"]["validation"] = validation
                self.write_metadata(metadata)
                with self.assertRaisesRegex(ValueError, "contiguous"):
                    ZCData(self.directory, input_profile="core4")
        self.write_metadata(baseline)
        data = ZCData(self.directory, input_profile="core4")
        data.metadata["chronological_split"]["validation"] = [6, 10]
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            data.fixed_supervised_split(1)

    def test_auxiliary_manifest_is_exact_and_sha_syntax_is_validated(self) -> None:
        baseline = self.metadata()
        mutations = ("missing", "extra", "bad_digest")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                metadata = json.loads(json.dumps(baseline))
                auxiliary = metadata["auxiliary_files"]
                if mutation == "missing":
                    del auxiliary["native_time_months.npy"]
                elif mutation == "extra":
                    auxiliary["other.npy"] = {
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                    }
                else:
                    auxiliary["nino3_index.npy"]["sha256"] = "not-a-digest"
                self.write_metadata(metadata)
                with self.assertRaises(ValueError):
                    ZCData(self.directory, input_profile="core4")

    def test_auxiliary_header_shape_and_dtype_are_validated(self) -> None:
        metadata = self.metadata()
        path = self.directory / "native_time_months.npy"
        np.save(path, np.arange(12, dtype="<f4"), allow_pickle=False)
        metadata["auxiliary_files"][path.name]["size_bytes"] = path.stat().st_size
        self.write_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "native time"):
            ZCData(self.directory, input_profile="core4")

    def test_scientific_semantic_mutations_are_rejected(self) -> None:
        baseline = self.metadata()
        mutations = ("grid", "nino3", "phase", "field", "split_claim")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                metadata = json.loads(json.dumps(baseline))
                if mutation == "grid":
                    metadata["grid"]["longitude_degrees_east"][0] += 1.0
                elif mutation == "nino3":
                    metadata["nino3"]["grid_point_count"] = 65
                elif mutation == "phase":
                    metadata["phase"]["formula"] = "different formula"
                elif mutation == "field":
                    metadata["fields"][6]["fortran"] = "UO"
                else:
                    metadata["chronological_split"]["normalization"] = "changed"
                self.write_metadata(metadata)
                with self.assertRaises(ValueError):
                    ZCData(self.directory, input_profile="core4")


if __name__ == "__main__":
    unittest.main()
