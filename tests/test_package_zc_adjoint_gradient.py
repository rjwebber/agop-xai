"""Tests for release packaging of the raw packed-state ZC adjoint."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import scripts.package_zc_adjoint_gradient as PACKAGE
from scripts.package_zc_adjoint_gradient import (
    DEFAULT_MANIFEST,
    load_manifest,
    load_producer_metadata,
    package_gradient,
    read_raw_gradient,
)


class AdjointGradientPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.segments = load_manifest(DEFAULT_MANIFEST)
        cls.length = int(cls.manifest["arrays"]["real32"]["length"])
        cls.by_name = {segment["name"]: segment for segment in cls.segments}

    def test_package_preserves_fortran_indexing_and_control_semantics(self) -> None:
        gradient = np.zeros(self.length, dtype=np.float32)
        to = self.by_name["TO"]
        h1 = self.by_name["H1"]
        qeo = self.by_name["QEO"]
        to_local = 4 + 30 * 7
        gradient[to["start"] + to_local] = np.float32(-3.5)
        gradient[h1["start"] + 2] = np.float32(2.0)
        gradient[qeo["start"]] = np.float32(0.25)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "case_gradient.bin"
            gradient.astype("<f4").tofile(binary)
            report_path, archive_path, report = package_gradient(
                binary,
                byte_order="little",
            )

            self.assertTrue(report_path.is_file())
            self.assertTrue(archive_path.is_file())
            with np.load(archive_path, allow_pickle=False) as archive:
                self.assertEqual(archive["TO"].shape, (30, 34))
                self.assertTrue(archive["TO"].flags.f_contiguous)
                self.assertEqual(float(archive["TO"][4, 7]), -3.5)
                np.testing.assert_array_equal(archive["packed_gradient"], gradient)
                control_mask = archive["independent_control_mask"]
                self.assertTrue(bool(control_mask[to["start"]]))
                self.assertFalse(bool(control_mask[h1["start"]]))
                self.assertFalse(bool(control_mask[qeo["start"]]))

            loaded = json.loads(report_path.read_text())
            self.assertEqual(loaded, report)
            self.assertEqual(report["archive"]["sha256"], self._digest(archive_path))
            self.assertEqual(report["format_validation"]["status"], "passed")
            self.assertFalse(report["adjoint_certification"]["assessed_by_packager"])
            self.assertEqual(
                report["adjoint_certification"]["producer_declared_status"],
                "not_provided",
            )
            self.assertEqual(report["input"]["byte_order_requested"], "little")
            self.assertEqual(report["packed_gradient"]["nonzero_count"], 3)
            self.assertEqual(report["packed_gradient"]["max_abs"], 3.5)
            location = report["packed_gradient"]["max_abs_location"]
            self.assertEqual(location["segment"], "TO")
            self.assertEqual(location["local_python_index"], [4, 7])
            self.assertEqual(location["local_fortran_index"], [5, 8])
            self.assertFalse(
                report["coordinate_semantics"][
                    "rescaled_or_standardized_by_this_program"
                ]
            )

            segments = {item["name"]: item for item in report["segments"]}
            self.assertTrue(segments["TO"]["independent_control"])
            self.assertFalse(segments["H1"]["independent_control"])
            self.assertEqual(segments["QEO"]["activity"], "passive")
            self.assertEqual(
                report["independent_control_gradient_summary"]["nonzero_count"], 1
            )
            self.assertEqual(
                report["independent_control_mask"]["true_count"],
                int(np.count_nonzero(control_mask)),
            )
            self.assertEqual(report["by_activity"]["passive"]["nonzero_count"], 1)

    def test_big_endian_input_is_supported_explicitly(self) -> None:
        gradient = np.linspace(-1.0, 1.0, self.length, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "big.bin"
            gradient.astype(">f4").tofile(path)
            loaded = read_raw_gradient(
                path,
                expected_length=self.length,
                byte_order="big",
            )
        np.testing.assert_array_equal(loaded, gradient)

    def test_wrong_size_and_nonfinite_streams_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short = root / "short.bin"
            np.zeros(self.length - 1, dtype=np.float32).tofile(short)
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                read_raw_gradient(
                    short,
                    expected_length=self.length,
                    byte_order="native",
                )

            nonfinite = root / "nonfinite.bin"
            values = np.zeros(self.length, dtype=np.float32)
            values[123] = np.nan
            values.tofile(nonfinite)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                read_raw_gradient(
                    nonfinite,
                    expected_length=self.length,
                    byte_order="native",
                )

    def test_existing_bundle_is_not_partly_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "case.bin"
            np.zeros(self.length, dtype=np.float32).tofile(binary)
            report_path, archive_path, report = package_gradient(binary)
            self.assertEqual(report["format_validation"]["status"], "passed")
            self.assertEqual(report["packed_gradient"]["nonzero_count"], 0)
            self.assertEqual(
                report["adjoint_certification"]["producer_declared_status"],
                "not_provided",
            )
            report_before = report_path.read_bytes()
            archive_before = archive_path.read_bytes()
            with self.assertRaises(FileExistsError):
                package_gradient(binary)
            self.assertEqual(report_path.read_bytes(), report_before)
            self.assertEqual(archive_path.read_bytes(), archive_before)

    def test_two_file_publication_failure_restores_old_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "case.bin"
            output_prefix = root / "release"
            np.zeros(self.length, dtype="<f4").tofile(binary)
            report_path, archive_path, _ = package_gradient(
                binary,
                output_prefix=output_prefix,
            )
            old_report = report_path.read_bytes()
            old_archive = archive_path.read_bytes()

            changed = np.zeros(self.length, dtype="<f4")
            changed[0] = 1.0
            changed.tofile(binary)
            original_replace = PACKAGE.os.replace
            injected = False

            def fail_report_commit(source: Path, destination: Path) -> None:
                nonlocal injected
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    not injected
                    and ".bundle-incomplete-" in source_path.parent.name
                    and destination_path.name == report_path.name
                ):
                    injected = True
                    raise OSError("injected report commit failure")
                original_replace(source, destination)

            with (
                mock.patch.object(
                    PACKAGE.os, "replace", side_effect=fail_report_commit
                ),
                self.assertRaisesRegex(OSError, "injected report commit"),
            ):
                package_gradient(
                    binary,
                    output_prefix=output_prefix,
                    overwrite=True,
                )

            self.assertTrue(injected)
            self.assertEqual(report_path.read_bytes(), old_report)
            self.assertEqual(archive_path.read_bytes(), old_archive)
            self.assertEqual(list(root.glob(".release.bundle-incomplete-*")), [])
            self.assertEqual(list(root.glob(".*.backup-*")), [])

    def test_signal_after_backup_move_restores_old_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "case.bin"
            output_prefix = root / "release"
            np.zeros(self.length, dtype="<f4").tofile(binary)
            report_path, archive_path, _ = package_gradient(
                binary, output_prefix=output_prefix
            )
            old_report = report_path.read_bytes()
            old_archive = archive_path.read_bytes()
            original_replace = PACKAGE.os.replace
            injected = False

            def interrupt_after_backup(source: Path, destination: Path) -> None:
                nonlocal injected
                source_path = Path(source)
                destination_path = Path(destination)
                original_replace(source, destination)
                if (
                    not injected
                    and source_path == archive_path
                    and destination_path.parent.name.startswith(
                        ".release.bundle-incomplete-"
                    )
                ):
                    injected = True
                    raise KeyboardInterrupt("injected after backup move")

            with (
                mock.patch.object(
                    PACKAGE.os, "replace", side_effect=interrupt_after_backup
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "after backup move"),
            ):
                package_gradient(
                    binary, output_prefix=output_prefix, overwrite=True
                )

            self.assertTrue(injected)
            self.assertEqual(report_path.read_bytes(), old_report)
            self.assertEqual(archive_path.read_bytes(), old_archive)
            self.assertEqual(list(root.glob(".release.bundle-incomplete-*")), [])

    def test_signal_after_first_install_restores_old_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "case.bin"
            output_prefix = root / "release"
            np.zeros(self.length, dtype="<f4").tofile(binary)
            report_path, archive_path, _ = package_gradient(
                binary, output_prefix=output_prefix
            )
            old_report = report_path.read_bytes()
            old_archive = archive_path.read_bytes()
            changed = np.zeros(self.length, dtype="<f4")
            changed[0] = 1.0
            changed.tofile(binary)
            original_replace = PACKAGE.os.replace
            injected = False

            def interrupt_after_install(source: Path, destination: Path) -> None:
                nonlocal injected
                source_path = Path(source)
                destination_path = Path(destination)
                original_replace(source, destination)
                if (
                    not injected
                    and source_path.parent.name.startswith(
                        ".release.bundle-incomplete-"
                    )
                    and source_path.name == archive_path.name
                    and destination_path == archive_path
                ):
                    injected = True
                    raise KeyboardInterrupt("injected after first install")

            with (
                mock.patch.object(
                    PACKAGE.os, "replace", side_effect=interrupt_after_install
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "after first install"),
            ):
                package_gradient(
                    binary, output_prefix=output_prefix, overwrite=True
                )

            self.assertTrue(injected)
            self.assertEqual(report_path.read_bytes(), old_report)
            self.assertEqual(archive_path.read_bytes(), old_archive)
            self.assertEqual(list(root.glob(".release.bundle-incomplete-*")), [])

    def test_input_mutation_before_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "case.bin"
            output_prefix = root / "release"
            np.zeros(self.length, dtype="<f4").tofile(binary)
            original_write_npz = PACKAGE.write_npz

            def write_then_mutate(*args: object, **kwargs: object) -> None:
                original_write_npz(*args, **kwargs)
                changed = np.zeros(self.length, dtype="<f4")
                changed[0] = 1.0
                changed.tofile(binary)

            with (
                mock.patch.object(
                    PACKAGE, "write_npz", side_effect=write_then_mutate
                ),
                self.assertRaisesRegex(RuntimeError, "changed before commit"),
            ):
                package_gradient(binary, output_prefix=output_prefix)

            self.assertFalse((root / "release.json").exists())
            self.assertFalse((root / "release.npz").exists())
            self.assertEqual(list(root.glob(".release.bundle-incomplete-*")), [])

    def test_input_mutation_during_commit_rolls_back_old_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "case.bin"
            output_prefix = root / "release"
            np.zeros(self.length, dtype="<f4").tofile(binary)
            report_path, archive_path, _ = package_gradient(
                binary, output_prefix=output_prefix
            )
            old_report = report_path.read_bytes()
            old_archive = archive_path.read_bytes()
            changed = np.zeros(self.length, dtype="<f4")
            changed[0] = 1.0
            changed.tofile(binary)
            original_replace = PACKAGE.os.replace
            injected = False

            def mutate_after_json_install(source: Path, destination: Path) -> None:
                nonlocal injected
                original_replace(source, destination)
                if not injected and Path(destination) == report_path:
                    injected = True
                    binary.write_bytes(b"x" * binary.stat().st_size)

            with (
                mock.patch.object(
                    PACKAGE.os, "replace", side_effect=mutate_after_json_install
                ),
                self.assertRaisesRegex(RuntimeError, "changed before commit"),
            ):
                package_gradient(
                    binary,
                    output_prefix=output_prefix,
                    overwrite=True,
                )

            self.assertTrue(injected)
            self.assertEqual(report_path.read_bytes(), old_report)
            self.assertEqual(archive_path.read_bytes(), old_archive)
            self.assertEqual(list(root.glob(".release.bundle-incomplete-*")), [])

    def test_concurrent_output_change_is_preserved_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "case.bin"
            output_prefix = root / "release"
            np.zeros(self.length, dtype="<f4").tofile(binary)
            report_path, archive_path, _ = package_gradient(
                binary, output_prefix=output_prefix
            )
            old_archive = archive_path.read_bytes()
            original_write_npz = PACKAGE.write_npz

            def write_then_replace_report(*args: object, **kwargs: object) -> None:
                original_write_npz(*args, **kwargs)
                report_path.write_text("concurrent owner\n", encoding="utf-8")

            with (
                mock.patch.object(
                    PACKAGE, "write_npz", side_effect=write_then_replace_report
                ),
                self.assertRaisesRegex(RuntimeError, "changed while"),
            ):
                package_gradient(
                    binary,
                    output_prefix=output_prefix,
                    overwrite=True,
                )

            self.assertEqual(report_path.read_text(), "concurrent owner\n")
            self.assertEqual(archive_path.read_bytes(), old_archive)
            self.assertEqual(list(root.glob(".release.bundle-incomplete-*")), [])

    def test_snapshot_authentication_hashes_content_not_only_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"original")
            snapshot = PACKAGE._snapshot_file(path)
            original_stat = path.stat()
            path.write_bytes(b"mutated!")
            os.utime(
                path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            self.assertEqual(path.stat().st_size, snapshot.size)
            self.assertEqual(path.stat().st_mtime_ns, snapshot.mtime_ns)
            self.assertFalse(PACKAGE._path_matches_snapshot(path, snapshot))

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _producer_document(
        self,
        root: Path,
        *,
        status: str,
        evidence_kinds: tuple[str, ...],
    ) -> dict[str, object]:
        checkpoint = root / "initial_state.bin"
        path = root / "certified_path.bin"
        executable = root / "zc_adjoint"
        gradient = root / "producer_gradient.bin"
        checkpoint.write_bytes(b"checkpoint")
        path.write_bytes(b"path")
        executable.write_bytes(b"executable")
        np.zeros(self.length, dtype="<f4").tofile(gradient)
        evidence: list[dict[str, str]] = []
        for kind in evidence_kinds:
            artifact = root / f"{kind}.json"
            artifact.write_text(json.dumps({"kind": kind}), encoding="utf-8")
            evidence.append(
                {"kind": kind, "path": artifact.name, "sha256": self._digest(artifact)}
            )
        source_manifests: dict[str, dict[str, str]] = {}
        for kind in ("prepared", "tangent", "reverse"):
            manifest = root / f"{kind}_source_manifest.sha256"
            manifest.write_text(f"{'a' * 64}  {kind}.f\n", encoding="utf-8")
            source_manifests[kind] = {
                "path": manifest.name,
                "sha256": self._digest(manifest),
            }
        return {
            "schema_version": 2,
            "objective": {
                "name": "canonical_nino3",
                "definition": "Mean of the 66 terminal TO cells in the locked box.",
                "units": "degrees Celsius",
                "terminal_seed": "uniform 1/66 on the 66 terminal TO cells",
            },
            "transitions": 31,
            "initial_checkpoint": {
                "label": "extreme_el_nino",
                "path": checkpoint.name,
                "sha256": self._digest(checkpoint),
            },
            "certified_path": {
                "path": path.name,
                "sha256": self._digest(path),
            },
            "gradient": {
                "path": gradient.name,
                "sha256": self._digest(gradient),
            },
            "source_manifests": source_manifests,
            "reverse_executable": {
                "path": executable.name,
                "sha256": self._digest(executable),
            },
            "tapenade": {
                "version": "3.16",
                "revision": "0449e37b7da896cb2c38b63c8a34f10c87a84534",
                "archive_sha256": "d" * 64,
            },
            "certification": {
                "status": status,
                "scope": "Locked real32 31-transition scalar objective.",
                "evidence": evidence,
            },
        }

    def test_certified_producer_metadata_is_verified_and_sanitized(self) -> None:
        kinds = (
            "primal_replay",
            "tangent_reverse_dot",
            "independent_finite_difference_or_taylor",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "producer.json"
            metadata_path.write_text(
                json.dumps(
                    self._producer_document(
                        root, status="certified", evidence_kinds=kinds
                    )
                ),
                encoding="utf-8",
            )
            clean = load_producer_metadata(metadata_path)
            self.assertEqual(
                clean["certification"]["producer_declared_status"], "certified"
            )
            self.assertNotIn("path", clean["initial_checkpoint"]["artifact"])

            gradient_path = root / "producer_gradient.bin"
            _, _, report = package_gradient(
                gradient_path,
                producer_metadata_path=metadata_path,
            )
            self.assertFalse(report["adjoint_certification"]["assessed_by_packager"])
            self.assertEqual(
                report["adjoint_certification"]["producer_declared_status"],
                "certified",
            )
            self.assertEqual(
                report["producer_provenance"]["metadata_file"]["sha256"],
                self._digest(metadata_path),
            )

    def test_certified_metadata_requires_all_evidence_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "producer.json"
            metadata_path.write_text(
                json.dumps(
                    self._producer_document(
                        root,
                        status="certified",
                        evidence_kinds=("primal_replay", "tangent_reverse_dot"),
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing required validation"):
                load_producer_metadata(metadata_path)

    def test_legacy_metadata_with_unbound_manifest_hashes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._producer_document(
                root,
                status="provisional",
                evidence_kinds=("primal_replay",),
            )
            document["schema_version"] = 1
            document["source_manifests"] = {
                "prepared_sha256": "a" * 64,
                "tangent_sha256": "b" * 64,
                "reverse_sha256": "c" * 64,
            }
            metadata_path = root / "producer.json"
            metadata_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version must equal 2"):
                load_producer_metadata(metadata_path)

    def test_producer_artifact_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._producer_document(
                root,
                status="provisional",
                evidence_kinds=("primal_replay",),
            )
            document["certified_path"]["sha256"] = "0" * 64
            metadata_path = root / "producer.json"
            metadata_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_producer_metadata(metadata_path)

    def test_metadata_for_a_different_gradient_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._producer_document(
                root,
                status="provisional",
                evidence_kinds=("primal_replay",),
            )
            metadata_path = root / "producer.json"
            metadata_path.write_text(json.dumps(document), encoding="utf-8")

            different_gradient = root / "different_gradient.bin"
            values = np.zeros(self.length, dtype="<f4")
            values[0] = 1.0
            values.tofile(different_gradient)
            with self.assertRaisesRegex(ValueError, "gradient hash does not match"):
                package_gradient(
                    different_gradient,
                    producer_metadata_path=metadata_path,
                )

    def test_outputs_cannot_replace_metadata_or_verified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kinds = (
                "primal_replay",
                "tangent_reverse_dot",
                "independent_finite_difference_or_taylor",
            )
            document = self._producer_document(
                root,
                status="provisional",
                evidence_kinds=kinds,
            )
            gradient = root / "producer_gradient.bin"

            metadata_path = root / "release.json"
            metadata_path.write_text(json.dumps(document), encoding="utf-8")
            metadata_before = metadata_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "collides"):
                package_gradient(
                    gradient,
                    output_prefix=root / "release",
                    producer_metadata_path=metadata_path,
                    overwrite=True,
                )
            self.assertEqual(metadata_path.read_bytes(), metadata_before)
            self.assertFalse((root / "release.npz").exists())

            metadata_path = root / "producer.json"
            metadata_path.write_text(json.dumps(document), encoding="utf-8")
            evidence = root / "primal_replay.json"
            evidence_before = evidence.read_bytes()
            with self.assertRaisesRegex(ValueError, "collides"):
                package_gradient(
                    gradient,
                    output_prefix=root / "primal_replay",
                    producer_metadata_path=metadata_path,
                    overwrite=True,
                )
            self.assertEqual(evidence.read_bytes(), evidence_before)
            self.assertFalse((root / "primal_replay.npz").exists())

    def test_output_hardlink_to_verified_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = self._producer_document(
                root,
                status="provisional",
                evidence_kinds=("primal_replay",),
            )
            metadata_path = root / "producer.json"
            metadata_path.write_text(json.dumps(document), encoding="utf-8")
            evidence = root / "primal_replay.json"
            alias = root / "release.json"
            os_link_supported = True
            try:
                alias.hardlink_to(evidence)
            except OSError:
                os_link_supported = False
            if not os_link_supported:
                self.skipTest("hard links are not supported by this filesystem")
            evidence_before = evidence.read_bytes()
            with self.assertRaisesRegex(ValueError, "collides"):
                package_gradient(
                    root / "producer_gradient.bin",
                    output_prefix=root / "release",
                    producer_metadata_path=metadata_path,
                    overwrite=True,
                )
            self.assertEqual(evidence.read_bytes(), evidence_before)


if __name__ == "__main__":
    unittest.main()
