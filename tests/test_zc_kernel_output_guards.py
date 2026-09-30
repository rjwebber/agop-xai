"""Destructive-output safety tests for the explicit-state kernel tools."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


BUILD = _load_module(
    "zc_kernel_build_safety",
    ROOT / "adjoint/fortran_kernel/build_kernel.py",
)
VALIDATE = _load_module(
    "zc_kernel_replay_safety",
    ROOT / "adjoint/fortran_kernel/validate_replay.py",
)


class TinyManifestGenerator:
    SCRIPT_VERSION = "test-generator-1"
    GENERATION_SCHEMA_VERSION = "test-schema-1"

    @staticmethod
    def source_manifest(source_dir: Path) -> dict[str, str]:
        return {
            str(path.relative_to(source_dir)): BUILD.sha256_file(path)
            for path in sorted(source_dir.rglob("*"))
            if path.is_file()
        }

    @staticmethod
    def manifest_sha256(manifest: dict[str, str]) -> str:
        payload = json.dumps(manifest, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


class KernelBuildOutputGuardTests(unittest.TestCase):
    def test_rejects_source_relationships_and_repository_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "upstream-source"
            source.mkdir(parents=True)
            for unsafe in (
                project,
                project.parent,
                source,
                source / "nested-build",
            ):
                with self.subTest(build=unsafe), self.assertRaises(ValueError):
                    BUILD.guarded_validate_build(
                        unsafe,
                        overwrite=True,
                        source_input=source,
                        project_root=project,
                    )

    def test_requires_marker_before_accepting_nonempty_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            build.mkdir(parents=True)
            sentinel = build / "keep.txt"
            sentinel.write_text("user data\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                BUILD.guarded_validate_build(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data\n")

            (build / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            BUILD.guarded_validate_build(
                build,
                overwrite=True,
                source_input=source,
                project_root=project,
            )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data\n")
            self.assertEqual(
                (build / BUILD.BUILD_MARKER).read_text(encoding="utf-8"),
                BUILD.BUILD_MARKER_CONTENT,
            )

    def test_transactional_build_failure_preserves_existing_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            build.mkdir(parents=True)
            (build / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            sentinel = build / "known-good.txt"
            sentinel.write_text("old build\n", encoding="utf-8")

            def fail_after_writing_staging(staging: Path) -> None:
                (staging / "new-build.txt").write_text("incomplete\n")
                raise RuntimeError("injected staged-build failure")

            with self.assertRaisesRegex(RuntimeError, "injected staged-build"):
                BUILD.build_transactionally(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                    build_action=fail_after_writing_staging,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "old build\n")
            self.assertFalse((build / "new-build.txt").exists())
            self.assertEqual(
                list(build.parent.glob(f".{build.name}.staging-*")), []
            )
            self.assertEqual(list(build.parent.glob(f".{build.name}.backup-*")), [])

    def test_publish_failure_restores_existing_build_and_cleans_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            build.mkdir(parents=True)
            (build / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            sentinel = build / "known-good.txt"
            sentinel.write_text("old build\n", encoding="utf-8")

            def complete_staged_build(staging: Path) -> str:
                (staging / "new-build.txt").write_text("complete\n")
                return "complete"

            original_replace = BUILD.os.replace
            injected = False

            def fail_staging_publish(source_path: Path, destination: Path) -> None:
                nonlocal injected
                source_object = Path(source_path)
                if (
                    not injected
                    and ".staging-" in source_object.name
                    and Path(destination).name == build.name
                ):
                    injected = True
                    raise OSError("injected atomic-publish failure")
                original_replace(source_path, destination)

            with (
                mock.patch.object(
                    BUILD.os, "replace", side_effect=fail_staging_publish
                ),
                self.assertRaisesRegex(OSError, "injected atomic-publish"),
            ):
                BUILD.build_transactionally(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                    build_action=complete_staged_build,
                )

            self.assertTrue(injected)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "old build\n")
            self.assertFalse((build / "new-build.txt").exists())
            self.assertEqual(
                list(build.parent.glob(f".{build.name}.staging-*")), []
            )
            self.assertEqual(list(build.parent.glob(f".{build.name}.backup-*")), [])
            self.assertEqual(
                list(build.parent.glob(f".{build.name}.failed-publish-*")), []
            )

    def test_signal_after_target_backup_restores_existing_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            build.mkdir(parents=True)
            (build / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            sentinel = build / "known-good.txt"
            sentinel.write_text("old build\n", encoding="utf-8")
            original_replace = BUILD.os.replace
            injected = False

            def interrupt_after_backup(source_path: Path, destination: Path) -> None:
                nonlocal injected
                source_object = Path(source_path)
                destination_object = Path(destination)
                original_replace(source_path, destination)
                if (
                    not injected
                    and source_object.name == build.name
                    and ".backup-" in destination_object.name
                ):
                    injected = True
                    raise KeyboardInterrupt("injected after target backup")

            def complete_staged_build(staging: Path) -> None:
                (staging / "new-build.txt").write_text("complete\n")

            with (
                mock.patch.object(
                    BUILD.os, "replace", side_effect=interrupt_after_backup
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "after target backup"),
            ):
                BUILD.build_transactionally(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                    build_action=complete_staged_build,
                )

            self.assertTrue(injected)
            self.assertEqual(sentinel.read_text(), "old build\n")
            self.assertEqual(list(build.parent.glob(f".{build.name}.*-*")), [])

    def test_signal_after_staged_install_restores_existing_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            build.mkdir(parents=True)
            (build / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            sentinel = build / "known-good.txt"
            sentinel.write_text("old build\n", encoding="utf-8")
            original_replace = BUILD.os.replace
            injected = False

            def interrupt_after_install(source_path: Path, destination: Path) -> None:
                nonlocal injected
                source_object = Path(source_path)
                destination_object = Path(destination)
                original_replace(source_path, destination)
                if (
                    not injected
                    and ".staging-" in source_object.name
                    and destination_object.name == build.name
                ):
                    injected = True
                    raise KeyboardInterrupt("injected after staged install")

            def complete_staged_build(staging: Path) -> None:
                (staging / "new-build.txt").write_text("complete\n")

            with (
                mock.patch.object(
                    BUILD.os, "replace", side_effect=interrupt_after_install
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "after staged install"),
            ):
                BUILD.build_transactionally(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                    build_action=complete_staged_build,
                )

            self.assertTrue(injected)
            self.assertEqual(sentinel.read_text(), "old build\n")
            self.assertFalse((build / "new-build.txt").exists())
            self.assertEqual(list(build.parent.glob(f".{build.name}.*-*")), [])

    def test_staging_mutation_after_action_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            build.mkdir(parents=True)
            (build / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            sentinel = build / "known-good.txt"
            sentinel.write_text("old build\n", encoding="utf-8")
            original_publish = BUILD._publish_staged_build

            def mutate_then_publish(
                staging: Path, target: Path, **kwargs: object
            ) -> None:
                (staging / "new-build.txt").write_text("mutated\n")
                original_publish(staging, target, **kwargs)

            def complete_staged_build(staging: Path) -> None:
                (staging / "new-build.txt").write_text("complete\n")

            with (
                mock.patch.object(
                    BUILD,
                    "_publish_staged_build",
                    side_effect=mutate_then_publish,
                ),
                self.assertRaisesRegex(RuntimeError, "staging directory content"),
            ):
                BUILD.build_transactionally(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                    build_action=complete_staged_build,
                )

            self.assertEqual(sentinel.read_text(), "old build\n")
            self.assertEqual(list(build.parent.glob(f".{build.name}.*-*")), [])

    def test_concurrent_target_content_change_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            build.mkdir(parents=True)
            (build / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            sentinel = build / "known-good.txt"
            sentinel.write_text("old build\n", encoding="utf-8")

            def change_target_during_build(staging: Path) -> None:
                (staging / "new-build.txt").write_text("complete\n")
                sentinel.write_text("concurrent build\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "content changed while"):
                BUILD.build_transactionally(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                    build_action=change_target_during_build,
                )

            self.assertEqual(sentinel.read_text(), "concurrent build\n")
            self.assertFalse((build / "new-build.txt").exists())
            self.assertEqual(list(build.parent.glob(f".{build.name}.*-*")), [])

    def test_staged_source_mutation_is_rejected_without_replacing_build(self) -> None:
        class TinyGenerator:
            @staticmethod
            def source_manifest(source_dir: Path) -> dict[str, str]:
                return {
                    path.name: BUILD.sha256_file(path)
                    for path in source_dir.iterdir()
                    if path.is_file()
                }

            @staticmethod
            def manifest_sha256(manifest: dict[str, str]) -> str:
                return repr(sorted(manifest.items()))

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            source_file = source / "model.F"
            source_file.write_text("original source\n", encoding="utf-8")
            build.mkdir(parents=True)
            (build / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            sentinel = build / "known-good.txt"
            sentinel.write_text("old build\n", encoding="utf-8")
            generator = TinyGenerator()
            expected_manifest = generator.source_manifest(source)
            expected_sha256 = generator.manifest_sha256(expected_manifest)

            def mutate_after_copy(staging: Path) -> None:
                snapshot = staging / "source"
                snapshot.mkdir()
                staged_file = snapshot / source_file.name
                staged_file.write_bytes(source_file.read_bytes())
                staged_file.write_text("changed after preflight\n", encoding="utf-8")
                BUILD.verify_staged_source_manifest(
                    snapshot,
                    expected_manifest=expected_manifest,
                    expected_sha256=expected_sha256,
                    generator=generator,
                )

            with self.assertRaisesRegex(RuntimeError, "differs from the preflight"):
                BUILD.build_transactionally(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                    build_action=mutate_after_copy,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "old build\n")
            self.assertEqual(
                list(build.parent.glob(f".{build.name}.staging-*")), []
            )
            self.assertEqual(list(build.parent.glob(f".{build.name}.backup-*")), [])

    def test_symlinked_marker_cannot_authorize_build_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            build = project / "outputs/build"
            source.mkdir(parents=True)
            build.mkdir(parents=True)
            sentinel = build / "keep.txt"
            sentinel.write_text("user data\n", encoding="utf-8")
            marker_target = project / "marker-content.txt"
            marker_target.write_text(BUILD.BUILD_MARKER_CONTENT, encoding="utf-8")
            (build / BUILD.BUILD_MARKER).symlink_to(marker_target)

            with self.assertRaises(ValueError):
                BUILD.guarded_validate_build(
                    build,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                )
            self.assertEqual(sentinel.read_text(), "user data\n")
            self.assertEqual(marker_target.read_text(), BUILD.BUILD_MARKER_CONTENT)

    def test_rejects_build_symlink_and_preserves_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "source"
            target = project / "outputs/owned-target"
            link = project / "outputs/apparent-build"
            source.mkdir(parents=True)
            target.mkdir(parents=True)
            sentinel = target / "keep.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            (target / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            link.symlink_to(target, target_is_directory=True)

            with self.assertRaises(ValueError):
                BUILD.guarded_validate_build(
                    link,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me\n")

    def test_rejects_outside_and_parent_symlink_builds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            project = temporary_root / "project"
            source = project / "source"
            source.mkdir(parents=True)

            outside = temporary_root / "outside-build"
            outside.mkdir()
            outside_sentinel = outside / "keep.txt"
            outside_sentinel.write_text("outside\n", encoding="utf-8")
            (outside / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "strict descendant"):
                BUILD.guarded_validate_build(
                    outside,
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                )
            self.assertEqual(outside_sentinel.read_text(), "outside\n")

            target_parent = project / "real-target"
            target = target_parent / "child"
            target.mkdir(parents=True)
            target_sentinel = target / "keep.txt"
            target_sentinel.write_text("linked\n", encoding="utf-8")
            (target / BUILD.BUILD_MARKER).write_text(
                BUILD.BUILD_MARKER_CONTENT, encoding="utf-8"
            )
            apparent_parent = project / "outputs/parent-link"
            apparent_parent.parent.mkdir()
            apparent_parent.symlink_to(target_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                BUILD.guarded_validate_build(
                    apparent_parent / "child",
                    overwrite=True,
                    source_input=source,
                    project_root=project,
                )
            self.assertEqual(target_sentinel.read_text(), "linked\n")


class ReplayBuildProvenanceTests(unittest.TestCase):
    def test_replay_rejects_post_stage_input_mutation(self) -> None:
        validator_source = (
            ROOT / "adjoint/fortran_kernel/validate_replay.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(
            validator_source.count("verify_staged_validation_inputs("), 3
        )
        for victim in ("source", "reference", "kernel", "metadata", "checkpoint"):
            with (
                self.subTest(victim=victim),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                build = root / "build"
                source = build / "source"
                source.mkdir(parents=True)
                (source / "model.F").write_text("model\n", encoding="utf-8")
                reference = build / "zc_reference"
                kernel = build / "zc_kernel_replay"
                reference.write_bytes(b"reference")
                kernel.write_bytes(b"kernel")
                metadata = root / "metadata.json"
                metadata.write_text("{}\n", encoding="utf-8")
                restart = root / "case.hst"
                restart.write_bytes(b"restart")
                output = root / "output"
                output.mkdir()
                provenance = {
                    "staged_source_file_sha256": TinyManifestGenerator.source_manifest(
                        source
                    ),
                    "reference_executable_sha256": VALIDATE.sha256_file(reference),
                    "kernel_executable_sha256": VALIDATE.sha256_file(kernel),
                }
                staged = VALIDATE.stage_validation_inputs(
                    output_dir=output,
                    build_dir=build,
                    metadata_path=metadata,
                    checkpoint_cases=[
                        {
                            "label": "case",
                            "path": restart,
                            "expected_sha256": VALIDATE.sha256_file(restart),
                        }
                    ],
                    build_provenance=provenance,
                    generator=TinyManifestGenerator,
                )
                (
                    staged_source,
                    staged_reference,
                    staged_kernel,
                    staged_metadata,
                    cases,
                ) = staged
                binding = VALIDATE.verify_staged_validation_inputs(
                    source_dir=staged_source,
                    reference_executable=staged_reference,
                    kernel_executable=staged_kernel,
                    metadata_path=staged_metadata,
                    checkpoint_cases=cases,
                    build_provenance=provenance,
                    metadata_sha256=VALIDATE.sha256_file(metadata),
                    generator=TinyManifestGenerator,
                )
                self.assertEqual(
                    binding["executable_sha256"]["zc_kernel_replay"],
                    provenance["kernel_executable_sha256"],
                )
                victims = {
                    "source": staged_source / "model.F",
                    "reference": staged_reference,
                    "kernel": staged_kernel,
                    "metadata": staged_metadata,
                    "checkpoint": Path(cases[0]["path"]),
                }
                victims[victim].write_bytes(b"mutated")
                with self.assertRaisesRegex(RuntimeError, "staged replay"):
                    VALIDATE.verify_staged_validation_inputs(
                        source_dir=staged_source,
                        reference_executable=staged_reference,
                        kernel_executable=staged_kernel,
                        metadata_path=staged_metadata,
                        checkpoint_cases=cases,
                        build_provenance=provenance,
                        metadata_sha256=VALIDATE.sha256_file(metadata),
                        generator=TinyManifestGenerator,
                    )

    def test_validation_stages_only_the_authenticated_source_inventory(self) -> None:
        class RuntimeGenerator(TinyManifestGenerator):
            @staticmethod
            def source_manifest(source_dir: Path) -> dict[str, str]:
                return {
                    path.relative_to(source_dir).as_posix(): BUILD.sha256_file(path)
                    for path in sorted(source_dir.rglob("*"))
                    if path.is_file()
                    and path.suffix != ".o"
                    and path.name not in {".DS_Store", "zeqfc1"}
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            source = build / "source"
            source.mkdir(parents=True)
            model = source / "model.F"
            model.write_text("model\n", encoding="utf-8")
            (source / "stale.o").write_bytes(b"unbound object")
            (source / "zeqfc1").write_bytes(b"unbound executable")
            (source / ".DS_Store").write_bytes(b"unbound metadata")
            reference = build / "zc_reference"
            kernel = build / "zc_kernel_replay"
            reference.write_bytes(b"reference")
            kernel.write_bytes(b"kernel")
            metadata = root / "metadata.json"
            metadata.write_text("{}\n", encoding="utf-8")
            restart = root / "case.hst"
            restart.write_bytes(b"restart")
            output = root / "output"
            output.mkdir()
            expected_source = RuntimeGenerator.source_manifest(source)
            provenance = {
                "staged_source_file_sha256": expected_source,
                "reference_executable_sha256": VALIDATE.sha256_file(reference),
                "kernel_executable_sha256": VALIDATE.sha256_file(kernel),
            }
            staged_source, _, _, _, _ = VALIDATE.stage_validation_inputs(
                output_dir=output,
                build_dir=build,
                metadata_path=metadata,
                checkpoint_cases=[
                    {
                        "label": "case",
                        "path": restart,
                        "expected_sha256": VALIDATE.sha256_file(restart),
                    }
                ],
                build_provenance=provenance,
                generator=RuntimeGenerator,
            )
            self.assertTrue((staged_source / "model.F").is_file())
            self.assertFalse((staged_source / "stale.o").exists())
            self.assertFalse((staged_source / "zeqfc1").exists())
            self.assertFalse((staged_source / ".DS_Store").exists())
            self.assertEqual(
                RuntimeGenerator.source_manifest(staged_source), expected_source
            )
            self.assertEqual(
                {
                    path.relative_to(staged_source).as_posix(): (
                        VALIDATE.sha256_file(path)
                    )
                    for path in staged_source.rglob("*")
                    if path.is_file()
                },
                expected_source,
            )

            model.write_text("mutated\n", encoding="utf-8")
            output_after_mutation = root / "output-after-mutation"
            output_after_mutation.mkdir()
            with self.assertRaisesRegex(RuntimeError, "input changed before staging"):
                VALIDATE.stage_validation_inputs(
                    output_dir=output_after_mutation,
                    build_dir=build,
                    metadata_path=metadata,
                    checkpoint_cases=[
                        {
                            "label": "case",
                            "path": restart,
                            "expected_sha256": VALIDATE.sha256_file(restart),
                        }
                    ],
                    build_provenance=provenance,
                    generator=RuntimeGenerator,
                )

    def test_verified_file_copy_rejects_changed_staged_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"expected")
            expected = VALIDATE.sha256_file(source)
            original_copy = VALIDATE.shutil.copy2

            def corrupt_copy(source_path: Path, destination_path: Path) -> None:
                original_copy(source_path, destination_path)
                Path(destination_path).write_bytes(b"corrupt")

            with (
                mock.patch.object(
                    VALIDATE.shutil,
                    "copy2",
                    side_effect=corrupt_copy,
                ),
                self.assertRaisesRegex(RuntimeError, "staged input differs"),
            ):
                VALIDATE.copy_verified_file(
                    source,
                    destination,
                    expected_sha256=expected,
                )

    def _fixture(
        self, root: Path
    ) -> tuple[Path, TinyManifestGenerator, Path, Path]:
        build = root / "build"
        source = build / "source"
        source.mkdir(parents=True)
        (source / "model.F").write_text("model source\n", encoding="utf-8")
        generator = TinyManifestGenerator()
        source_manifest = generator.source_manifest(source)
        source_manifest_path = build / "source_manifest.json"
        source_manifest_path.write_text(
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_manifest_digest_path = build / "source_manifest.sha256"
        source_manifest_digest_path.write_text(
            generator.manifest_sha256(source_manifest) + "\n",
            encoding="utf-8",
        )
        reference = build / "zc_reference"
        kernel = build / "zc_kernel_replay"
        reference.write_bytes(b"reference executable")
        kernel.write_bytes(b"kernel executable")
        builder = root / "build_kernel.py"
        generator_path = root / "generate_fresh_zc_dataset.py"
        builder.write_text("builder\n", encoding="utf-8")
        generator_path.write_text("generator\n", encoding="utf-8")
        report = {
            "schema_version": VALIDATE.EXPECTED_BUILD_REPORT_SCHEMA,
            "kernel_version": VALIDATE.EXPECTED_KERNEL_VERSION,
            "reference_executable_sha256": VALIDATE.sha256_file(reference),
            "kernel_executable_sha256": VALIDATE.sha256_file(kernel),
            "source_manifest_file_sha256": VALIDATE.sha256_file(
                source_manifest_path
            ),
            "source_manifest_digest_file_sha256": VALIDATE.sha256_file(
                source_manifest_digest_path
            ),
            "source_manifest_sha256": generator.manifest_sha256(source_manifest),
            "fresh_patch_sha256": {},
            "kernel_patch_sha256": {},
            "kernel_support_source_sha256": {},
            "builder_source_sha256": {
                "build_kernel.py": VALIDATE.sha256_file(builder),
                "generate_fresh_zc_dataset.py": VALIDATE.sha256_file(
                    generator_path
                ),
            },
            "generator": {
                "script_version": generator.SCRIPT_VERSION,
                "generation_schema_version": generator.GENERATION_SCHEMA_VERSION,
            },
        }
        (build / "build_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return build, generator, builder, generator_path

    def _validate(
        self, fixture: tuple[Path, TinyManifestGenerator, Path, Path]
    ) -> dict[str, object]:
        build, generator, builder, generator_path = fixture
        return VALIDATE.validate_build_provenance(
            build,
            generator=generator,
            builder_path=builder,
            generator_path=generator_path,
        )

    def test_build_report_binds_executables_source_and_live_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            evidence = self._validate(fixture)
            self.assertEqual(
                evidence["kernel_version"], VALIDATE.EXPECTED_KERNEL_VERSION
            )
            self.assertEqual(
                evidence["generator_script_version"],
                TinyManifestGenerator.SCRIPT_VERSION,
            )

    def test_tampered_executable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            (fixture[0] / "zc_kernel_replay").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "executable hash mismatch"):
                self._validate(fixture)

    def test_tampered_staged_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            (fixture[0] / "source/model.F").write_text(
                "tampered source\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "staged build source"):
                self._validate(fixture)

    def test_tampered_source_manifest_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            (fixture[0] / "source_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "source_manifest.json hash"):
                self._validate(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            (fixture[0] / "source_manifest.sha256").write_text(
                f"{'0' * 64}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "source_manifest.sha256 hash"
            ):
                self._validate(fixture)

    def test_stale_report_version_and_live_generator_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            report_path = fixture[0] / "build_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["kernel_version"] = "obsolete"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "kernel_version"):
                self._validate(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            fixture[3].write_text("changed generator\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "builder/generator hashes"):
                self._validate(fixture)

    def test_locked_checkpoint_triples_reject_mutable_labels_and_nt(self) -> None:
        reference_sha256 = "f" * 64
        metadata = {
            "schema_version": "fresh-zc-interpretability-v1",
            "script_version": "1.0.0",
            "event_restart_checkpoints": [
                {
                    "label": label,
                    "checkpoint_file": f"{label}.hst",
                    "checkpoint_sha256": VALIDATE.LOCKED_CHECKPOINTS[label][
                        "sha256"
                    ],
                    "pre_input_checkpoint_nt": VALIDATE.LOCKED_CHECKPOINTS[label][
                        "pre_nt"
                    ],
                    "event_replay_executable_sha256": reference_sha256,
                }
                for label in ("extreme_el_nino", "extreme_la_nina")
            ],
        }
        cases = VALIDATE.validate_processed_metadata(
            metadata,
            reference_executable_sha256=reference_sha256,
        )
        self.assertEqual(len(cases), 2)
        metadata["event_restart_checkpoints"][0][
            "pre_input_checkpoint_nt"
        ] += 1
        with self.assertRaisesRegex(RuntimeError, "checkpoint triple mismatch"):
            VALIDATE.validate_processed_metadata(
                metadata,
                reference_executable_sha256=reference_sha256,
            )


class ReplayValidationOutputGuardTests(unittest.TestCase):
    def test_rejects_input_relationships_and_repository_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            build = project / "build"
            data = project / "data"
            build.mkdir(parents=True)
            data.mkdir()
            for unsafe in (
                project,
                project.parent,
                build,
                build / "nested-output",
                data,
                data / "nested-output",
            ):
                with self.subTest(output=unsafe), self.assertRaises(ValueError):
                    VALIDATE.guarded_prepare_output(
                        unsafe,
                        overwrite=True,
                        protected_paths=(build, data),
                        project_root=project,
                    )

    def test_requires_marker_before_replacing_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            build = project / "build"
            data = project / "data"
            output = project / "outputs/replay"
            build.mkdir(parents=True)
            data.mkdir()
            output.mkdir(parents=True)
            sentinel = output / "keep.txt"
            sentinel.write_text("user data\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                VALIDATE.guarded_prepare_output(
                    output,
                    overwrite=True,
                    protected_paths=(build, data),
                    project_root=project,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data\n")

            (output / VALIDATE.OUTPUT_MARKER).write_text(
                VALIDATE.OUTPUT_MARKER_CONTENT, encoding="utf-8"
            )
            VALIDATE.guarded_prepare_output(
                output,
                overwrite=True,
                protected_paths=(build, data),
                project_root=project,
            )
            self.assertFalse(sentinel.exists())
            self.assertEqual(
                (output / VALIDATE.OUTPUT_MARKER).read_text(encoding="utf-8"),
                VALIDATE.OUTPUT_MARKER_CONTENT,
            )

    def test_symlinked_marker_cannot_authorize_output_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            build = project / "build"
            data = project / "data"
            output = project / "outputs/replay"
            build.mkdir(parents=True)
            data.mkdir()
            output.mkdir(parents=True)
            sentinel = output / "keep.txt"
            sentinel.write_text("user data\n", encoding="utf-8")
            marker_target = project / "marker-content.txt"
            marker_target.write_text(
                VALIDATE.OUTPUT_MARKER_CONTENT, encoding="utf-8"
            )
            (output / VALIDATE.OUTPUT_MARKER).symlink_to(marker_target)

            with self.assertRaises(ValueError):
                VALIDATE.guarded_prepare_output(
                    output,
                    overwrite=True,
                    protected_paths=(build, data),
                    project_root=project,
                )
            self.assertEqual(sentinel.read_text(), "user data\n")
            self.assertEqual(
                marker_target.read_text(), VALIDATE.OUTPUT_MARKER_CONTENT
            )

    def test_rejects_output_symlink_and_preserves_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            build = project / "build"
            data = project / "data"
            target = project / "outputs/owned-target"
            link = project / "outputs/apparent-output"
            build.mkdir(parents=True)
            data.mkdir()
            target.mkdir(parents=True)
            sentinel = target / "keep.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            (target / VALIDATE.OUTPUT_MARKER).write_text(
                VALIDATE.OUTPUT_MARKER_CONTENT, encoding="utf-8"
            )
            link.symlink_to(target, target_is_directory=True)

            with self.assertRaises(ValueError):
                VALIDATE.guarded_prepare_output(
                    link,
                    overwrite=True,
                    protected_paths=(build, data),
                    project_root=project,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me\n")

    def test_entry_points_preserve_raw_output_path_for_guard(self) -> None:
        build_source = (ROOT / "adjoint/fortran_kernel/build_kernel.py").read_text(
            encoding="utf-8"
        )
        replay_source = (
            ROOT / "adjoint/fortran_kernel/validate_replay.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("build_root = args.build_dir.resolve()", build_source)
        self.assertNotIn("output_dir = args.output_dir.resolve()", replay_source)
        self.assertIn("transactional_output(", replay_source)
        self.assertIn("with output_transaction as output_dir:", replay_source)

    def test_rejects_outside_and_parent_symlink_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            project = temporary_root / "project"
            build = project / "build"
            data = project / "data"
            build.mkdir(parents=True)
            data.mkdir()

            outside = temporary_root / "outside-validation"
            outside.mkdir()
            outside_sentinel = outside / "keep.txt"
            outside_sentinel.write_text("outside\n", encoding="utf-8")
            (outside / VALIDATE.OUTPUT_MARKER).write_text(
                VALIDATE.OUTPUT_MARKER_CONTENT, encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "strict descendant"):
                VALIDATE.guarded_prepare_output(
                    outside,
                    overwrite=True,
                    protected_paths=(build, data),
                    project_root=project,
                )
            self.assertEqual(outside_sentinel.read_text(), "outside\n")

            target_parent = project / "real-target"
            target = target_parent / "child"
            target.mkdir(parents=True)
            target_sentinel = target / "keep.txt"
            target_sentinel.write_text("linked\n", encoding="utf-8")
            (target / VALIDATE.OUTPUT_MARKER).write_text(
                VALIDATE.OUTPUT_MARKER_CONTENT, encoding="utf-8"
            )
            apparent_parent = project / "outputs/parent-link"
            apparent_parent.parent.mkdir()
            apparent_parent.symlink_to(target_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                VALIDATE.guarded_prepare_output(
                    apparent_parent / "child",
                    overwrite=True,
                    protected_paths=(build, data),
                    project_root=project,
                )
            self.assertEqual(target_sentinel.read_text(), "linked\n")


if __name__ == "__main__":
    unittest.main()
