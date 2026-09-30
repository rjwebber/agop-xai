"""Output ownership and deletion-safety tests for full-tangent audits."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "adjoint/full_tangent_audit"
SPEC = importlib.util.spec_from_file_location(
    "zc_full_tangent_output_safety", AUDIT_ROOT / "output_safety.py"
)
assert SPEC is not None and SPEC.loader is not None
SAFETY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SAFETY)

VALIDATORS = (
    "validate_tangent.py",
    "validate_composed_tangent.py",
    "validate_generic_adjoint_dot.py",
    "validate_adjoint_dot.py",
    "validate_scalar_taylor.py",
)


class FullTangentOutputGuardTests(unittest.TestCase):
    def test_transaction_stages_in_private_system_temp_on_same_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            private_temp = project / "private-temp"
            protected.mkdir(parents=True)
            output.parent.mkdir(parents=True)
            private_temp.mkdir()

            with (
                mock.patch.object(
                    SAFETY.tempfile,
                    "gettempdir",
                    return_value=str(private_temp),
                ),
                SAFETY.transactional_output(
                    output,
                    overwrite=False,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                ) as staging,
            ):
                self.assertEqual(staging.parent.parent, private_temp)
                self.assertEqual(staging.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual(staging.stat().st_mode & 0o777, 0o700)
                self.assertEqual(
                    list(output.parent.glob(f".{output.name}.staging-*")), []
                )
                (staging / "result.txt").write_text("complete\n", encoding="utf-8")

            self.assertEqual(
                (output / "result.txt").read_text(encoding="utf-8"), "complete\n"
            )

    def test_private_staging_still_rejects_scientific_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            private_temp = project / "private-temp"
            protected.mkdir(parents=True)
            output.parent.mkdir(parents=True)
            private_temp.mkdir()
            original_tree_manifest = SAFETY._tree_manifest
            staging_path: Path | None = None
            injected = False

            def mutate_after_manifest(path: Path) -> dict[str, str]:
                nonlocal injected
                result = original_tree_manifest(path)
                if path == staging_path and not injected:
                    injected = True
                    (path / "scientific.txt").write_text(
                        "mutated\n", encoding="utf-8"
                    )
                return result

            with (
                mock.patch.object(
                    SAFETY.tempfile,
                    "gettempdir",
                    return_value=str(private_temp),
                ),
                mock.patch.object(
                    SAFETY,
                    "_tree_manifest",
                    side_effect=mutate_after_manifest,
                ),
                self.assertRaisesRegex(RuntimeError, "staged output changed"),
                SAFETY.transactional_output(
                    output,
                    overwrite=False,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                ) as staging,
            ):
                staging_path = staging
                (staging / "scientific.txt").write_text(
                    "original\n", encoding="utf-8"
                )

            self.assertTrue(injected)
            self.assertFalse(output.exists())

    def test_overwrite_backup_uses_the_same_private_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            private_temp = project / "private-temp"
            protected.mkdir(parents=True)
            output.mkdir(parents=True)
            private_temp.mkdir()
            (output / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            (output / "old.txt").write_text("old\n", encoding="utf-8")
            original_replace = SAFETY.os.replace
            observed_backup: Path | None = None

            def record_backup(source: Path, destination: Path) -> None:
                nonlocal observed_backup
                if (
                    Path(source).resolve() == output.resolve()
                    and ".backup-" in Path(destination).name
                ):
                    observed_backup = Path(destination)
                original_replace(source, destination)

            with (
                mock.patch.object(
                    SAFETY.tempfile,
                    "gettempdir",
                    return_value=str(private_temp),
                ),
                mock.patch.object(SAFETY.os, "replace", side_effect=record_backup),
                SAFETY.transactional_output(
                    output,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                ) as staging,
            ):
                (staging / "new.txt").write_text("new\n", encoding="utf-8")

            self.assertIsNotNone(observed_backup)
            assert observed_backup is not None
            self.assertEqual(
                observed_backup.parent.parent.resolve(), private_temp.resolve()
            )
            self.assertFalse(observed_backup.exists())
            self.assertFalse((output / "old.txt").exists())
            self.assertEqual((output / "new.txt").read_text(), "new\n")

    def test_transaction_failure_preserves_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            protected.mkdir(parents=True)
            output.mkdir(parents=True)
            (output / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            sentinel = output / "known-good.txt"
            sentinel.write_text("old\n", encoding="utf-8")

            with (
                self.assertRaisesRegex(RuntimeError, "injected"),
                SAFETY.transactional_output(
                    output,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                ) as staging,
            ):
                (staging / "partial.txt").write_text("partial\n")
                raise RuntimeError("injected validation failure")

            self.assertEqual(sentinel.read_text(), "old\n")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*-*")), [])

    def test_transaction_signal_after_backup_restores_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            protected.mkdir(parents=True)
            output.mkdir(parents=True)
            (output / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            sentinel = output / "known-good.txt"
            sentinel.write_text("old\n", encoding="utf-8")
            original_replace = SAFETY.os.replace
            injected = False

            def interrupt_after_backup(source: Path, destination: Path) -> None:
                nonlocal injected
                original_replace(source, destination)
                if (
                    not injected
                    and Path(source).name == output.name
                    and ".backup-" in Path(destination).name
                ):
                    injected = True
                    raise KeyboardInterrupt("injected publish signal")

            with (
                mock.patch.object(
                    SAFETY.os,
                    "replace",
                    side_effect=interrupt_after_backup,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "publish signal"),
                SAFETY.transactional_output(
                    output,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                ) as staging,
            ):
                (staging / "new.txt").write_text("new\n")

            self.assertTrue(injected)
            self.assertEqual(sentinel.read_text(), "old\n")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*-*")), [])

    def test_transaction_signal_after_install_restores_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            protected.mkdir(parents=True)
            output.mkdir(parents=True)
            (output / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            sentinel = output / "known-good.txt"
            sentinel.write_text("old\n", encoding="utf-8")
            original_replace = SAFETY.os.replace
            injected = False

            def interrupt_after_install(source: Path, destination: Path) -> None:
                nonlocal injected
                original_replace(source, destination)
                if (
                    not injected
                    and ".staging-" in Path(source).name
                    and Path(destination).name == output.name
                ):
                    injected = True
                    raise KeyboardInterrupt("injected install signal")

            with (
                mock.patch.object(
                    SAFETY.os,
                    "replace",
                    side_effect=interrupt_after_install,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "install signal"),
                SAFETY.transactional_output(
                    output,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                ) as staging,
            ):
                (staging / "new.txt").write_text("new\n")

            self.assertTrue(injected)
            self.assertEqual(sentinel.read_text(), "old\n")
            self.assertFalse((output / "new.txt").exists())
            self.assertEqual(list(output.parent.glob(f".{output.name}.*-*")), [])

    def test_transaction_rejects_concurrent_target_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            protected.mkdir(parents=True)
            output.mkdir(parents=True)
            (output / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            sentinel = output / "known-good.txt"
            sentinel.write_text("old\n", encoding="utf-8")

            with (
                self.assertRaisesRegex(RuntimeError, "changed while"),
                SAFETY.transactional_output(
                    output,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                ) as staging,
            ):
                (staging / "new.txt").write_text("new\n")
                sentinel.write_text("concurrent\n", encoding="utf-8")

            self.assertEqual(sentinel.read_text(), "concurrent\n")
            self.assertFalse((output / "new.txt").exists())
            self.assertEqual(list(output.parent.glob(f".{output.name}.*-*")), [])

    def test_rejects_input_relationships_and_repository_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            build = project / "build"
            checkpoint = project / "checkpoint"
            build.mkdir(parents=True)
            checkpoint.mkdir()
            for unsafe in (
                project,
                project.parent,
                build,
                build / "nested-output",
                checkpoint,
                checkpoint / "nested-output",
            ):
                with self.subTest(output=unsafe), self.assertRaises(ValueError):
                    SAFETY.guarded_prepare_output(
                        unsafe,
                        overwrite=True,
                        protected_paths=(build, checkpoint),
                        project_root=project,
                        owner="test-validator",
                    )

    def test_requires_the_calling_validators_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            protected.mkdir(parents=True)
            output.mkdir(parents=True)
            sentinel = output / "keep.txt"
            sentinel.write_text("user data\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                SAFETY.guarded_prepare_output(
                    output,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                )
            self.assertTrue(sentinel.is_file())

            (output / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-b"), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                SAFETY.guarded_prepare_output(
                    output,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                )
            self.assertTrue(sentinel.is_file())

            (output / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            SAFETY.guarded_prepare_output(
                output,
                overwrite=True,
                protected_paths=(protected,),
                project_root=project,
                owner="validator-a",
            )
            self.assertFalse(sentinel.exists())
            self.assertEqual(
                (output / SAFETY.OUTPUT_MARKER).read_text(encoding="utf-8"),
                SAFETY.marker_content("validator-a"),
            )

    def test_symlinked_marker_cannot_authorize_output_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            output = project / "outputs/audit"
            protected.mkdir(parents=True)
            output.mkdir(parents=True)
            sentinel = output / "keep.txt"
            sentinel.write_text("user data\n", encoding="utf-8")
            marker_target = project / "marker-content.txt"
            marker_target.write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            (output / SAFETY.OUTPUT_MARKER).symlink_to(marker_target)

            with self.assertRaises(ValueError):
                SAFETY.guarded_prepare_output(
                    output,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                )
            self.assertEqual(sentinel.read_text(), "user data\n")
            self.assertEqual(
                marker_target.read_text(), SAFETY.marker_content("validator-a")
            )

    def test_rejects_output_symlink_and_preserves_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            protected = project / "input"
            target = project / "outputs/owned-target"
            link = project / "outputs/apparent-output"
            protected.mkdir(parents=True)
            target.mkdir(parents=True)
            sentinel = target / "keep.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            (target / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            link.symlink_to(target, target_is_directory=True)

            with self.assertRaises(ValueError):
                SAFETY.guarded_prepare_output(
                    link,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me\n")

    def test_rejects_outside_and_parent_symlink_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            project = temporary_root / "project"
            protected = project / "input"
            protected.mkdir(parents=True)

            outside = temporary_root / "outside-audit"
            outside.mkdir()
            outside_sentinel = outside / "keep.txt"
            outside_sentinel.write_text("outside\n", encoding="utf-8")
            (outside / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "strict descendant"):
                SAFETY.guarded_prepare_output(
                    outside,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                )
            self.assertEqual(outside_sentinel.read_text(), "outside\n")

            target_parent = project / "real-target"
            target = target_parent / "child"
            target.mkdir(parents=True)
            target_sentinel = target / "keep.txt"
            target_sentinel.write_text("linked\n", encoding="utf-8")
            (target / SAFETY.OUTPUT_MARKER).write_text(
                SAFETY.marker_content("validator-a"), encoding="utf-8"
            )
            apparent_parent = project / "outputs/parent-link"
            apparent_parent.parent.mkdir()
            apparent_parent.symlink_to(target_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                SAFETY.guarded_prepare_output(
                    apparent_parent / "child",
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=project,
                    owner="validator-a",
                )
            self.assertEqual(target_sentinel.read_text(), "linked\n")

    def test_all_destructive_validators_use_transactional_publication(self) -> None:
        for name in VALIDATORS:
            source = (AUDIT_ROOT / name).read_text(encoding="utf-8")
            with self.subTest(validator=name):
                self.assertIn(
                    "from output_safety import transactional_output", source
                )
                self.assertIn("transactional_output(", source)
                self.assertNotIn("shutil.rmtree(output)", source)

    def test_validators_do_not_silently_select_an_old_generation(self) -> None:
        for name in VALIDATORS:
            source = (AUDIT_ROOT / name).read_text(encoding="utf-8")
            with self.subTest(validator=name):
                self.assertNotIn("coupled_run21", source)
                self.assertNotIn("outputs/zc_adjoint/kernel_build/source", source)
                self.assertNotIn("args.output_dir.resolve()", source)

        for name in ("validate_tangent.py", "validate_composed_tangent.py"):
            source = (AUDIT_ROOT / name).read_text(encoding="utf-8")
            with self.subTest(validator=name):
                self.assertIn('"--kernel-source"', source)
                self.assertIn('"--runtime-source"', source)
                self.assertIn('"--tangent-executable"', source)
                self.assertIn('"--primal-executable"', source)
                self.assertIn("source=kernel_source", source)
                self.assertIn("source = staged_runtime_source", source)
                self.assertIn("verify_replay_runtime_source(", source)
                self.assertGreaterEqual(
                    source.count("verify_private_staged_inputs("), 2
                )
                self.assertIn("staged_manifest", source)

        scalar = (AUDIT_ROOT / "validate_scalar_taylor.py").read_text(encoding="utf-8")
        self.assertIn('"--kernel-source"', scalar)
        self.assertIn('"--runtime-source"', scalar)
        self.assertIn('"--replay-validation-dir"', scalar)
        self.assertIn('"--cases-root"', scalar)
        self.assertIn("source=kernel_source", scalar)
        self.assertIn("source = staged_runtime_source", scalar)
        self.assertIn("verify_replay_runtime_source(", scalar)
        self.assertGreaterEqual(scalar.count("verify_private_staged_inputs("), 2)
        self.assertIn("staged_manifest", scalar)

        scalar_dot = (AUDIT_ROOT / "validate_adjoint_dot.py").read_text(
            encoding="utf-8"
        )
        generic_dot = (
            AUDIT_ROOT / "validate_generic_adjoint_dot.py"
        ).read_text(encoding="utf-8")
        for name, source in (
            ("validate_adjoint_dot.py", scalar_dot),
            ("validate_generic_adjoint_dot.py", generic_dot),
        ):
            with self.subTest(validator=name):
                self.assertIn("verify_producer_runtime(", source)
                self.assertIn("stage_producer_runtime(", source)
                self.assertIn("cwd=staged_run_dir", source)
                self.assertIn("verify_staged_producer_runtime(", source)
        self.assertIn("str(staged_reverse_executable)", scalar_dot)
        self.assertIn("re-executed reverse gradient differs", scalar_dot)


if __name__ == "__main__":
    unittest.main()
