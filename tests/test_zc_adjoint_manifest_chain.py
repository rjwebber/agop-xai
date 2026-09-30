"""Adversarial tests for the staged ZC/Tapenade provenance chain."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = ROOT / "adjoint/tapenade_toolchain"
SCRIPTS = TOOLCHAIN / "scripts"
GUARD = SCRIPTS / "build_guard.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root: Path, manifest: Path, paths: list[Path]) -> None:
    lines = []
    for path in sorted(paths):
        lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")
    manifest.write_text("".join(lines), encoding="utf-8")


def version_values() -> dict[str, str]:
    values = {}
    for line in (TOOLCHAIN / "VERSION.env").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def make_fake_tapenade(home: Path, toolchain: Path = TOOLCHAIN) -> None:
    values = version_values()
    (home / "bin/linux").mkdir(parents=True)
    (home / "ADFirstAidKit").mkdir()
    launcher = home / "bin/tapenade"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        f"echo 'Tapenade {values['TAPENADE_VERSION']} develop'\n"
        f"echo 'Revision: {values['TAPENADE_REVISION']}'\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    parser = home / "bin/linux/fortranParser"
    parser.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    parser.chmod(0o755)
    for name in ("adStack.c", "adStack.h", "adComplex.h"):
        (home / "ADFirstAidKit" / name).write_text(f"/* {name} */\n")
    (home / "resources.jar").write_bytes(b"pinned-resource\n")

    seal_fake_tapenade(home, toolchain)


def seal_fake_tapenade(home: Path, toolchain: Path = TOOLCHAIN) -> None:
    values = version_values()
    for old_sidecar in (
        home / ".zc_tapenade_tree_manifest.sha256",
        home / ".zc_tapenade_install_receipt.txt",
    ):
        if old_sidecar.exists():
            old_sidecar.unlink()

    tree_manifest = home / ".zc_tapenade_tree_manifest.sha256"
    payloads = [
        path
        for path in home.rglob("*")
        if path.is_file() and not path.name.startswith(".zc_tapenade_")
    ]
    write_manifest(home, tree_manifest, payloads)
    receipt = home / ".zc_tapenade_install_receipt.txt"
    receipt.write_text(
        "\n".join(
            (
                "receipt_schema=zc-tapenade-install-receipt-v1",
                f"tapenade_version={values['TAPENADE_VERSION']}",
                f"tapenade_revision={values['TAPENADE_REVISION']}",
                f"expected_archive_sha256={values['TAPENADE_ARCHIVE_SHA256']}",
                f"observed_archive_sha256={values['TAPENADE_ARCHIVE_SHA256']}",
                f"tree_manifest_sha256={sha256(tree_manifest)}",
                f"tree_file_count={len(payloads)}",
                "installer_script_sha256="
                f"{sha256(toolchain / 'scripts/install_tapenade_linux.sh')}",
                f"build_guard_sha256={sha256(toolchain / 'scripts/build_guard.sh')}",
                f"version_env_sha256={sha256(toolchain / 'VERSION.env')}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def run_guard(command: str, *arguments: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail; source "$1"; {command}',
            "bash",
            str(GUARD),
            *map(str, arguments),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


class ManifestGuardTests(unittest.TestCase):
    def test_resolved_command_state_accepts_ubuntu_style_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            executable = root / "cpp-real"
            executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            alias = root / "cpp"
            alias.symlink_to(executable)
            result = run_guard(
                'resolved="$(zc_resolve_path "$2")"; '
                'zc_capture_path_state "$resolved"',
                alias,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith("file:"))

    def test_manifest_tamper_fails_before_stage_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inputs = root / "inputs"
            build = root / "build"
            stage = build / "stage"
            inputs.mkdir()
            stage.mkdir(parents=True)
            payload = inputs / "payload.txt"
            payload.write_text("original\n", encoding="utf-8")
            manifest = inputs / "manifest.sha256"
            write_manifest(inputs, manifest, [payload])
            payload.write_text("tampered\n", encoding="utf-8")
            sentinel = stage / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")

            result = run_guard(
                'export ADJOINT_OVERWRITE=1; '
                'zc_verify_sha256_manifest "$2" "$3" >/dev/null; '
                'zc_prepare_stage_dir "$4" "$5"',
                inputs,
                manifest,
                build,
                stage,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_exact_manifest_rejects_unmanifested_resource(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            payload = root / "payload.jar"
            payload.write_bytes(b"declared")
            manifest = root / "tree.sha256"
            write_manifest(root, manifest, [payload])
            (root / "injected.jar").write_bytes(b"injected")
            result = run_guard(
                'zc_verify_sha256_manifest_exact "$2" "$3" tree.sha256',
                root,
                manifest,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unexpected", result.stderr)

    def test_tapenade_receipt_rejects_tree_change_and_fake_home(self) -> None:
        values = version_values()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            verified = root / "verified"
            verified.mkdir()
            make_fake_tapenade(verified)
            command = (
                'zc_verify_tapenade_install "$2" "$3" "$4" "$5" "$6"'
            )
            arguments = (
                verified,
                TOOLCHAIN,
                values["TAPENADE_VERSION"],
                values["TAPENADE_REVISION"],
                values["TAPENADE_ARCHIVE_SHA256"],
            )
            self.assertEqual(run_guard(command, *arguments).returncode, 0)
            (verified / "resources.jar").write_bytes(b"changed")
            self.assertNotEqual(run_guard(command, *arguments).returncode, 0)

            fake = root / "fake"
            (fake / "bin").mkdir(parents=True)
            (fake / "bin/tapenade").write_text("not manifested\n")
            fake_arguments = (fake, *arguments[1:])
            self.assertNotEqual(run_guard(command, *fake_arguments).returncode, 0)

    def test_build_claim_rejects_lexical_parent_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            toolchain = Path(temporary) / "toolchain"
            build_root = toolchain / "build"
            real_parent = build_root / "real-parent"
            real_parent.mkdir(parents=True)
            sentinel = real_parent / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            alias = build_root / "alias"
            alias.symlink_to(real_parent, target_is_directory=True)
            result = run_guard(
                'zc_claim_build_dir "$2" "$3" >/dev/null',
                toolchain,
                alias / "child",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse((real_parent / "child").exists())

    def test_stage_publication_rolls_back_each_injected_failure(self) -> None:
        failpoints = (
            "stage_before_old_move",
            "stage_after_old_move",
            "stage_before_new_move",
            "stage_after_new_move",
            "stage_before_commit",
        )
        for failpoint in failpoints:
            with (
                self.subTest(failpoint=failpoint),
                tempfile.TemporaryDirectory() as tmp,
            ):
                build = Path(tmp) / "build"
                target = build / "target"
                staged = build / "candidate"
                target.mkdir(parents=True)
                staged.mkdir()
                (target / "payload.txt").write_text("old\n", encoding="utf-8")
                (staged / "payload.txt").write_text("new\n", encoding="utf-8")
                expected = run_guard(
                    'zc_capture_path_state "$2"', target
                ).stdout.strip()
                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'set -euo pipefail; source "$1"; '
                        'zc_publish_stage_directory "$2" "$3" "$4" "$5" standard',
                        "bash",
                        str(GUARD),
                        str(build),
                        str(target),
                        str(staged),
                        expected,
                    ],
                    env={
                        **os.environ,
                        "ADJOINT_OVERWRITE": "1",
                        "ZC_TOOLCHAIN_TESTING": "1",
                        "ZC_TEST_FAILPOINT": failpoint,
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(
                    (target / "payload.txt").read_text(encoding="utf-8"), "old\n"
                )
                self.assertFalse(any(build.glob(".zc-stage-backup.*")))

    def test_stage_publication_rolls_back_signal_after_new_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build"
            target = build / "target"
            staged = build / "candidate"
            target.mkdir(parents=True)
            staged.mkdir()
            (target / "payload.txt").write_text("old\n", encoding="utf-8")
            (staged / "payload.txt").write_text("new\n", encoding="utf-8")
            expected = run_guard('zc_capture_path_state "$2"', target).stdout.strip()
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$1"; '
                    'zc_publish_stage_directory "$2" "$3" "$4" "$5" standard',
                    "bash",
                    str(GUARD),
                    str(build),
                    str(target),
                    str(staged),
                    expected,
                ],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "ZC_TOOLCHAIN_TESTING": "1",
                    "ZC_TEST_FAILPOINT": "stage_after_new_move",
                    "ZC_TEST_FAIL_MODE": "signal",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                (target / "payload.txt").read_text(encoding="utf-8"), "old\n"
            )

    def test_stage_workspace_cleanup_rejects_inode_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build"
            build.mkdir()
            workspace = build / ".zc-test-work.123456"
            workspace.mkdir()
            (workspace / ".zc_stage_workspace").write_text(
                "managed-zc-stage-workspace-v1\n", encoding="utf-8"
            )
            identity = run_guard('zc_path_identity "$2"', workspace).stdout.strip()
            displaced = build / "displaced"
            workspace.rename(displaced)
            workspace.mkdir()
            (workspace / ".zc_stage_workspace").write_text(
                "managed-zc-stage-workspace-v1\n", encoding="utf-8"
            )
            sentinel = workspace / "keep.txt"
            sentinel.write_text("replacement\n", encoding="utf-8")
            result = run_guard(
                'zc_remove_stage_workspace "$2" "$3" "$4"',
                build,
                workspace,
                identity,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "replacement\n")

    def test_stage_publication_refuses_replaced_commit_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp).resolve() / "build"
            target = build / "target"
            staged = build / "candidate"
            target.mkdir(parents=True)
            staged.mkdir()
            (target / "payload.txt").write_text("old\n", encoding="utf-8")
            (staged / "payload.txt").write_text("new\n", encoding="utf-8")
            expected = run_guard('zc_capture_path_state "$2"', target).stdout.strip()
            (target / "payload.txt").write_text("concurrent\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$1"; '
                    'zc_publish_stage_directory "$2" "$3" "$4" "$5" standard',
                    "bash",
                    str(GUARD),
                    str(build),
                    str(target),
                    str(staged),
                    expected,
                ],
                env={**os.environ, "ADJOINT_OVERWRITE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                (target / "payload.txt").read_text(encoding="utf-8"),
                "concurrent\n",
            )

    def test_stage_publication_verifies_object_actually_moved_to_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp).resolve() / "build"
            target = build / "target"
            staged = build / "candidate"
            target.mkdir(parents=True)
            staged.mkdir()
            (target / "payload.txt").write_text("old\n", encoding="utf-8")
            (staged / "payload.txt").write_text("new\n", encoding="utf-8")
            expected = run_guard('zc_capture_path_state "$2"', target).stdout.strip()
            hook = build / "replace-after-check.sh"
            hook.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "mv \"$1\" \"$1.before-race\"\n"
                "mkdir \"$1\"\n"
                "printf 'concurrent\\n' > \"$1/payload.txt\"\n",
                encoding="utf-8",
            )
            hook.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$1"; '
                    'zc_publish_stage_directory "$2" "$3" "$4" "$5" standard',
                    "bash",
                    str(GUARD),
                    str(build),
                    str(target),
                    str(staged),
                    expected,
                ],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "ZC_TOOLCHAIN_TESTING": "1",
                    "ZC_TEST_STAGE_AFTER_TARGET_RECHECK_HOOK": str(hook),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((target / "payload.txt").read_text(), "concurrent\n")
            self.assertEqual(
                (build / "target.before-race/payload.txt").read_text(), "old\n"
            )
            self.assertFalse(any(build.glob(".zc-stage-backup.*")))

    def test_stage_backup_marker_loss_is_preserved_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp).resolve() / "build"
            target = build / "target"
            staged = build / "candidate"
            target.mkdir(parents=True)
            staged.mkdir()
            (target / "payload.txt").write_text("old\n", encoding="utf-8")
            (staged / "payload.txt").write_text("new\n", encoding="utf-8")
            expected = run_guard('zc_capture_path_state "$2"', target).stdout.strip()
            hook = build / "remove-backup-marker.sh"
            hook.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "rm \"$1/.zc_stage_backup\"\n",
                encoding="utf-8",
            )
            hook.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$1"; '
                    'zc_publish_stage_directory "$2" "$3" "$4" "$5" standard',
                    "bash",
                    str(GUARD),
                    str(build),
                    str(target),
                    str(staged),
                    expected,
                ],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "ZC_TOOLCHAIN_TESTING": "1",
                    "ZC_TEST_STAGE_BEFORE_COMMIT_HOOK": str(hook),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            backups = list(build.glob(".zc-stage-backup.*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(
                (backups[0] / "original/payload.txt").read_text(), "old\n"
            )
            self.assertEqual((target / "payload.txt").read_text(), "new\n")

    def test_stage_backup_replacement_survives_term_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp).resolve() / "build"
            target = build / "target"
            staged = build / "candidate"
            target.mkdir(parents=True)
            staged.mkdir()
            (target / "payload.txt").write_text("old\n", encoding="utf-8")
            (staged / "payload.txt").write_text("new\n", encoding="utf-8")
            expected = run_guard('zc_capture_path_state "$2"', target).stdout.strip()
            hook = build / "replace-backup-dir.sh"
            hook.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "mv \"$1\" \"$1.displaced\"\n"
                "mkdir \"$1\"\n"
                "printf 'replacement\\n' > \"$1/keep.txt\"\n",
                encoding="utf-8",
            )
            hook.chmod(0o755)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$1"; '
                    'zc_publish_stage_directory "$2" "$3" "$4" "$5" standard',
                    "bash",
                    str(GUARD),
                    str(build),
                    str(target),
                    str(staged),
                    expected,
                ],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "ZC_TOOLCHAIN_TESTING": "1",
                    "ZC_TEST_STAGE_BEFORE_COMMIT_HOOK": str(hook),
                    "ZC_TEST_FAILPOINT": "stage_before_commit",
                    "ZC_TEST_FAIL_MODE": "signal",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            displaced = list(build.glob(".zc-stage-backup.*.displaced"))
            replacement = [
                path
                for path in build.glob(".zc-stage-backup.*")
                if not path.name.endswith(".displaced")
            ]
            self.assertEqual(len(displaced), 1)
            self.assertEqual(len(replacement), 1)
            self.assertEqual(
                (displaced[0] / "original/payload.txt").read_text(), "old\n"
            )
            self.assertEqual(
                (replacement[0] / "keep.txt").read_text(), "replacement\n"
            )
            self.assertEqual((target / "payload.txt").read_text(), "new\n")

    def test_managed_build_signal_restores_authenticated_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp).resolve() / "build"
            target = build / "target"
            staged = build / "candidate"
            for directory, payload in ((target, "old\n"), (staged, "new\n")):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / ".zc_tapenade_build").write_text(
                    "managed-zc-tapenade-build-v1\n", encoding="utf-8"
                )
                (directory / "payload.txt").write_text(payload, encoding="utf-8")
            expected = run_guard('zc_capture_path_state "$2"', target).stdout.strip()
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$1"; '
                    'zc_publish_stage_directory "$2" "$3" "$4" "$5" managed',
                    "bash",
                    str(GUARD),
                    str(build),
                    str(target),
                    str(staged),
                    expected,
                ],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "ZC_TOOLCHAIN_TESTING": "1",
                    "ZC_TEST_FAILPOINT": "stage_after_new_move",
                    "ZC_TEST_FAIL_MODE": "signal",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                (target / "payload.txt").read_text(encoding="utf-8"), "old\n"
            )


class StagePreservationTests(unittest.TestCase):
    def setUp(self) -> None:
        (TOOLCHAIN / "build").mkdir(exist_ok=True)

    def _managed_run(self, prefix: str) -> tempfile.TemporaryDirectory[str]:
        temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=TOOLCHAIN / "build")
        run = Path(temporary.name)
        (run / ".zc_tapenade_build").write_text(
            "managed-zc-tapenade-build-v1\n", encoding="utf-8"
        )
        return temporary

    def test_generate_tamper_preserves_existing_generated_stage(self) -> None:
        with (
            self._managed_run(".generate-manifest-test.") as run_name,
            tempfile.TemporaryDirectory() as tap_name,
        ):
            run = Path(run_name)
            prepared = run / "coupled_prepared"
            generated = run / "coupled_generated"
            prepared.mkdir()
            generated.mkdir()
            payload = prepared / "zc_kernel_api.f"
            payload.write_text("original\n", encoding="utf-8")
            manifest = prepared / "prepared_source_manifest.sha256"
            write_manifest(prepared, manifest, [payload])
            payload.write_text("tampered\n", encoding="utf-8")
            sentinel = generated / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            tapenade = Path(tap_name) / "tapenade"
            tapenade.mkdir()
            make_fake_tapenade(tapenade)

            result = subprocess.run(
                [str(SCRIPTS / "generate_coupled_nino3.sh"), str(tapenade), str(run)],
                env={**os.environ, "ADJOINT_OVERWRITE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_prepare_detects_staged_input_mutation_and_preserves_old(self) -> None:
        kernel = ROOT / "outputs/zc_adjoint/kernel_build_run23/source"
        if not kernel.is_dir():
            self.skipTest("certified kernel source is unavailable")
        with (
            self._managed_run(".prepare-mutation-test.") as run_name,
            tempfile.TemporaryDirectory() as bin_name,
        ):
            run = Path(run_name)
            prepared = run / "coupled_prepared"
            prepared.mkdir()
            sentinel = prepared / "keep.txt"
            sentinel.write_text("old-good-stage\n", encoding="utf-8")
            fake_patch = Path(bin_name) / "patch"
            fake_patch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "/usr/bin/patch \"$@\"\n"
                "if [[ ! -e ${MUTATE_ONCE} ]]; then\n"
                "  directory=''\n"
                "  arguments=(\"$@\")\n"
                "  for ((i=0; i<${#arguments[@]}; i++)); do\n"
                "    if [[ ${arguments[$i]} == --directory ]]; then\n"
                "      directory=${arguments[$((i+1))]}\n"
                "    fi\n"
                "  done\n"
                "  if [[ -n $directory ]]; then\n"
                "    victim=$(dirname \"$directory\")/inputs/kernel_source/akcalc.F\n"
                "    chmod u+w \"$victim\"\n"
                "    printf 'mutation\\n' >> \"$victim\"\n"
                "    : > \"${MUTATE_ONCE}\"\n"
                "  fi\n"
                "fi\n",
                encoding="utf-8",
            )
            fake_patch.chmod(0o755)
            result = subprocess.run(
                [
                    str(SCRIPTS / "prepare_coupled_sources.sh"),
                    str(kernel),
                    str(run),
                ],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "PATH": f"{bin_name}:{os.environ['PATH']}",
                    "MUTATE_ONCE": str(Path(bin_name) / "mutated.once"),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"), "old-good-stage\n"
            )

    def test_generate_detects_staged_input_mutation_and_preserves_old(self) -> None:
        kernel = ROOT / "outputs/zc_adjoint/kernel_build_run23/source"
        if not kernel.is_dir():
            self.skipTest("certified kernel source is unavailable")
        values = version_values()
        with (
            self._managed_run(".generate-mutation-test.") as run_name,
            tempfile.TemporaryDirectory() as tap_name,
        ):
            run = Path(run_name)
            prepare = subprocess.run(
                [
                    str(SCRIPTS / "prepare_coupled_sources.sh"),
                    str(kernel),
                    str(run),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(prepare.returncode, 0, prepare.stdout + prepare.stderr)
            generated = run / "coupled_generated"
            generated.mkdir()
            sentinel = generated / "keep.txt"
            sentinel.write_text("old-good-stage\n", encoding="utf-8")

            tapenade = Path(tap_name).resolve() / "tapenade"
            tapenade.mkdir()
            make_fake_tapenade(tapenade)
            launcher = tapenade / "bin/tapenade"
            launcher.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [[ ${1:-} == -version ]]; then\n"
                f"  echo 'Tapenade {values['TAPENADE_VERSION']} develop'\n"
                f"  echo 'Revision: {values['TAPENADE_REVISION']}'\n"
                "  exit 0\n"
                "fi\n"
                "out=''\n"
                "include=''\n"
                "mode=''\n"
                "while [[ $# -gt 0 ]]; do\n"
                "  case $1 in\n"
                "    -O) out=$2; shift 2;;\n"
                "    -I) include=$2; shift 2;;\n"
                "    -tangent|-reverse) mode=$1; shift;;\n"
                "    *) shift;;\n"
                "  esac\n"
                "done\n"
                "mkdir -p \"$out\"\n"
                "if [[ $mode == -tangent ]]; then\n"
                "  printf 'generated tangent\\n' > \"$out/zc_kernel_nino3_d.f\"\n"
                "else\n"
                "  printf 'generated reverse\\n' > \"$out/zc_kernel_nino3_b.f\"\n"
                "fi\n"
                "if [[ ! -e ${MUTATE_ONCE} ]]; then\n"
                "  chmod u+w \"$include/akcalc.f\"\n"
                "  printf 'mutation\\n' >> \"$include/akcalc.f\"\n"
                "  : > \"${MUTATE_ONCE}\"\n"
                "fi\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            seal_fake_tapenade(tapenade)
            result = subprocess.run(
                [str(SCRIPTS / "generate_coupled_nino3.sh"), str(tapenade), str(run)],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "MUTATE_ONCE": str(Path(tap_name) / "mutated.once"),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"), "old-good-stage\n"
            )


class InstallerSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        (TOOLCHAIN / "build").mkdir(exist_ok=True)

    def _managed_run(self, prefix: str) -> tempfile.TemporaryDirectory[str]:
        temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=TOOLCHAIN / "build")
        run = Path(temporary.name)
        (run / ".zc_tapenade_build").write_text(
            "managed-zc-tapenade-build-v1\n", encoding="utf-8"
        )
        return temporary

    def _fake_toolchain_and_archive(self, root: Path) -> tuple[Path, Path]:
        toolchain = root / "toolchain"
        scripts = toolchain / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(SCRIPTS / "install_tapenade_linux.sh", scripts)
        shutil.copy2(GUARD, scripts)

        version = "9.99"
        revision = "test-revision"
        payload_parent = root / "archive-payload"
        payload = payload_parent / f"tapenade_{version}"
        (payload / "bin/linux").mkdir(parents=True)
        (payload / "ADFirstAidKit").mkdir()
        launcher = payload / "bin/tapenade"
        launcher.write_text(
            "#!/usr/bin/env bash\n"
            f"echo 'Tapenade {version} develop'\n"
            f"echo 'Revision: {revision}'\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        parser = payload / "bin/linux/fortranParser"
        parser.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        parser.chmod(0o755)
        for name in ("adStack.c", "adStack.h", "adComplex.h"):
            (payload / "ADFirstAidKit" / name).write_text(
                f"/* {name} */\n", encoding="utf-8"
            )
        archive = root / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(payload, arcname=payload.name)
        digest = sha256(archive)
        (toolchain / "VERSION.env").write_text(
            f"TAPENADE_VERSION={version}\n"
            f"TAPENADE_REVISION={revision}\n"
            f"TAPENADE_ARCHIVE_URL={archive.resolve().as_uri()}\n"
            f"TAPENADE_ARCHIVE_SHA256={digest}\n",
            encoding="utf-8",
        )
        return toolchain, archive

    def test_installer_rejects_symlinked_parent_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            toolchain, _ = self._fake_toolchain_and_archive(root)
            real_parent = root / "real-parent"
            real_parent.mkdir()
            alias = root / "parent-alias"
            alias.symlink_to(real_parent, target_is_directory=True)
            result = subprocess.run(
                [str(toolchain / "scripts/install_tapenade_linux.sh"), str(alias)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)

            install_parent = root / "install"
            install_parent.mkdir()
            dangling = install_parent / "cache.tar"
            dangling.symlink_to(root / "missing.tar")
            result = subprocess.run(
                [
                    str(toolchain / "scripts/install_tapenade_linux.sh"),
                    str(install_parent),
                ],
                env={**os.environ, "TAPENADE_ARCHIVE": str(dangling)},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(dangling.is_symlink())

    def test_installer_rejects_hardlinked_archive_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            toolchain, source = self._fake_toolchain_and_archive(root)
            install_parent = root / "install"
            install_parent.mkdir()
            cache = install_parent / "cache.tar"
            os.link(source, cache)
            result = subprocess.run(
                [
                    str(toolchain / "scripts/install_tapenade_linux.sh"),
                    str(install_parent),
                ],
                env={**os.environ, "TAPENADE_ARCHIVE": str(cache)},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(cache.exists())
            self.assertEqual(cache.read_bytes(), source.read_bytes())

    def test_interrupted_download_leaves_no_published_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            toolchain, _ = self._fake_toolchain_and_archive(root)
            install_parent = root / "install"
            install_parent.mkdir()
            cache = install_parent / "cache.tar"
            result = subprocess.run(
                [
                    str(toolchain / "scripts/install_tapenade_linux.sh"),
                    str(install_parent),
                ],
                env={
                    **os.environ,
                    "TAPENADE_ARCHIVE": str(cache),
                    "ZC_TOOLCHAIN_TESTING": "1",
                    "ZC_TEST_FAILPOINT": "installer_after_archive_link",
                    "ZC_TEST_FAIL_MODE": "signal",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(cache.exists())
            self.assertFalse(any(install_parent.glob(".tapenade-download.*")))
            self.assertFalse(any(install_parent.glob(".tapenade-archive.*")))
            self.assertFalse(any(install_parent.glob(".tapenade-extract.*")))

    def test_installer_success_publishes_verified_unique_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            toolchain, _ = self._fake_toolchain_and_archive(root)
            install_parent = root / "install"
            install_parent.mkdir()
            cache = install_parent / "cache.tar"
            result = subprocess.run(
                [
                    str(toolchain / "scripts/install_tapenade_linux.sh"),
                    str(install_parent),
                ],
                env={**os.environ, "TAPENADE_ARCHIVE": str(cache)},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            install_dir = Path(result.stdout.splitlines()[-1])
            self.assertTrue(
                (install_dir / ".zc_tapenade_install_receipt.txt").is_file()
            )
            self.assertTrue(
                (install_dir / ".zc_tapenade_tree_manifest.sha256").is_file()
            )
            self.assertEqual(cache.stat().st_nlink, 1)

    def test_instrument_tamper_does_not_patch_generated_source(self) -> None:
        with self._managed_run(".instrument-manifest-test.") as run_name:
            run = Path(run_name)
            generated = run / "coupled_generated"
            reverse = generated / "reverse"
            reverse.mkdir(parents=True)
            source = reverse / "zc_kernel_api_b.f"
            source.write_text("original generated source\n", encoding="utf-8")
            manifest = generated / "generated_source_manifest.pre_instrument.sha256"
            write_manifest(generated, manifest, [source])
            (generated / "generation_provenance.txt").write_text("stage=test\n")
            (generated / "generation_manifest_binding.txt").write_text(
                "binding_schema=zc-generation-manifest-binding-v1\n"
            )
            source.write_text("tampered generated source\n", encoding="utf-8")

            result = subprocess.run(
                [str(SCRIPTS / "instrument_generated_reverse.sh"), str(reverse)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                source.read_text(encoding="utf-8"), "tampered generated source\n"
            )

    def test_compile_tamper_preserves_existing_compiled_stage(self) -> None:
        with (
            self._managed_run(".compile-manifest-test.") as run_name,
            tempfile.TemporaryDirectory() as tap_name,
            tempfile.TemporaryDirectory() as kernel_name,
        ):
            run = Path(run_name)
            prepared = run / "coupled_prepared"
            generated = run / "coupled_generated"
            compiled = run / "coupled_compiled"
            (prepared / "ZC_lib_routines").mkdir(parents=True)
            (generated / "reverse").mkdir(parents=True)
            (generated / "tangent").mkdir(parents=True)
            compiled.mkdir()
            sentinel = compiled / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")

            primal = [
                "zc_kernel_api",
                "ssta",
                "ztmfc1",
                "cforce",
                "mloop",
                "akcalc",
                "bndary",
                "uhcalc",
                "uhinit",
                "tridag",
            ]
            for name in primal:
                (prepared / f"{name}.f").write_text("source\n")
            (prepared / "ZC_lib_routines/fft2c.f").write_text("source\n")
            payload = prepared / "zc_kernel_api.f"
            prepared_manifest = prepared / "prepared_source_manifest.sha256"
            write_manifest(prepared, prepared_manifest, [payload])
            for name in (
                "prepared_manifest_binding.txt",
                "preparation_provenance.txt",
                "upstream_preprocessed_manifest.sha256",
            ):
                (prepared / name).write_text("placeholder\n")
            payload.write_text("tampered\n")

            for relative in (
                "reverse/zc_kernel_api_b.f",
                "tangent/zc_kernel_api_d.f",
            ):
                target = generated / relative
                target.write_text("generated\n")
            for name in (
                "generated_source_manifest.pre_instrument.sha256",
                "generated_source_manifest.sha256",
                "generation_provenance.txt",
                "generation_manifest_binding.txt",
                "instrumentation_provenance.txt",
                "instrumentation_manifest_binding.txt",
            ):
                (generated / name).write_text("placeholder\n")

            kernel = Path(kernel_name)
            for name in [
                "openfl",
                "setup",
                "constc",
                "nrdhist",
                "initdat",
                "setup2",
                "close_files",
            ]:
                (kernel / f"{name}.F").write_text("passive\n")
            for name in ("zeq.common", "modified_means.common"):
                (kernel / name).write_text("include\n")
            (kernel / "ZC_lib_routines").mkdir()
            for name in [
                "GGNQF",
                "GGUBFS",
                "MDNRIS",
                "MERFI",
                "UERTST",
                "UGETIO",
                "USPKD",
            ]:
                (kernel / "ZC_lib_routines" / f"{name}.F").write_text("library\n")

            tapenade = Path(tap_name) / "tapenade"
            tapenade.mkdir()
            make_fake_tapenade(tapenade)
            result = subprocess.run(
                [
                    str(SCRIPTS / "compile_coupled_adjoint.sh"),
                    str(tapenade),
                    str(kernel),
                    str(run),
                ],
                env={**os.environ, "ADJOINT_OVERWRITE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")


class WrapperRollbackTests(unittest.TestCase):
    def test_post_backup_failure_restores_previous_managed_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "adjoint/tapenade_toolchain"
            scripts = fake / "scripts"
            build_root = fake / "build"
            scripts.mkdir(parents=True)
            build_root.mkdir()
            shutil.copy2(SCRIPTS / "run_coupled_toolchain_linux.sh", scripts)
            shutil.copy2(GUARD, scripts)
            shutil.copy2(TOOLCHAIN / "VERSION.env", fake)
            (scripts / "install_tapenade_linux.sh").write_text("installer\n")
            for name in (
                "prepare_coupled_sources.sh",
                "generate_coupled_nino3.sh",
                "instrument_generated_reverse.sh",
            ):
                path = scripts / name
                path.write_text("#!/usr/bin/env bash\nexit 0\n")
                path.chmod(0o755)
            compile_script = scripts / "compile_coupled_adjoint.sh"
            compile_script.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'invalid\\n' > \"$3/.zc_tapenade_build\"\n",
                encoding="utf-8",
            )
            compile_script.chmod(0o755)
            for relative in (
                "coupled/zc_kernel_nino3.F",
                "coupled/zc_adjoint_path_driver.F",
                "coupled/zc_nino3_adjoint_driver.F",
                "../full_tangent_audit/tangent_driver.F",
                "../full_tangent_audit/tangent_path_driver.F",
                "patches/test.patch",
            ):
                target = fake / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("input\n")

            kernel = Path(temporary) / "kernel"
            kernel.mkdir()
            for name in [
                "zc_kernel_api",
                "ssta",
                "ztmfc1",
                "cforce",
                "mloop",
                "akcalc",
                "bndary",
                "uhcalc",
                "uhinit",
                "tridag",
            ]:
                (kernel / f"{name}.F").write_text("source\n")
            for name in [
                "zc_kernel_state.inc",
                "zeq.common",
                "modified_means.common",
                "openfl.F",
                "setup.F",
                "constc.F",
                "nrdhist.F",
                "initdat.F",
                "setup2.F",
                "close_files.F",
            ]:
                (kernel / name).write_text("input\n")
            (kernel / "ZC_lib_routines").mkdir()
            for name in [
                "FFT2C",
                "GGNQF",
                "GGUBFS",
                "MDNRIS",
                "MERFI",
                "UERTST",
                "UGETIO",
                "USPKD",
            ]:
                (kernel / "ZC_lib_routines" / f"{name}.F").write_text("library\n")

            tapenade = Path(temporary) / "tapenade"
            tapenade.mkdir()
            make_fake_tapenade(tapenade, fake)
            target_build = build_root / "target"
            target_build.mkdir()
            (target_build / ".zc_tapenade_build").write_text(
                "managed-zc-tapenade-build-v1\n"
            )
            sentinel = target_build / "keep.txt"
            sentinel.write_text("old-good-build\n")

            result = subprocess.run(
                [
                    str(scripts / "run_coupled_toolchain_linux.sh"),
                    str(kernel),
                    str(tapenade),
                    str(target_build),
                ],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "FC": "true",
                    "CC": "true",
                    "AR": "true",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "old-good-build\n")
            self.assertFalse(any(build_root.glob(".coupled-backup.*")))
            self.assertFalse(any(build_root.glob(".coupled-build.*")))


class TangentAuditGuardTests(unittest.TestCase):
    def _make_bound_run(self, root: Path) -> tuple[Path, Path, Path]:
        run = root / "run"
        compiled = run / "coupled_compiled"
        snapshot = compiled / "build_input_snapshot"
        generated = snapshot / "generated/tangent"
        prepared = snapshot / "prepared"
        audit_inputs = snapshot / "audit"
        kernel = snapshot / "kernel_source"
        for directory in (generated, prepared, audit_inputs, kernel):
            directory.mkdir(parents=True, exist_ok=True)
        for name in ("zc_kernel_api_d.f", "zc_kernel_nino3_d.f"):
            (generated / name).write_text(f"source {name}\n", encoding="utf-8")
        (prepared / "zc_kernel_state.inc").write_text("state\n", encoding="utf-8")
        for name in ("tangent_path_driver.F", "tangent_driver.F"):
            (audit_inputs / name).write_text(f"driver {name}\n", encoding="utf-8")
        for name in (
            "close_files",
            "UGETIO",
            "constc",
            "setup",
            "openfl",
            "MDNRIS",
            "setup2",
            "USPKD",
            "nrdhist",
            "GGUBFS",
            "GGNQF",
            "MERFI",
            "initdat",
            "UERTST",
        ):
            (kernel / f"{name}.F").write_text(f"passive {name}\n", encoding="utf-8")

        compiler = root / "fake-fortran"
        compiler.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ ${1:-} == --version ]]; then echo 'fake Fortran 1'; exit 0; fi\n"
            "if [[ -n ${MUTATE_PATH:-} && ! -e ${MUTATE_ONCE:-} ]]; then\n"
            "  chmod u+w \"$MUTATE_PATH\"\n"
            "  printf 'mutation\\n' >> \"$MUTATE_PATH\"\n"
            "  : > \"$MUTATE_ONCE\"\n"
            "fi\n"
            "out=''\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  if [[ $1 == -o ]]; then out=$2; shift 2; else shift; fi\n"
            "done\n"
            "[[ -n $out ]]\n"
            "printf 'fake object or executable\\n' > \"$out\"\n"
            "chmod +x \"$out\"\n",
            encoding="utf-8",
        )
        compiler.chmod(0o755)
        input_manifest = compiled / "build_input_manifest.sha256"
        write_manifest(
            snapshot,
            input_manifest,
            [path for path in snapshot.rglob("*") if path.is_file()],
        )
        build_manifest = compiled / "build_manifest.txt"
        build_manifest.write_text(
            f"build_input_manifest_sha256={sha256(input_manifest)}\n"
            f"compiler_path={compiler}\n"
            f"compiler_sha256={sha256(compiler)}\n",
            encoding="utf-8",
        )
        return run, compiler, generated / "zc_kernel_api_d.f"

    @staticmethod
    def _write_old_output_set(output: Path) -> list[Path]:
        output_set = [
            output,
            Path(f"{output}_one_step"),
            Path(f"{output}.build_manifest.txt"),
            Path(f"{output}.build_inputs.sha256"),
        ]
        for member in output_set:
            member.write_text("old-good-output\n", encoding="utf-8")
        return output_set

    def test_tangent_publication_rolls_back_each_member_failure(self) -> None:
        audit = ROOT / "adjoint/full_tangent_audit"
        (audit / "build").mkdir(exist_ok=True)
        failpoints = [
            *(f"tangent_after_backup_{index}" for index in range(4)),
            *(f"tangent_after_publish_{index}" for index in range(4)),
            "tangent_before_commit",
        ]
        for failpoint in failpoints:
            with (
                self.subTest(failpoint=failpoint),
                tempfile.TemporaryDirectory() as root_name,
                tempfile.TemporaryDirectory(
                    prefix=".tangent-transaction-test.", dir=audit / "build"
                ) as output_parent_name,
            ):
                run, _, _ = self._make_bound_run(Path(root_name))
                output = Path(output_parent_name) / "tangent"
                output_set = self._write_old_output_set(output)
                result = subprocess.run(
                    [str(audit / "build_run_tangent.sh"), str(run), "O3", str(output)],
                    env={
                        **os.environ,
                        "ADJOINT_OVERWRITE": "1",
                        "ZC_TOOLCHAIN_TESTING": "1",
                        "ZC_TEST_FAILPOINT": failpoint,
                    },
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                for member in output_set:
                    self.assertEqual(
                        member.read_text(encoding="utf-8"), "old-good-output\n"
                    )
                self.assertFalse(any((audit / "build").glob(".tangent-backup.*")))

    def test_tangent_detects_snapshot_mutation_before_publication(self) -> None:
        audit = ROOT / "adjoint/full_tangent_audit"
        (audit / "build").mkdir(exist_ok=True)
        with (
            tempfile.TemporaryDirectory() as root_name,
            tempfile.TemporaryDirectory(
                prefix=".tangent-mutation-test.", dir=audit / "build"
            ) as output_parent_name,
        ):
            root = Path(root_name)
            run, _, mutate_path = self._make_bound_run(root)
            output = Path(output_parent_name) / "tangent"
            output_set = self._write_old_output_set(output)
            result = subprocess.run(
                [str(audit / "build_run_tangent.sh"), str(run), "O3", str(output)],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "MUTATE_PATH": str(mutate_path),
                    "MUTATE_ONCE": str(root / "mutated.once"),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            for member in output_set:
                self.assertEqual(
                    member.read_text(encoding="utf-8"), "old-good-output\n"
                )

    def test_tangent_signal_rolls_back_partial_output_set(self) -> None:
        audit = ROOT / "adjoint/full_tangent_audit"
        (audit / "build").mkdir(exist_ok=True)
        with (
            tempfile.TemporaryDirectory() as root_name,
            tempfile.TemporaryDirectory(
                prefix=".tangent-signal-test.", dir=audit / "build"
            ) as output_parent_name,
        ):
            run, _, _ = self._make_bound_run(Path(root_name))
            output = Path(output_parent_name) / "tangent"
            output_set = self._write_old_output_set(output)
            result = subprocess.run(
                [str(audit / "build_run_tangent.sh"), str(run), "O3", str(output)],
                env={
                    **os.environ,
                    "ADJOINT_OVERWRITE": "1",
                    "ZC_TOOLCHAIN_TESTING": "1",
                    "ZC_TEST_FAILPOINT": "tangent_after_publish_2",
                    "ZC_TEST_FAIL_MODE": "signal",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            for member in output_set:
                self.assertEqual(
                    member.read_text(encoding="utf-8"), "old-good-output\n"
                )

    def test_unmanifested_tangent_source_preserves_existing_output_set(self) -> None:
        audit = ROOT / "adjoint/full_tangent_audit"
        (audit / "build").mkdir(exist_ok=True)
        with (
            tempfile.TemporaryDirectory() as run_name,
            tempfile.TemporaryDirectory(
                prefix=".tangent-manifest-test.", dir=audit / "build"
            ) as output_parent_name,
        ):
            run = Path(run_name)
            compiled = run / "coupled_compiled"
            snapshot = compiled / "build_input_snapshot"
            generated = snapshot / "generated/tangent"
            prepared = snapshot / "prepared"
            audit_inputs = snapshot / "audit"
            for directory in (generated, prepared, audit_inputs):
                directory.mkdir(parents=True, exist_ok=True)
            required = [
                generated / "zc_kernel_api_d.f",
                generated / "zc_kernel_nino3_d.f",
                prepared / "zc_kernel_state.inc",
                audit_inputs / "tangent_path_driver.F",
                audit_inputs / "tangent_driver.F",
            ]
            for path in required:
                path.write_text(f"declared {path.name}\n", encoding="utf-8")
            input_manifest = compiled / "build_input_manifest.sha256"
            write_manifest(snapshot, input_manifest, required)
            (compiled / "build_manifest.txt").write_text(
                "build_input_manifest_sha256="
                f"{sha256(input_manifest)}\n",
                encoding="utf-8",
            )
            (generated / "unmanifested_d.f").write_text(
                "unmanifested\n", encoding="utf-8"
            )

            output = Path(output_parent_name) / "tangent"
            output_set = [
                output,
                Path(f"{output}_one_step"),
                Path(f"{output}.build_manifest.txt"),
                Path(f"{output}.build_inputs.sha256"),
            ]
            for member in output_set:
                member.write_text("old-good-output\n", encoding="utf-8")
            result = subprocess.run(
                [
                    str(audit / "build_run_tangent.sh"),
                    str(run),
                    "O3",
                    str(output),
                ],
                env={**os.environ, "ADJOINT_OVERWRITE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            for member in output_set:
                self.assertEqual(
                    member.read_text(encoding="utf-8"), "old-good-output\n"
                )


if __name__ == "__main__":
    unittest.main()
