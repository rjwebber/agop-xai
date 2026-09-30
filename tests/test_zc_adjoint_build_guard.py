"""Behavioral safety tests for managed Tapenade build directories."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = ROOT / "adjoint/tapenade_toolchain"
BUILD_ROOT = TOOLCHAIN / "build"
GUARD = TOOLCHAIN / "scripts/build_guard.sh"
COUPLED_WRAPPER = TOOLCHAIN / "scripts/run_coupled_toolchain_linux.sh"


class BuildGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.created_build_root = not BUILD_ROOT.exists()
        BUILD_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.created_build_root:
            BUILD_ROOT.rmdir()

    def test_build_symlink_is_rejected_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temporary:
            root = Path(temporary)
            target = root / "owned-target"
            target.mkdir()
            sentinel = target / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            link = BUILD_ROOT / f"{root.name}-build-link"
            link.symlink_to(target, target_is_directory=True)
            try:
                completed = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; ADJOINT_OVERWRITE=1; '
                        'zc_claim_build_dir "$2" "$3"',
                        "build-guard-test",
                        str(GUARD),
                        str(TOOLCHAIN),
                        str(link),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertRegex(completed.stderr, r"symbolic[- ]link")
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")
            finally:
                link.unlink(missing_ok=True)

    def test_stage_symlink_is_rejected_without_deleting_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temporary:
            build = Path(temporary)
            (build / ".zc_tapenade_build").write_text(
                "managed-zc-tapenade-build-v1\n", encoding="utf-8"
            )
            victim = build / "victim"
            victim.mkdir()
            payload = victim / "keep.txt"
            payload.write_text("keep\n", encoding="utf-8")
            stage = build / "coupled_compiled"
            stage.symlink_to(victim, target_is_directory=True)

            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; ADJOINT_OVERWRITE=1; '
                    'zc_prepare_stage_dir "$2" "$3"',
                    "build-guard-test",
                    str(GUARD),
                    str(build),
                    str(stage),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertRegex(completed.stderr, r"symbolic[- ]link")
            self.assertEqual(payload.read_text(encoding="utf-8"), "keep\n")

    def test_unowned_nonempty_build_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temporary:
            build = Path(temporary)
            (build / "user-file.txt").write_text("keep\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; zc_claim_build_dir "$2" "$3"',
                    "build-guard-test",
                    str(GUARD),
                    str(TOOLCHAIN),
                    str(build),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not owned", completed.stderr)
            self.assertTrue((build / "user-file.txt").is_file())

    def test_symlinked_marker_cannot_authorize_build_deletion(self) -> None:
        with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temporary:
            build = Path(temporary)
            sentinel = build / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            marker_target = build.parent / f"{build.name}-marker-content.txt"
            marker_target.write_text(
                "managed-zc-tapenade-build-v1\n", encoding="utf-8"
            )
            (build / ".zc_tapenade_build").symlink_to(marker_target)
            try:
                completed = subprocess.run(
                    [
                        "bash",
                        "-c",
                        'source "$1"; ADJOINT_OVERWRITE=1; '
                        'zc_claim_build_dir "$2" "$3"',
                        "build-guard-test",
                        str(GUARD),
                        str(TOOLCHAIN),
                        str(build),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("not owned", completed.stderr)
                self.assertEqual(sentinel.read_text(), "keep\n")
                self.assertEqual(
                    marker_target.read_text(), "managed-zc-tapenade-build-v1\n"
                )
            finally:
                marker_target.unlink(missing_ok=True)

    def test_wrapper_preflight_failure_preserves_existing_build(self) -> None:
        with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temporary:
            build = Path(temporary)
            (build / ".zc_tapenade_build").write_text(
                "managed-zc-tapenade-build-v1\n", encoding="utf-8"
            )
            sentinel = build / "known-good-result.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            missing_kernel = build.parent / "missing-kernel-source"
            missing_tapenade = build.parent / "missing-tapenade"

            completed = subprocess.run(
                [
                    str(COUPLED_WRAPPER),
                    str(missing_kernel),
                    str(missing_tapenade),
                    str(build),
                ],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "ADJOINT_OVERWRITE": "1"},
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("input is missing", completed.stderr)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
