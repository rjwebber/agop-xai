"""Contract and numerical tests for the isolated variable-window drivers."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = ROOT / "adjoint" / "variable_window"
CANONICAL = (
    ROOT / "adjoint" / "tapenade_toolchain" / "build" / "coupled_run23_final3"
)
COMPILED = CANONICAL / "coupled_compiled"
CASE = ROOT / "outputs" / "zc_adjoint" / "run23_final3_case_replay" / "extreme_el_nino"
RUNTIME = CASE / "runtime"
REFERENCE_PATH = CASE / "artifacts" / "zc_31step_path.bin"
REFERENCE_GRADIENT = CASE / "artifacts" / "zc_nino3_gradient.bin"
NREAL = 59_148
STATE_BYTES = 278_864
TAPE_BYTES_PER_STEP = 3 * 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


class VariableWindowContractTests(unittest.TestCase):
    def test_drivers_are_dynamic_and_isolated_from_canonical_sources(self) -> None:
        path_source = (VARIABLE / "zc_variable_path_driver.F").read_text(
            encoding="utf-8"
        )
        reverse_source = (VARIABLE / "zc_variable_adjoint_driver.F").read_text(
            encoding="utf-8"
        )
        tangent_source = (VARIABLE / "zc_variable_tangent_driver.F").read_text(
            encoding="utf-8"
        )
        runtime_source = (VARIABLE / "zc_variable_runtime.F").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("PARAMETER (TARGET_STEPS=31", path_source)
        self.assertNotIn("PARAMETER (TARGET_STEPS=31", reverse_source)
        self.assertNotIn("PARAMETER (TARGET_STEPS=31", tangent_source)
        self.assertIn("GET_COMMAND_ARGUMENT(3,STEPS_TEXT)", path_source)
        self.assertIn("DO 20 N=1,NSTEPS", path_source)
        self.assertIn("DO 60 N=NSTEPS,1,-1", reverse_source)
        self.assertIn("DO 100 N=1,NSTEPS", tangent_source)
        self.assertIn("NSTEPS.GT.MAX_STEPS", runtime_source)

        build_script = (VARIABLE / "build_variable_window.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("verify_canonical_objects", build_script)
        self.assertIn("canonical_final3_objects.sha256", build_script)
        self.assertNotIn("adjoint/controlled_window", build_script)
        self.assertNotIn("rm -rf -- \"${compiled}", build_script)

    def test_canonical_allowlist_is_unique_and_path_safe(self) -> None:
        entries: list[tuple[str, str]] = []
        for line in (
            VARIABLE / "canonical_final3_objects.sha256"
        ).read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            self.assertEqual(len(digest), 64)
            self.assertTrue(
                all(character in "0123456789abcdef" for character in digest)
            )
            path = Path(relative)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            entries.append((digest, relative))
        self.assertEqual(len(entries), 45)
        self.assertEqual(len({relative for _, relative in entries}), len(entries))
        self.assertIn((
            "8501b8159096afd17156c2bd4ff27bafcdd387866c58d0d8bb3bb6ac6ce5f7e9",
            "build_manifest.txt",
        ), entries)


class VariableWindowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (
            COMPILED / "build_manifest.txt",
            COMPILED / "zc_adjoint_make_path",
            COMPILED / "zc_nino3_adjoint",
            RUNTIME / "kernel_initial_state.bin",
            REFERENCE_PATH,
            REFERENCE_GRADIENT,
        )
        if any(not path.is_file() for path in required):
            raise unittest.SkipTest("local authentic final3 build/case is unavailable")
        build = manifest_values(COMPILED / "build_manifest.txt")
        compiler = Path(build.get("compiler_path", ""))
        if (
            not compiler.is_file()
            or compiler.is_symlink()
            or sha256(compiler) != build.get("compiler_sha256")
        ):
            raise unittest.SkipTest("exact canonical final3 compiler is unavailable")

    def run_driver(
        self, executable: Path, *arguments: os.PathLike[str] | str, cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(executable), *(str(argument) for argument in arguments)],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
            timeout=60,
        )

    def test_31step_compatibility_and_40step_transpose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_dir = root / "build"
            subprocess.run(
                [
                    str(VARIABLE / "build_variable_window.sh"),
                    str(CANONICAL),
                    str(build_dir),
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
                timeout=60,
            )
            path_executable = build_dir / "zc_variable_make_path"
            reverse_executable = build_dir / "zc_variable_adjoint"
            tangent_executable = build_dir / "zc_variable_tangent"

            runtime10 = root / "runtime10"
            shutil.copytree(RUNTIME, runtime10)
            path10 = root / "path10.bin"
            self.run_driver(
                path_executable,
                "kernel_initial_state.bin",
                path10,
                "10",
                cwd=runtime10,
            )
            self.assertEqual(
                path10.stat().st_size,
                7 * 4 + 11 * STATE_BYTES + 10 * TAPE_BYTES_PER_STEP,
            )
            self.assertIn(
                "transitions=10",
                (root / "path10.bin.txt").read_text(encoding="utf-8"),
            )

            runtime31 = root / "runtime31"
            shutil.copytree(RUNTIME, runtime31)
            path31 = root / "path31.bin"
            self.run_driver(
                path_executable,
                "kernel_initial_state.bin",
                path31,
                "31",
                cwd=runtime31,
            )
            self.assertEqual(sha256(path31), sha256(REFERENCE_PATH))
            prefix31 = root / "adj31"
            self.run_driver(
                reverse_executable,
                path31,
                prefix31,
                cwd=runtime31,
            )
            self.assertEqual(
                sha256(root / "adj31_gradient.bin"), sha256(REFERENCE_GRADIENT)
            )

            runtime40 = root / "runtime40"
            shutil.copytree(RUNTIME, runtime40)
            path40 = root / "path40.bin"
            self.run_driver(
                path_executable,
                "kernel_initial_state.bin",
                path40,
                "40",
                cwd=runtime40,
            )
            self.assertEqual(
                path40.stat().st_size,
                7 * 4 + 41 * STATE_BYTES + 40 * TAPE_BYTES_PER_STEP,
            )
            self.assertIn(
                "transitions=40",
                (root / "path40.bin.txt").read_text(encoding="utf-8"),
            )

            rng = np.random.default_rng(20_260_906)
            direction = (
                rng.normal(size=NREAL).astype(np.float32) * np.float32(1.0e-3)
            )
            terminal_seed = rng.normal(size=NREAL).astype(np.float32) / np.float32(
                math.sqrt(NREAL)
            )
            direction_path = root / "direction.bin"
            seed_path = root / "terminal_seed.bin"
            direction.tofile(direction_path)
            terminal_seed.tofile(seed_path)
            tangent_prefix = root / "tan40"
            reverse_prefix = root / "adj40"
            self.run_driver(
                tangent_executable,
                path40,
                direction_path,
                tangent_prefix,
                cwd=runtime40,
            )
            self.run_driver(
                reverse_executable,
                path40,
                reverse_prefix,
                seed_path,
                cwd=runtime40,
            )
            jv = np.fromfile(root / "tan40_jv.bin", dtype="<f4")
            gradient = np.fromfile(root / "adj40_gradient.bin", dtype="<f4")
            self.assertEqual(jv.shape, (NREAL,))
            self.assertEqual(gradient.shape, (NREAL,))
            self.assertTrue(np.isfinite(jv).all())
            self.assertTrue(np.isfinite(gradient).all())
            left = float(
                np.dot(terminal_seed.astype(np.float64), jv.astype(np.float64))
            )
            right = float(
                np.dot(direction.astype(np.float64), gradient.astype(np.float64))
            )
            relative_error = abs(left - right) / max(abs(left), abs(right), 1.0e-300)
            self.assertLess(relative_error, 2.0e-5)


if __name__ == "__main__":
    unittest.main()
