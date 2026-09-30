"""Static contract tests for the source-transformed ZC adjoint build."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ADJOINT_ROOT = REPOSITORY_ROOT / "adjoint"
ZC_V3_ROOT = REPOSITORY_ROOT / "data/processed/zc-v3"


class CoupledNino3HeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            ADJOINT_ROOT / "tapenade_toolchain/coupled/zc_kernel_nino3.F"
        ).read_text()
        self.manifest = json.loads(
            (ADJOINT_ROOT / "fortran_kernel/state_manifest.json").read_text()
        )

    def test_head_uses_checkpoint_to_target_transition_count(self) -> None:
        match = re.search(r"PARAMETER \(TARGET_STEPS=(\d+),", self.source)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 31)

    def test_head_uses_canonical_center_inclusive_nino3_box(self) -> None:
        self.assertRegex(self.source, r"DO\s+20\s+J=21,31")
        self.assertRegex(self.source, r"DO\s+10\s+I=13,18")
        self.assertRegex(self.source, r"REAL\s+NINO3\s*\n\s*REAL\*8\s+NINO3_SUM")
        self.assertRegex(self.source, r"NINO3_SUM=0\.0D0")
        self.assertRegex(
            self.source,
            r"NINO3_SUM=NINO3_SUM\+DBLE\(STATE_R_OUT\(K\)\)",
        )
        self.assertRegex(self.source, r"NINO3=NINO3_SUM/66\.0D0")
        self.assertNotRegex(self.source, r"NINO3=NINO3\+STATE_R_OUT\(K\)/66\.0")
        self.assertNotRegex(self.source, r"DO\s+20\s+J=20,30")

    def test_double_reduction_matches_every_released_target_bitwise(self) -> None:
        """Emulate the Fortran REAL*8 reduction on the complete zc-v3 series."""

        field_path = ZC_V3_ROOT / "sst_anomaly.npy"
        target_path = ZC_V3_ROOT / "nino3_index.npy"
        if not field_path.is_file() or not target_path.is_file():
            self.skipTest("complete zc-v3 processed arrays are not installed")

        fields = np.load(field_path, mmap_mode="r")
        released = np.load(target_path, mmap_mode="r")
        self.assertEqual(fields.shape, (released.size, 20, 27))

        # Raw Fortran I=13:18 is stored north-to-south as released rows 12:7;
        # raw J=21:31 is stored as released columns 15:25.  Iterating longitude
        # outside latitude reproduces the scalar head's accumulation order.
        chunk_size = 32_768
        for start in range(0, released.size, chunk_size):
            stop = min(start + chunk_size, released.size)
            total = np.zeros(stop - start, dtype=np.float64)
            for column in range(15, 26):
                for row in range(12, 6, -1):
                    values = fields[start:stop, row, column]
                    total += np.asarray(values, dtype=np.float64)
            emulated = np.asarray(total / 66.0, dtype=np.float32)
            expected = np.asarray(released[start:stop], dtype=np.float32)
            self.assertTrue(
                np.array_equal(emulated.view(np.uint32), expected.view(np.uint32)),
                msg=f"canonical Nino-3 reduction mismatch in [{start}:{stop}]",
            )

    def test_packed_to_offset_matches_state_manifest(self) -> None:
        match = re.search(r"TO_OFFSET=(\d+)\)", self.source)
        self.assertIsNotNone(match)
        target = self.manifest["named_offsets_for_target_and_attribution"]["TO"]
        start = int(target["python_slice"].split(":", maxsplit=1)[0])
        self.assertEqual(int(match.group(1)), start)


class TapenadeProvenanceTests(unittest.TestCase):
    def test_pinned_toolchain_records_version_revision_and_archive_hash(self) -> None:
        values: dict[str, str] = {}
        version_file = ADJOINT_ROOT / "tapenade_toolchain/VERSION.env"
        for line in version_file.read_text().splitlines():
            if line and not line.startswith("#"):
                key, value = line.split("=", maxsplit=1)
                values[key] = value
        self.assertEqual(values["TAPENADE_VERSION"], "3.16")
        self.assertRegex(values["TAPENADE_REVISION"], r"^[0-9a-f]{40}$")
        self.assertRegex(values["TAPENADE_ARCHIVE_SHA256"], r"^[0-9a-f]{64}$")
        self.assertTrue(values["TAPENADE_ARCHIVE_URL"].startswith("https://"))


class CoupledBuildScriptContractTests(unittest.TestCase):
    def setUp(self) -> None:
        scripts = ADJOINT_ROOT / "tapenade_toolchain/scripts"
        self.compile_script = (scripts / "compile_coupled_adjoint.sh").read_text()
        self.release_script = (scripts / "run_coupled_toolchain_linux.sh").read_text()
        self.prepare_script = (scripts / "prepare_coupled_sources.sh").read_text()
        self.generate_script = (scripts / "generate_coupled_nino3.sh").read_text()
        self.build_guard = (scripts / "build_guard.sh").read_text()
        self.case_runner = (scripts / "run_coupled_case.py").read_text()
        self.instrument_script = (
            scripts / "instrument_generated_reverse.sh"
        ).read_text()
        self.install_script = (scripts / "install_tapenade_linux.sh").read_text()
        self.tangent_build_script = (
            ADJOINT_ROOT / "full_tangent_audit/build_run_tangent.sh"
        ).read_text()

    def test_primal_and_derivative_compilation_modes_are_distinct(self) -> None:
        # O3 is part of the authentic single-precision trajectory. Derivative
        # debugging deliberately uses O0 plus runtime bounds checking.
        self.assertRegex(
            self.compile_script,
            r"debug\)\s+fflags=\(-std=legacy -O0 -g -fcheck=all",
        )
        self.assertRegex(
            self.compile_script,
            r"primal_fflags=\(-std=legacy -O3 -fbacktrace",
        )
        self.assertIn(
            'compile_primal_fortran "${compile_prepared_dir}/${name}.f"',
            self.compile_script,
        )
        self.assertIn(
            'compile_primal_fortran "${compile_kernel_source}/${name}.F"',
            self.compile_script,
        )
        self.assertIn(
            'for source_file in "${compile_generated_dir}/reverse/"*.f',
            self.compile_script,
        )
        self.assertIn(
            'compile_fortran "${source_file}" "${reverse_objects}/${name}.o"',
            self.compile_script,
        )

    def test_compile_refuses_uninstrumented_or_undefined_scratch_reverse(self) -> None:
        compile_markers = (
            "tape(3) = kernel_atm_reset",
            "h_minus_eng = 0.0",
            "beta_factor(i, j) = 0.0",
            "c2 = 0.0D0",
        )
        for marker in compile_markers:
            self.assertIn(marker, self.compile_script)

        # The instrumentation gate checks every generated file touched by the
        # defined-scratch patch, not just the representative compile markers.
        for marker in (
            "a = 0.0",
            "at = 0.0",
            "rx = 0.0",
            "dn = 0.0",
            "d(j) = 0.0",
            "r1(j) = 0.0",
        ):
            self.assertIn(marker, self.instrument_script)
        self.assertIn("generated_reverse_defined_scratch.patch", self.instrument_script)

    def test_nonempty_build_targets_require_explicit_overwrite(self) -> None:
        self.assertIn("has_payload=0", self.release_script)
        self.assertIn("zc_publish_stage_directory", self.release_script)
        self.assertIn("zc_assert_stage_replacement_allowed", self.compile_script)
        self.assertIn('find "${build_dir}"', self.build_guard)
        self.assertIn('${ADJOINT_OVERWRITE:-0}', self.build_guard)
        self.assertIn("ADJOINT_OVERWRITE=1", self.build_guard)
        self.assertIn("managed-zc-tapenade-build-v1", self.build_guard)
        self.assertIn("return 2", self.build_guard)

    def test_release_wrapper_builds_before_replacing_previous_result(self) -> None:
        fresh_build = self.release_script.index(
            '"${toolchain_dir}/scripts/prepare_coupled_sources.sh"'
        )
        publish = self.release_script.index(
            'zc_publish_stage_directory "${toolchain_dir}/build"'
        )
        self.assertLess(fresh_build, publish)
        self.assertIn('ADFirstAidKit/adStack.h', self.release_script)
        self.assertIn('ADFirstAidKit/adComplex.h', self.release_script)
        self.assertRegex(self.release_script, r'Java 17 is required')

    def test_destructive_build_targets_are_scoped_to_the_toolchain_build(self) -> None:
        self.assertIn('"${toolchain_dir}/build/"*', self.build_guard)
        self.assertIn("build target must be a strict child", self.build_guard)
        self.assertIn("zc_paths_overlap", self.build_guard)
        for script in (
            self.release_script,
            self.prepare_script,
            self.generate_script,
            self.compile_script,
        ):
            self.assertIn('source "${toolchain_dir}/scripts/build_guard.sh"', script)
            self.assertIn("zc_claim_build_dir", script)

    def test_case_runner_only_overwrites_a_managed_output_directory(self) -> None:
        self.assertIn('OUTPUT_MARKER = ".zc_adjoint_case_output"', self.case_runner)
        self.assertIn("refusing to delete a nonempty directory", self.case_runner)

    def test_audit_tangent_uses_only_the_bound_build_snapshot(self) -> None:
        script = self.tangent_build_script
        self.assertIn('snapshot="${compiled}/build_input_snapshot"', script)
        self.assertIn('generated="${snapshot}/generated/tangent"', script)
        self.assertIn('kernel_source="${snapshot}/kernel_source"', script)
        self.assertIn('driver="${snapshot}/audit/tangent_path_driver.F"', script)
        self.assertIn(
            'one_step_driver="${snapshot}/audit/tangent_driver.F"', script
        )
        self.assertIn(
            'zc_verify_sha256_manifest_exact "${snapshot}" "${input_manifest}"',
            script,
        )
        self.assertIn("build_input_manifest_sha256", script)
        self.assertIn("compiler_sha256", script)
        self.assertIn("builder_script_sha256", script)
        self.assertNotIn("outputs/zc_adjoint/kernel_build/source", script)

    def test_audit_tangent_output_is_scoped_and_explicit(self) -> None:
        script = self.tangent_build_script
        self.assertIn("usage: build_run_tangent.sh RUN_DIR O0|O2|O3 OUTPUT", script)
        self.assertIn('"${audit_dir}/build/"*', script)
        self.assertIn("tangent output may not be a symbolic link", script)
        self.assertIn("tangent output-set member may not be a symbolic link", script)
        self.assertIn('one_step_output="${output}_one_step"', script)
        self.assertIn("Publish only after both executables", script)
        self.assertIn("ADJOINT_OVERWRITE=1", script)

    def test_compile_is_bound_to_an_immutable_input_snapshot(self) -> None:
        self.assertIn("build_input_manifest.sha256", self.compile_script)
        self.assertIn("build_input_snapshot", self.compile_script)
        self.assertIn(
            'compile_prepared_dir="${input_snapshot}/prepared"',
            self.compile_script,
        )
        self.assertIn(
            'compile_generated_dir="${input_snapshot}/generated"',
            self.compile_script,
        )
        self.assertIn("compile_recipe_sha256", self.compile_script)
        self.assertIn(
            'audit/tangent_driver.F',
            self.compile_script,
        )

    def test_generation_enforces_java17_and_fresh_pinned_extraction(self) -> None:
        self.assertIn("Always extract a fresh tree", self.install_script)
        self.assertIn("mktemp -d", self.install_script)
        self.assertNotIn(
            'if [[ ! -x "${install_dir}/bin/linux/fortranParser" ]]',
            self.install_script,
        )
        self.assertIn("Java 17 is required", self.generate_script)
        self.assertIn("java_sha256", self.generate_script)
        self.assertIn("tapenade_executable_sha256", self.generate_script)
        self.assertIn("fortran_parser_sha256", self.generate_script)
        self.assertIn(
            'tapenade/ADFirstAidKit/adStack.h',
            self.compile_script,
        )

    def test_every_stage_verifies_manifest_chain_before_replacement(self) -> None:
        self.assertLess(
            self.generate_script.index("zc_verify_sha256_manifest_exact"),
            self.generate_script.index("zc_publish_stage_directory"),
        )
        self.assertLess(
            self.instrument_script.index("zc_verify_sha256_manifest_exact"),
            self.instrument_script.index('"${patch_path}" --quiet'),
        )
        self.assertLess(
            self.compile_script.index("zc_verify_sha256_manifest_exact"),
            self.compile_script.index("zc_publish_stage_directory"),
        )
        for script in (
            self.prepare_script,
            self.generate_script,
            self.instrument_script,
            self.compile_script,
        ):
            self.assertIn("manifest_binding", script)
        self.assertIn("kernel_source_input_manifest", self.prepare_script)
        self.assertIn(
            'zc_verify_sha256_manifest "${kernel_source}"', self.compile_script
        )

    def test_tapenade_install_receipt_binds_complete_extracted_tree(self) -> None:
        self.assertIn(".zc_tapenade_tree_manifest.sha256", self.install_script)
        self.assertIn("observed_archive_sha256", self.install_script)
        self.assertIn("zc_verify_tapenade_install", self.install_script)
        self.assertIn("zc_verify_tapenade_install", self.generate_script)
        self.assertIn("zc_verify_tapenade_install", self.release_script)
        self.assertIn("zc_verify_sha256_manifest_exact", self.build_guard)

    def test_compilation_uses_controlled_empty_working_directory(self) -> None:
        self.assertIn(
            'controlled_cwd="${compile_dir}/controlled_cwd"',
            self.compile_script,
        )
        self.assertIn('(cd "${controlled_cwd}" &&', self.compile_script)
        self.assertIn("controlled_compile_cwd", self.compile_script)
        self.assertIn(
            'compile_cwd="${stage}/controlled_cwd"',
            self.tangent_build_script,
        )
        self.assertIn('(cd "${compile_cwd}" &&', self.tangent_build_script)

    def test_platform_flags_cover_tapenade_c_runtime_compile(self) -> None:
        self.assertIn(
            '"${cc_path}" "${cflags[@]}" "${platform_flags[@]}" -c',
            self.compile_script,
        )
        self.assertIn(
            "c_compile=<CC> <C_FLAGS> <PLATFORM_FLAGS>", self.compile_script
        )

    def test_prepare_resolves_symlinked_cpp_and_patch_commands(self) -> None:
        self.assertIn(
            'cpp_path="$(zc_resolve_path "${cpp_path}")"',
            self.prepare_script,
        )
        self.assertIn(
            'patch_path="$(zc_resolve_path "$(command -v patch)")"',
            self.prepare_script,
        )

    def test_compile_patch_inventory_is_built_and_verified_from_snapshot(self) -> None:
        snapshot_manifest = (
            '${input_snapshot}/manifests/toolchain_patch_manifest.sha256'
        )
        self.assertIn(snapshot_manifest, self.compile_script)
        self.assertIn(
            '"${input_snapshot}/toolchain/patches"', self.compile_script
        )
        self.assertIn('"${snapshot_patch_manifest}"', self.compile_script)
        self.assertLess(
            self.compile_script.index(
                'cd "${input_snapshot}/toolchain/patches"'
            ),
            self.compile_script.index("verify_compile_inputs_unchanged()"),
        )

    def test_case_runner_is_bound_into_build_and_reports_its_hash(self) -> None:
        self.assertIn("run_coupled_case.py", self.compile_script)
        self.assertIn("bound_case_runner_sha256", self.case_runner)
        self.assertIn("runner_sha256 != bound_runner_sha256", self.case_runner)

    def test_publication_paths_are_transactional_and_signal_aware(self) -> None:
        self.assertIn(".zc-stage-backup.XXXXXX", self.build_guard)
        self.assertIn("cleanup_stage_publication", self.build_guard)
        self.assertIn("trap 'exit 143' TERM", self.build_guard)
        self.assertIn("zc_publish_stage_directory", self.release_script)
        self.assertIn(".tangent-backup.XXXXXX", self.tangent_build_script)
        self.assertIn("cleanup_publication", self.tangent_build_script)
        self.assertIn('ln "${staged_outputs[${index}]}"', self.tangent_build_script)


class ReverseDriverContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            ADJOINT_ROOT
            / "tapenade_toolchain/coupled/zc_nino3_adjoint_driver.F"
        ).read_text()
        self.path_source = (
            ADJOINT_ROOT
            / "tapenade_toolchain/coupled/zc_adjoint_path_driver.F"
        ).read_text()

    def test_reverse_uses_an_independent_authentic_primal_path(self) -> None:
        self.assertIn("CALL ZC_AD_READ_PATH(", self.source)
        self.assertNotIn("CALL ZC_KERNEL_WINDOW(", self.source)
        self.assertIn("CALL ZC_KERNEL_WINDOW(", self.path_source)
        self.assertNotIn("CALL ZC_KERNEL_WINDOW_FWD(", self.path_source)
        self.assertNotIn("CALL ZC_KERNEL_WINDOW_BWD(", self.path_source)
        self.assertIn("producer=normalized authentic primal kernel", self.path_source)

    def test_driver_runs_one_forward_sweep_before_its_reverse(self) -> None:
        forward_call = self.source.index("CALL ZC_KERNEL_WINDOW_FWD(")
        reverse_call = self.source.index("CALL ZC_KERNEL_WINDOW_BWD(")
        self.assertLess(forward_call, reverse_call)

    def test_driver_reports_primal_scalar_from_the_forward_state(self) -> None:
        self.assertRegex(self.source, r"DOUBLE PRECISION NORM2,NINO3_SUM")
        self.assertRegex(self.source, r"NINO3_SUM=0\.0D0")
        self.assertRegex(self.source, r"DO\s+\d+\s+J=21,31")
        self.assertRegex(self.source, r"DO\s+\d+\s+I=13,18")
        self.assertRegex(
            self.source,
            r"NINO3_SUM=NINO3_SUM\+DBLE\(STATE_R_"
            r"(?:OUT\(K\)|PATH\(K,TARGET_STEPS\))\)",
        )
        self.assertRegex(self.source, r"NINO3=NINO3_SUM/66\.0D0")
        self.assertIn("'final_nino3_celsius=',NINO3", self.source)

    def test_driver_seeds_only_the_canonical_target_box(self) -> None:
        self.assertRegex(
            self.source,
            r"DO\s+(\d+)\s+I=1,ZC_STATE_NREAL\s+"
            r"STATE_R_OUTB\(I\)=0\.0\s+\1\s+CONTINUE",
        )
        self.assertRegex(self.source, r"DO\s+\d+\s+J=21,31")
        self.assertRegex(self.source, r"DO\s+\d+\s+I=13,18")
        self.assertRegex(
            self.source,
            r"K=TO_OFFSET\+I\+30\*\(J-1\)\s+"
            r"STATE_R_OUTB\(K\)=1\.0/66\.0",
        )

    def test_generated_reverse_overwrites_the_input_cotangent(self) -> None:
        """The user driver may leave STATE_R_INB uninitialized only if BWD zeros it."""

        candidates = list(
            (ADJOINT_ROOT / "tapenade_toolchain/build").glob(
                "coupled_run*/coupled_generated/reverse/zc_kernel_api_b.f"
            )
        )
        if not candidates:
            self.skipTest("no generated reverse wrapper is installed")

        def generation_number(path: Path) -> int:
            match = re.match(r"coupled_run(\d+)", path.parents[2].name)
            if match is None:
                return -1
            return int(match.group(1))

        generated = max(
            candidates,
            key=lambda path: (generation_number(path), path.parents[2].name),
        ).read_text()
        window = generated[
            generated.index("SUBROUTINE ZC_KERNEL_WINDOW_BWD") :
            generated.index("SUBROUTINE ZC_KERNEL_PACK_BWD")
        ]
        self.assertRegex(
            window,
            r"DO ii1=1,zc_state_nreal\s+"
            r"state_r_inb\(ii1\) = 0\.0\s+ENDDO\s+"
            r"CALL ZC_KERNEL_UNPACK_BWD",
        )

    def test_driver_does_not_assume_reverse_recomputes_the_scalar(self) -> None:
        self.assertNotIn("CALL ZC_KERNEL_NINO3_B(", self.source)
        self.assertNotIn("NINO3_DUMMY", self.source)


if __name__ == "__main__":
    unittest.main()
