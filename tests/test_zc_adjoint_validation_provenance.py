"""Adversarial provenance tests for tangent/adjoint validation reports."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "adjoint/full_tangent_audit"
sys.path.insert(0, str(AUDIT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "zc_validate_adjoint_dot_provenance",
    AUDIT_ROOT / "validate_adjoint_dot.py",
)
assert SPEC is not None and SPEC.loader is not None
DOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOT)


def write_manifest(root: Path, path: Path, files: list[Path]) -> None:
    path.write_text(
        "".join(
            f"{DOT.sha256(file)}  {file.relative_to(root).as_posix()}\n"
            for file in sorted(files)
        ),
        encoding="utf-8",
    )


class ScalarProducerBindingTests(unittest.TestCase):
    def test_bound_command_executes_from_the_private_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "private-runtime"
            runtime.mkdir()
            (runtime / "fc.data").write_text("bound\n", encoding="utf-8")
            source_executable = root / "fake-derivative"
            source_executable.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "test -f fc.data\n"
                "pwd > \"$1\"\n",
                encoding="utf-8",
            )
            staged_executable = root / "staged/fake-derivative"
            DOT.stage_verified_file(
                source_executable,
                staged_executable,
                DOT.sha256(source_executable),
            )
            staged_executable.chmod(0o700)
            observed_cwd = root / "observed-cwd.txt"
            DOT.run_bound_command(
                [str(staged_executable), str(observed_cwd)],
                cwd=runtime,
                log_path=root / "run.log",
                label="fake derivative",
            )
            self.assertEqual(
                Path(observed_cwd.read_text(encoding="utf-8").strip()),
                runtime.resolve(),
            )

    def test_scalar_gradient_must_match_the_reverse_producer_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "path.bin"
            gradient = root / "gradient.bin"
            reverse = root / "reverse"
            coupled_manifest = root / "build_manifest.txt"
            coupled_inputs = root / "build_input_manifest.sha256"
            report_path = root / "run_report.json"
            path.write_bytes(b"path")
            gradient.write_bytes(b"gradient")
            reverse.write_bytes(b"reverse")
            coupled_inputs.write_text(
                f"{DOT.sha256(path)}  path.bin\n", encoding="utf-8"
            )
            coupled_manifest.write_text(
                f"build_input_manifest_sha256={DOT.sha256(coupled_inputs)}\n",
                encoding="utf-8",
            )
            report = {
                "schema_version": 1,
                "status": "completed",
                "scientific_contract": {
                    "transitions": 31,
                    "real_state_length": DOT.NREAL,
                    "objective": "canonical Nino-3 SST anomaly",
                    "configuration_verified": True,
                },
                "build": {
                    "adjoint_executable_sha256": DOT.sha256(reverse),
                    "executed_staged_executable_sha256": {
                        reverse.name: DOT.sha256(reverse)
                    },
                    "build_manifest_sha256": DOT.sha256(coupled_manifest),
                    "build_input_manifest_sha256": DOT.sha256(coupled_inputs),
                },
                "results": {
                    "output_sha256": {
                        "artifacts/zc_31step_path.bin": DOT.sha256(path),
                        "artifacts/zc_nino3_gradient.bin": DOT.sha256(gradient),
                    }
                },
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")

            binding = DOT.verify_scalar_gradient_producer(
                run_report_path=report_path,
                path_file=path,
                gradient_file=gradient,
                reverse_executable=reverse,
                coupled_build_manifest=coupled_manifest,
                coupled_build_input_manifest=coupled_inputs,
            )
            self.assertEqual(binding["gradient_sha256"], DOT.sha256(gradient))

            gradient.write_bytes(b"different")
            with self.assertRaisesRegex(RuntimeError, "supplied gradient"):
                DOT.verify_scalar_gradient_producer(
                    run_report_path=report_path,
                    path_file=path,
                    gradient_file=gradient,
                    reverse_executable=reverse,
                    coupled_build_manifest=coupled_manifest,
                    coupled_build_input_manifest=coupled_inputs,
                )

            gradient.write_bytes(b"gradient")
            report["build"]["executed_staged_executable_sha256"][reverse.name] = (
                "0" * 64
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "executed reverse bytes"):
                DOT.verify_scalar_gradient_producer(
                    run_report_path=report_path,
                    path_file=path,
                    gradient_file=gradient,
                    reverse_executable=reverse,
                    coupled_build_manifest=coupled_manifest,
                    coupled_build_input_manifest=coupled_inputs,
                )

            report["build"]["executed_staged_executable_sha256"][reverse.name] = (
                DOT.sha256(reverse)
            )
            report["build"]["build_input_manifest_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "input manifest"):
                DOT.verify_scalar_gradient_producer(
                    run_report_path=report_path,
                    path_file=path,
                    gradient_file=gradient,
                    reverse_executable=reverse,
                    coupled_build_manifest=coupled_manifest,
                    coupled_build_input_manifest=coupled_inputs,
                )

    def test_producer_runtime_is_exactly_bound_and_privately_staged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "case"
            runtime = case / "runtime"
            runtime.mkdir(parents=True)
            (runtime / "fc.data").write_text("config\n", encoding="utf-8")
            (runtime / "Data").mkdir()
            (runtime / "Data/ocean.dat").write_bytes(b"ocean")
            manifest = {
                "fc.data": DOT.sha256(runtime / "fc.data"),
                "Data/ocean.dat": DOT.sha256(runtime / "Data/ocean.dat"),
            }
            report = case / "run_report.json"
            report.write_text(
                json.dumps(
                    {"checkpoint_input": {"staged_file_sha256": manifest}}
                ),
                encoding="utf-8",
            )

            observed, binding = DOT.verify_producer_runtime(runtime, report)
            self.assertEqual(observed, manifest)
            self.assertEqual(binding["runtime_file_count"], 2)
            staged = case / "private-runtime"
            DOT.stage_producer_runtime(runtime, staged, observed)
            self.assertEqual(DOT.sha256(staged / "fc.data"), manifest["fc.data"])
            self.assertEqual(
                DOT.sha256(staged / "Data/ocean.dat"), manifest["Data/ocean.dat"]
            )
            self.assertTrue((staged / "EOF_data/to").is_dir())

            with self.assertRaisesRegex(RuntimeError, "producer case's runtime"):
                DOT.verify_producer_runtime(case, report)
            (runtime / "fc.data").write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "runtime manifest mismatch"):
                DOT.verify_producer_runtime(runtime, report)
            (runtime / "fc.data").unlink()
            marker = case / "same-config.txt"
            marker.write_text("config\n", encoding="utf-8")
            (runtime / "fc.data").symlink_to(marker)
            with self.assertRaisesRegex(RuntimeError, "runtime path became a symlink"):
                DOT.verify_producer_runtime(runtime, report)


class TangentBuildBindingTests(unittest.TestCase):
    def test_private_staged_input_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = root / "runtime-source"
            tree.mkdir()
            source = tree / "model.F"
            source.write_text("model\n", encoding="utf-8")
            executable = root / "one-step"
            executable.write_bytes(b"one-step")
            trees = {
                "runtime source": (
                    tree,
                    DOT.tangent_audit.tree_file_hashes(tree),
                )
            }
            files = {
                "one-step executable": (executable, DOT.sha256(executable))
            }
            binding = DOT.tangent_audit.verify_private_staged_inputs(
                trees=trees,
                files=files,
            )
            self.assertEqual(binding["trees"]["runtime source"]["file_count"], 1)

            source.write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "runtime source"):
                DOT.tangent_audit.verify_private_staged_inputs(
                    trees=trees,
                    files=files,
                )
            source.write_text("model\n", encoding="utf-8")
            executable.write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "one-step executable"):
                DOT.tangent_audit.verify_private_staged_inputs(
                    trees=trees,
                    files=files,
                )

    def test_tangent_build_requires_exact_coupled_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            tangent_source = snapshot / "generated/tangent/source_d.f"
            tangent_source.parent.mkdir(parents=True)
            tangent_source.write_text("source\n", encoding="utf-8")
            tangent = root / "tangent"
            tangent.write_bytes(b"executable")
            tangent_inputs = root / "tangent.build_inputs.sha256"
            write_manifest(snapshot, tangent_inputs, [tangent_source])
            coupled_inputs = root / "coupled_inputs.sha256"
            write_manifest(snapshot, coupled_inputs, [tangent_source])
            coupled_manifest = root / "coupled_manifest.txt"
            coupled_manifest.write_text(
                "build_input_manifest_sha256="
                f"{DOT.sha256(coupled_inputs)}\n",
                encoding="utf-8",
            )
            tangent_manifest = root / "tangent.build_manifest.txt"
            tangent_manifest.write_text(
                "schema=zc-tangent-audit-build-v1\n"
                "optimization=O3\n"
                f"path_executable_sha256={DOT.sha256(tangent)}\n"
                f"tangent_build_inputs_sha256={DOT.sha256(tangent_inputs)}\n"
                "coupled_build_manifest_sha256="
                f"{DOT.sha256(coupled_manifest)}\n"
                "coupled_build_input_manifest_sha256="
                f"{DOT.sha256(coupled_inputs)}\n",
                encoding="utf-8",
            )

            evidence = DOT.verify_tangent_build(
                tangent_executable=tangent,
                tangent_build_manifest=tangent_manifest,
                tangent_build_inputs=tangent_inputs,
                coupled_build_manifest=coupled_manifest,
                coupled_build_input_manifest=coupled_inputs,
                coupled_build_snapshot=snapshot,
            )
            self.assertEqual(evidence["optimization"], "O3")

            extra = snapshot / "undeclared.txt"
            extra.write_text("not bound\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact snapshot inventory"):
                DOT.verify_tangent_build(
                    tangent_executable=tangent,
                    tangent_build_manifest=tangent_manifest,
                    tangent_build_inputs=tangent_inputs,
                    coupled_build_manifest=coupled_manifest,
                    coupled_build_input_manifest=coupled_inputs,
                    coupled_build_snapshot=snapshot,
                )

    def test_replay_report_must_bind_the_exact_primal_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primal = root / "zc_kernel_replay"
            build_report = root / "build_report.json"
            replay_report = root / "replay_report.json"
            primal.write_bytes(b"primal")
            build_report.write_text("{}\n", encoding="utf-8")
            report = {
                "schema_version": 2,
                "status": "passed",
                "build_report_sha256": DOT.sha256(build_report),
                "kernel_executable_sha256": DOT.sha256(primal),
                "executed_staged_executable_sha256": {
                    "zc_kernel_replay": DOT.sha256(primal)
                },
                "build_provenance": {
                    "build_report_sha256": DOT.sha256(build_report),
                    "kernel_executable_sha256": DOT.sha256(primal),
                },
                "locked_checkpoint_contract": {
                    "case": {"pre_nt": 1, "sha256": "0" * 64}
                },
                "cases": [{"checkpoint_label": "case", "passed": True}],
            }
            replay_report.write_text(json.dumps(report), encoding="utf-8")
            binding = DOT.tangent_audit.verify_replay_report_binding(
                replay_report,
                kernel_build_report=build_report,
                primal=primal,
            )
            self.assertEqual(binding["primal_executable_sha256"], DOT.sha256(primal))

            report["executed_staged_executable_sha256"]["zc_kernel_replay"] = (
                "0" * 64
            )
            replay_report.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "executed staged primal"):
                DOT.tangent_audit.verify_replay_report_binding(
                    replay_report,
                    kernel_build_report=build_report,
                    primal=primal,
                )
            report["executed_staged_executable_sha256"]["zc_kernel_replay"] = (
                DOT.sha256(primal)
            )
            replay_report.write_text(json.dumps(report), encoding="utf-8")

            primal.write_bytes(b"other primal")
            with self.assertRaisesRegex(RuntimeError, "different primal"):
                DOT.tangent_audit.verify_replay_report_binding(
                    replay_report,
                    kernel_build_report=build_report,
                    primal=primal,
                )

    def test_full_tangent_requires_the_one_step_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            source = snapshot / "kernel_source"
            source.mkdir(parents=True)
            generation = snapshot / "generated/generation_provenance.txt"
            generation.parent.mkdir(parents=True)
            generation.write_text("generation\n", encoding="utf-8")
            kernel_report = snapshot / "kernel_parent/build_report.json"
            kernel_report.parent.mkdir(parents=True)
            primal = root / "primal"
            one_step = root / "one-step"
            path_driver = root / "path-driver"
            generator = root / "generate_fresh_zc_dataset.py"
            for path, payload in (
                (primal, b"primal"),
                (one_step, b"one-step"),
                (path_driver, b"path-driver"),
                (generator, b"generator"),
            ):
                path.write_bytes(payload)
            kernel_report.write_text(
                json.dumps(
                    {
                        "kernel_executable_sha256": DOT.sha256(primal),
                        "builder_source_sha256": {
                            "generate_fresh_zc_dataset.py": DOT.sha256(generator)
                        },
                    }
                ),
                encoding="utf-8",
            )
            tangent_inputs = root / "tangent-inputs.sha256"
            coupled_inputs = root / "coupled-inputs.sha256"
            tangent_inputs.write_text("inputs\n", encoding="utf-8")
            coupled_inputs.write_text("inputs\n", encoding="utf-8")
            coupled_manifest = root / "coupled-manifest.txt"
            coupled_manifest.write_text(
                f"build_input_manifest_sha256={DOT.sha256(coupled_inputs)}\n"
                f"kernel_parent_build_report_sha256={DOT.sha256(kernel_report)}\n",
                encoding="utf-8",
            )
            tangent_manifest = root / "tangent-manifest.txt"
            tangent_manifest.write_text(
                "schema=zc-tangent-audit-build-v1\n"
                "optimization=O3\n"
                f"one_step_executable_sha256={DOT.sha256(one_step)}\n"
                f"path_executable_sha256={DOT.sha256(path_driver)}\n"
                f"tangent_build_inputs_sha256={DOT.sha256(tangent_inputs)}\n"
                f"coupled_build_manifest_sha256={DOT.sha256(coupled_manifest)}\n"
                "coupled_build_input_manifest_sha256="
                f"{DOT.sha256(coupled_inputs)}\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                DOT.tangent_audit,
                "verify_sha256_manifest",
                side_effect=(1, 1),
            ):
                result = DOT.tangent_audit.verify_audit_build(
                    tangent=one_step,
                    primal=primal,
                    source=source,
                    generation_provenance=generation,
                    tangent_build_manifest=tangent_manifest,
                    tangent_build_inputs=tangent_inputs,
                    coupled_build_manifest=coupled_manifest,
                    coupled_build_input_manifest=coupled_inputs,
                    coupled_build_snapshot=snapshot,
                    kernel_build_report=kernel_report,
                    generator_path=generator,
                )
            self.assertEqual(result["optimization"], "O3")
            self.assertEqual(
                result["one_step_executable_sha256"], DOT.sha256(one_step)
            )
            self.assertEqual(result["primal_executable_sha256"], DOT.sha256(primal))
            with (
                mock.patch.object(
                    DOT.tangent_audit,
                    "verify_sha256_manifest",
                    side_effect=(1, 1),
                ),
                self.assertRaisesRegex(
                    RuntimeError, "one_step_executable_sha256"
                ),
            ):
                DOT.tangent_audit.verify_audit_build(
                    tangent=path_driver,
                    primal=primal,
                    source=source,
                    generation_provenance=generation,
                    tangent_build_manifest=tangent_manifest,
                    tangent_build_inputs=tangent_inputs,
                    coupled_build_manifest=coupled_manifest,
                    coupled_build_input_manifest=coupled_inputs,
                    coupled_build_snapshot=snapshot,
                    kernel_build_report=kernel_report,
                    generator_path=generator,
                )

    def test_replay_runtime_source_is_bound_to_staged_source_manifest(self) -> None:
        class Generator:
            @staticmethod
            def source_manifest(source: Path) -> dict[str, str]:
                return {
                    path.relative_to(source).as_posix(): DOT.sha256(path)
                    for path in sorted(source.rglob("*"))
                    if path.is_file()
                    and path.suffix != ".o"
                    and path.name not in {".DS_Store", "zeqfc1"}
                }

            @staticmethod
            def manifest_sha256(manifest: dict[str, str]) -> str:
                return hashlib.sha256(
                    json.dumps(manifest, sort_keys=True).encode()
                ).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            replay = Path(temporary) / "replay"
            runtime_source = replay / "verified_inputs/source"
            (runtime_source / "Data").mkdir(parents=True)
            (runtime_source / "fc.data").write_text("config\n", encoding="utf-8")
            (runtime_source / "Data/ocean.dat").write_bytes(b"ocean")
            expected = Generator.source_manifest(runtime_source)
            kernel_parent = replay / "kernel-build"
            kernel_parent.mkdir()
            source_manifest = kernel_parent / "source_manifest.json"
            source_manifest.write_text(
                json.dumps(expected, sort_keys=True), encoding="utf-8"
            )
            build_report = kernel_parent / "build_report.json"
            build_report.write_text(
                json.dumps(
                    {
                        "source_manifest_file_sha256": DOT.sha256(source_manifest),
                        "source_manifest_sha256": Generator.manifest_sha256(expected),
                        "fresh_patch_sha256": {},
                        "kernel_patch_sha256": {},
                        "kernel_support_source_sha256": {},
                    }
                ),
                encoding="utf-8",
            )
            report = replay / "replay_report.json"
            report.write_text(
                json.dumps(
                    {
                        "build_provenance": {
                            "build_report_sha256": DOT.sha256(build_report),
                            "staged_source_file_sha256": expected,
                            "staged_source_manifest_sha256": (
                                Generator.manifest_sha256(expected)
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )
            binding = DOT.tangent_audit.verify_replay_runtime_source(
                runtime_source,
                report,
                kernel_build_report=build_report,
                generator=Generator,
            )
            self.assertEqual(binding["runtime_source_manifest_entry_count"], 2)

            other_source = replay / "other-source"
            (other_source / "Data").mkdir(parents=True)
            (other_source / "fc.data").write_text("config\n", encoding="utf-8")
            (other_source / "Data/ocean.dat").write_bytes(b"ocean")
            with self.assertRaisesRegex(RuntimeError, "verified_inputs/source"):
                DOT.tangent_audit.verify_replay_runtime_source(
                    other_source,
                    report,
                    kernel_build_report=build_report,
                    generator=Generator,
                )

            document = json.loads(report.read_text(encoding="utf-8"))
            document["build_provenance"]["staged_source_manifest_sha256"] = "0" * 64
            report.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "digest is inconsistent"):
                DOT.tangent_audit.verify_replay_runtime_source(
                    runtime_source,
                    report,
                    kernel_build_report=build_report,
                    generator=Generator,
                )
            document["build_provenance"]["staged_source_manifest_sha256"] = (
                Generator.manifest_sha256(expected)
            )
            report.write_text(json.dumps(document), encoding="utf-8")

            (runtime_source / "fc.data").write_text("mutated\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "build provenance"):
                DOT.tangent_audit.verify_replay_runtime_source(
                    runtime_source,
                    report,
                    kernel_build_report=build_report,
                    generator=Generator,
                )

            (runtime_source / "fc.data").unlink()
            same_config = replay / "same-config.txt"
            same_config.write_text("config\n", encoding="utf-8")
            (runtime_source / "fc.data").symlink_to(same_config)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                DOT.tangent_audit.verify_replay_runtime_source(
                    runtime_source,
                    report,
                    kernel_build_report=build_report,
                    generator=Generator,
                )
            (runtime_source / "fc.data").unlink()
            (runtime_source / "fc.data").write_text("config\n", encoding="utf-8")
            (runtime_source / "extra.o").write_bytes(b"unauthenticated")
            with self.assertRaisesRegex(RuntimeError, "complete-tree inventory"):
                DOT.tangent_audit.verify_replay_runtime_source(
                    runtime_source,
                    report,
                    kernel_build_report=build_report,
                    generator=Generator,
                )


if __name__ == "__main__":
    unittest.main()
