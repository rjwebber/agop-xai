"""Unit tests for the release-facing coupled ZC adjoint case runner."""

from __future__ import annotations

import array
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "adjoint/tapenade_toolchain/scripts/run_coupled_case.py"
SPEC = importlib.util.spec_from_file_location("run_coupled_case", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class BinaryContractTests(unittest.TestCase):
    def test_path_byte_count_matches_known_31_step_layout(self) -> None:
        self.assertEqual(RUNNER.expected_path_bytes(), 8_924_048)

    def test_float32_seed_requires_exact_length_and_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.bin"
            with valid.open("wb") as stream:
                array.array("f", [0.0, 1.25, -2.5]).tofile(stream)
            self.assertEqual(
                list(RUNNER.read_float32_stream(valid, 3)), [0.0, 1.25, -2.5]
            )

            with self.assertRaises(ValueError):
                RUNNER.read_float32_stream(valid, 4)

            nonfinite = root / "nonfinite.bin"
            with nonfinite.open("wb") as stream:
                array.array("f", [float("nan")]).tofile(stream)
            with self.assertRaises(ValueError):
                RUNNER.read_float32_stream(nonfinite, 1)


class ProvenanceTests(unittest.TestCase):
    def test_staged_runtime_is_reverified_after_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "runtime"
            (runtime / "Data").mkdir(parents=True)
            config = runtime / "fc.data"
            ocean = runtime / "Data/ocean.dat"
            config.write_text("config\n", encoding="utf-8")
            ocean.write_bytes(b"ocean")
            expected = {
                "fc.data": RUNNER.sha256_file(config),
                "Data/ocean.dat": RUNNER.sha256_file(ocean),
            }
            self.assertEqual(
                RUNNER.verify_staged_runtime(runtime, expected), expected
            )
            ocean.write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "changed during execution"):
                RUNNER.verify_staged_runtime(runtime, expected)

    def test_manifest_verifier_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.txt"
            payload.write_text("first\n", encoding="utf-8")
            manifest = root / "manifest.sha256"
            manifest.write_text(
                f"{RUNNER.sha256_file(payload)}  payload.txt\n", encoding="utf-8"
            )
            self.assertEqual(RUNNER.verify_sha256_manifest(root, manifest), 1)
            payload.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                RUNNER.verify_sha256_manifest(root, manifest)

    def test_manifest_verifier_rejects_escape_and_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload.txt"
            payload.write_text("payload\n", encoding="utf-8")
            digest = RUNNER.sha256_file(payload)
            manifest = root / "manifest.sha256"
            manifest.write_text(f"{digest}  ../payload.txt\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                RUNNER.verify_sha256_manifest(root, manifest)
            manifest.write_text(
                f"{digest}  payload.txt\n{digest}  payload.txt\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                RUNNER.verify_sha256_manifest(root, manifest)

    def test_runtime_canonicalization_masks_only_dynamic_times(self) -> None:
        fc = "NSTART = 3\nTFIND  = 1\nTZERO = 2\nTENDD = 3\nNIC = 0\n"
        canonical = RUNNER.canonical_runtime_text("fc.data", fc)
        self.assertIn("TFIND  = <dynamic>", canonical)
        self.assertIn("TZERO = <dynamic>", canonical)
        self.assertIn("TENDD = <dynamic>", canonical)
        self.assertIn("NSTART = 3", canonical)
        self.assertIn("NIC = 0", canonical)

    def test_build_input_manifest_detects_snapshot_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            toolchain = Path(temporary) / "toolchain"
            build = toolchain / "build/run23"
            compiled = build / "coupled_compiled"
            snapshot = compiled / "build_input_snapshot"
            prepared = snapshot / "prepared"
            generated = snapshot / "generated"
            patches = snapshot / "toolchain/patches"
            for directory in (snapshot, prepared, generated, patches):
                directory.mkdir(parents=True, exist_ok=True)

            contracts = {
                "prepared_source_manifest": (prepared, "source.f"),
                "generated_source_manifest": (generated, "reverse/source.f"),
                "toolchain_patch_manifest": (patches, "change.patch"),
            }
            values: dict[str, str] = {}
            for key, (manifest_root, relative) in contracts.items():
                target = manifest_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"{key}\n", encoding="utf-8")
                if key == "prepared_source_manifest":
                    manifest = prepared / "prepared_source_manifest.sha256"
                elif key == "generated_source_manifest":
                    manifest = generated / "generated_source_manifest.sha256"
                else:
                    manifest = compiled / "toolchain_patch_manifest.sha256"
                manifest.write_text(
                    f"{RUNNER.sha256_file(target)}  {relative}\n", encoding="utf-8"
                )
                values[f"{key}_sha256"] = RUNNER.sha256_file(manifest)
            recipe = snapshot / "manifests/compile_recipe.txt"
            recipe.parent.mkdir()
            recipe.write_text("recipe\n", encoding="utf-8")
            values["compile_recipe_sha256"] = RUNNER.sha256_file(recipe)
            prepared_manifest = prepared / "prepared_source_manifest.sha256"
            generated_manifest = generated / "generated_source_manifest.sha256"
            prepared_binding = prepared / "prepared_manifest_binding.txt"
            prepared_binding.write_text(
                "binding_schema=zc-prepared-manifest-binding-v1\n"
                "prepared_source_manifest_sha256="
                f"{RUNNER.sha256_file(prepared_manifest)}\n",
                encoding="utf-8",
            )
            generated_pre_manifest = (
                generated / "generated_source_manifest.pre_instrument.sha256"
            )
            generated_pre_manifest.write_text(generated_manifest.read_text())
            generation_provenance = generated / "generation_provenance.txt"
            generation_provenance.write_text(
                "prepared_manifest_binding_sha256="
                f"{RUNNER.sha256_file(prepared_binding)}\n"
            )
            generation_binding = generated / "generation_manifest_binding.txt"
            generation_binding.write_text(
                "binding_schema=zc-generation-manifest-binding-v1\n"
                "generated_pre_manifest_sha256="
                f"{RUNNER.sha256_file(generated_pre_manifest)}\n"
                "generation_provenance_sha256="
                f"{RUNNER.sha256_file(generation_provenance)}\n"
                "prepared_manifest_binding_sha256="
                f"{RUNNER.sha256_file(prepared_binding)}\n",
                encoding="utf-8",
            )
            instrumentation_provenance = generated / "instrumentation_provenance.txt"
            instrumentation_provenance.write_text(
                "generation_manifest_binding_sha256="
                f"{RUNNER.sha256_file(generation_binding)}\n"
            )
            instrumentation_binding = generated / "instrumentation_manifest_binding.txt"
            instrumentation_binding.write_text(
                "binding_schema=zc-instrumentation-manifest-binding-v1\n"
                "generated_source_manifest_sha256="
                f"{RUNNER.sha256_file(generated_manifest)}\n"
                "instrumentation_provenance_sha256="
                f"{RUNNER.sha256_file(instrumentation_provenance)}\n"
                "generation_manifest_binding_sha256="
                f"{RUNNER.sha256_file(generation_binding)}\n"
                "generated_pre_manifest_sha256="
                f"{RUNNER.sha256_file(generated_pre_manifest)}\n",
                encoding="utf-8",
            )
            tapenade_snapshot = snapshot / "tapenade"
            tapenade_snapshot.mkdir()
            tapenade_receipt = tapenade_snapshot / ".zc_tapenade_install_receipt.txt"
            tapenade_tree = tapenade_snapshot / ".zc_tapenade_tree_manifest.sha256"
            tapenade_receipt.write_text("receipt\n")
            tapenade_tree.write_text("tree\n")
            for key, path in (
                ("prepared_manifest_binding_sha256", prepared_binding),
                ("generation_manifest_binding_sha256", generation_binding),
                (
                    "instrumentation_manifest_binding_sha256",
                    instrumentation_binding,
                ),
                ("tapenade_install_receipt_sha256", tapenade_receipt),
                ("tapenade_tree_manifest_sha256", tapenade_tree),
            ):
                values[key] = RUNNER.sha256_file(path)
            tangent = snapshot / "generated/tangent"
            reverse = snapshot / "generated/reverse"
            tangent.mkdir(exist_ok=True)
            reverse.mkdir(exist_ok=True)
            (tangent / "source_d.f").write_text("tangent\n", encoding="utf-8")
            (reverse / "source_b.f").write_text("reverse\n", encoding="utf-8")
            compile_roots = {
                "prepared_compile_inputs": prepared,
                "tangent_compile_inputs": tangent,
                "reverse_compile_inputs": reverse,
            }
            for key, compile_root in compile_roots.items():
                manifest = compiled / f"{key}.sha256"
                manifest.write_text(
                    "".join(
                        f"{RUNNER.sha256_file(path)}  "
                        f"{path.relative_to(compile_root).as_posix()}\n"
                        for path in sorted(compile_root.rglob("*"))
                        if path.is_file()
                    ),
                    encoding="utf-8",
                )
                values[f"{key}_sha256"] = RUNNER.sha256_file(manifest)
                values[f"{key}_root"] = {
                    "prepared_compile_inputs": "build_input_snapshot/prepared",
                    "tangent_compile_inputs": (
                        "build_input_snapshot/generated/tangent"
                    ),
                    "reverse_compile_inputs": (
                        "build_input_snapshot/generated/reverse"
                    ),
                }[key]
            declared = snapshot / "declared/input.txt"
            declared.parent.mkdir()
            declared.write_text("declared\n", encoding="utf-8")
            build_input_manifest = compiled / "build_input_manifest.sha256"
            snapshot_files = sorted(
                path for path in snapshot.rglob("*") if path.is_file()
            )
            build_input_manifest.write_text(
                "".join(
                    f"{RUNNER.sha256_file(path)}  "
                    f"{path.relative_to(snapshot).as_posix()}\n"
                    for path in snapshot_files
                ),
                encoding="utf-8",
            )
            values["build_input_manifest_sha256"] = RUNNER.sha256_file(
                build_input_manifest
            )
            (compiled / "build_manifest.txt").write_text(
                "".join(f"{key}={value}\n" for key, value in values.items()),
                encoding="utf-8",
            )

            _, counts = RUNNER.verify_build_provenance(build)
            self.assertEqual(counts["build_input_manifest_sha256"], len(snapshot_files))
            (snapshot / "declared/input.txt").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                RUNNER.verify_build_provenance(build)

    def test_output_guard_rejects_ancestors_and_descendants_of_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = root / "build"
            checkpoint = root / "checkpoint"
            build.mkdir()
            checkpoint.mkdir()
            for unsafe in (
                root,
                build,
                checkpoint,
                build / "nested-output",
                checkpoint / "nested-output",
            ):
                with self.subTest(output=unsafe), self.assertRaises(ValueError):
                    RUNNER.guarded_prepare_output(
                        unsafe,
                        overwrite=True,
                        protected_paths=(build.resolve(), checkpoint.resolve()),
                        project_root=root,
                    )

            outside = root.parent / f"{root.name}-outside-output"
            outside.mkdir()
            marker = outside / RUNNER.OUTPUT_MARKER
            marker.write_text(RUNNER.OUTPUT_MARKER_CONTENT, encoding="utf-8")
            sentinel = outside / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            try:
                with self.assertRaisesRegex(ValueError, "strict descendant"):
                    RUNNER.guarded_prepare_output(
                        outside,
                        overwrite=True,
                        protected_paths=(build.resolve(), checkpoint.resolve()),
                        project_root=root,
                    )
                self.assertEqual(sentinel.read_text(), "keep\n")
            finally:
                sentinel.unlink()
                marker.unlink()
                outside.rmdir()

    def test_output_guard_refuses_to_delete_a_protected_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root / "checkpoint" / "case"
            protected.mkdir(parents=True)
            with self.assertRaises(ValueError):
                RUNNER.guarded_prepare_output(
                    root,
                    overwrite=True,
                    protected_paths=(protected,),
                    project_root=root,
                )

    def test_output_guard_rejects_a_symbolic_link_without_touching_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            sentinel = target / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            link = root / "output-link"
            link.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                RUNNER.guarded_prepare_output(
                    link,
                    overwrite=True,
                    protected_paths=(),
                    project_root=root,
                )
            self.assertTrue(link.is_symlink())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_output_guard_rejects_parent_symbolic_link_without_touching_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            output = target / "child"
            output.mkdir()
            marker = output / RUNNER.OUTPUT_MARKER
            marker.write_text(RUNNER.OUTPUT_MARKER_CONTENT, encoding="utf-8")
            sentinel = output / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            parent_link = root / "parent-link"
            parent_link.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                RUNNER.guarded_prepare_output(
                    parent_link / "child",
                    overwrite=True,
                    protected_paths=(),
                    project_root=root,
                )
            self.assertTrue(parent_link.is_symlink())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_symlinked_marker_cannot_authorize_output_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep\n", encoding="utf-8")
            marker_target = root / "marker-content.txt"
            marker_target.write_text(RUNNER.OUTPUT_MARKER_CONTENT, encoding="utf-8")
            (output / RUNNER.OUTPUT_MARKER).symlink_to(marker_target)

            with self.assertRaises(ValueError):
                RUNNER.guarded_prepare_output(
                    output,
                    overwrite=True,
                    protected_paths=(),
                    project_root=root,
                )
            self.assertEqual(sentinel.read_text(), "keep\n")
            self.assertEqual(marker_target.read_text(), RUNNER.OUTPUT_MARKER_CONTENT)


if __name__ == "__main__":
    unittest.main()
