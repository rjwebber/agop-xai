"""Release-boundary tests for the fresh ZC generator."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_fresh_zc_dataset.py"
SPEC = importlib.util.spec_from_file_location("fresh_zc_generator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class FreshZCGeneratorReleaseTests(unittest.TestCase):
    def make_bound_workspace(
        self, root: Path
    ) -> tuple[SimpleNamespace, Path, Path, Path]:
        source = root / "upstream"
        source.mkdir()
        (source / "source.F").write_text("C upstream source\n")
        workspace = root / "workspace"
        executable = workspace / "source" / "zeqfc1"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"executable")
        args = SimpleNamespace(
            source_dir=source,
            workspace=workspace,
            output_dir=root / "output",
            spinup_years=1,
            retained_years=1,
            checkpoint_years=1,
            event_lead_months=1,
        )
        config = GENERATOR.production_configuration(args)
        source_digest = GENERATOR.manifest_sha256(GENERATOR.source_manifest(source))
        build = {
            "upstream": {"local_source_manifest_sha256": source_digest},
            "executable_sha256": GENERATOR.sha256_file(executable),
        }
        preflight = {
            "report_schema_version": GENERATOR.GENERATION_SCHEMA_VERSION,
            "generator": GENERATOR.generator_identity(),
            "status": "passed",
            "requested_production_configuration": config,
            "build": build,
        }
        preflight_path = workspace / "preflight_report.json"
        GENERATOR.write_json(preflight_path, preflight)

        run_dir = workspace / "runs" / "production"
        run_dir.mkdir(parents=True)
        field_stream = run_dir / "fresh_fields.data"
        field_stream.write_bytes(
            b"\0" * (config["retained_steps"] * config["stream_bytes_per_step"])
        )
        history = run_dir / "outhst"
        history.write_bytes(b"a" * 64 + b"b" * 64)
        production = {
            "report_schema_version": GENERATOR.GENERATION_SCHEMA_VERSION,
            "generator": GENERATOR.generator_identity(),
            "status": "integration_complete_pending_finalize",
            "configuration": config,
            "binding": {
                "preflight_report_sha256": GENERATOR.sha256_file(preflight_path),
                "source_manifest_sha256": source_digest,
                "executable_sha256": GENERATOR.sha256_file(executable),
            },
            "field_stream_sha256": GENERATOR.sha256_file(field_stream),
            "history_sha256": GENERATOR.sha256_file(history),
            "artifacts": {
                "field_stream": {
                    "file": field_stream.name,
                    "size_bytes": field_stream.stat().st_size,
                    "sha256": GENERATOR.sha256_file(field_stream),
                    "shape": [
                        config["retained_steps"],
                        13,
                        20,
                        27,
                    ],
                    "dtype": "<f4",
                    "layout": config["stream_layout"],
                    "bytes_per_step": config["stream_bytes_per_step"],
                },
                "history": {
                    "file": history.name,
                    "size_bytes": history.stat().st_size,
                    "sha256": GENERATOR.sha256_file(history),
                    "checkpoint_count": 2,
                    "checkpoint_chunk_bytes": 64,
                },
            },
        }
        production_path = workspace / "production_report.json"
        production_path.write_text(json.dumps(production))
        return args, workspace, field_stream, history

    def test_production_configuration_binds_user_visible_and_binary_layout(
        self,
    ) -> None:
        args = SimpleNamespace(
            spinup_years=100,
            retained_years=12_000,
            checkpoint_years=10,
            event_lead_months=7,
        )
        config = GENERATOR.production_configuration(args)
        self.assertEqual(config["spinup_years"], 100)
        self.assertEqual(config["retained_years"], 12_000)
        self.assertEqual(config["event_lead_months"], 7)
        self.assertEqual(config["field_count"], 13)
        self.assertEqual(config["field_shape"], [20, 27])
        self.assertEqual(config["field_dtype"], "<f4")
        self.assertEqual(config["stream_bytes_per_step"], 13 * 20 * 27 * 4)

    def test_event_checkpoint_name_uses_configurable_lead(self) -> None:
        self.assertEqual(
            GENERATOR.event_checkpoint_filename("extreme_el_nino", 21),
            "extreme_el_nino_7month_pre_input.hst",
        )
        with self.assertRaises(RuntimeError):
            GENERATOR.event_checkpoint_filename("event", 20)

    def test_thermocline_depth_metadata_uses_source_native_meters(self) -> None:
        thermocline = next(
            field
            for field in GENERATOR.FIELDS
            if field["name"] == "thermocline_depth"
        )
        self.assertEqual(thermocline["fortran"], "H1")
        self.assertEqual(thermocline["units"], "m")

    def test_validate_file_record_detects_size_and_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            path.write_bytes(b"certified")
            record = {
                "size_bytes": path.stat().st_size,
                "sha256": GENERATOR.sha256_file(path),
            }
            GENERATOR.validate_file_record(path, record, "artifact")
            path.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                GENERATOR.validate_file_record(path, record, "artifact")

    def test_finalize_binding_accepts_exact_identity_and_rejects_cli_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, workspace, field_stream, history = self.make_bound_workspace(
                Path(temporary)
            )
            result = GENERATOR.validate_production_binding(args, workspace)
            self.assertEqual(result[3:], (field_stream, history))
            changed = SimpleNamespace(**vars(args))
            changed.event_lead_months = 2
            with self.assertRaises(RuntimeError):
                GENERATOR.validate_production_binding(changed, workspace)

    def test_atomic_publish_replaces_only_complete_sibling_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            destination = parent / "release"
            destination.mkdir()
            (destination / "value.txt").write_text("old")
            staging = parent / ".release.staging-test"
            staging.mkdir()
            (staging / "value.txt").write_text("new")
            GENERATOR.atomic_publish_directory(staging, destination, overwrite=True)
            self.assertEqual((destination / "value.txt").read_text(), "new")
            self.assertFalse(staging.exists())
            self.assertEqual(list(parent.glob(".release.backup-*")), [])

    def test_atomic_publish_refuses_overwrite_without_touching_destination(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            destination = parent / "release"
            destination.mkdir()
            (destination / "value.txt").write_text("old")
            staging = parent / ".release.staging-test"
            staging.mkdir()
            (staging / "value.txt").write_text("new")
            with self.assertRaises(FileExistsError):
                GENERATOR.atomic_publish_directory(
                    staging, destination, overwrite=False
                )
            self.assertEqual((destination / "value.txt").read_text(), "old")
            self.assertEqual((staging / "value.txt").read_text(), "new")


if __name__ == "__main__":
    unittest.main()
