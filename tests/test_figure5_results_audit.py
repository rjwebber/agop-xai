"""Adversarial tests for the independent fresh Figure 5 result audit."""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import audit_figure5_results as audit


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_relative(row: dict[str, str]) -> str:
    path = (
        Path("figure5")
        / "core4"
        / row["architecture"]
        / f"lead-{int(row['lead_months']):02d}m"
        / f"years-{float(row['train_years_requested']):g}"
    )
    if row["panel"] == "lead_time":
        path /= "common-max-lead-12m"
    return (path / f"seed-{int(row['seed']):06d}").as_posix()


def _row(
    panel: str,
    architecture: str,
    lead: int,
    years: float,
    repetition: int,
) -> dict[str, str]:
    target_period = (230, 240) if panel == "training_size" else (236, 246)
    row = {
        "panel": panel,
        "architecture": architecture,
        "lead_months": str(lead),
        "train_years_requested": str(float(years)),
        "development_years_actual": str(float(years)),
        "fit_years_actual": str(float(years)),
        "embargo_years_actual": "0.0",
        "validation_years_actual": "1000.0",
        "repetition": str(repetition),
        "seed": str(audit.BASE_SEED + repetition),
        "test_r2": "0.75",
        "test_target_start_step": str(target_period[0]),
        "test_target_stop_step_exclusive": str(target_period[1]),
        "checkpoint_sha256": "0" * 64,
        "artifact_directory": "",
    }
    row["artifact_directory"] = _artifact_relative(row)
    return row


def _canonical_rows(
    left_repetitions: int = audit.DEFAULT_LEFT_REPETITIONS,
    right_repetitions: int = audit.DEFAULT_RIGHT_REPETITIONS,
) -> list[dict[str, str]]:
    rows = []
    for architecture in audit.ARCHITECTURES:
        for years in audit.TRAINING_YEARS:
            for repetition in range(left_repetitions):
                rows.append(
                    _row("training_size", architecture, 10, years, repetition)
                )
        for years in audit.RIGHT_YEARS:
            for lead in audit.LEADS:
                for repetition in range(right_repetitions):
                    rows.append(
                        _row("lead_time", architecture, lead, years, repetition)
                    )
    return rows


def _sidecar(
    left_repetitions: int = audit.DEFAULT_LEFT_REPETITIONS,
    right_repetitions: int = audit.DEFAULT_RIGHT_REPETITIONS,
) -> dict[str, object]:
    return {
        "schema_version": audit.RESULT_SCHEMA_VERSION,
        "figure": 5,
        "data": {
            "input_profile": "core4",
            "metadata_sha256": "a" * 64,
        },
        "training_config": copy.deepcopy(audit.EXPECTED_TRAINING_CONFIG),
        "base_seed": audit.BASE_SEED,
        "left_panel": {
            "lead_months": 10,
            "training_years": list(audit.TRAINING_YEARS),
            "repetitions": left_repetitions,
        },
        "right_panel": {
            "lead_months": list(audit.LEADS),
            "training_years": list(audit.RIGHT_YEARS),
            "repetitions": right_repetitions,
            "common_period_max_lead_months": 12,
        },
        "result_rows": audit._result_row_count(
            left_repetitions,
            right_repetitions,
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Figure5GridAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results = self.root / "results.csv"
        self.artifacts = self.root / "artifacts"
        self.rows = _canonical_rows()
        _write_rows(self.results, self.rows)
        self.results.with_suffix(".json").write_text(
            json.dumps(_sidecar()),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _audit_grid(self) -> int:
        with mock.patch.object(audit, "_validate_artifact"):
            return audit.audit_results(self.results, self.artifacts)

    def test_canonical_grid_and_truthful_main_message_pass(self) -> None:
        self.assertEqual(self._audit_grid(), 144)
        output = io.StringIO()
        arguments = [
            "audit_figure5_results.py",
            "--results",
            str(self.results),
            "--artifacts-dir",
            str(self.artifacts),
        ]
        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(audit, "_validate_artifact"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(audit.main(), 0)
        self.assertIn("144 distinct complete artifacts", output.getvalue())

    def test_noncanonical_or_missing_sidecar_is_rejected(self) -> None:
        sidecar_path = self.results.with_suffix(".json")
        sidecar = _sidecar()
        sidecar["training_config"]["batch_size"] = 1024  # type: ignore[index]
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "canonical training configuration"):
            self._audit_grid()
        sidecar_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "result sidecar"):
            self._audit_grid()

    def test_repeated_or_mismatched_artifact_directory_is_rejected(self) -> None:
        self.rows[1]["artifact_directory"] = self.rows[0]["artifact_directory"]
        _write_rows(self.results, self.rows)
        with self.assertRaisesRegex(ValueError, "artifact_directory"):
            self._audit_grid()

    def test_right_panel_dates_must_be_global_not_series_local(self) -> None:
        for row in self.rows:
            if (
                row["panel"] == "lead_time"
                and row["architecture"] == "mlp"
                and float(row["train_years_requested"]) == 50.0
            ):
                row["test_target_start_step"] = "237"
                row["test_target_stop_step_exclusive"] = "247"
        _write_rows(self.results, self.rows)
        with self.assertRaisesRegex(ValueError, "across all runs"):
            self._audit_grid()

    def test_left_panel_dates_must_be_shared(self) -> None:
        self.rows[0]["test_target_start_step"] = "231"
        self.rows[0]["test_target_stop_step_exclusive"] = "241"
        _write_rows(self.results, self.rows)
        with self.assertRaisesRegex(ValueError, "Left-panel target dates"):
            self._audit_grid()

    def test_seed_must_match_repetition(self) -> None:
        self.rows[0]["seed"] = "99"
        self.rows[0]["artifact_directory"] = _artifact_relative(self.rows[0])
        _write_rows(self.results, self.rows)
        with self.assertRaisesRegex(ValueError, "seed does not match"):
            self._audit_grid()

    def test_five_seed_grid_uses_sidecar_or_explicit_requirements(self) -> None:
        self.rows = _canonical_rows(left_repetitions=5, right_repetitions=5)
        _write_rows(self.results, self.rows)
        self.results.with_suffix(".json").write_text(
            json.dumps(_sidecar(left_repetitions=5, right_repetitions=5)),
            encoding="utf-8",
        )
        self.assertEqual(self._audit_grid(), 480)
        with mock.patch.object(audit, "_validate_artifact"):
            self.assertEqual(
                audit.audit_results(
                    self.results,
                    self.artifacts,
                    expected_left_repetitions=5,
                    expected_right_repetitions=5,
                ),
                480,
            )
            with self.assertRaisesRegex(ValueError, "explicitly required"):
                audit.audit_results(
                    self.results,
                    self.artifacts,
                    expected_left_repetitions=3,
                    expected_right_repetitions=5,
                )

    def test_sidecar_repetitions_and_derived_row_count_are_strict(self) -> None:
        sidecar_path = self.results.with_suffix(".json")
        sidecar = _sidecar()
        sidecar["left_panel"]["repetitions"] = 0  # type: ignore[index]
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self._audit_grid()

        sidecar = _sidecar()
        sidecar["result_rows"] = audit.RESULT_ROWS - 1
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "declared complete grid"):
            self._audit_grid()


class Figure5ArtifactAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sidecar = _sidecar()
        self.row = _canonical_rows()[0]
        self.directory = self.root.joinpath(*Path(self.row["artifact_directory"]).parts)
        self.directory.mkdir(parents=True)
        checkpoint = self.directory / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        self.row["checkpoint_sha256"] = _digest(checkpoint)
        (self.directory / "normalization.npz").write_bytes(b"normalization")

        lead_steps = int(self.row["lead_months"]) * 3
        test_start = int(self.row["test_target_start_step"]) - lead_steps
        np.savez(
            self.directory / "indices.npz",
            fit_inputs=np.arange(0, 10, dtype=np.int64),
            standardization_inputs=np.arange(0, 100, dtype=np.int64),
            validation_inputs=np.arange(100, 110, dtype=np.int64),
            test_inputs=np.arange(test_start, test_start + 10, dtype=np.int64),
        )
        self.spec = {
            "architecture": "mlp",
            "lead_months": 10,
            "train_years": 50.0,
            "seed": 42,
            "train_fraction": 0.9,
            "validation_fraction": 0.2,
            "common_period_max_lead_months": None,
            "input_profile": "core4",
        }
        metrics = {
            "schema_version": 1,
            "spec": self.spec,
            "training_config": self.sidecar["training_config"],
            "model": {"checkpoint_sha256": self.row["checkpoint_sha256"]},
            "test_r2": 0.75,
            "selection": {
                "fixed_blocks": {
                    "train": [0, 100],
                    "validation": [100, 200],
                    "test": [200, 300],
                }
            },
            "evaluation": {
                "test_target_start_step": 230,
                "test_target_stop_step_exclusive": 240,
            },
        }
        (self.directory / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
        self.completion = {
            "schema_version": 1,
            "spec": self.spec,
            "training_config": self.sidecar["training_config"],
            "data_metadata_sha256": self.sidecar["data"]["metadata_sha256"],
            "files": {
                filename: _digest(self.directory / filename)
                for filename in audit.ARTIFACT_FILES
            },
        }
        self._write_completion()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_completion(self) -> None:
        (self.directory / "completed.json").write_text(
            json.dumps(self.completion),
            encoding="utf-8",
        )

    def test_artifact_is_linked_to_row_spec_seed_and_checkpoint(self) -> None:
        audit._validate_artifact(
            self.row,
            self.directory,
            sidecar=self.sidecar,
        )

        self.completion["spec"] = {**self.spec, "seed": 43}
        self._write_completion()
        with self.assertRaisesRegex(ValueError, "specification"):
            audit._validate_artifact(
                self.row,
                self.directory,
                sidecar=self.sidecar,
            )

    def test_csv_checkpoint_must_match_manifest_and_file(self) -> None:
        self.row["checkpoint_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "CSV checkpoint digest"):
            audit._validate_artifact(
                self.row,
                self.directory,
                sidecar=self.sidecar,
            )


if __name__ == "__main__":
    unittest.main()
