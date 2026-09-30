"""Publication-table tests for the fresh core4 Table I renderer."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import make_fresh_table1 as table1
from zc_xai.fresh_xai_outputs import (
    FRESH_TABLE_ARCHITECTURES,
    FRESH_TABLE_METHODS,
    FRESH_TABLE_SCHEMA_VERSION,
    FRESH_XAI_TRAINING_POPULATION,
)
from zc_xai.io import sha256_file


def _score_rows() -> list[dict[str, object]]:
    rows = []
    for method_index, method in enumerate(FRESH_TABLE_METHODS):
        for architecture_index, architecture in enumerate(
            FRESH_TABLE_ARCHITECTURES
        ):
            value = 0.1 + 0.1 * method_index + 0.01 * architecture_index
            rows.append(
                {
                    "method": method,
                    "architecture": architecture,
                    "sensitivity": value,
                    "attribution": value + 0.01,
                    "robustness": value + 0.02,
                    "coherence_spatial_only": value + 0.03,
                    "exhaustive": True,
                    "neighbor_percent": 1.0,
                    "evaluated_count": 3960,
                    "neighborhood_count": 3960,
                }
            )
    return rows


def _write_scores(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metadata(scores_path: Path) -> dict[str, object]:
    return {
        "schema_version": FRESH_TABLE_SCHEMA_VERSION,
        "status": "complete",
        "run_identity": {
            "input_profile": "core4",
            "expected_architectures": list(FRESH_TABLE_ARCHITECTURES),
            "data": {"metadata_sha256": "data-digest"},
            "robustness": {
                "nearest_percent": 1.0,
                "candidate_population": FRESH_XAI_TRAINING_POPULATION,
                "sampling": "none: every member of the nearest-percent population",
            },
            "methods": {
                "integrated_gradients": {
                    "gradient_count_per_explanation": 1024
                },
                "gradient_shap": {
                    "distinct_empirical_backgrounds": 1024,
                    "reference_population": FRESH_XAI_TRAINING_POPULATION,
                },
                "gradient_batch_size": 1024,
            },
        },
        "output_files": {
            "scores": {
                "file": scores_path.name,
                "sha256": sha256_file(scores_path),
            }
        },
    }


class FreshTable1RendererTests(unittest.TestCase):
    def test_defaults_are_publication_paths(self) -> None:
        args = table1.build_parser().parse_args([])
        self.assertEqual(
            args.table_dir,
            Path("artifacts/zc-v3/manuscript/table1_figure6"),
        )
        self.assertEqual(
            args.output,
            Path("outputs/manuscript/tables/table1_xai_scores.tex"),
        )

    def test_scores_require_complete_exhaustive_grid(self) -> None:
        with TemporaryDirectory() as temporary:
            scores_path = Path(temporary) / "scores.csv"
            rows = _score_rows()
            _write_scores(scores_path, rows)
            scores = table1.read_scores(scores_path)
            self.assertEqual(len(scores), 12)
            latex = table1.latex_table(scores)
            self.assertIn(r"\begin{tabular}{l|ccc|ccc|ccc|ccc|}", latex)
            self.assertIn("GradientSHAP", latex)
            self.assertEqual(latex.count(r"\textbf{"), 12)

            rows[0]["exhaustive"] = False
            _write_scores(scores_path, rows)
            with self.assertRaisesRegex(ValueError, "not exhaustive"):
                table1.read_scores(scores_path)

    def test_renderer_validates_bundle_and_writes_sidecar(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bundle = directory / "bundle"
            bundle.mkdir()
            scores_path = bundle / "table1_scores.csv"
            _write_scores(scores_path, _score_rows())
            (bundle / "table1_metadata.json").write_text(
                json.dumps(_metadata(scores_path)), encoding="utf-8"
            )
            output = directory / "table1.tex"
            with patch.object(
                sys,
                "argv",
                [
                    "make_fresh_table1.py",
                    "--table-dir",
                    str(bundle),
                    "--output",
                    str(output),
                ],
            ):
                self.assertEqual(table1.main(), 0)
            self.assertTrue(output.is_file())
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["table"], 1)
            self.assertEqual(sidecar["xai_configuration"]["gradient_batch_size"], 1024)
            self.assertEqual(sidecar["output"]["sha256"], sha256_file(output))

            scores_path.write_text("corrupt", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed its checksum"):
                table1.load_completed_scores(bundle)


if __name__ == "__main__":
    unittest.main()
