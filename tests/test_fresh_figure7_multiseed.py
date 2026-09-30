"""Focused tests for the revised five-seed Figure 7 score pipeline."""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from matplotlib.axes import Axes

from scripts import generate_fresh_figure7_data as generator
from scripts import make_fresh_figure7 as figure7
from zc_xai.artifacts import CompletedBundle
from zc_xai.fresh_case_studies import FreshXAISettings, settings_metadata
from zc_xai.fresh_xai_outputs import FRESH_XAI_BUNDLE_SCHEMA_VERSION
from zc_xai.io import sha256_file


def _digest(value: int) -> str:
    return f"{value:064x}"


def _metadata(seed: int, *, figure: int = 7) -> dict:
    cases = {}
    experiments = {}
    target_step = 50_000
    for lead in figure7.EXPECTED_LEADS:
        case_id = f"el_nino_lead_{lead:02d}"
        lead_key = f"lead_{lead:02d}"
        checkpoint = _digest(seed * 100 + lead)
        cases[case_id] = {
            "case_id": case_id,
            "lead_months": lead,
            "input_step": target_step - 3 * lead,
            "target_step": target_step,
            "target_nino3_c": 4.25,
            "selection_rule": "fixed test maximum",
            "selection_split": "common target dates in fixed test block",
            "checkpoint_sha256": checkpoint,
            "event_input_standardized_sha256": _digest(20_000 + lead),
            "runtime": {},
        }
        experiments[lead_key] = {
            "spec": {
                "architecture": "cnn",
                "lead_months": lead,
                "train_years": 10_000.0,
                "seed": seed,
                "common_period_max_lead_months": 12,
                "input_profile": "core4",
                "train_fraction": 0.9,
                "validation_fraction": 0.2,
            },
            "training_config": {
                "batch_size": 256,
                "learning_rate": 1.0e-3,
                "weight_decay": 1.0e-4,
            },
            "common_target_dates": {
                "maximum_lead_months": 12,
                "fit_target_indices_sha256": _digest(1),
                "validation_target_indices_sha256": _digest(2),
                "test_target_indices_sha256": _digest(3),
            },
            "fit_indices_sha256": _digest(100 + lead),
            "standardization_indices_sha256": _digest(4),
            "validation_indices_sha256": _digest(200 + lead),
            "test_indices_sha256": _digest(300 + lead),
            "normalization_count": 360_000,
            "normalization_mean_sha256": _digest(5),
            "normalization_scale_sha256": _digest(6),
            "spatial_input_shape": [4, 20, 27],
            "phase_features": 2,
            "checkpoint_sha256": checkpoint,
        }
    return {
        "schema_version": FRESH_XAI_BUNDLE_SCHEMA_VERSION,
        "artifact": generator.ARTIFACT,
        "figure": figure,
        "data": {"metadata_sha256": _digest(10)},
        "input_profile": "core4",
        "spatial_fields": [
            "sst_anomaly",
            "thermocline_depth",
            "zonal_ocean_current",
            "meridional_ocean_current",
        ],
        "phase_features": 2,
        "lead_months": list(figure7.EXPECTED_LEADS),
        "case_order": [
            f"el_nino_lead_{lead:02d}" for lead in figure7.EXPECTED_LEADS
        ],
        "cases": cases,
        "experiments": experiments,
        "anchor": {
            "target_step": target_step,
            "target_nino3_c": 4.25,
            "common_test_target_indices_sha256": _digest(3),
        },
        "comparison_period": {
            "common_period_max_lead_months": 12,
            "shared_fit_target_dates": True,
            "shared_validation_target_dates": True,
            "shared_test_target_dates": True,
            "shared_normalization": True,
            "normalization_population": "fixed training block",
        },
        "xai_configuration": settings_metadata(FreshXAISettings()),
        "source_sha256": {
            "generator": sha256_file(Path(generator.__file__)),
            "fresh_case_studies.py": sha256_file(
                generator.REPOSITORY_ROOT / "src" / "zc_xai" / "fresh_case_studies.py"
            ),
            "fresh_figure7.py": sha256_file(
                generator.REPOSITORY_ROOT / "src" / "zc_xai" / "fresh_figure7.py"
            ),
            "xai.py": sha256_file(
                generator.REPOSITORY_ROOT / "src" / "zc_xai" / "xai.py"
            ),
        },
    }


def _seed_scores(scale: float = 0.1) -> dict:
    return {
        (lead, method): {
            metric: scale for metric, _label in figure7.METRICS
        }
        for lead in figure7.EXPECTED_LEADS
        for method in figure7.METHOD_ORDER
    }


def _bundle(root: Path, seed: int, metadata: dict | None = None) -> CompletedBundle:
    directory = root / f"seed-{seed:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.json"
    completion_path = directory / "completed.json"
    metadata_path.write_text("{}", encoding="utf-8")
    completion_path.write_text("{}", encoding="utf-8")
    return CompletedBundle(
        directory=directory,
        metadata_path=metadata_path,
        completion_path=completion_path,
        metadata=_metadata(seed) if metadata is None else metadata,
        files={},
        sha256={},
    )


class FreshFigure7MultiseedTests(unittest.TestCase):
    def test_aggregate_is_equal_weight_seed_mean_with_sample_uncertainty(self) -> None:
        seed_scores = {
            seed: _seed_scores(scale=float(seed - 42))
            for seed in figure7.CANONICAL_SEEDS
        }
        rows = figure7._aggregate_scores(seed_scores)
        self.assertEqual(len(rows), 12 * 4)
        first = rows[0]
        self.assertEqual(first["seed_count"], 5)
        self.assertEqual(first["seeds"], "42;43;44;45;46")
        self.assertEqual(first["sensitivity_mean"], 2.0)
        self.assertAlmostEqual(first["sensitivity_sample_sd"], math.sqrt(2.5))
        self.assertAlmostEqual(first["sensitivity_sem"], math.sqrt(0.5))
        self.assertEqual(first["sensitivity_min"], 0.0)
        self.assertEqual(first["sensitivity_max"], 4.0)

    def test_aggregate_rejects_a_missing_or_extra_seed(self) -> None:
        scores = {seed: _seed_scores() for seed in figure7.CANONICAL_SEEDS[:-1]}
        with self.assertRaisesRegex(ValueError, "exactly one.*42--46"):
            figure7._aggregate_scores(scores)
        scores[47] = _seed_scores()
        with self.assertRaisesRegex(ValueError, "exactly one.*42--46"):
            figure7._aggregate_scores(scores)

    def test_metadata_validation_locks_model_seed_and_training_only_xai(self) -> None:
        contract, checkpoints = figure7._validate_seed_metadata(
            _metadata(44), 44, legacy_figure9=False
        )
        self.assertEqual(contract["lead_months"], list(range(1, 13)))
        self.assertEqual(len(checkpoints), 12)

        wrong_seed = _metadata(44)
        wrong_seed["experiments"]["lead_03"]["spec"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "wrong model specification"):
            figure7._validate_seed_metadata(
                wrong_seed, 44, legacy_figure9=False
            )

        development_xai = _metadata(44)
        development_xai["xai_configuration"][
            "robustness_candidate_population"
        ] = "development predictors"
        with self.assertRaisesRegex(ValueError, "training-only XAI"):
            figure7._validate_seed_metadata(
                development_xai, 44, legacy_figure9=False
            )

    def test_bundle_aggregation_rejects_cross_seed_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = {
                seed: _bundle(root, seed) for seed in figure7.CANONICAL_SEEDS
            }

            def loader(directory: Path, **_kwargs: object) -> CompletedBundle:
                seed = int(Path(directory).name.split("-")[-1])
                return bundles[seed]

            with mock.patch.object(
                figure7, "load_completed_bundle", side_effect=loader
            ):
                loaded, _checkpoints, _contract, legacy = (
                    figure7._load_seed_bundles(root, None)
                )
            self.assertEqual(tuple(loaded), figure7.CANONICAL_SEEDS)
            self.assertFalse(legacy)

            drifted = copy.deepcopy(bundles[46].metadata)
            drifted["experiments"]["lead_08"][
                "normalization_scale_sha256"
            ] = _digest(99)
            bundles[46] = _bundle(root, 46, drifted)
            with (
                mock.patch.object(
                    figure7, "load_completed_bundle", side_effect=loader
                ),
                self.assertRaisesRegex(ValueError, "provenance differs"),
            ):
                figure7._load_seed_bundles(root, None)

            duplicated = _metadata(46)
            copied_hash = bundles[42].metadata["experiments"]["lead_08"][
                "checkpoint_sha256"
            ]
            duplicated["experiments"]["lead_08"][
                "checkpoint_sha256"
            ] = copied_hash
            duplicated["cases"]["el_nino_lead_08"][
                "checkpoint_sha256"
            ] = copied_hash
            bundles[46] = _bundle(root, 46, duplicated)
            with (
                mock.patch.object(
                    figure7, "load_completed_bundle", side_effect=loader
                ),
                self.assertRaisesRegex(ValueError, "five distinct CNN checkpoints"),
            ):
                figure7._load_seed_bundles(root, None)

    def test_score_reader_requires_complete_exhaustive_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = _metadata(42)
            bundle = _bundle(root, 42, metadata)
            score_path = bundle.directory / "scores.csv"
            rows = []
            for lead in figure7.EXPECTED_LEADS:
                case = metadata["cases"][f"el_nino_lead_{lead:02d}"]
                for method in figure7.METHOD_ORDER:
                    rows.append(
                        {
                            "case_id": case["case_id"],
                            "method": method,
                            "architecture": "cnn",
                            "lead_months": lead,
                            "sensitivity": 0.1,
                            "attribution": 0.2,
                            "robustness": 0.3,
                            "coherence": 0.4,
                            "neighbor_percent": 1.0,
                            "neighborhood_count": 3600,
                            "robustness_evaluated_count": 3600,
                            "robustness_exhaustive": True,
                            "robustness_finite_population_se": 0.0,
                            "event_input_step": case["input_step"],
                            "event_target_step": case["target_step"],
                            "event_target_nino3_c": case["target_nino3_c"],
                            "checkpoint_sha256": case["checkpoint_sha256"],
                        }
                    )

            def write(selected: list[dict]) -> None:
                with score_path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(selected)

            bundle.files["scores"] = score_path
            checkpoints = {
                lead: metadata["cases"][f"el_nino_lead_{lead:02d}"][
                    "checkpoint_sha256"
                ]
                for lead in figure7.EXPECTED_LEADS
            }
            write(rows)
            parsed = figure7._read_seed_scores(bundle, 42, checkpoints)
            self.assertEqual(len(parsed), 48)

            write(rows[:-1])
            with self.assertRaisesRegex(ValueError, "score grid mismatch"):
                figure7._read_seed_scores(bundle, 42, checkpoints)

            rows[0]["robustness_exhaustive"] = False
            write(rows)
            with self.assertRaisesRegex(ValueError, "not exhaustive"):
                figure7._read_seed_scores(bundle, 42, checkpoints)

    def test_plot_draws_means_only_and_never_uncertainty_ribbons(self) -> None:
        seed_scores = {
            seed: _seed_scores(scale=(seed - 41) / 10.0)
            for seed in figure7.CANONICAL_SEEDS
        }
        rows = figure7._aggregate_scores(seed_scores)
        plotted: list[np.ndarray] = []
        original_plot = Axes.plot

        def record_plot(axis: Axes, *args: object, **kwargs: object) -> object:
            if kwargs.get("label") in figure7.METHOD_ORDER:
                plotted.append(np.asarray(args[1], dtype=np.float64))
            return original_plot(axis, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "figure7.png"
            with (
                mock.patch.object(Axes, "plot", new=record_plot),
                mock.patch.object(Axes, "fill_between") as fill_between,
            ):
                figure7._plot(rows, output, dpi=72, overwrite=False)
            self.assertTrue(output.is_file())
            self.assertEqual(len(plotted), 16)
            for values in plotted:
                np.testing.assert_allclose(values, 0.3)
            fill_between.assert_not_called()

    def test_renderer_sidecar_and_companion_csv_name_figure7(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = {
                seed: _bundle(root, seed) for seed in figure7.CANONICAL_SEEDS
            }
            checkpoints = {
                seed: {
                    lead: bundles[seed].metadata["cases"][
                        f"el_nino_lead_{lead:02d}"
                    ]["checkpoint_sha256"]
                    for lead in figure7.EXPECTED_LEADS
                }
                for seed in figure7.CANONICAL_SEEDS
            }
            output = root / "figure7_zcv3_core4.pdf"

            def fake_plot(
                _rows: list[dict],
                path: Path,
                *,
                dpi: int,
                overwrite: bool,
            ) -> None:
                del dpi, overwrite
                path.write_bytes(b"synthetic PDF")

            arguments = [
                "make_fresh_figure7.py",
                "--bundle-root",
                str(root),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(
                    figure7,
                    "_load_seed_bundles",
                    return_value=(
                        bundles,
                        checkpoints,
                        {"contract": "fixed"},
                        False,
                    ),
                ),
                mock.patch.object(
                    figure7,
                    "_read_seed_scores",
                    side_effect=[
                        _seed_scores((seed - 41) / 10.0)
                        for seed in figure7.CANONICAL_SEEDS
                    ],
                ),
                mock.patch.object(figure7, "_plot", side_effect=fake_plot),
            ):
                self.assertEqual(figure7.main(), 0)
            summary = root / "figure7_zcv3_core4_scores.csv"
            sidecar = json.loads(
                output.with_suffix(".json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary.is_file())
            self.assertEqual(sidecar["figure"], 7)
            self.assertEqual(sidecar["model_seeds"], [42, 43, 44, 45, 46])
            self.assertFalse(sidecar["aggregation"]["uncertainty_displayed"])
            self.assertEqual(sidecar["summary"]["file"], summary.name)

    def test_seed_generator_resume_requires_current_source_identity(self) -> None:
        arguments = generator.build_parser().parse_args([])
        arguments.output_dir = Path("unused")
        metadata = _metadata(42)
        fake = CompletedBundle(
            directory=Path("unused"),
            metadata_path=Path("unused/metadata.json"),
            completion_path=Path("unused/completed.json"),
            metadata=metadata,
            files={},
            sha256={},
        )
        with mock.patch.object(generator, "load_completed_bundle", return_value=fake):
            self.assertTrue(generator._is_matching_completed_bundle(arguments))
            fake.metadata["source_sha256"]["generator"] = _digest(999)
            self.assertFalse(generator._is_matching_completed_bundle(arguments))


if __name__ == "__main__":
    unittest.main()
