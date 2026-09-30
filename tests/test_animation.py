"""Fast checks for the public zc-v3 animation workflow."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "animate_zc_data.py"
SPEC = importlib.util.spec_from_file_location("animate_zc_data", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
animation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = animation
SPEC.loader.exec_module(animation)


class AnimationTests(unittest.TestCase):
    def test_default_profile_is_public_core4_data(self) -> None:
        arguments = animation.parse_args(["--data-dir", "data/processed/zc-v3"])
        self.assertEqual(arguments.input_profile, "core4")
        self.assertEqual(arguments.vector_kind, "ocean-current")
        self.assertEqual(arguments.train_years, 10_000.0)

    def test_normalization_artifact_path_matches_training_layout(self) -> None:
        self.assertEqual(
            animation.normalization_artifact_directory(
                "core4", "cnn", 10, 10_000.0, 42
            ),
            "models/core4/cnn/lead-10m/years-10000/seed-000042",
        )

    def test_standardized_field_uses_featurewise_fit_statistics(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        field = animation.StandardizedField(
            source=source,
            mean=np.full((3, 4), 2.0, dtype=np.float32),
            scale=np.full((3, 4), 2.0, dtype=np.float32),
        )
        np.testing.assert_allclose(field[1], (source[1] - 2.0) / 2.0)
        np.testing.assert_allclose(field[1, 1:, 2:], (source[1, 1:, 2:] - 2.0) / 2.0)

    def test_even_bwr_has_no_exact_white_lookup_entry(self) -> None:
        palette = animation._even_bwr()
        colors = palette(np.arange(animation.COLORMAP_LOOKUP_TABLE_SIZE))[:, :3]
        self.assertFalse(np.any(np.all(colors == 1.0, axis=1)))


if __name__ == "__main__":
    unittest.main()
