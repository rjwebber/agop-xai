#!/usr/bin/env python3
"""Typeset revised Table I from the completed fresh core4 XAI artifact."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from zc_xai.fresh_xai_outputs import (  # noqa: E402
    FRESH_EXPECTED_GRADIENT_COUNT,
    FRESH_IG_GRADIENT_COUNT,
    FRESH_ROBUSTNESS_NEIGHBOR_PERCENT,
    FRESH_TABLE_ARCHITECTURES,
    FRESH_TABLE_METHODS,
    FRESH_TABLE_SCHEMA_VERSION,
    FRESH_XAI_GRADIENT_BATCH_SIZE,
    FRESH_XAI_TRAINING_POPULATION,
)
from zc_xai.io import (  # noqa: E402
    atomic_output_path,
    load_json,
    sha256_file,
    write_json,
)

METRICS = (
    ("sensitivity", "S_s"),
    ("attribution", "S_a"),
    ("robustness", "S_r"),
    ("coherence_spatial_only", "S_c"),
)
ARCHITECTURE_LABELS = {"mlp": "MLP", "cnn": "CNN", "vit": "ViT"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create revised fresh-data Table I LaTeX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("artifacts/zc-v3/manuscript/table1_figure6"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/manuscript/tables/table1_xai_scores.tex"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _safe_bundle_file(table_dir: Path, record: Any, label: str) -> Path:
    values = _mapping(record, label)
    filename = values.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"{label} has an unsafe filename.")
    path = table_dir / filename
    if not path.is_file() or sha256_file(path) != values.get("sha256"):
        raise ValueError(f"{label} is missing or failed its checksum.")
    return path


def load_completed_scores(table_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Validate and return the score table from one completed Table I bundle."""

    directory = table_dir.expanduser().resolve()
    metadata_path = directory / "table1_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Table I metadata not found: {metadata_path}")
    metadata = load_json(metadata_path)
    identity = _mapping(metadata.get("run_identity"), "run_identity")
    if (
        metadata.get("schema_version") != FRESH_TABLE_SCHEMA_VERSION
        or metadata.get("status") != "complete"
        or identity.get("input_profile") != "core4"
        or identity.get("expected_architectures")
        != list(FRESH_TABLE_ARCHITECTURES)
    ):
        raise ValueError("Fresh Table I metadata is incomplete or incompatible.")

    robustness = _mapping(identity.get("robustness"), "run_identity.robustness")
    methods = _mapping(identity.get("methods"), "run_identity.methods")
    integrated_gradients = _mapping(
        methods.get("integrated_gradients"),
        "run_identity.methods.integrated_gradients",
    )
    gradient_shap = _mapping(
        methods.get("gradient_shap"), "run_identity.methods.gradient_shap"
    )
    if (
        robustness.get("nearest_percent")
        != FRESH_ROBUSTNESS_NEIGHBOR_PERCENT
        or robustness.get("candidate_population")
        != FRESH_XAI_TRAINING_POPULATION
        or robustness.get("sampling")
        != "none: every member of the nearest-percent population"
        or methods.get("gradient_batch_size") != FRESH_XAI_GRADIENT_BATCH_SIZE
        or integrated_gradients.get("gradient_count_per_explanation")
        != FRESH_IG_GRADIENT_COUNT
        or gradient_shap.get("distinct_empirical_backgrounds")
        != FRESH_EXPECTED_GRADIENT_COUNT
        or gradient_shap.get("reference_population")
        != FRESH_XAI_TRAINING_POPULATION
    ):
        raise ValueError("Fresh Table I does not use the final XAI configuration.")

    outputs = _mapping(metadata.get("output_files"), "output_files")
    scores_path = _safe_bundle_file(directory, outputs.get("scores"), "scores")
    return scores_path, metadata_path, metadata


def read_scores(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows: dict[tuple[str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            method = source.get("method")
            architecture = source.get("architecture")
            if method not in FRESH_TABLE_METHODS:
                raise ValueError(f"Unexpected Table I method {method!r}.")
            if architecture not in FRESH_TABLE_ARCHITECTURES:
                raise ValueError(
                    f"Unexpected Table I architecture {architecture!r}."
                )
            key = (method, architecture)
            if key in rows:
                raise ValueError(f"Duplicate Table I score row {key}.")
            try:
                values = {metric: float(source[metric]) for metric, _ in METRICS}
                exhaustive = source["exhaustive"] == "True"
                neighbor_percent = float(source["neighbor_percent"])
                evaluated_count = int(source["evaluated_count"])
                neighborhood_count = int(source["neighborhood_count"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Malformed Table I score row {key}.") from error
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Nonfinite Table I score row {key}.")
            if (
                not exhaustive
                or neighbor_percent != FRESH_ROBUSTNESS_NEIGHBOR_PERCENT
                or evaluated_count <= 0
                or evaluated_count != neighborhood_count
            ):
                raise ValueError(
                    f"Table I robustness is not exhaustive nearest 1% for {key}."
                )
            rows[key] = values
    expected = {
        (method, architecture)
        for method in FRESH_TABLE_METHODS
        for architecture in FRESH_TABLE_ARCHITECTURES
    }
    if set(rows) != expected:
        raise ValueError(f"Missing Table I rows: {sorted(expected - set(rows))}")
    return rows


def _format(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 0.1 or magnitude == 0.0:
        return f"{value:.2f}"
    return f"{value:.3f}" if magnitude >= 0.01 else f"{value:.4f}"


def latex_table(rows: dict[tuple[str, str], dict[str, float]]) -> str:
    maxima = {
        (architecture, metric): max(
            rows[(method, architecture)][metric]
            for method in FRESH_TABLE_METHODS
        )
        for architecture in FRESH_TABLE_ARCHITECTURES
        for metric, _ in METRICS
    }
    architecture_header = " & ".join(
        ARCHITECTURE_LABELS[architecture]
        for _metric, _symbol in METRICS
        for architecture in FRESH_TABLE_ARCHITECTURES
    )
    lines = [
        r"\begin{tabular}{l|ccc|ccc|ccc|ccc|}",
        r"\hline",
        " & "
        + " & ".join(
            rf"\multicolumn{{3}}{{c|}}{{${symbol}$}}"
            for _metric, symbol in METRICS
        )
        + r" \\",
        r"\cline{2-4} \cline{5-7} \cline{8-10} \cline{11-13}",
        " & " + architecture_header + r" \\",
        r"\hline",
    ]
    for method in FRESH_TABLE_METHODS:
        cells = []
        for metric, _symbol in METRICS:
            for architecture in FRESH_TABLE_ARCHITECTURES:
                value = rows[(method, architecture)][metric]
                text = _format(value)
                if math.isclose(
                    value,
                    maxima[(architecture, metric)],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    text = rf"\textbf{{{text}}}"
                cells.append(text)
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\hline", r"\end{tabular}", ""))
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".tex":
        raise ValueError("--output must end in .tex")
    sidecar = output.with_suffix(".json")
    if not args.overwrite and (output.exists() or sidecar.exists()):
        raise FileExistsError("Table or sidecar exists; use --overwrite.")
    scores_path, metadata_path, metadata = load_completed_scores(args.table_dir)
    rows = read_scores(scores_path)
    with (
        atomic_output_path(output, overwrite=args.overwrite) as temporary,
        temporary.open("w", encoding="utf-8") as stream,
    ):
        stream.write(latex_table(rows))
    write_json(
        sidecar,
        {
            "schema_version": 1,
            "table": 1,
            "methods": list(FRESH_TABLE_METHODS),
            "architectures": list(FRESH_TABLE_ARCHITECTURES),
            "metrics": [metric for metric, _symbol in METRICS],
            "xai_configuration": {
                "gradient_batch_size": FRESH_XAI_GRADIENT_BATCH_SIZE,
                "integrated_gradients": FRESH_IG_GRADIENT_COUNT,
                "gradient_shap": FRESH_EXPECTED_GRADIENT_COUNT,
                "neighbor_percent": FRESH_ROBUSTNESS_NEIGHBOR_PERCENT,
                "neighbor_sampling": "none; exhaustive finite population",
                "reference_population": FRESH_XAI_TRAINING_POPULATION,
            },
            "source": {
                "table1_metadata_sha256": sha256_file(metadata_path),
                "scores_file": scores_path.name,
                "scores_sha256": sha256_file(scores_path),
                "data_metadata_sha256": metadata["run_identity"]["data"][
                    "metadata_sha256"
                ],
            },
            "output": {"file": output.name, "sha256": sha256_file(output)},
        },
        overwrite=args.overwrite,
    )
    print(f"Fresh Table I saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
