#!/usr/bin/env python3
"""Build the portable, source-free run23 ZC adjoint evidence bundle.

This curator deliberately publishes validation reports, manifests, and
gradient packages, but not upstream ZC source, Tapenade-generated source,
model executables, checkpoints, or raw path files. Project-authored ZC-derived
patches and coupled drivers are published separately as repository source and
are not duplicated in this evidence bundle.
The omitted producer artifacts are still bound by SHA-256 in the published
records.  An authorized user can reconstruct them with the documented build
and case recipes, then re-run the strict gradient packager.

The result is evidence for a configuration-specific, major-tape-conditioned
research derivative.  The script does not promote any input report to global
derivative certification and refuses evidence that claims otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.package_zc_adjoint_gradient import package_gradient  # noqa: E402

SCRIPT_VERSION = "1.0.1"
INDEX_SCHEMA_VERSION = 2
CASE_LABELS = ("extreme_el_nino", "extreme_la_nina", "neutral_member_04")
CASE_TITLES = {
    "extreme_el_nino": "Extreme El Nino",
    "extreme_la_nina": "Extreme La Nina",
    "neutral_member_04": "Neutral member 04",
}
REQUIRED_AUDITS = (
    "scalar_dot_extreme_el_nino",
    "scalar_dot_extreme_la_nina",
    "scalar_dot_neutral_member_04",
    "generic_dot_extreme_el_nino",
    "generic_dot_extreme_la_nina",
    "generic_dot_neutral_member_04",
    "tangent_one_step_independent",
    "tangent_one_step_active",
    "tangent_composed_31step",
    "scalar_taylor_random_full",
    "scalar_taylor_aligned_el",
    "scalar_taylor_aligned_la",
    "scalar_taylor_aligned_neutral",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MESSAGE_CODE = re.compile(r"^\s*\d+\s+.*?\(([A-Z]{2}\d{2})\)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--kernel-build-dir",
        type=Path,
        default=Path("outputs/zc_adjoint/kernel_build_run23_final3"),
    )
    parser.add_argument(
        "--replay-validation-dir",
        type=Path,
        default=Path("outputs/zc_adjoint/kernel_replay_validation_run23_final3"),
    )
    parser.add_argument(
        "--case-root",
        type=Path,
        default=Path("outputs/zc_adjoint/run23_final3_case_replay"),
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("outputs/zc_adjoint/run23_final3_numerical_audit"),
    )
    parser.add_argument(
        "--coupled-build-dir",
        type=Path,
        default=Path(
            "adjoint/tapenade_toolchain/build/"
            "coupled_run23_final3/coupled_compiled"
        ),
    )
    parser.add_argument(
        "--tangent-executable",
        type=Path,
        default=Path(
            "adjoint/full_tangent_audit/build/"
            "zc_tangent_path_driver_run23_final3_o3"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("adjoint/tapenade_toolchain/results/run23"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def resolve_from(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (root / path).resolve()


def require_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Required regular input is missing: {path}")
    return path


def relpath(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def _portable_string(value: str, *, root: Path) -> str:
    """Replace local identities and ephemeral transaction roots in text."""

    replacements = (
        (str(root), "<PROJECT_ROOT>"),
        (
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
            "<MACOS_SDK>",
        ),
        ("/opt/homebrew/Cellar/gcc/13.1.0/bin/gfortran-13", "gfortran-13"),
        ("/usr/bin/cc", "cc"),
        ("/usr/bin/ar", "ar"),
        ("/usr/bin/patch", "patch"),
    )
    result = value
    for source, target in replacements:
        result = result.replace(source, target)
    result = re.sub(
        r"/(?:Users|home)/[^/\s\"']+/agop-xai",
        "<PROJECT_ROOT>",
        result,
    )
    result = re.sub(
        r"/dsmlp/home-[^/\s\"']+/\d+/\d+/[^/\s\"']+/agop-xai",
        "<PROJECT_ROOT>",
        result,
    )
    result = re.sub(r"/Users/[^/\s\"']+", "<LOCAL_HOME>", result)
    result = re.sub(r"/home/[^/\s\"']+", "<CLUSTER_HOME>", result)
    result = re.sub(
        r"/dsmlp/home-[^/\s\"']+/\d+/\d+/[^/\s\"']+",
        "<CLUSTER_HOME>",
        result,
    )
    # Validator reports intentionally record their private transaction path.
    # Retain the final artifact basename while removing host-specific temp roots.
    result = re.sub(
        r"/var/folders/[^\s\"']+?/(verified_inputs(?:/[^\s\"']*)?)",
        r"<PRIVATE_VALIDATION_STAGE>/\1",
        result,
    )
    result = re.sub(
        r"/private/tmp/[^\s\"']+?/(verified_inputs(?:/[^\s\"']*)?)",
        r"<PRIVATE_VALIDATION_STAGE>/\1",
        result,
    )
    return result


def sanitize(value: Any, *, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item, root=root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item, root=root) for item in value]
    if isinstance(value, str):
        return _portable_string(value, root=root)
    return value


def sanitize_json(
    source: Path,
    destination: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    original = load_json(require_file(source))
    clean = sanitize(original, root=root)
    write_json(destination, clean)
    return {
        "source_sha256": sha256_file(source),
        "published_sha256": sha256_file(destination),
        "sanitization": (
            "JSON values preserved except local project/home/compiler/SDK and "
            "private validation-stage paths were replaced by explicit tokens; "
            "key order and whitespace were canonicalized"
        ),
    }


def require_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest.")
    return value


def verify_report_contract(report: dict[str, Any], *, label: str) -> None:
    if report.get("status") != "completed_not_certified":
        raise ValueError(f"{label} does not have completed_not_certified status.")
    if report.get("acceptance_thresholds") is not None:
        raise ValueError(f"{label} unexpectedly applies an acceptance threshold.")
    if report.get("certification_assessed") is not False:
        raise ValueError(f"{label} unexpectedly claims certification.")


def report_step_summary(case: dict[str, Any]) -> dict[str, Any]:
    steps = case["steps"]
    stable = [step for step in steps if step["both_major_tapes_match_baseline"]]
    best = (
        min(stable, key=lambda step: step["centered_fd_relative_defect"])
        if stable
        else None
    )
    return {
        "case": case["case"],
        "direction_mode": case["direction_mode"],
        "direction_scope": case["direction_scope"],
        "tested_step_sizes": [step["h"] for step in steps],
        "major_tape_stable_step_sizes": [step["h"] for step in stable],
        "best_major_tape_stable_centered_fd": (
            {
                "h": best["h"],
                "relative_defect": best["centered_fd_relative_defect"],
                "absolute_defect": best["centered_fd_absolute_defect"],
                "input_rounding_relative_l2": best[
                    "centered_input_direction_rounding_relative_l2"
                ],
            }
            if best is not None
            else None
        ),
    }


def build_transpose_summary(
    audit_reports: dict[str, dict[str, Any]],
    audit_paths: dict[str, Path],
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case in CASE_LABELS:
        scalar_key = f"scalar_dot_{case}"
        generic_key = f"generic_dot_{case}"
        scalar = audit_reports[scalar_key]
        generic = audit_reports[generic_key]
        cases[case] = {
            "scalar_nino3_head": {
                "seeds": len(scalar["tests"]),
                "maximum_bilinear_relative_defect": scalar[
                    "maximum_bilinear_relative_defect"
                ],
                "maximum_norm_product_scaled_defect": scalar[
                    "maximum_norm_scaled_transpose_defect"
                ],
                "source_report_sha256": sha256_file(audit_paths[scalar_key]),
            },
            "dense_generic_terminal_seed": {
                "seeds": len(generic["tests"]),
                "maximum_bilinear_relative_defect": generic[
                    "maximum_bilinear_relative_defect"
                ],
                "maximum_norm_product_scaled_defect": generic[
                    "maximum_norm_scaled_transpose_defect"
                ],
                "source_report_sha256": sha256_file(audit_paths[generic_key]),
            },
        }
    return {
        "schema_version": 1,
        "status": "completed_not_certified",
        "certification_assessed": False,
        "claim": (
            "Same-generation 31-transition real32 tangent/reverse transpose "
            "comparisons on three locked checkpoints. No automatic acceptance "
            "threshold was applied. Both cancellation-sensitive bilinear and "
            "norm-product-scaled defects are reported."
        ),
        "cases": cases,
        "global_maxima": {
            "scalar_bilinear_relative": max(
                item["scalar_nino3_head"]["maximum_bilinear_relative_defect"]
                for item in cases.values()
            ),
            "scalar_norm_product_scaled": max(
                item["scalar_nino3_head"]["maximum_norm_product_scaled_defect"]
                for item in cases.values()
            ),
            "generic_bilinear_relative": max(
                item["dense_generic_terminal_seed"][
                    "maximum_bilinear_relative_defect"
                ]
                for item in cases.values()
            ),
            "generic_norm_product_scaled": max(
                item["dense_generic_terminal_seed"][
                    "maximum_norm_product_scaled_defect"
                ]
                for item in cases.values()
            ),
        },
    }


def build_fd_summary(
    audit_reports: dict[str, dict[str, Any]],
    audit_paths: dict[str, Path],
) -> dict[str, Any]:
    one_step: dict[str, Any] = {}
    for scope, key in (
        ("independent_control", "tangent_one_step_independent"),
        ("active_carried_state", "tangent_one_step_active"),
    ):
        report = audit_reports[key]
        one_step[scope] = {
            "source_report_sha256": sha256_file(audit_paths[key]),
            "cases": {
                case["checkpoint"]: {
                    "h": case["centered_difference_steps"][0]["h"],
                    "full_output_relative_l2_error": case[
                        "centered_difference_steps"
                    ][0]["full_relative_l2_error"],
                    "full_output_cosine": case["centered_difference_steps"][0][
                        "full_cosine"
                    ],
                    "major_tape_matches_baseline": case[
                        "centered_difference_steps"
                    ][0]["major_tape_matches_baseline"],
                }
                for case in report["cases"]
            },
        }

    aligned: dict[str, Any] = {}
    for public_name, key in (
        ("extreme_el_nino", "scalar_taylor_aligned_el"),
        ("extreme_la_nina", "scalar_taylor_aligned_la"),
        ("neutral_member_04", "scalar_taylor_aligned_neutral"),
    ):
        report = audit_reports[key]
        item = report_step_summary(report["cases"][0])
        item["source_report_sha256"] = sha256_file(audit_paths[key])
        aligned[public_name] = item

    composed = audit_reports["tangent_composed_31step"]
    composed_summary = {
        "source_report_sha256": sha256_file(audit_paths["tangent_composed_31step"]),
        "interpretation": (
            "Diagnostic only, not a pass: random full-state segment-scaled "
            "directions change the recorded major tape for the warm and cold "
            "cases; the neutral case remains tape-stable but is noisy in "
            "production float32 arithmetic."
        ),
        "cases": {
            case["checkpoint"]: {
                "tested_step_sizes": [
                    step["h"] for step in case["centered_difference_steps"]
                ],
                "major_tape_stable_step_sizes": [
                    step["h"]
                    for step in case["centered_difference_steps"]
                    if step["major_tape_matches_baseline"]
                ],
            }
            for case in composed["cases"]
        },
    }
    random_scalar = audit_reports["scalar_taylor_random_full"]
    return {
        "schema_version": 1,
        "status": "provisional_evidence_not_certification",
        "certification_assessed": False,
        "one_step_centered_finite_difference": one_step,
        "aligned_scalar_31_transition": aligned,
        "random_scalar_31_transition_source_report_sha256": sha256_file(
            audit_paths["scalar_taylor_random_full"]
        ),
        "random_scalar_31_transition": [
            report_step_summary(case) for case in random_scalar["cases"]
        ],
        "composed_full_state_31_transition": composed_summary,
        "limitations": [
            "No clean multi-level h^2 Taylor regime was established.",
            "Input rounding and trajectory arithmetic are real32.",
            "The three-slot external tape does not expose internal sign, cap, "
            "upwind, CFL-controlling-cell, or threshold-distance information.",
            "Every tested aligned La Nina perturbation changes NSST at step 15.",
        ],
    }


def build_warning_summary(generated_root: Path) -> dict[str, Any]:
    files = {
        "tangent": require_file(generated_root / "tangent/zc_kernel_nino3_d.msg"),
        "reverse": require_file(generated_root / "reverse/zc_kernel_nino3_b.msg"),
    }
    results: dict[str, Any] = {}
    for mode, path in files.items():
        counts: Counter[str] = Counter()
        line_count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            line_count += 1
            match = MESSAGE_CODE.match(line)
            counts[match.group(1) if match else "TOOL_MESSAGE"] += 1
        results[mode] = {
            "message_count": line_count,
            "message_code_counts": dict(sorted(counts.items())),
            "source_message_file_sha256": sha256_file(path),
        }
    return {
        "schema_version": 1,
        "tapenade": {
            "version": "3.16",
            "revision": "0449e37b7da896cb2c38b63c8a34f10c87a84534",
            "archive_sha256": (
                "534b83598c508a3202880016403f84e352e4b00bdc4c6b934bb39ccbdddb0963"
            ),
        },
        "messages": results,
        "important_interpretation": [
            "Tapenade reported implicit external declarations for GGNQF and "
            "GGUBFS; the build links the independently hashed support routines.",
            "Tapenade reported legacy mixed-kind assignments and a complex "
            "constructor cost-analysis message.",
            "I/O warnings describe dormant diagnostic writes in the prepared "
            "source; the differentiated kernel path is run I/O-free.",
            "Generation warnings were retained and were not treated as proof "
            "of derivative correctness; numerical evidence is reported separately.",
        ],
        "raw_message_files_published": False,
    }


def producer_document(
    *,
    case: str,
    case_dir: Path,
    public_case_dir: Path,
    provenance_dir: Path,
    evidence_dir: Path,
    reverse_executable: Path,
) -> dict[str, Any]:
    run_report = require_file(case_dir / "run_report.json")
    report = load_json(run_report)
    if report.get("status") != "completed":
        raise ValueError(f"Case {case} did not complete.")
    contract = report["scientific_contract"]
    if (
        contract.get("configuration_verified") is not True
        or contract.get("transitions") != 31
        or contract.get("real_state_length") != 59148
    ):
        raise ValueError(f"Case {case} has the wrong scientific contract.")

    checkpoint = require_file(case_dir / "runtime/zeq9fsu.hst")
    path = require_file(case_dir / "artifacts/zc_31step_path.bin")
    gradient = require_file(case_dir / "artifacts/zc_nino3_gradient.bin")
    hashes = report["results"]["output_sha256"]
    for key, file_path in (
        ("artifacts/zc_31step_path.bin", path),
        ("artifacts/zc_nino3_gradient.bin", gradient),
    ):
        if sha256_file(file_path) != require_hash(hashes[key], label=f"{case}:{key}"):
            raise ValueError(f"Case {case} output changed: {key}")

    return {
        "schema_version": 2,
        "objective": {
            "name": "canonical_nino3",
            "definition": (
                "Mean of 66 final TO cells at Fortran I=13:18, J=21:31 "
                "after 31 coupled ten-day transitions."
            ),
            "units": "degrees Celsius",
            "terminal_seed": (
                "uniform float32 1/66 on the 66 terminal TO cells; zero elsewhere"
            ),
        },
        "transitions": 31,
        "initial_checkpoint": {
            "label": case,
            "path": relpath(checkpoint, public_case_dir),
            "sha256": sha256_file(checkpoint),
        },
        "certified_path": {
            "path": relpath(path, public_case_dir),
            "sha256": sha256_file(path),
        },
        "gradient": {
            "path": relpath(gradient, public_case_dir),
            "sha256": sha256_file(gradient),
        },
        "source_manifests": {
            kind: {
                "path": relpath(
                    provenance_dir / f"{kind}_compile_inputs.sha256",
                    public_case_dir,
                ),
                "sha256": sha256_file(
                    provenance_dir / f"{kind}_compile_inputs.sha256"
                ),
            }
            for kind in ("prepared", "tangent", "reverse")
        },
        "reverse_executable": {
            "path": relpath(reverse_executable, public_case_dir),
            "sha256": sha256_file(reverse_executable),
        },
        "tapenade": {
            "version": "3.16",
            "revision": "0449e37b7da896cb2c38b63c8a34f10c87a84534",
            "archive_sha256": (
                "534b83598c508a3202880016403f84e352e4b00bdc4c6b934bb39ccbdddb0963"
            ),
        },
        "certification": {
            "status": "provisional",
            "scope": (
                "provisional_major_tape_conditioned: locked Standard, NSEG=1, "
                "NATM=2, NIC=0, default-real 31-transition branchwise derivative; "
                "not global certification"
            ),
            "evidence": [
                {
                    "kind": "primal_replay",
                    "path": relpath(
                        evidence_dir / "primal_replay_report.json", public_case_dir
                    ),
                    "sha256": sha256_file(evidence_dir / "primal_replay_report.json"),
                },
                {
                    "kind": "tangent_reverse_dot",
                    "path": relpath(
                        evidence_dir / "transpose_summary.json", public_case_dir
                    ),
                    "sha256": sha256_file(evidence_dir / "transpose_summary.json"),
                },
                {
                    "kind": "independent_finite_difference_or_taylor",
                    "path": relpath(
                        evidence_dir / "finite_difference_summary.json",
                        public_case_dir,
                    ),
                    "sha256": sha256_file(
                        evidence_dir / "finite_difference_summary.json"
                    ),
                },
            ],
        },
    }


def build_readme() -> str:
    return """# Run23 Zebiak--Cane adjoint evidence

This is the portable, source-free evidence bundle for the run23 discrete
adjoint.  Its precise status is **`provisional_major_tape_conditioned`**.
It supports a branchwise derivative of the locked Standard Zebiak--Cane
Fortran calculation for the three recorded checkpoints; it is not global
derivative certification.

The authentic explicit-state primal passes 15 bitwise replay cases, three
fresh-process segmented replays, and six omitted-workspace poison tests.  The
same-generation real32 tangent and reverse pass scalar-head and dense-terminal
transpose comparisons.  One-step centered differences are especially strong
on the conservative independent-control mask.  Long-window scalar checks are
useful but limited by float32 rounding and switching: the best major-tape-
stable aligned defects are about 3.36e-4 for the extreme warm case and 1.11e-2
for the neutral case; every tested aligned cold-case perturbation changes NSST
at step 15.  No clean multi-level quadratic Taylor regime was established.

## Contents

- `INDEX.json` is the human- and machine-readable claim boundary and inventory.
- `evidence/` contains sanitized detailed reports and compact summaries.  The
  index records both each original report hash and its published sanitized
  hash, along with the sanitizer rule.
- `provenance/` contains the explicit state manifest, exact hash inventories,
  source-free build summaries, and Tapenade warning counts.
- `cases/*/gradient.npz` contains the packed gradient plus each named manifest
  segment.  `gradient.json` records its coordinate semantics and embeds the
  sanitized v2 producer provenance.  `producer_metadata.json` is the exact v2
  producer document used by the strict packager.
- `MANIFEST.sha256` covers every published file other than itself.  Finder
  metadata is never included.

## What is deliberately absent

The bundle does not publish upstream ZC source, Tapenade-generated source,
executables, checkpoints, or raw trajectory paths. Those artifacts are
identified by hashes in the evidence chain. Project-authored ZC-derived
patches and coupled drivers are published separately in the repository rather
than copied into this evidence bundle.

## Reproducing with authorized ZC source

1. Build and replay the explicit-state primal using the commands in
   `../../README.md` and `../../../fortran_kernel/README.md`.  Require the
   15/3/6 replay gates to pass for the exact three checkpoint hashes in
   `INDEX.json`.
2. Prepare the coupled sources locally, run pinned Tapenade 3.16 revision
   `0449e37b7da896cb2c38b63c8a34f10c87a84534` on Linux, instrument the
   generated reverse, and compile on the Mac.  Compare the resulting three
   compile-input inventories and executable hashes with `provenance/` and
   `INDEX.json`.
3. Run each checkpoint with `run_coupled_case.py`.  The released case record
   names the expected path, gradient, checkpoint, and reverse-executable
   hashes.  It uses 31 coupled ten-day transitions and the final `TO` mean over
   Fortran `I=13:18,J=21:31`.
4. Re-run the numerical validators described in
   `../../../full_tangent_audit/README.md`.  Compare original report hashes in
   `INDEX.json`; do not compare timing fields across machines.
5. From the repository root, re-run
   `python adjoint/tapenade_toolchain/scripts/curate_run23_release.py --overwrite`.
   The curator verifies the producer/evidence chain, rebuilds all named gradient
   packages, strips local paths from published reports, and writes the top-level
   manifest.

The gradient has 59,148 raw packed Fortran coordinates.  It is not a gradient
in the standardized 2,162-feature neural-network space, and raw cross-variable
norms are not feature importance.  A comparison with AGOP additionally needs
the differentiated observation/time-alignment map, the frozen normalization
transpose, and a balanced native-state lift or declared control map.

This warning matters numerically: the largest raw-gradient segment is carried
atmospheric memory for each released case (see `INDEX.json`).  Only 28,591
native entries belong to the first conservative independent-control mask.
Plotting all 59,148 coefficients as a physical precursor pattern would be
misleading.
"""


def tree_manifest(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Published bundle may not contain a symlink: {path}")
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        if path.name == ".DS_Store":
            raise ValueError(f"Finder metadata entered the staged release: {path}")
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + "\n"


def scan_release(root: Path) -> None:
    forbidden_names = {".DS_Store"}
    forbidden_suffixes = {".F", ".f", ".o", ".a"}
    forbidden_exact = {
        "zc_nino3_adjoint",
        "zc_adjoint_make_path",
        "zc_kernel_replay",
        "zc_reference",
    }
    forbidden_text = (
        re.compile(r"/Users/[^/\s\"']+"),
        re.compile(r"/home/[^/\s\"']+"),
        re.compile(r"/dsmlp/home-[^/\s\"']+/\d+/\d+/[^/\s\"']+"),
        re.compile(r"/var/folders/"),
        re.compile(r"/private/tmp/"),
    )
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Release contains a symlink: {path}")
        if not path.is_file():
            continue
        if (
            path.name in forbidden_names | forbidden_exact
            or path.suffix in forbidden_suffixes
        ):
            raise ValueError(f"Forbidden source/build artifact in release: {path}")
        if path.suffix in {".json", ".md", ".txt", ".sha256"}:
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden_text:
                if pattern.search(text):
                    raise ValueError(
                        f"Private path matching {pattern.pattern!r} remains in {path}"
                    )


def curate(args: argparse.Namespace) -> Path:
    root = args.project_root.expanduser().resolve()
    kernel_build = resolve_from(root, args.kernel_build_dir)
    replay_dir = resolve_from(root, args.replay_validation_dir)
    case_root = resolve_from(root, args.case_root)
    audit_root = resolve_from(root, args.audit_root)
    coupled_build = resolve_from(root, args.coupled_build_dir)
    tangent_executable = require_file(resolve_from(root, args.tangent_executable))
    output = resolve_from(root, args.output_dir)

    build_report_path = require_file(kernel_build / "build_report.json")
    replay_path = require_file(replay_dir / "replay_report.json")
    replay = load_json(replay_path)
    if replay.get("status") != "passed":
        raise ValueError("The final3 primal replay report did not pass.")
    if not (
        len(replay.get("cases", [])) == 15
        and len(replay.get("segmented_replay_cases", [])) == 3
        and len(replay.get("workspace_poison_cases", [])) == 6
    ):
        raise ValueError(
            "The replay report does not contain the required 15/3/6 gates."
        )
    if replay.get("build_report_sha256") != sha256_file(build_report_path):
        raise ValueError("Replay report is not bound to the supplied build report.")

    state_manifest = require_file(root / "adjoint/fortran_kernel/state_manifest.json")
    state = load_json(state_manifest)
    real_segments = state["arrays"]["real32"]["segments"]
    counts = {
        activity: sum(
            int(segment["stop"]) - int(segment["start"])
            for segment in real_segments
            if segment["activity"] == activity
        )
        for activity in ("active", "diagnostic", "passive")
    }
    independent_count = sum(
        int(segment["stop"]) - int(segment["start"])
        for segment in real_segments
        if segment["independent_control"]
    )
    if counts != {"active": 38315, "diagnostic": 18910, "passive": 1923}:
        raise ValueError(f"Unexpected real-state activity counts: {counts}")
    if independent_count != 28591:
        raise ValueError(f"Unexpected independent-control count: {independent_count}")

    audit_paths = {
        name: require_file(audit_root / name / "report.json")
        for name in REQUIRED_AUDITS
    }
    audit_reports = {name: load_json(path) for name, path in audit_paths.items()}
    for name, report in audit_reports.items():
        verify_report_contract(report, label=name)
        if report.get("state_manifest_sha256") != sha256_file(state_manifest):
            raise ValueError(f"{name} is bound to another state manifest.")

    reverse_executable = require_file(coupled_build / "zc_nino3_adjoint")
    build_manifest = require_file(coupled_build / "build_manifest.txt")
    compile_inventories = {
        kind: require_file(coupled_build / f"{kind}_compile_inputs.sha256")
        for kind in ("prepared", "tangent", "reverse")
    }
    generated_root = coupled_build / "build_input_snapshot/generated"

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The candidate is a sibling of the final target. Relative paths embedded
    # in producer metadata therefore remain valid after the atomic rename.
    staged = Path(tempfile.mkdtemp(prefix=".run23-curation-", dir=output.parent))
    try:
        evidence_dir = staged / "evidence"
        provenance_dir = staged / "provenance"
        cases_dir = staged / "cases"
        evidence_dir.mkdir()
        provenance_dir.mkdir()
        cases_dir.mkdir()

        file_records: dict[str, Any] = {}
        file_records["evidence/primal_replay_report.json"] = sanitize_json(
            replay_path,
            evidence_dir / "primal_replay_report.json",
            root=root,
        )
        file_records["provenance/kernel_build_report.json"] = sanitize_json(
            build_report_path,
            provenance_dir / "kernel_build_report.json",
            root=root,
        )
        for name, path in audit_paths.items():
            published = evidence_dir / f"{name}.json"
            file_records[f"evidence/{name}.json"] = sanitize_json(
                path,
                published,
                root=root,
            )

        zero_seed_report = audit_root / "zero_terminal_seed_smoke/run_report.json"
        if zero_seed_report.is_file() and not zero_seed_report.is_symlink():
            zero_seed = load_json(zero_seed_report)
            zero_results = zero_seed.get("results", {})
            zero_contract = zero_seed.get("scientific_contract", {})
            if not (
                zero_seed.get("status") == "completed"
                and zero_contract.get("configuration_verified") is True
                and zero_contract.get("objective")
                == "custom terminal real-state linear functional"
                and zero_results.get("gradient_l2_norm_recomputed_float64") == 0.0
                and zero_results.get("gradient_max_abs_recomputed") == 0.0
                and zero_results.get("gradient_nonzero_count") == 0
                and float(
                    zero_results["gradient_metadata"]["objective_value"]
                )
                == 0.0
            ):
                raise ValueError("The zero-terminal-seed smoke report is invalid.")
            file_records["evidence/zero_terminal_seed_smoke.json"] = sanitize_json(
                zero_seed_report,
                evidence_dir / "zero_terminal_seed_smoke.json",
                root=root,
            )

        transpose_summary = build_transpose_summary(audit_reports, audit_paths)
        write_json(evidence_dir / "transpose_summary.json", transpose_summary)
        fd_summary = build_fd_summary(audit_reports, audit_paths)
        write_json(evidence_dir / "finite_difference_summary.json", fd_summary)
        warning_summary = build_warning_summary(generated_root)
        write_json(provenance_dir / "tapenade_warning_summary.json", warning_summary)

        shutil.copy2(state_manifest, provenance_dir / "state_manifest.json")
        upstream_manifest = require_file(kernel_build / "source_manifest.json")
        shutil.copy2(
            upstream_manifest,
            provenance_dir / "upstream_source_manifest.json",
        )
        for kind, source in compile_inventories.items():
            shutil.copy2(source, provenance_dir / f"{kind}_compile_inputs.sha256")
        portable_build = _portable_string(
            build_manifest.read_text(encoding="utf-8"), root=root
        )
        (provenance_dir / "coupled_build_manifest.sanitized.txt").write_text(
            portable_build, encoding="utf-8"
        )

        case_index: dict[str, Any] = {}
        for case in CASE_LABELS:
            source_case = case_root / case
            public_case = cases_dir / case
            public_case.mkdir()
            run_report_path = require_file(source_case / "run_report.json")
            run_report = load_json(run_report_path)
            if run_report["build"]["adjoint_executable_sha256"] != sha256_file(
                reverse_executable
            ):
                raise ValueError(f"Case {case} is bound to another reverse executable.")
            for kind, inventory in compile_inventories.items():
                recorded = run_report["build"][f"{kind}_compile_inputs_sha256"]
                if recorded != sha256_file(inventory):
                    raise ValueError(
                        f"Case {case} is bound to another {kind} inventory."
                    )
            file_records[f"cases/{case}/producer_run_report.json"] = sanitize_json(
                run_report_path,
                public_case / "producer_run_report.json",
                root=root,
            )

            producer = producer_document(
                case=case,
                case_dir=source_case,
                public_case_dir=public_case,
                provenance_dir=provenance_dir,
                evidence_dir=evidence_dir,
                reverse_executable=reverse_executable,
            )
            producer_path = public_case / "producer_metadata.json"
            write_json(producer_path, producer)
            gradient_path = source_case / "artifacts/zc_nino3_gradient.bin"
            package_gradient(
                gradient_path,
                manifest_path=state_manifest,
                output_prefix=public_case / "gradient",
                byte_order="little",
                producer_metadata_path=producer_path,
                overwrite=False,
            )
            package_report = load_json(public_case / "gradient.json")
            if (
                package_report["adjoint_certification"]["producer_declared_status"]
                != "provisional"
            ):
                raise ValueError(f"Packaged case {case} lost provisional status.")
            dominant_segment = max(
                package_report["segments"],
                key=lambda item: item["squared_l2_fraction_of_packed_gradient"],
            )
            total_squared = package_report["packed_gradient"]["squared_l2_norm"]
            independent_squared = package_report[
                "independent_control_gradient_summary"
            ]["squared_l2_norm"]
            case_index[case] = {
                "title": CASE_TITLES[case],
                "initial_checkpoint_sha256": producer["initial_checkpoint"]["sha256"],
                "forward_path_sha256": producer["certified_path"]["sha256"],
                "raw_gradient_sha256": producer["gradient"]["sha256"],
                "final_nino3_celsius": float(
                    run_report["results"]["path_metadata"]["final_nino3_celsius"]
                ),
                "reverse_cpu_seconds": float(
                    run_report["results"]["gradient_metadata"]["cpu_seconds"]
                ),
                "gradient_l2_norm": run_report["results"][
                    "gradient_l2_norm_recomputed_float64"
                ],
                "gradient_package_report_sha256": sha256_file(
                    public_case / "gradient.json"
                ),
                "gradient_package_npz_sha256": sha256_file(
                    public_case / "gradient.npz"
                ),
                "producer_metadata_sha256": sha256_file(producer_path),
                "source_run_report_sha256": sha256_file(run_report_path),
                "raw_gradient_interpretation_warning": {
                    "dominant_manifest_segment": dominant_segment["name"],
                    "dominant_segment_squared_l2_fraction": dominant_segment[
                        "squared_l2_fraction_of_packed_gradient"
                    ],
                    "independent_control_squared_l2_fraction": (
                        independent_squared / total_squared
                        if total_squared > 0.0
                        else None
                    ),
                    "raw_mixed_unit_norm_is_not_feature_importance": True,
                },
            }

        latest_evidence_time = max(
            str(report.get("created_utc", "")) for report in audit_reports.values()
        )
        index = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "curator_script_version": SCRIPT_VERSION,
            "run_id": "run23-final3",
            "status": "provisional_major_tape_conditioned",
            "certification": {
                "assessed": False,
                "globally_certified": False,
                "automatic_acceptance_thresholds_applied": False,
            },
            "curation_time": {
                "value": latest_evidence_time,
                "semantics": (
                    "latest created_utc value among the immutable input evidence; "
                    "not wall-clock execution time of this deterministic curator"
                ),
            },
            "scientific_contract": {
                "model": "authentic Zebiak-Cane Fortran implementation",
                "configuration": {
                    "grid": "Standard",
                    "NSEG": 1,
                    "NATM": 2,
                    "NIC": 0,
                    "stochastic_westerly_wind_bursts": False,
                    "arithmetic": "production-compatible default REAL (real32)",
                },
                "state_semantics": "pre-input native checkpoint",
                "transitions": 31,
                "step_length": "10 days",
                "objective": (
                    "mean final TO over 66 cells, Fortran I=13:18,J=21:31"
                ),
                "real_state": {
                    "total": 59148,
                    **counts,
                    "conservative_independent_control": independent_count,
                },
            },
            "claim_boundary": {
                "supported": [
                    "bitwise authentic-primal replay in all 15 locked window cases",
                    "fresh-process state closure in all three 40-step segmented cases",
                    "omitted-workspace closure in all six poison cases",
                    (
                        "same-generation scalar and generic-terminal tangent/reverse "
                        "transpose comparisons on three paths"
                    ),
                    (
                        "one-step authentic-primal centered differences on "
                        "conservative independent-control and broader active-state "
                        "directions"
                    ),
                    (
                        "provisional 31-transition aligned scalar directional "
                        "checks for the warm and neutral cases"
                    ),
                    (
                        "arbitrary 59,148-coordinate terminal REAL-state linear-"
                        "functional reverse seed API, including an exact all-zero "
                        "smoke test when listed in evidence"
                    ),
                ],
                "not_supported": [
                    "global differentiability across branch changes",
                    (
                        "stability of unrecorded internal sign, cap, upwind, or "
                        "threshold branches"
                    ),
                    "a clean multi-level h^2 Taylor regime",
                    "a real64 verification twin",
                    (
                        "a validated derivative for another model configuration or "
                        "initialization mode"
                    ),
                    (
                        "coordinatewise equivalence to standardized 2162-feature "
                        "neural-network AGOP"
                    ),
                    (
                        "independently perturbing every active carried-state "
                        "coordinate as a balanced physical control"
                    ),
                ],
            },
            "raw_gradient_interpretation": {
                "warning": (
                    "The 59,148-coordinate gradient mixes units and includes "
                    "carried atmospheric memory. Its raw squared norm is dominated "
                    "by Q0O for the warm and neutral cases and by UBAR for the cold "
                    "case. Do not plot or interpret the full raw vector as physical "
                    "feature importance."
                ),
                "conservative_independent_control_count": independent_count,
                "balanced_standardized_observation_lift_available": False,
            },
            "forward_gate": {
                "status": "passed",
                "window_cases": 15,
                "segmented_replay_cases": 3,
                "workspace_poison_cases": 6,
                "build_report_original_sha256": sha256_file(build_report_path),
                "replay_report_original_sha256": sha256_file(replay_path),
                "reference_executable_sha256": replay[
                    "reference_executable_sha256"
                ],
                "kernel_executable_sha256": replay["kernel_executable_sha256"],
            },
            "derivative_build": {
                "reverse_executable_sha256": sha256_file(reverse_executable),
                "o3_audit_tangent_executable_sha256": sha256_file(tangent_executable),
                "coupled_build_manifest_original_sha256": sha256_file(build_manifest),
                "compile_input_inventory_file_sha256": {
                    kind: sha256_file(path)
                    for kind, path in compile_inventories.items()
                },
                "prepared_source_manifest_sha256": (
                    "02eb3455bccad4c911ea659a8f192ac100f63ac2b7366ecce6f51f75d905db13"
                ),
                "generated_source_manifest_sha256": (
                    "33ff5b6ac2a693f233d16d59bb0035a1d7efa74733bededda2c03957ae607750"
                ),
                "tapenade_archive_sha256": warning_summary["tapenade"][
                    "archive_sha256"
                ],
                "tapenade_revision": warning_summary["tapenade"]["revision"],
            },
            "transpose_summary": transpose_summary["global_maxima"],
            "finite_difference_summary": {
                "independent_control_one_step_full_output_relative_l2_range": [
                    min(
                        item["full_output_relative_l2_error"]
                        for item in fd_summary[
                            "one_step_centered_finite_difference"
                        ]["independent_control"]["cases"].values()
                    ),
                    max(
                        item["full_output_relative_l2_error"]
                        for item in fd_summary[
                            "one_step_centered_finite_difference"
                        ]["independent_control"]["cases"].values()
                    ),
                ],
                "active_carried_one_step_full_output_relative_l2_range": [
                    min(
                        item["full_output_relative_l2_error"]
                        for item in fd_summary[
                            "one_step_centered_finite_difference"
                        ]["active_carried_state"]["cases"].values()
                    ),
                    max(
                        item["full_output_relative_l2_error"]
                        for item in fd_summary[
                            "one_step_centered_finite_difference"
                        ]["active_carried_state"]["cases"].values()
                    ),
                ],
                "aligned_el_nino_best_stable_relative_defect": aligned_value(
                    fd_summary, "extreme_el_nino"
                ),
                "aligned_neutral_best_stable_relative_defect": aligned_value(
                    fd_summary, "neutral_member_04"
                ),
                "aligned_la_nina": (
                    "no tested two-sided major-tape-stable h; NSST changes at step 15"
                ),
                "composed_random_full_state_classification": (
                    "diagnostic_not_a_pass"
                ),
            },
            "cases": case_index,
            "published_report_provenance": file_records,
            "distribution": {
                "source_free": True,
                "raw_zc_source_published": False,
                "generated_source_published": False,
                "source_bearing_patches_published": False,
                "executables_published": False,
                "checkpoint_or_raw_path_files_published": False,
                "gradient_packages_published": True,
                "reason": (
                    "This evidence bundle remains source-free; project-authored "
                    "ZC-derived patches and coupled drivers are published "
                    "separately as repository source."
                ),
            },
            "sanitization": {
                "method": (
                    "recursive JSON value substitution for local project/home, "
                    "compiler/SDK, and private validation-stage paths; all original "
                    "and published report hashes are recorded"
                ),
                "finder_metadata_included": False,
            },
        }
        write_json(staged / "INDEX.json", index)
        (staged / "README.md").write_text(build_readme(), encoding="utf-8")
        scan_release(staged)
        (staged / "MANIFEST.sha256").write_text(
            tree_manifest(staged), encoding="utf-8"
        )
        scan_release(staged)

        backup = None
        if output.exists():
            backup = output.parent / f".{output.name}-prior-curation"
            if backup.exists() or backup.is_symlink():
                raise FileExistsError(f"Refusing existing backup path: {backup}")
            os.replace(output, backup)
        try:
            os.replace(staged, output)
        except BaseException:
            if backup is not None and not output.exists():
                os.replace(backup, output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return output
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def aligned_value(fd_summary: dict[str, Any], case: str) -> float | None:
    best = fd_summary["aligned_scalar_31_transition"][case][
        "best_major_tape_stable_centered_fd"
    ]
    return None if best is None else float(best["relative_defect"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = curate(args)
    print(f"Curated source-free run23 bundle: {output}")
    print(f"Manifest: {output / 'MANIFEST.sha256'}")
    print("Status: provisional_major_tape_conditioned (not globally certified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
