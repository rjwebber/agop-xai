#!/usr/bin/env python3
"""Validate the combined GitHub/Zenodo release tree without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    ".gitattributes",
    ".github/workflows/ci.yml",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-DATA",
    "README.md",
    "RELEASE_CONTENTS.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
}

FIGURE_STEMS = (
    "figure3_noaa_ersstv5",
    "figure4_cnn_forecasts",
    "figure5_predictive_skill",
    "figure6_fresh_core4",
    "figure7_zcv3_core4",
    "figure8_el_nino_multilead",
    "figure9_la_nina_multilead",
    "figure10_extreme_agop_dose_response",
    "figure11_agop_warm_cold_paired_trajectories",
    "figure12_xai_method_mean_trajectories",
)

OUTPUT_FILES = {
    "README.md",
    "SHA256SUMS",
    *(f"manuscript/figures/{stem}.pdf" for stem in FIGURE_STEMS),
    *(f"manuscript/figures/{stem}.json" for stem in FIGURE_STEMS),
    "manuscript/figures/figure4_cnn_forecasts.csv",
    "manuscript/figures/figure7_zcv3_core4_scores.csv",
    "manuscript/tables/table1_xai_scores.json",
    "manuscript/tables/table1_xai_scores.tex",
    "manuscript/source_data/figure5/figure5_zcv3_results.csv",
    "manuscript/source_data/figure5/figure5_zcv3_results.json",
    "manuscript/source_data/diagnostics/agop_field_mass_audit.csv",
    "manuscript/source_data/diagnostics/agop_field_mass_audit.json",
    "manuscript/benchmarks/xai_cpu4/agop_report.json",
    "manuscript/benchmarks/xai_cpu4/benchmark.log",
    "manuscript/benchmarks/xai_cpu4/method_report.json",
    "video/zc_standardized_ocean_currents.json",
    "video/zc_standardized_ocean_currents.mp4",
}

STEERING_FILES = {
    "README.md",
    "SHA256SUMS",
    "accepted_reports.zip",
    "manifest.json",
    "steering_plot_data.npz",
}

COVARIANCE_FILES = {
    "README.md",
    "dense_manifest.json",
    "manifest.json",
    "phase_moments.npz",
}

CHECKSUM_ROOTS = (
    "data/processed/zc-v3",
    "data/external/noaa",
    "artifacts/zc-v3",
    "outputs",
)

GITHUB_TEXT_ROOTS = (
    ".gitattributes",
    ".github",
    "README.md",
    "CHANGELOG.md",
    "RELEASE_CONTENTS.md",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-DATA",
    "pyproject.toml",
    ".gitignore",
    "src",
    "scripts",
    "tests",
    "docs",
    "adjoint",
    "examples",
)

PUBLIC_DATA_ROOTS = ("data", "artifacts", "outputs")

TEXT_SUFFIXES = {
    "",
    ".c",
    ".cff",
    ".csv",
    ".f",
    ".f90",
    ".json",
    ".log",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

GENERATED_DIR_NAMES = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "scratch",
    "tmp",
}

IGNORED_LOCAL_DIRS = {".git", ".tox", ".venv", "venv"}
IGNORED_LOCAL_FILES = {".DS_Store"}

# Construct this pattern in pieces so the verifier does not indict its own source.
LEGACY_NUMBERED_TABLE = re.compile(
    r"(?i)(?<![a-z])" + "table" + r"(?:[ _~-]*2)(?![0-9])"
)
NONPORTABLE = (
    (re.compile(r"/Users/"), "machine-local /Users path"),
    (re.compile(r"(?<![A-Za-z0-9_.-])scratch/"), "scratch path"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_files(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name not in IGNORED_LOCAL_FILES
    }


def compare_inventory(
    directory: Path, expected: set[str], label: str, errors: list[str]
) -> None:
    if not directory.is_dir():
        errors.append(f"missing {label} directory: {directory}")
        return
    actual = relative_files(directory)
    for name in sorted(expected - actual):
        errors.append(f"missing {label} file: {directory / name}")
    for name in sorted(actual - expected):
        errors.append(f"unexpected {label} file: {directory / name}")


def parse_checksum_manifest(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    if not path.is_file():
        return entries, [f"missing checksum manifest: {path}"]
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+(.+)", raw_line)
        if match is None:
            errors.append(f"invalid checksum line {path}:{line_number}")
            continue
        digest, raw_name = match.groups()
        if raw_name.startswith("*"):
            raw_name = raw_name[1:]
        if raw_name.startswith("./"):
            raw_name = raw_name[2:]
        pure = PurePosixPath(raw_name)
        if pure.is_absolute() or ".." in pure.parts or raw_name in {"", "."}:
            errors.append(f"unsafe checksum path {raw_name!r} in {path}")
            continue
        name = pure.as_posix()
        if name in entries:
            errors.append(f"duplicate checksum path {name!r} in {path}")
            continue
        entries[name] = digest.lower()
    return entries, errors


def verify_checksum_tree(directory: Path, errors: list[str]) -> None:
    manifest = directory / "SHA256SUMS"
    entries, parse_errors = parse_checksum_manifest(manifest)
    errors.extend(parse_errors)
    if parse_errors:
        return
    actual = relative_files(directory) - {"SHA256SUMS"}
    listed = set(entries)
    for name in sorted(listed - actual):
        errors.append(f"checksum entry has no file: {directory / name}")
    # A parent manifest may omit a nested manifest because the nested tree is
    # checked independently below. Including and hashing it is also valid.
    unlisted_payloads = {
        name for name in actual - listed if PurePosixPath(name).name != "SHA256SUMS"
    }
    for name in sorted(unlisted_payloads):
        errors.append(f"file is missing from checksum manifest: {directory / name}")
    for name in sorted(actual & listed):
        observed = sha256_file(directory / name)
        if observed != entries[name]:
            errors.append(
                f"checksum mismatch: {directory / name} "
                f"(expected {entries[name]}, observed {observed})"
            )


def iter_tree(root: Path):
    """Yield paths while pruning local environments and version-control state."""
    for directory, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(
            name for name in dir_names if name not in IGNORED_LOCAL_DIRS
        )
        base = Path(directory)
        for name in file_names:
            if name not in IGNORED_LOCAL_FILES:
                yield base / name


def text_files(path: Path):
    if path.is_file():
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path
        return
    if not path.is_dir():
        return
    for candidate in iter_tree(path):
        if candidate.suffix.lower() in TEXT_SUFFIXES:
            yield candidate


def read_text(path: Path, errors: list[str]) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"cannot read expected text file {path}: {exc}")
        return None


def check_portable_text(path: Path, text: str, errors: list[str]) -> None:
    for pattern, description in NONPORTABLE:
        if pattern.search(text):
            errors.append(f"{description} in public file: {path}")


def verify_accepted_reports(bundle: Path, errors: list[str]) -> None:
    archive = bundle / "accepted_reports.zip"
    manifest_path = bundle / "manifest.json"
    if not archive.is_file() or not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_count = int(manifest["accepted_reports"]["member_count"])
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ) as exc:
        errors.append(f"invalid compact steering manifest {manifest_path}: {exc}")
        return
    try:
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            if len(members) != expected_count:
                errors.append(
                    f"accepted-report count mismatch in {archive}: "
                    f"expected {expected_count}, found {len(members)}"
                )
            seen: set[str] = set()
            for member in members:
                name = member.filename
                pure = PurePosixPath(name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or name in seen
                    or not name.endswith(".json")
                    or member.is_dir()
                ):
                    errors.append(f"invalid accepted-report member: {archive}!{name}")
                    continue
                seen.add(name)
                if LEGACY_NUMBERED_TABLE.search(name):
                    errors.append(f"legacy numbered-table name: {archive}!{name}")
                try:
                    text = handle.read(member).decode("utf-8")
                    json.loads(text)
                except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
                    errors.append(
                        f"invalid accepted-report JSON {archive}!{name}: {exc}"
                    )
                    continue
                check_portable_text(Path(f"{archive}!{name}"), text, errors)
                if LEGACY_NUMBERED_TABLE.search(text):
                    errors.append(
                        f"legacy numbered-table nomenclature in {archive}!{name}"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"invalid accepted-report archive {archive}: {exc}")


def verify_metadata(root: Path, errors: list[str]) -> None:
    for name in sorted(ROOT_FILES):
        if not (root / name).is_file():
            errors.append(f"missing required root file: {name}")
    license_text = (
        read_text(root / "LICENSE", errors) if (root / "LICENSE").is_file() else ""
    )
    data_license_text = (
        read_text(root / "LICENSE-DATA", errors)
        if (root / "LICENSE-DATA").is_file()
        else ""
    )
    if license_text is not None and "BSD 3-Clause License" not in license_text:
        errors.append("LICENSE is not the BSD 3-Clause license")
    if data_license_text is not None and "CC BY 4.0" not in data_license_text:
        errors.append("LICENSE-DATA does not identify CC BY 4.0")

    version_files = {
        "pyproject.toml": (
            root / "pyproject.toml",
            lambda text: str(tomllib.loads(text)["project"]["version"]),
        ),
        "CITATION.cff": (
            root / "CITATION.cff",
            lambda text: _required_match(
                r'^version:\s*["\']?([^"\'\s]+)', text, "CITATION.cff version"
            ),
        ),
        "src/zc_xai/__init__.py": (
            root / "src/zc_xai/__init__.py",
            lambda text: _required_match(
                r'^__version__\s*=\s*["\']([^"\']+)', text, "package version"
            ),
        ),
        "CHANGELOG.md": (
            root / "CHANGELOG.md",
            lambda text: _required_match(
                r"^##\s+([0-9]+\.[0-9]+\.[0-9]+)\b", text, "changelog version"
            ),
        ),
    }
    observed_versions: dict[str, str] = {}
    for label, (path, parser) in version_files.items():
        if not path.is_file():
            continue
        text = read_text(path, errors)
        if text is None:
            continue
        try:
            observed_versions[label] = parser(text)
        except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"cannot read release version from {label}: {exc}")
    if len(set(observed_versions.values())) > 1:
        detail = ", ".join(
            f"{label}={version}" for label, version in observed_versions.items()
        )
        errors.append(f"release-version mismatch: {detail}")


def _required_match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"missing {label}")
    return match.group(1)


def verify_generated_directories(root: Path, errors: list[str]) -> None:
    for directory, dir_names, _ in os.walk(root):
        dir_names[:] = sorted(
            name for name in dir_names if name not in IGNORED_LOCAL_DIRS
        )
        for name in dir_names:
            if (
                name in GENERATED_DIR_NAMES
                or name.startswith("build-")
                or name.endswith(".egg-info")
            ):
                errors.append(
                    f"generated/local directory remains: {Path(directory) / name}"
                )


def verify_no_private_file(root: Path, errors: list[str]) -> None:
    for path in iter_tree(root):
        if path.name.casefold() == "grads_1.data":
            errors.append(f"private legacy input is present: {path}")


def verify_legacy_nomenclature(root: Path, errors: list[str]) -> None:
    paths: list[Path] = []
    for name in GITHUB_TEXT_ROOTS:
        candidate = root / name
        if candidate.exists():
            paths.extend(text_files(candidate))
    for name in ("artifacts", "outputs"):
        candidate = root / name
        if candidate.exists():
            paths.extend(text_files(candidate))
    for path in sorted(set(paths)):
        relative = path.relative_to(root).as_posix()
        if LEGACY_NUMBERED_TABLE.search(relative):
            errors.append(f"legacy numbered-table path: {relative}")
        text = read_text(path, errors)
        if text is not None and LEGACY_NUMBERED_TABLE.search(text):
            errors.append(f"legacy numbered-table nomenclature in: {relative}")


def verify_portability(root: Path, errors: list[str]) -> None:
    for name in PUBLIC_DATA_ROOTS:
        public_root = root / name
        if not public_root.exists():
            continue
        for path in text_files(public_root):
            text = read_text(path, errors)
            if text is not None:
                check_portable_text(path, text, errors)


def verify_release(root: Path) -> list[str]:
    errors: list[str] = []
    verify_metadata(root, errors)
    compare_inventory(root / "data/raw", {"README.md"}, "raw-data", errors)
    compare_inventory(root / "outputs", OUTPUT_FILES, "output", errors)

    bundle = root / "artifacts/zc-v3/manuscript/steering/final_plot_data"
    compare_inventory(bundle, STEERING_FILES, "compact steering", errors)
    covariance = (
        root
        / "artifacts/zc-v3/manuscript/covariance"
        / "training-years10000-phases00-35"
    )
    compare_inventory(covariance, COVARIANCE_FILES, "all-phase covariance", errors)

    verify_generated_directories(root, errors)
    verify_no_private_file(root, errors)
    verify_portability(root, errors)
    verify_legacy_nomenclature(root, errors)
    verify_accepted_reports(bundle, errors)

    # The compact bundle has its own independently verifiable checksum manifest.
    verify_checksum_tree(bundle, errors)
    for relative in CHECKSUM_ROOTS:
        directory = root / relative
        if directory.is_dir():
            verify_checksum_tree(directory, errors)
        else:
            errors.append(f"missing checksum root: {directory}")
    return sorted(set(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation of the staged GitHub/Zenodo release."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of this script directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    errors = verify_release(root)
    if errors:
        print(f"Release verification FAILED ({len(errors)} issue(s)):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Release verification passed.")
    print("Validated GitHub metadata/source boundaries and the Zenodo data tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
