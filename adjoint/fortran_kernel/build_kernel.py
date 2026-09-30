#!/usr/bin/env python3
"""Build an isolated explicit-state kernel beside the authentic ZC reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

KERNEL_VERSION = "0.1.4"
FORTRAN_FLAGS = "-std=legacy -O3 -I. -ffixed-line-length-none"
BUILD_MARKER = ".zc_kernel_build_output"
BUILD_MARKER_CONTENT = "managed-zc-kernel-build-output-v1\n"
_BuildResult = TypeVar("_BuildResult")
KERNEL_OBJECTS = (
    "zc_kernel_driver.o",
    "zc_kernel_api.o",
    "openfl.o",
    "setup.o",
    "constc.o",
    "mloop.o",
    "cforce.o",
    "ztmfc1.o",
    "akcalc.o",
    "bndary.o",
    "uhcalc.o",
    "uhinit.o",
    "tridag.o",
    "nrdhist.o",
    "initdat.o",
    "setup2.o",
    "ssta.o",
    "plotit.o",
    "close_files.o",
    "FFT2C.o",
    "GGNQF.o",
    "GGUBFS.o",
    "MDNRIS.o",
    "MERFI.o",
    "UERTST.o",
    "UGETIO.o",
    "USPKD.o",
)


class DirectoryIdentity(NamedTuple):
    """Stable identity of one authenticated build directory."""

    device: int
    inode: int


def _directory_identity(path: Path) -> DirectoryIdentity | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"expected a real build directory: {path}")
    status = path.stat()
    return DirectoryIdentity(device=int(status.st_dev), inode=int(status.st_ino))


def _directory_matches(path: Path, expected: DirectoryIdentity | None) -> bool:
    try:
        return _directory_identity(path) == expected
    except RuntimeError:
        return False


def _tree_manifest(root: Path) -> dict[str, dict[str, Any]]:
    """Hash an exact regular-file/directory inventory without following links."""

    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            raise RuntimeError(f"build tree contains a symbolic link: {path}")
        if path.is_dir():
            manifest[relative] = {"type": "directory"}
        elif path.is_file():
            manifest[relative] = {
                "type": "file",
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        else:
            raise RuntimeError(f"build tree contains a special file: {path}")
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new)


def load_generator(project_root: Path):
    scripts = project_root / "scripts"
    sys.path.insert(0, str(scripts))
    import generate_fresh_zc_dataset  # noqa: PLC0415

    return generate_fresh_zc_dataset


def compiler_environment() -> tuple[str, str]:
    compiler = shutil.which("gfortran")
    make = shutil.which("make")
    if compiler is None or make is None:
        raise RuntimeError("gfortran and GNU make are required")
    return str(Path(compiler).resolve()), str(Path(make).resolve())


def command_identity(path: str) -> dict[str, str]:
    """Return the exact executable identity used by this build."""

    executable = Path(path)
    version = subprocess.run(
        [path, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    return {
        "path": str(executable),
        "sha256": sha256_file(executable),
        "version": version,
    }


def guarded_validate_build(
    build_root: Path,
    *,
    overwrite: bool,
    source_input: Path,
    project_root: Path,
) -> Path:
    """Validate a requested build destination without changing the filesystem."""

    raw = Path(os.path.abspath(build_root.expanduser()))
    project_raw = Path(os.path.abspath(project_root.expanduser()))
    project_resolved = project_raw.resolve()
    if project_raw in raw.parents:
        lexical_project = project_raw
    elif project_resolved in raw.parents:
        # A previously validated path is returned in resolved form. Support
        # revalidation on platforms such as macOS where /var aliases
        # /private/var, while still checking every component below the root.
        lexical_project = project_resolved
    else:
        raise ValueError(
            f"build path must be a strict descendant of the project root: "
            f"{project_raw}"
        )
    relative = raw.relative_to(lexical_project)
    candidates: list[Path] = []
    candidate = lexical_project
    for part in relative.parts:
        candidate /= part
        candidates.append(candidate)
    if any(candidate.is_symlink() for candidate in candidates):
        raise ValueError(f"build path may not traverse a symbolic link: {raw}")
    resolved = raw.resolve()
    source = source_input.resolve()
    project = project_root.resolve()
    home = Path.home().resolve()
    dangerous = {Path("/").resolve(), home, project}
    if project not in resolved.parents:
        raise ValueError(f"build path must resolve inside the project root: {project}")
    if resolved in dangerous or resolved in home.parents or resolved in project.parents:
        raise ValueError(f"refusing unsafe build directory: {resolved}")
    if (
        resolved == source
        or resolved in source.parents
        or source in resolved.parents
    ):
        raise ValueError(
            "build directory may not contain, or be inside, the source directory"
        )
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(
            f"build path exists and is not a directory: {resolved}"
        )
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"build directory is nonempty: {resolved}; pass --overwrite "
                "to replace it"
            )
        marker = resolved / BUILD_MARKER
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.read_text(encoding="utf-8") != BUILD_MARKER_CONTENT
        ):
            raise ValueError(
                "refusing to delete a nonempty directory: invalid kernel-build "
                f"marker; {resolved}"
            )
    return resolved


def _owned_or_empty_directory(path: Path) -> bool:
    """Return whether ``path`` is an empty or authenticated managed directory."""

    if path.is_symlink() or not path.is_dir():
        return False
    if not any(path.iterdir()):
        return True
    marker = path / BUILD_MARKER
    return (
        not marker.is_symlink()
        and marker.is_file()
        and marker.read_text(encoding="utf-8") == BUILD_MARKER_CONTENT
    )


def _remove_owned_directory(
    path: Path,
    *,
    expected_identity: DirectoryIdentity | None = None,
    expected_manifest: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Remove a staging/backup directory only after authenticating ownership."""

    if not path.exists():
        return
    if not _owned_or_empty_directory(path):
        raise RuntimeError(
            f"refusing to remove unauthenticated build directory: {path}"
        )
    if expected_identity is not None and _directory_identity(path) != expected_identity:
        raise RuntimeError(f"refusing to remove replaced build directory: {path}")
    if expected_manifest is not None and _tree_manifest(path) != expected_manifest:
        raise RuntimeError(f"refusing to remove changed build directory: {path}")
    shutil.rmtree(path)


def _unique_sibling(parent: Path, *, stem: str, kind: str) -> Path:
    """Reserve and return an authenticated unique sibling directory."""

    path = Path(tempfile.mkdtemp(prefix=f".{stem}.{kind}-", dir=parent))
    (path / BUILD_MARKER).write_text(BUILD_MARKER_CONTENT, encoding="utf-8")
    return path


class BuildRollbackError(RuntimeError):
    """A failed publication retained recovery directories for manual review."""


def _publish_staged_build(
    staging: Path,
    target: Path,
    *,
    expected_target_identity: DirectoryIdentity | None,
    expected_target_manifest: dict[str, dict[str, Any]] | None,
    expected_staging_identity: DirectoryIdentity,
    expected_staging_manifest: dict[str, dict[str, Any]],
) -> None:
    """Publish ``staging`` with authenticated rollback around every rename."""

    if not _owned_or_empty_directory(staging):
        raise RuntimeError(f"staging directory lost its ownership marker: {staging}")
    if _directory_identity(staging) != expected_staging_identity:
        raise RuntimeError(f"staging directory identity changed: {staging}")
    if _tree_manifest(staging) != expected_staging_manifest:
        raise RuntimeError(f"staging directory content changed: {staging}")
    if _directory_identity(target) != expected_target_identity:
        raise RuntimeError(f"build destination changed concurrently: {target}")
    if expected_target_manifest is not None and (
        _tree_manifest(target) != expected_target_manifest
    ):
        raise RuntimeError(f"build destination content changed concurrently: {target}")

    backup: Path | None = None
    if expected_target_identity is not None:
        if not _owned_or_empty_directory(target):
            raise RuntimeError(
                f"build destination changed or lost its ownership marker: {target}"
            )
        if _tree_manifest(target) != expected_target_manifest:
            raise RuntimeError(
                f"build destination changed immediately before backup: {target}"
            )
        backup = _unique_sibling(
            target.parent,
            stem=target.name,
            kind="backup",
        )
        # Reserve an unpredictable same-filesystem name, record it before the
        # move, then remove only the authenticated empty reservation.
        _remove_owned_directory(
            backup,
            expected_identity=_directory_identity(backup),
            expected_manifest=_tree_manifest(backup),
        )

    try:
        if backup is not None:
            os.replace(target, backup)
            if _directory_identity(backup) != expected_target_identity:
                raise RuntimeError(
                    "the directory moved to backup was not the validated target"
                )
            if _tree_manifest(backup) != expected_target_manifest:
                raise RuntimeError(
                    "the directory moved to backup had unvalidated content"
                )
        elif target.exists() or target.is_symlink():
            raise RuntimeError(f"build destination appeared concurrently: {target}")

        if not _owned_or_empty_directory(staging):
            raise RuntimeError("staging ownership changed immediately before publish")
        if _directory_identity(staging) != expected_staging_identity:
            raise RuntimeError("staging identity changed immediately before publish")
        if _tree_manifest(staging) != expected_staging_manifest:
            raise RuntimeError("staging content changed immediately before publish")
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"build destination appeared before publish: {target}")
        os.replace(staging, target)
        if _directory_identity(target) != expected_staging_identity:
            raise RuntimeError("published build has the wrong directory identity")
        if _tree_manifest(target) != expected_staging_manifest:
            raise RuntimeError("published build content differs from staged content")
    except BaseException as publication_error:
        rollback_errors: list[BaseException] = []
        if _directory_matches(target, expected_staging_identity):
            try:
                if staging.exists() or staging.is_symlink():
                    raise RuntimeError(
                        "cannot recover published build over an occupied staging path"
                    )
                os.replace(target, staging)
            except BaseException as error:  # pragma: no cover - catastrophic race
                rollback_errors.append(error)
        elif target.exists() or target.is_symlink():
            original_still_present = bool(
                expected_target_identity is not None
                and _directory_matches(target, expected_target_identity)
                and (backup is None or not backup.exists())
            )
            if not original_still_present:
                rollback_errors.append(
                    RuntimeError(
                        "refusing to move an unauthenticated concurrent build target"
                    )
                )

        if backup is not None and backup.exists():
            backup_identity = _directory_identity(backup)
            if target.exists() or target.is_symlink():
                rollback_errors.append(
                    RuntimeError("cannot restore build backup over an occupied target")
                )
            else:
                try:
                    os.replace(backup, target)
                except BaseException as error:  # pragma: no cover
                    rollback_errors.append(error)
                else:
                    # If another managed target was swapped in between our
                    # authentication and rename, restore the exact directory
                    # we actually moved. It may not be the initial identity,
                    # but it must never be discarded.
                    if not _directory_matches(target, backup_identity):
                        rollback_errors.append(
                            RuntimeError("restored build backup has wrong identity")
                        )
        elif expected_target_identity is not None and (
            not _directory_matches(target, expected_target_identity)
        ):
            rollback_errors.append(RuntimeError("prior build was not restored"))

        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise BuildRollbackError(
                "Build publication failed and rollback was incomplete: "
                f"{details}. Inspect {staging} and {backup}."
            ) from publication_error
        raise
    if backup is not None and backup.exists():
        try:
            _remove_owned_directory(
                backup,
                expected_identity=expected_target_identity,
                expected_manifest=expected_target_manifest,
            )
        except Exception as error:  # pragma: no cover - nonfatal cleanup failure
            warnings.warn(
                f"Published build is valid, but old backup remains at {backup}: "
                f"{error}",
                RuntimeWarning,
                stacklevel=2,
            )


def build_transactionally(
    build_root: Path,
    *,
    overwrite: bool,
    source_input: Path,
    project_root: Path,
    build_action: Callable[[Path], _BuildResult],
) -> tuple[Path, _BuildResult]:
    """Build in a unique sibling and publish only after complete success."""

    target = guarded_validate_build(
        build_root,
        overwrite=overwrite,
        source_input=source_input,
        project_root=project_root,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    initial_target_identity = _directory_identity(target)
    initial_target_manifest = (
        _tree_manifest(target) if initial_target_identity is not None else None
    )
    guarded_validate_build(
        target,
        overwrite=overwrite,
        source_input=source_input,
        project_root=project_root,
    )
    if _directory_identity(target) != initial_target_identity:
        raise RuntimeError(
            f"build destination changed during initial validation: {target}"
        )
    if initial_target_manifest is not None and (
        _tree_manifest(target) != initial_target_manifest
    ):
        raise RuntimeError(
            f"build destination content changed during initial validation: {target}"
        )
    staging = _unique_sibling(
        target.parent,
        stem=target.name,
        kind="staging",
    )
    initial_staging_identity = _directory_identity(staging)
    assert initial_staging_identity is not None
    preserve_staging = False
    try:
        result = build_action(staging)
        # Revalidate immediately before publication. A target created or
        # replaced while compilation ran must satisfy the original contract.
        guarded_validate_build(
            target,
            overwrite=overwrite,
            source_input=source_input,
            project_root=project_root,
        )
        if _directory_identity(target) != initial_target_identity:
            raise RuntimeError(
                f"build destination changed while staging was compiled: {target}"
            )
        if initial_target_manifest is not None and (
            _tree_manifest(target) != initial_target_manifest
        ):
            raise RuntimeError(
                "build destination content changed while staging was compiled: "
                f"{target}"
            )
        staging_identity = _directory_identity(staging)
        if staging_identity is None:
            raise RuntimeError("build action removed its staging directory")
        staging_manifest = _tree_manifest(staging)
        try:
            _publish_staged_build(
                staging,
                target,
                expected_target_identity=initial_target_identity,
                expected_target_manifest=initial_target_manifest,
                expected_staging_identity=staging_identity,
                expected_staging_manifest=staging_manifest,
            )
        except BuildRollbackError:
            preserve_staging = True
            raise
    except BaseException:
        if staging.exists() and not preserve_staging:
            _remove_owned_directory(
                staging,
                expected_identity=initial_staging_identity,
            )
        raise
    return target, result


def link_flags() -> list[str]:
    if platform.system() != "Darwin":
        return []
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        raise RuntimeError("xcrun is required for the macOS SDK path")
    sdk = subprocess.run(
        [xcrun, "--show-sdk-path"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return ["-isysroot", sdk]


def patch_kernel_io_and_tape(source_dir: Path) -> dict[str, str]:
    """Suppress active-kernel output and expose two executed branch counts."""

    ssta_path = source_dir / "ssta.F"
    ssta = ssta_path.read_text()
    ssta = replace_one(
        ssta,
        "      common/fresh_output/ISTEP_fresh\n",
        "      common/fresh_output/ISTEP_fresh\n"
        "      LOGICAL ZC_KERNEL_MODE\n"
        "      COMMON/zc_kernel_mode/ZC_KERNEL_MODE\n"
        "      INTEGER KERNEL_NSST,KERNEL_ATM_ITER,KERNEL_ATM_RESET\n"
        "      COMMON/zc_kernel_tape/KERNEL_NSST,KERNEL_ATM_ITER,\n"
        "     A KERNEL_ATM_RESET\n",
        "SSTA kernel declarations",
    )
    ssta = replace_one(
        ssta,
        "      CALL GETN(NSST)\n",
        "      CALL GETN(NSST)\n      KERNEL_NSST=NSST\n",
        "SSTA substep tape",
    )
    ssta = replace_one(
        ssta,
        "     $     T .le. time_end_writing_grads_data\n"
        "     $     ) then\n",
        "     $     T .le. time_end_writing_grads_data\n"
        "     $     .and. .not.ZC_KERNEL_MODE\n"
        "     $     ) then\n",
        "SSTA compact-output guard",
    )
    ssta_path.write_text(ssta)

    atmosphere_path = source_dir / "ztmfc1.F"
    atmosphere = atmosphere_path.read_text()
    atmosphere = replace_one(
        atmosphere,
        "      COMMON/ATMPAR/ALPHA,EPS,BETA,CTOL,IMIN,IMAX,ISTEP\n"
        "      LOGICAL ISTEP\n\n"
        "      COMMON/commTIME/IT,MP,TY\n",
        "      COMMON/ATMPAR/ALPHA,EPS,BETA,CTOL,IMIN,IMAX,ISTEP\n"
        "      LOGICAL ISTEP\n\n"
        "      COMMON/commTIME/IT,MP,TY\n"
        "      LOGICAL ZC_KERNEL_MODE\n"
        "      COMMON/zc_kernel_mode/ZC_KERNEL_MODE\n"
        "      INTEGER KERNEL_NSST,KERNEL_ATM_ITER,KERNEL_ATM_RESET\n"
        "      COMMON/zc_kernel_tape/KERNEL_NSST,KERNEL_ATM_ITER,\n"
        "     A KERNEL_ATM_RESET\n",
        "ZATMC kernel declarations",
    )
    atmosphere = replace_one(
        atmosphere,
        "      if(ty .lt. .1) then\n"
        "      if (print_timestep_diagnostics) write(6,403) t,nino3\n"
        "      write(63,'(g16.8)') nino3\n",
        "      if(ty .lt. .1 .and. .not.ZC_KERNEL_MODE) then\n"
        "      if (print_timestep_diagnostics) write(6,403) t,nino3\n"
        "      write(63,'(g16.8)') nino3\n",
        "ZATMC diagnostic-output guard",
    )
    atmosphere = replace_one(
        atmosphere,
        " 401  ISTEP=.TRUE.\n",
        " 401  ISTEP=.TRUE.\n      KERNEL_ATM_RESET=0\n",
        "ZATMC reset tape initialization",
    )
    atmosphere = replace_one(
        atmosphere,
        "      IF (ABS(NINO3) .LE. 0.1) then\n"
        "         ISTEP=.FALSE.\n"
        "         time_of_last_initialization=t\n",
        "      IF (ABS(NINO3) .LE. 0.1) then\n"
        "         ISTEP=.FALSE.\n"
        "         KERNEL_ATM_RESET=1\n"
        "         time_of_last_initialization=t\n",
        "ZATMC threshold reset tape",
    )
    atmosphere = replace_one(
        atmosphere,
        "         write(6,'(\" ztmfc1.f: Eli: initializing ISTEP after 4 years.\"\n"
        "     $        ,\"; t=\",f12.3)') t\n"
        "         ISTEP=.FALSE.\n"
        "         time_of_last_initialization=t\n",
        "         if (.not.ZC_KERNEL_MODE)\n"
        "     $     write(6,'(\" ztmfc1.f: Eli: initializing ISTEP after 4 years.\"\n"
        "     $        ,\"; t=\",f12.3)') t\n"
        "         ISTEP=.FALSE.\n"
        "         KERNEL_ATM_RESET=1\n"
        "         time_of_last_initialization=t\n",
        "ZATMC forced reset tape",
    )
    atmosphere = replace_one(
        atmosphere,
        "      ICWR=IC+1\n",
        "      ICWR=IC+1\n      KERNEL_ATM_ITER=ICWR\n",
        "ZATMC iteration tape",
    )
    atmosphere_path.write_text(atmosphere)
    return {
        "ssta.F": sha256_file(ssta_path),
        "ztmfc1.F": sha256_file(atmosphere_path),
    }


def run(command: list[str], cwd: Path, log: list[str]) -> None:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    log.append("$ " + " ".join(command))
    log.append(completed.stdout)
    log.append(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {command!r}")


def verify_staged_source_manifest(
    source_dir: Path,
    *,
    expected_manifest: dict[str, str],
    expected_sha256: str,
    generator: Any,
) -> None:
    """Reject a source snapshot that changed after the preflight read."""

    observed_manifest = generator.source_manifest(source_dir)
    observed_sha256 = generator.manifest_sha256(observed_manifest)
    if observed_manifest != expected_manifest or observed_sha256 != expected_sha256:
        raise RuntimeError(
            "staged upstream source differs from the preflight source manifest"
        )


def copy_verified_support_source(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """Copy one authored support file and verify the staged bytes."""

    shutil.copy2(source, destination)
    observed_sha256 = sha256_file(destination)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"staged kernel support source changed during copy: {source.name}"
        )


def build_kernel_tree(
    build_root: Path,
    *,
    source_input: Path,
    source_manifest: dict[str, str],
    source_manifest_sha256: str,
    compiler: str,
    make: str,
    compiler_identity: dict[str, str],
    make_identity: dict[str, str],
    flags: list[str],
    linker_flags: list[str],
    support_sources: dict[str, str],
    builder_sources: dict[str, str],
    here: Path,
    generator: Any,
) -> dict[str, object]:
    """Construct one complete kernel build inside an owned staging directory."""

    source_dir = build_root / "source"
    shutil.copytree(
        source_input,
        source_dir,
        ignore=shutil.ignore_patterns("*.o", "zeqfc1", ".DS_Store"),
    )
    verify_staged_source_manifest(
        source_dir,
        expected_manifest=source_manifest,
        expected_sha256=source_manifest_sha256,
        generator=generator,
    )
    source_manifest_path = build_root / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_manifest_digest_path = build_root / "source_manifest.sha256"
    source_manifest_digest_path.write_text(
        source_manifest_sha256 + "\n", encoding="utf-8"
    )

    fresh_patches = generator.patch_source(source_dir)
    log: list[str] = []
    started = time.monotonic()

    # First compile the unmodified-equations fresh source as the oracle.
    make_command = [
        make,
        "-j4",
        f"FORTRAN={compiler}",
        f"FLAGS={FORTRAN_FLAGS}",
        f"LINKFLAGS={' '.join(linker_flags)}",
    ]
    run(make_command, source_dir, log)
    reference = build_root / "zc_reference"
    shutil.copy2(source_dir / "zeqfc1", reference)

    kernel_patches = patch_kernel_io_and_tape(source_dir)
    for name in ("zc_kernel_state.inc", "zc_kernel_api.F", "zc_kernel_driver.F"):
        copy_verified_support_source(
            here / name,
            source_dir / name,
            expected_sha256=support_sources[name],
        )

    # Only these production objects differ in the kernel build.
    for name in ("ssta.F", "ztmfc1.F", "zc_kernel_api.F", "zc_kernel_driver.F"):
        run([compiler, *flags, "-c", name], source_dir, log)
    run(
        [
            compiler,
            "-o",
            str(build_root / "zc_kernel_replay"),
            *linker_flags,
            *KERNEL_OBJECTS,
        ],
        source_dir,
        log,
    )

    expected_final_source = dict(source_manifest)
    expected_final_source.update(fresh_patches)
    expected_final_source.update(kernel_patches)
    expected_final_source.update(support_sources)
    observed_final_source = generator.source_manifest(source_dir)
    if observed_final_source != expected_final_source:
        raise RuntimeError(
            "compiled staged source no longer matches its bound source/patch inputs"
        )
    if command_identity(compiler) != compiler_identity:
        raise RuntimeError("Fortran compiler identity changed during the build")
    if command_identity(make) != make_identity:
        raise RuntimeError("make identity changed during the build")
    observed_support_sources = {
        name: sha256_file(here / name) for name in support_sources
    }
    if observed_support_sources != support_sources:
        raise RuntimeError("kernel support source changed during the build")
    observed_builder_sources = {
        "build_kernel.py": sha256_file(Path(__file__).resolve()),
        "generate_fresh_zc_dataset.py": sha256_file(
            Path(generator.__file__).resolve()
        ),
    }
    if observed_builder_sources != builder_sources:
        raise RuntimeError("builder or fresh-data generator changed during the build")

    (build_root / "build.log").write_text("\n".join(log))
    report: dict[str, object] = {
        "schema_version": 2,
        "kernel_version": KERNEL_VERSION,
        "source_dir": str(source_input),
        "source_manifest_sha256": source_manifest_sha256,
        "fresh_patch_sha256": fresh_patches,
        "kernel_patch_sha256": kernel_patches,
        "compiler": compiler_identity,
        "make": make_identity,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "flags": FORTRAN_FLAGS,
        "linker_flags": linker_flags,
        "kernel_support_source_sha256": support_sources,
        "builder_source_sha256": builder_sources,
        "generator": {
            "script_version": generator.SCRIPT_VERSION,
            "generation_schema_version": generator.GENERATION_SCHEMA_VERSION,
        },
        "source_manifest_file_sha256": sha256_file(source_manifest_path),
        "source_manifest_digest_file_sha256": sha256_file(
            source_manifest_digest_path
        ),
        "reference_executable_sha256": sha256_file(reference),
        "kernel_executable_sha256": sha256_file(build_root / "zc_kernel_replay"),
        "state_layout": {
            "real": 59148,
            "complex": 5280,
            "double": 1,
            "integer": 4,
            "passive_time": 2,
            "branch_tape_width": 3,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (build_root / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    source_input = args.source_dir.resolve()
    build_requested = args.build_dir

    # Resolve every tool and inspect the source manifest before the only
    # potentially destructive operation.  A bad input must never erase a
    # previously successful managed build.
    if not source_input.is_dir():
        raise FileNotFoundError(f"source directory is missing: {source_input}")
    generator = load_generator(project_root)
    source_manifest = generator.source_manifest(source_input)
    source_manifest_sha256 = generator.manifest_sha256(source_manifest)
    compiler, make = compiler_environment()
    compiler_identity = command_identity(compiler)
    make_identity = command_identity(make)
    flags = FORTRAN_FLAGS.split()
    linker_flags = link_flags()
    support_sources = {
        name: sha256_file(here / name)
        for name in ("zc_kernel_state.inc", "zc_kernel_api.F", "zc_kernel_driver.F")
    }
    builder_sources = {
        "build_kernel.py": sha256_file(Path(__file__).resolve()),
        "generate_fresh_zc_dataset.py": sha256_file(
            project_root / "scripts" / "generate_fresh_zc_dataset.py"
        ),
    }
    for name in ("zc_kernel_state.inc", "zc_kernel_api.F", "zc_kernel_driver.F"):
        required = here / name
        if not required.is_file():
            raise FileNotFoundError(f"kernel support source is missing: {required}")

    _, report = build_transactionally(
        build_requested,
        overwrite=args.overwrite,
        source_input=source_input,
        project_root=project_root,
        build_action=lambda staging: build_kernel_tree(
            staging,
            source_input=source_input,
            source_manifest=source_manifest,
            source_manifest_sha256=source_manifest_sha256,
            compiler=compiler,
            make=make,
            compiler_identity=compiler_identity,
            make_identity=make_identity,
            flags=flags,
            linker_flags=linker_flags,
            support_sources=support_sources,
            builder_sources=builder_sources,
            here=here,
            generator=generator,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
