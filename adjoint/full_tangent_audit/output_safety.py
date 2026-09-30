"""Managed-directory guard shared by the full-tangent audit programs."""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import NamedTuple

OUTPUT_MARKER = ".zc_full_tangent_audit_output"


class DirectoryIdentity(NamedTuple):
    device: int
    inode: int


class OutputRollbackError(RuntimeError):
    """Raised when an interrupted publish could not be rolled back safely."""


def marker_content(owner: str) -> str:
    return f"managed-zc-full-tangent-audit-output-v1:{owner}\n"


def _identity(path: Path) -> DirectoryIdentity | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"managed output is not a regular directory: {path}")
    stat = path.stat(follow_symlinks=False)
    return DirectoryIdentity(stat.st_dev, stat.st_ino)


def _tree_manifest(path: Path) -> dict[str, str]:
    import hashlib

    result: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"managed output contains a symbolic link: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"managed output contains a special file: {item}")
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[item.relative_to(path).as_posix()] = digest.hexdigest()
    return result


def _validate_target(
    output: Path,
    *,
    overwrite: bool,
    protected_paths: tuple[Path, ...],
    project_root: Path,
    owner: str,
    marker_name: str,
    marker_value: str,
) -> tuple[Path, DirectoryIdentity | None, dict[str, str] | None]:
    """Authenticate a prospective target without mutating it."""

    raw = Path(os.path.abspath(output.expanduser()))
    project_raw = Path(os.path.abspath(project_root.expanduser()))
    if project_raw not in raw.parents:
        raise ValueError(
            "output path must be a strict descendant of the project root: "
            f"{project_raw}"
        )
    candidate = project_raw
    for part in raw.relative_to(project_raw).parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"output path may not traverse a symbolic link: {raw}")
    resolved = raw.resolve()
    protected = tuple(item.resolve() for item in protected_paths)
    project = project_root.resolve()
    home = Path.home().resolve()
    dangerous = {Path("/").resolve(), home, project}
    if project not in resolved.parents:
        raise ValueError(f"output path must resolve inside the project root: {project}")
    if resolved in dangerous or resolved in home.parents or resolved in project.parents:
        raise ValueError(f"refusing unsafe output directory: {resolved}")
    if any(
        resolved == item
        or resolved in item.parents
        or (item.is_dir() and item in resolved.parents)
        for item in protected
    ):
        raise ValueError(
            "output directory may not contain, or be inside, a protected input"
        )
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output path exists and is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is nonempty: {resolved}; pass --overwrite "
                "to replace it"
            )
        marker = resolved / marker_name
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.read_text(encoding="utf-8") != marker_value
        ):
            raise ValueError(
                "refusing to replace a nonempty directory: invalid audit-output "
                f"marker for {owner}; {resolved}"
            )
    identity = _identity(resolved)
    manifest = _tree_manifest(resolved) if identity is not None else None
    return resolved, identity, manifest


def _matches(
    path: Path,
    identity: DirectoryIdentity | None,
    manifest: dict[str, str] | None,
) -> bool:
    try:
        return _identity(path) == identity and (
            identity is None or _tree_manifest(path) == manifest
        )
    except (OSError, ValueError):
        return False


def _remove_owned_staging(
    staging: Path,
    identity: DirectoryIdentity,
    marker_name: str,
    marker_value: str,
) -> None:
    if _identity(staging) != identity:
        raise OutputRollbackError(
            f"refusing to remove replaced staging directory: {staging}"
        )
    marker = staging / marker_name
    if (
        marker.is_symlink()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8") != marker_value
    ):
        raise OutputRollbackError(
            f"refusing to remove unauthenticated staging directory: {staging}"
        )
    shutil.rmtree(staging)


def _private_transaction_workspace(target: Path) -> Path:
    """Create a private same-filesystem workspace for staging and rollback."""

    prefix = f".{target.name}.transaction-"
    temporary_root = Path(tempfile.gettempdir())
    candidate: Path | None = None
    try:
        candidate = Path(tempfile.mkdtemp(prefix=prefix, dir=temporary_root))
        candidate_stat = candidate.stat(follow_symlinks=False)
        target_parent_stat = target.parent.stat(follow_symlinks=False)
        same_drive = (
            not candidate.drive
            or not target.parent.drive
            or candidate.drive.casefold() == target.parent.drive.casefold()
        )
        if candidate_stat.st_dev == target_parent_stat.st_dev and same_drive:
            return candidate
        candidate.rmdir()
        candidate = None
    except OSError:
        if candidate is not None:
            with suppress(OSError):
                candidate.rmdir()

    # A sibling is the only generally portable location when the system
    # temporary directory lives on another filesystem. It remains mode 0700
    # and the same manifest/identity gates apply.
    return Path(tempfile.mkdtemp(prefix=prefix, dir=target.parent))


def _remove_empty_workspace(
    workspace: Path, expected_identity: DirectoryIdentity
) -> None:
    """Remove only the still-owned, empty transaction workspace."""

    if _identity(workspace) != expected_identity:
        raise OutputRollbackError(
            f"refusing to remove replaced transaction workspace: {workspace}"
        )
    if any(workspace.iterdir()):
        raise OutputRollbackError(
            f"refusing to remove nonempty transaction workspace: {workspace}"
        )
    workspace.rmdir()


@contextmanager
def transactional_output(
    output: Path,
    *,
    overwrite: bool,
    protected_paths: tuple[Path, ...],
    project_root: Path,
    owner: str,
    marker_name: str = OUTPUT_MARKER,
    marker_value: str | None = None,
) -> Iterator[Path]:
    """Build in private staging and atomically publish with rollback."""

    expected_marker = marker_value or marker_content(owner)
    target, old_identity, old_manifest = _validate_target(
        output,
        overwrite=overwrite,
        protected_paths=protected_paths,
        project_root=project_root,
        owner=owner,
        marker_name=marker_name,
        marker_value=expected_marker,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace = _private_transaction_workspace(target)
    workspace_identity = _identity(workspace)
    assert workspace_identity is not None
    staging = workspace / f".{target.name}.staging-new"
    staging.mkdir(mode=0o700)
    (staging / marker_name).write_text(expected_marker, encoding="utf-8")
    initial_staging_identity = _identity(staging)
    assert initial_staging_identity is not None
    backup = workspace / f".{target.name}.backup-old"
    try:
        yield staging
    except BaseException:
        _remove_owned_staging(
            staging,
            initial_staging_identity,
            marker_name,
            expected_marker,
        )
        _remove_empty_workspace(workspace, workspace_identity)
        raise

    # Reauthenticate both the lexical path and original directory immediately
    # before committing.  A content or directory swap during the calculation
    # is a conflict, never permission to overwrite the new object.
    target_now, _, _ = _validate_target(
        output,
        overwrite=overwrite,
        protected_paths=protected_paths,
        project_root=project_root,
        owner=owner,
        marker_name=marker_name,
        marker_value=expected_marker,
    )
    if target_now != target or not _matches(target, old_identity, old_manifest):
        _remove_owned_staging(
            staging,
            initial_staging_identity,
            marker_name,
            expected_marker,
        )
        _remove_empty_workspace(workspace, workspace_identity)
        raise RuntimeError("managed output changed while validation was running")
    staging_identity = _identity(staging)
    staging_manifest = _tree_manifest(staging)
    target_move_attempted = False
    stage_move_attempted = False
    try:
        if old_identity is not None:
            target_move_attempted = True
            os.replace(target, backup)
            if not _matches(backup, old_identity, old_manifest):
                raise RuntimeError("output changed during the authenticated swap")
        if not _matches(staging, staging_identity, staging_manifest):
            raise RuntimeError("staged output changed before publication")
        stage_move_attempted = True
        os.replace(staging, target)
        if not _matches(target, staging_identity, staging_manifest):
            raise RuntimeError("published output differs from staged result")
    except BaseException as error:
        problems: list[str] = []
        try:
            if stage_move_attempted and _matches(
                target, staging_identity, staging_manifest
            ):
                os.replace(target, staging)
        except BaseException as rollback_error:  # pragma: no cover - rare OS fault
            problems.append(f"could not quarantine new output: {rollback_error}")
        try:
            if target_move_attempted and backup.exists():
                if target.exists():
                    problems.append("target occupied during rollback")
                elif not _matches(backup, old_identity, old_manifest):
                    problems.append("backup failed identity/content authentication")
                else:
                    os.replace(backup, target)
        except BaseException as rollback_error:  # pragma: no cover - rare OS fault
            problems.append(f"could not restore prior output: {rollback_error}")
        if problems:
            raise OutputRollbackError("; ".join(problems)) from error
        if staging.exists() and _matches(staging, staging_identity, staging_manifest):
            _remove_owned_staging(
                staging,
                initial_staging_identity,
                marker_name,
                expected_marker,
            )
        if workspace.exists() and not any(workspace.iterdir()):
            _remove_empty_workspace(workspace, workspace_identity)
        raise
    if backup.exists():
        try:
            if not _matches(backup, old_identity, old_manifest):
                raise OutputRollbackError(
                    "published output is valid, but the old-output backup "
                    f"changed before cleanup: {backup}"
                )
            assert old_identity is not None
            _remove_owned_staging(
                backup,
                old_identity,
                marker_name,
                expected_marker,
            )
        except (OSError, ValueError, OutputRollbackError) as error:
            # Publication is already complete. Never turn a cleanup race into
            # permission to delete an unauthenticated replacement.
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)
    if workspace.exists() and not any(workspace.iterdir()):
        try:
            _remove_empty_workspace(workspace, workspace_identity)
        except (OSError, ValueError, OutputRollbackError) as error:
            warnings.warn(str(error), RuntimeWarning, stacklevel=2)


def guarded_prepare_output(
    output: Path,
    *,
    overwrite: bool,
    protected_paths: tuple[Path, ...],
    project_root: Path,
    owner: str,
) -> Path:
    """Create an owned output directory, refusing unsafe recursive deletion."""

    raw = Path(os.path.abspath(output.expanduser()))
    project_raw = Path(os.path.abspath(project_root.expanduser()))
    if project_raw not in raw.parents:
        raise ValueError(
            f"output path must be a strict descendant of the project root: "
            f"{project_raw}"
        )
    relative = raw.relative_to(project_raw)
    candidates: list[Path] = []
    candidate = project_raw
    for part in relative.parts:
        candidate /= part
        candidates.append(candidate)
    if any(candidate.is_symlink() for candidate in candidates):
        raise ValueError(f"output path may not traverse a symbolic link: {raw}")
    resolved = raw.resolve()
    protected = tuple(item.resolve() for item in protected_paths)
    project = project_root.resolve()
    home = Path.home().resolve()
    dangerous = {Path("/").resolve(), home, project}
    if project not in resolved.parents:
        raise ValueError(f"output path must resolve inside the project root: {project}")
    if resolved in dangerous or resolved in home.parents or resolved in project.parents:
        raise ValueError(f"refusing unsafe output directory: {resolved}")
    if any(
        resolved == item
        or resolved in item.parents
        or (item.is_dir() and item in resolved.parents)
        for item in protected
    ):
        raise ValueError(
            "output directory may not contain, or be inside, a protected input"
        )
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"output path exists and is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is nonempty: {resolved}; pass --overwrite "
                "to replace it"
            )
        marker = resolved / OUTPUT_MARKER
        expected = marker_content(owner)
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.read_text(encoding="utf-8") != expected
        ):
            raise ValueError(
                "refusing to delete a nonempty directory: invalid audit-output "
                f"marker for {owner}; {resolved}"
            )
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / OUTPUT_MARKER).write_text(marker_content(owner), encoding="utf-8")
    return resolved
