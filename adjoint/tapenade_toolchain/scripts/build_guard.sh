#!/usr/bin/env bash
# Shared ownership and path-safety checks for Tapenade build stages.

zc_resolve_path() {
  python3 -c \
    'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1"
}

zc_assert_lexical_strict_child_no_symlinks() {
  local base="$1"
  local requested="$2"
  local label="${3:-path}"
  python3 - "${base}" "${requested}" "${label}" <<'PY'
import os
import pathlib
import sys

base = pathlib.Path(os.path.abspath(sys.argv[1]))
raw_requested = pathlib.Path(sys.argv[2])
if ".." in raw_requested.parts:
    raise SystemExit(
        f"ERROR: {sys.argv[3]} may not contain parent traversal: {raw_requested}"
    )
requested = pathlib.Path(os.path.abspath(raw_requested))
label = sys.argv[3]
try:
    relative = requested.relative_to(base)
except ValueError as error:
    raise SystemExit(
        f"ERROR: {label} must be a strict child of {base}: {requested}"
    ) from error
if not relative.parts:
    raise SystemExit(f"ERROR: {label} may not equal protected root: {base}")
current = base
for part in relative.parts:
    current = current / part
    if current.is_symlink():
        raise SystemExit(
            f"ERROR: {label} has a symbolic-link component below {base}: {current}"
        )
print(requested)
PY
}

zc_assert_no_symlink_components() {
  local requested="$1"
  local label="${2:-path}"
  python3 - "${requested}" "${label}" <<'PY'
import os
import pathlib
import sys

raw = pathlib.Path(sys.argv[1])
label = sys.argv[2]
if ".." in raw.parts:
    raise SystemExit(f"ERROR: {label} may not contain parent traversal: {raw}")
path = pathlib.Path(os.path.abspath(raw))
current = pathlib.Path(path.anchor)
for part in path.parts[1:]:
    current = current / part
    if current.is_symlink():
        raise SystemExit(
            f"ERROR: {label} has a symbolic-link component: {current}"
        )
print(path)
PY
}

zc_path_identity() {
  python3 - "$1" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
info = os.lstat(path)
if stat.S_ISLNK(info.st_mode):
    raise SystemExit(f"ERROR: path identity target is a symlink: {path}")
print(f"{info.st_dev}:{info.st_ino}:{stat.S_IFMT(info.st_mode)}")
PY
}

zc_capture_path_state() {
  python3 - "$1" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])


def hash_regular_file(candidate: pathlib.Path) -> tuple[os.stat_result, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(f"ERROR: path-state entry is not regular: {candidate}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.lstat(candidate)
    identity_before = (before.st_dev, before.st_ino, before.st_size)
    identity_after = (after.st_dev, after.st_ino, after.st_size)
    identity_current = (current.st_dev, current.st_ino, current.st_size)
    if identity_before != identity_after or identity_after != identity_current:
        raise SystemExit(f"ERROR: path-state entry changed while hashing: {candidate}")
    return current, digest.hexdigest()


try:
    root_info = os.lstat(path)
except FileNotFoundError:
    print("absent")
    raise SystemExit(0)
if stat.S_ISLNK(root_info.st_mode):
    raise SystemExit(f"ERROR: path-state root is a symbolic link: {path}")
if stat.S_ISREG(root_info.st_mode):
    current, digest = hash_regular_file(path)
    print(
        "file:"
        f"{current.st_dev}:{current.st_ino}:{stat.S_IMODE(current.st_mode)}:"
        f"{current.st_size}:{digest}"
    )
    raise SystemExit(0)
if not stat.S_ISDIR(root_info.st_mode):
    raise SystemExit(f"ERROR: path-state root is not a regular file/directory: {path}")

lines = [
    "ROOT "
    f"{root_info.st_dev}:{root_info.st_ino}:{stat.S_IMODE(root_info.st_mode)}"
]
for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
    relative = candidate.relative_to(path).as_posix()
    info = os.lstat(candidate)
    if stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"ERROR: path-state tree contains a symlink: {candidate}")
    if stat.S_ISDIR(info.st_mode):
        lines.append(
            f"D {relative} {info.st_dev}:{info.st_ino}:"
            f"{stat.S_IMODE(info.st_mode)}"
        )
    elif stat.S_ISREG(info.st_mode):
        current, digest = hash_regular_file(candidate)
        lines.append(
            f"F {relative} {current.st_dev}:{current.st_ino}:"
            f"{stat.S_IMODE(current.st_mode)}:{current.st_size}:{digest}"
        )
    else:
        raise SystemExit(f"ERROR: path-state tree contains a special file: {candidate}")
root_after = os.lstat(path)
if (root_after.st_dev, root_after.st_ino) != (root_info.st_dev, root_info.st_ino):
    raise SystemExit(f"ERROR: path-state root changed while hashing: {path}")
payload = "\n".join(lines).encode("utf-8")
print("directory:" + hashlib.sha256(payload).hexdigest())
PY
}

zc_verify_path_state() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(zc_capture_path_state "${path}")" || return
  if [[ "${observed}" != "${expected}" ]]; then
    printf 'ERROR: path changed after preflight: %s\n' "${path}" >&2
    printf 'Expected state %s\nObserved state %s\n' \
      "${expected}" "${observed}" >&2
    return 2
  fi
}

zc_assert_stage_replacement_allowed() {
  local target="$1"
  local transition="${2:-0}"
  if [[ -L "${target}" || ( -e "${target}" && ! -d "${target}" ) ]]; then
    printf 'ERROR: stage target is symlinked or not a directory: %s\n' \
      "${target}" >&2
    return 2
  fi
  if [[ -d "${target}" ]] &&
     find "${target}" -mindepth 1 -maxdepth 1 -print -quit | grep -q . &&
     [[ "${transition}" != 1 && "${ADJOINT_OVERWRITE:-0}" != 1 ]]; then
    printf 'ERROR: stage target is nonempty: %s\n' "${target}" >&2
    printf 'Set ADJOINT_OVERWRITE=1 to replace this exact stage.\n' >&2
    return 2
  fi
}

zc_authenticate_managed_build_marker() {
  local build_dir="$1"
  local marker="${build_dir}/.zc_tapenade_build"
  if [[ -L "${build_dir}" || ! -d "${build_dir}" || \
        -L "${marker}" || ! -f "${marker}" || \
        "$(cat "${marker}")" != managed-zc-tapenade-build-v1 ]]; then
    printf 'ERROR: managed build marker authentication failed: %s\n' \
      "${build_dir}" >&2
    return 2
  fi
}

zc_test_failpoint() {
  local label="$1"
  [[ "${ZC_TOOLCHAIN_TESTING:-0}" == 1 ]] || return 0
  [[ "${ZC_TEST_FAILPOINT:-}" == "${label}" ]] || return 0
  if [[ "${ZC_TEST_FAIL_MODE:-failure}" == signal ]]; then
    # A child shell's PPID is the actual PID even under Bash 3.2, where $$ is
    # inherited by subshells and BASHPID is unavailable.
    sh -c 'kill -TERM "$PPID"'
    sleep 1
  fi
  printf 'ERROR: injected toolchain failure at %s\n' "${label}" >&2
  return 97
}

zc_create_stage_workspace() {
  local build_dir="$1"
  local label="$2"
  local workspace
  case "${label}" in
    *[!A-Za-z0-9_-]*|'')
      printf 'ERROR: invalid staging-workspace label: %s\n' "${label}" >&2
      return 2
      ;;
  esac
  workspace="$(mktemp -d "${build_dir}/.zc-${label}-work.XXXXXX")"
  printf '%s\n' managed-zc-stage-workspace-v1 > \
    "${workspace}/.zc_stage_workspace"
  printf '%s\n' "${workspace}"
}

zc_remove_stage_workspace() {
  local build_dir="$1"
  local workspace="$2"
  local expected_identity="$3"
  local lexical
  [[ -n "${workspace}" ]] || return 0
  lexical="$(zc_assert_lexical_strict_child_no_symlinks \
    "${build_dir}" "${workspace}" 'stage workspace')" || return
  [[ -d "${lexical}" && ! -L "${lexical}" ]] || return 0
  [[ "$(zc_path_identity "${lexical}")" == "${expected_identity}" ]] || {
    printf 'ERROR: refusing to remove replaced staging workspace: %s\n' \
      "${lexical}" >&2
    return 2
  }
  [[ -f "${lexical}/.zc_stage_workspace" && \
     ! -L "${lexical}/.zc_stage_workspace" && \
     "$(cat "${lexical}/.zc_stage_workspace")" == \
       managed-zc-stage-workspace-v1 ]] || {
    printf 'ERROR: refusing to remove unauthenticated staging workspace: %s\n' \
      "${lexical}" >&2
    return 2
  }
  # Immutable input snapshots deliberately remove directory write permission.
  # Authenticate the complete tree before restoring owner write permission for
  # cleanup; never chmod or recurse through a symlinked/special entry.
  zc_capture_path_state "${lexical}" >/dev/null || return
  find "${lexical}" -type d -exec chmod u+w {} +
  rm -rf -- "${lexical}"
}

zc_publish_stage_directory() (
  set -euo pipefail
  # EXIT fires after this subshell function's local scope is unwound on Bash
  # 3.2. Keep transaction state in subshell-global variables so signal/failure
  # cleanup can always recover it.
  build_dir="$1"
  target="$2"
  staged="$3"
  expected_target_state="$4"
  publication_mode="${5:-standard}"
  target_lexical=""
  staged_lexical=""
  new_state=""
  backup=""
  backup_identity=""
  publication_started=0
  publication_committed=0
  restored=1

  authenticate_stage_backup() {
    local require_original="${1:-0}"
    local allow_failed="${2:-1}"
    local entry base_name saw_original=0 saw_failed=0
    [[ -n "${backup}" && -d "${backup}" && ! -L "${backup}" && \
       "$(zc_path_identity "${backup}")" == "${backup_identity}" && \
       -f "${backup}/.zc_stage_backup" && \
       ! -L "${backup}/.zc_stage_backup" && \
       "$(cat "${backup}/.zc_stage_backup")" == \
         managed-zc-stage-backup-v1 ]] || {
      printf 'ERROR: stage backup authentication failed: %s\n' \
        "${backup}" >&2
      return 2
    }
    while IFS= read -r entry; do
      base_name="$(basename "${entry}")"
      case "${base_name}" in
        .zc_stage_backup) ;;
        original)
          [[ "${expected_target_state}" != absent ]] || {
            printf 'ERROR: unexpected original entered stage backup\n' >&2
            return 2
          }
          zc_verify_path_state "${entry}" "${expected_target_state}" || return
          saw_original=1
          ;;
        failed-publication)
          [[ "${allow_failed}" == 1 ]] || {
            printf 'ERROR: failed payload is not allowed in committed backup\n' >&2
            return 2
          }
          zc_verify_path_state "${entry}" "${new_state}" || return
          saw_failed=1
          ;;
        *)
          printf 'ERROR: unrecognized stage-backup payload: %s\n' \
            "${entry}" >&2
          return 2
          ;;
      esac
    done < <(find "${backup}" -mindepth 1 -maxdepth 1 -print)
    if [[ "${require_original}" == 1 && "${saw_original}" != 1 ]]; then
      printf 'ERROR: authenticated stage backup lost its original payload\n' >&2
      return 2
    fi
    if [[ "${allow_failed}" != 1 && "${saw_failed}" == 1 ]]; then
      printf 'ERROR: unexpected failed payload in stage backup\n' >&2
      return 2
    fi
  }

  target_lexical="$(zc_assert_lexical_strict_child_no_symlinks \
    "${build_dir}" "${target}" 'stage publication target')"
  staged_lexical="$(zc_assert_lexical_strict_child_no_symlinks \
    "${build_dir}" "${staged}" 'staged publication source')"
  [[ "$(dirname "${target_lexical}")" == "${build_dir}" ]] || {
    printf 'ERROR: stage publication target must be an immediate build child\n' >&2
    exit 2
  }
  [[ -d "${staged_lexical}" && ! -L "${staged_lexical}" ]] || {
    printf 'ERROR: staged publication source is missing or symlinked\n' >&2
    exit 2
  }
  case "${publication_mode}" in
    standard|managed)
      zc_assert_stage_replacement_allowed "${target_lexical}" 0
      ;;
    transition)
      zc_assert_stage_replacement_allowed "${target_lexical}" 1
      ;;
    *)
      printf 'ERROR: invalid stage publication mode: %s\n' \
        "${publication_mode}" >&2
      exit 2
      ;;
  esac
  zc_verify_path_state "${target_lexical}" "${expected_target_state}"
  if [[ "${publication_mode}" == managed ]]; then
    zc_authenticate_managed_build_marker "${target_lexical}"
    zc_authenticate_managed_build_marker "${staged_lexical}"
  fi
  new_state="$(zc_capture_path_state "${staged_lexical}")"

  backup="$(mktemp -d "${build_dir}/.zc-stage-backup.XXXXXX")"
  printf '%s\n' managed-zc-stage-backup-v1 > "${backup}/.zc_stage_backup"
  backup_identity="$(zc_path_identity "${backup}")"

  cleanup_stage_publication() {
    local status="$?" current_state
    trap - EXIT HUP INT TERM
    if [[ "${publication_started}" -eq 1 && \
          "${publication_committed}" -eq 0 ]]; then
      if [[ -e "${target_lexical}" ]]; then
        current_state="$(zc_capture_path_state \
          "${target_lexical}" 2>/dev/null || true)"
        if [[ "${current_state}" == "${new_state}" ]]; then
          require_original=0
          [[ "${expected_target_state}" == absent ]] || require_original=1
          authenticate_stage_backup "${require_original}" 0 || {
            restored=0
            status=2
          }
          if [[ "${restored}" -ne 1 ]]; then
            printf 'ERROR: preserving unauthenticated stage backup\n' >&2
          elif [[ -e "${backup}/failed-publication" || \
                  -L "${backup}/failed-publication" ]]; then
            printf 'ERROR: stage backup already contains failed payload\n' >&2
            restored=0
            status=2
          else
            mv "${target_lexical}" "${backup}/failed-publication"
            zc_verify_path_state "${backup}/failed-publication" \
              "${new_state}" || {
              restored=0
              status=2
            }
          fi
        elif [[ "${current_state}" != "${expected_target_state}" ]]; then
          printf '%s\n' \
            "ERROR: refusing to displace an unexpected publication target:" \
            "${target_lexical}" >&2
          restored=0
          status=2
        fi
      fi
      if [[ -e "${backup}/original" ]]; then
        authenticate_stage_backup 1 1 || {
          restored=0
          status=2
        }
        zc_verify_path_state "${backup}/original" \
          "${expected_target_state}" || {
          restored=0
          status=2
        }
        if [[ ! -e "${target_lexical}" ]]; then
          if [[ "${restored}" -eq 1 ]]; then
            mv "${backup}/original" "${target_lexical}"
          fi
        elif [[ "$(zc_capture_path_state "${target_lexical}")" != \
                "${expected_target_state}" ]]; then
          restored=0
          status=2
        fi
      elif [[ "${expected_target_state}" == absent && \
              -e "${target_lexical}" ]]; then
        restored=0
        status=2
      fi
    fi
    if [[ -n "${backup}" && "${restored}" -eq 1 ]]; then
      authenticate_stage_backup 0 1 || {
        restored=0
        status=2
      }
    fi
    if [[ "${restored}" -eq 1 && -n "${backup}" && \
          -d "${backup}" ]]; then
      find "${backup}" -type d -exec chmod u+w {} +
      rm -rf -- "${backup}"
    fi
    exit "${status}"
  }
  trap cleanup_stage_publication EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM

  # Reauthenticate at the last possible point. Intent is recorded before each
  # atomic rename, so interruption between mutation and bookkeeping is safe.
  zc_verify_path_state "${target_lexical}" "${expected_target_state}"
  if [[ "${publication_mode}" == managed ]]; then
    zc_authenticate_managed_build_marker "${target_lexical}"
    zc_authenticate_managed_build_marker "${staged_lexical}"
  fi
  publication_started=1
  zc_test_failpoint stage_before_old_move
  zc_verify_path_state "${target_lexical}" "${expected_target_state}"
  if [[ "${publication_mode}" == managed ]]; then
    zc_authenticate_managed_build_marker "${target_lexical}"
  fi
  if [[ "${ZC_TOOLCHAIN_TESTING:-0}" == 1 && \
        -n "${ZC_TEST_STAGE_AFTER_TARGET_RECHECK_HOOK:-}" ]]; then
    [[ -x "${ZC_TEST_STAGE_AFTER_TARGET_RECHECK_HOOK}" && \
       ! -L "${ZC_TEST_STAGE_AFTER_TARGET_RECHECK_HOOK}" ]]
    "${ZC_TEST_STAGE_AFTER_TARGET_RECHECK_HOOK}" "${target_lexical}"
  fi
  if [[ -e "${target_lexical}" ]]; then
    mv "${target_lexical}" "${backup}/original"
    # A replacement can race the immediately preceding path check. Verify the
    # object that was actually moved before the candidate is allowed to occupy
    # the target. On mismatch, put that exact object back and abort.
    if [[ "${expected_target_state}" == absent ]] || \
       ! zc_verify_path_state "${backup}/original" \
         "${expected_target_state}"; then
      if [[ ! -e "${target_lexical}" && ! -L "${target_lexical}" ]]; then
        mv "${backup}/original" "${target_lexical}"
        publication_started=0
      fi
      printf 'ERROR: publication target changed during the backup rename\n' >&2
      exit 2
    fi
    if [[ "${publication_mode}" == managed ]]; then
      zc_authenticate_managed_build_marker "${backup}/original" || {
        if [[ ! -e "${target_lexical}" && ! -L "${target_lexical}" ]]; then
          mv "${backup}/original" "${target_lexical}"
          publication_started=0
        fi
        exit 2
      }
    fi
  fi
  zc_test_failpoint stage_after_old_move
  zc_test_failpoint stage_before_new_move
  zc_verify_path_state "${staged_lexical}" "${new_state}"
  if [[ "${publication_mode}" == managed ]]; then
    zc_authenticate_managed_build_marker "${staged_lexical}"
  fi
  [[ ! -e "${target_lexical}" && ! -L "${target_lexical}" ]] || {
    printf 'ERROR: publication target appeared before staged rename: %s\n' \
      "${target_lexical}" >&2
    exit 2
  }
  mv "${staged_lexical}" "${target_lexical}"
  zc_test_failpoint stage_after_new_move
  zc_verify_path_state "${target_lexical}" "${new_state}"
  if [[ "${ZC_TOOLCHAIN_TESTING:-0}" == 1 && \
        -n "${ZC_TEST_STAGE_BEFORE_COMMIT_HOOK:-}" ]]; then
    [[ -x "${ZC_TEST_STAGE_BEFORE_COMMIT_HOOK}" && \
       ! -L "${ZC_TEST_STAGE_BEFORE_COMMIT_HOOK}" ]]
    "${ZC_TEST_STAGE_BEFORE_COMMIT_HOOK}" "${backup}" \
      "${target_lexical}"
  fi
  zc_test_failpoint stage_before_commit
  zc_verify_path_state "${target_lexical}" "${new_state}"
  if [[ "${publication_mode}" == managed ]]; then
    zc_authenticate_managed_build_marker "${target_lexical}"
  fi
  require_original=0
  [[ "${expected_target_state}" == absent ]] || require_original=1
  authenticate_stage_backup "${require_original}" 0
  zc_capture_path_state "${backup}" >/dev/null
  publication_committed=1
  find "${backup}" -type d -exec chmod u+w {} +
  rm -rf -- "${backup}"
  backup=""
  trap - EXIT HUP INT TERM
)

zc_sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    python3 - "$1" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
  fi
}

zc_verify_single_link_regular_sha256() {
  local path="$1"
  local expected="$2"
  python3 - "${path}" "${expected}" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
try:
    initial = os.lstat(path)
except FileNotFoundError as error:
    raise SystemExit(f"ERROR: required archive is missing: {path}") from error
if not stat.S_ISREG(initial.st_mode) or stat.S_ISLNK(initial.st_mode):
    raise SystemExit(f"ERROR: archive must be a non-symlink regular file: {path}")
if initial.st_nlink != 1:
    raise SystemExit(f"ERROR: archive must have exactly one hard link: {path}")
flags = os.O_RDONLY
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(path, flags)
try:
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino):
        raise SystemExit(f"ERROR: archive changed before opening: {path}")
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    final_fd = os.fstat(descriptor)
finally:
    os.close(descriptor)
final_path = os.lstat(path)
identity = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_nlink)
if identity != (
    final_fd.st_dev,
    final_fd.st_ino,
    final_fd.st_size,
    final_fd.st_nlink,
) or identity != (
    final_path.st_dev,
    final_path.st_ino,
    final_path.st_size,
    final_path.st_nlink,
):
    raise SystemExit(f"ERROR: archive changed while hashing: {path}")
observed = digest.hexdigest()
if observed != expected:
    raise SystemExit(
        f"ERROR: archive checksum mismatch: expected {expected}, observed {observed}"
    )
print(observed)
PY
}

zc_unlink_if_identity() {
  local path="$1"
  local expected_identity="$2"
  python3 - "${path}" "${expected_identity}" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
try:
    info = os.lstat(path)
except FileNotFoundError:
    raise SystemExit(0)
if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
    raise SystemExit(f"ERROR: refusing to unlink non-regular/replaced file: {path}")
observed = f"{info.st_dev}:{info.st_ino}:{stat.S_IFMT(info.st_mode)}"
if observed != expected:
    raise SystemExit(f"ERROR: refusing to unlink replaced file: {path}")
os.unlink(path)
PY
}

zc_verify_sha256_manifest() {
  local root="$1"
  local manifest="$2"
  python3 - "${root}" "${manifest}" <<'PY'
import hashlib
import os
import pathlib
import re
import sys

root_arg = pathlib.Path(sys.argv[1])
manifest_arg = pathlib.Path(sys.argv[2])
if root_arg.is_symlink():
    raise SystemExit(f"ERROR: manifest root may not be a symbolic link: {root_arg}")
if manifest_arg.is_symlink():
    raise SystemExit(f"ERROR: manifest may not be a symbolic link: {manifest_arg}")
try:
    root = root_arg.resolve(strict=True)
    manifest = manifest_arg.resolve(strict=True)
except FileNotFoundError as error:
    raise SystemExit(f"ERROR: manifest input is missing: {error.filename}") from error
if not root.is_dir() or not manifest.is_file():
    raise SystemExit("ERROR: manifest root/file contract is invalid")

pattern = re.compile(r"^([0-9a-f]{64})  ([^\n]+)$")
seen: set[str] = set()
count = 0
for line_number, line in enumerate(
    manifest.read_text(encoding="utf-8").splitlines(), start=1
):
    match = pattern.fullmatch(line)
    if match is None:
        raise SystemExit(
            f"ERROR: malformed SHA-256 manifest line {line_number}: {manifest}"
        )
    expected, relative_text = match.groups()
    relative = pathlib.PurePosixPath(relative_text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in relative_text
    ):
        raise SystemExit(
            f"ERROR: unsafe manifest path at line {line_number}: {relative_text}"
        )
    normalized = relative.as_posix()
    if normalized in seen:
        raise SystemExit(f"ERROR: duplicate manifest path: {normalized}")
    seen.add(normalized)
    target = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"ERROR: manifest target uses a symbolic link: {current}")
    if not target.is_file():
        raise SystemExit(f"ERROR: manifest target is missing/not regular: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected:
        raise SystemExit(
            f"ERROR: SHA-256 mismatch for {normalized}: "
            f"expected {expected}, observed {observed}"
        )
    count += 1
if count == 0:
    raise SystemExit(f"ERROR: SHA-256 manifest is empty: {manifest}")
print(count)
PY
}

zc_verify_sha256_manifest_exact() {
  local root="$1"
  local manifest="$2"
  shift 2
  zc_verify_sha256_manifest "${root}" "${manifest}" >/dev/null || return
  python3 - "${root}" "${manifest}" "$@" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
manifest = pathlib.Path(sys.argv[2]).resolve(strict=True)
excluded = set(sys.argv[3:])
pattern = re.compile(r"^[0-9a-f]{64}  ([^\n]+)$")
declared = {
    pattern.fullmatch(line).group(1)
    for line in manifest.read_text(encoding="utf-8").splitlines()
}
actual: set[str] = set()
for candidate in root.rglob("*"):
    relative = candidate.relative_to(root).as_posix()
    if candidate.is_symlink():
        raise SystemExit(f"ERROR: verified tree contains a symbolic link: {relative}")
    if candidate.is_file():
        if relative not in excluded:
            actual.add(relative)
    elif not candidate.is_dir():
        raise SystemExit(f"ERROR: verified tree contains a special file: {relative}")
if declared != actual:
    missing = sorted(declared - actual)
    unexpected = sorted(actual - declared)
    raise SystemExit(
        "ERROR: exact tree manifest mismatch; "
        f"missing={missing[:5]}, unexpected={unexpected[:5]}"
    )
PY
}

zc_binding_value() {
  local binding="$1"
  local key="$2"
  python3 - "${binding}" "${key}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
if path.is_symlink() or not path.is_file():
    raise SystemExit(f"ERROR: binding is missing, non-regular, or symlinked: {path}")
matches = []
for line in path.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#"):
        continue
    if "=" not in line:
        raise SystemExit(f"ERROR: malformed provenance binding line: {line!r}")
    candidate, value = line.split("=", 1)
    if candidate == key:
        matches.append(value)
if len(matches) != 1 or not matches[0]:
    raise SystemExit(
        f"ERROR: binding must contain exactly one nonempty {key}= entry: {path}"
    )
print(matches[0])
PY
}

zc_verify_binding_value() {
  local binding="$1"
  local key="$2"
  local expected="$3"
  local observed
  observed="$(zc_binding_value "${binding}" "${key}")"
  if [[ "${observed}" != "${expected}" ]]; then
    printf 'ERROR: provenance binding mismatch for %s in %s\n' \
      "${key}" "${binding}" >&2
    printf 'Expected %s\nObserved %s\n' "${expected}" "${observed}" >&2
    return 2
  fi
}

zc_verify_binding_file_hash() {
  local binding="$1"
  local key="$2"
  local target="$3"
  if [[ -L "${target}" || ! -f "${target}" ]]; then
    printf 'ERROR: bound file is missing, non-regular, or symlinked: %s\n' \
      "${target}" >&2
    return 2
  fi
  zc_verify_binding_value "${binding}" "${key}" \
    "$(zc_sha256_file "${target}")"
}

zc_verify_tapenade_install() {
  local tapenade_home="$1"
  local toolchain_dir="$2"
  local expected_version="$3"
  local expected_revision="$4"
  local expected_archive_sha256="$5"
  local receipt="${tapenade_home}/.zc_tapenade_install_receipt.txt"
  local tree_manifest="${tapenade_home}/.zc_tapenade_tree_manifest.sha256"

  if [[ -L "${tapenade_home}" || ! -d "${tapenade_home}" ]]; then
    printf 'ERROR: Tapenade home is missing, non-directory, or symlinked: %s\n' \
      "${tapenade_home}" >&2
    return 2
  fi
  zc_verify_sha256_manifest_exact "${tapenade_home}" "${tree_manifest}" \
    .zc_tapenade_tree_manifest.sha256 .zc_tapenade_install_receipt.txt || return
  zc_verify_binding_value "${receipt}" receipt_schema \
    zc-tapenade-install-receipt-v1 || return
  zc_verify_binding_value "${receipt}" tapenade_version \
    "${expected_version}" || return
  zc_verify_binding_value "${receipt}" tapenade_revision \
    "${expected_revision}" || return
  zc_verify_binding_value "${receipt}" expected_archive_sha256 \
    "${expected_archive_sha256}" || return
  zc_verify_binding_value "${receipt}" observed_archive_sha256 \
    "${expected_archive_sha256}" || return
  zc_verify_binding_file_hash "${receipt}" tree_manifest_sha256 \
    "${tree_manifest}" || return
  zc_verify_binding_file_hash "${receipt}" installer_script_sha256 \
    "${toolchain_dir}/scripts/install_tapenade_linux.sh" || return
  zc_verify_binding_file_hash "${receipt}" build_guard_sha256 \
    "${toolchain_dir}/scripts/build_guard.sh" || return
  zc_verify_binding_file_hash "${receipt}" version_env_sha256 \
    "${toolchain_dir}/VERSION.env" || return
  zc_verify_binding_value "${receipt}" tree_file_count \
    "$(wc -l < "${tree_manifest}" | tr -d '[:space:]')" || return
}

zc_paths_overlap() {
  python3 - "$1" "$2" <<'PY'
import pathlib
import sys

left = pathlib.Path(sys.argv[1]).resolve()
right = pathlib.Path(sys.argv[2]).resolve()
raise SystemExit(not (left == right or left in right.parents or right in left.parents))
PY
}

zc_claim_build_dir() {
  local toolchain_dir="$1"
  local requested_build="$2"
  shift 2
  local build_dir marker protected lexical_build
  lexical_build="$(zc_assert_lexical_strict_child_no_symlinks \
    "${toolchain_dir}/build" "${requested_build}" 'build target')" || return
  if [[ -L "${requested_build}" ]]; then
    printf 'ERROR: build target may not be a symbolic link: %s\n' \
      "${requested_build}" >&2
    return 2
  fi
  build_dir="$(zc_resolve_path "${lexical_build}")"
  marker="${build_dir}/.zc_tapenade_build"

  case "${build_dir}" in
    "${toolchain_dir}/build/"*) ;;
    *)
      printf 'ERROR: build target must be a strict child of %s/build\n' \
        "${toolchain_dir}" >&2
      return 2
      ;;
  esac
  for protected in "$@"; do
    test -n "${protected}" || continue
    if zc_paths_overlap "${build_dir}" "${protected}"; then
      printf 'ERROR: build target overlaps a protected input: %s\n' \
        "${protected}" >&2
      return 2
    fi
  done

  if [[ -e "${build_dir}" && ! -d "${build_dir}" ]]; then
    printf 'ERROR: build target exists and is not a directory: %s\n' \
      "${build_dir}" >&2
    return 2
  fi
  if [[ -d "${build_dir}" ]] &&
     find "${build_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    if [[ -L "${marker}" ]] || [[ ! -f "${marker}" ]] ||
       [[ "$(cat "${marker}")" != 'managed-zc-tapenade-build-v1' ]]; then
      printf 'ERROR: nonempty build target is not owned by this toolchain: %s\n' \
        "${build_dir}" >&2
      return 2
    fi
  else
    mkdir -p "${build_dir}"
    printf '%s\n' managed-zc-tapenade-build-v1 > "${marker}"
  fi
  printf '%s\n' "${build_dir}"
}

zc_prepare_stage_dir() {
  local build_dir="$1"
  local stage_dir="$2"
  local build_resolved stage_resolved
  build_resolved="$(zc_resolve_path "${build_dir}")"
  if [[ -L "${stage_dir}" ]]; then
    printf 'ERROR: stage target may not be a symbolic link: %s\n' \
      "${stage_dir}" >&2
    return 2
  fi
  stage_resolved="$(zc_resolve_path "${stage_dir}")"
  if [[ "$(dirname "${stage_resolved}")" != "${build_resolved}" ]]; then
    printf 'ERROR: stage target must be an immediate build child: %s\n' \
      "${stage_resolved}" >&2
    return 2
  fi
  if [[ -e "${stage_resolved}" && ! -d "${stage_resolved}" ]]; then
    printf 'ERROR: stage target exists and is not a directory: %s\n' \
      "${stage_resolved}" >&2
    return 2
  fi
  if [[ -d "${stage_resolved}" ]] &&
     find "${stage_resolved}" -mindepth 1 -maxdepth 1 -print -quit |
       grep -q .; then
    if [[ "${ADJOINT_OVERWRITE:-0}" != '1' ]]; then
      printf 'ERROR: stage target is nonempty: %s\n' "${stage_resolved}" >&2
      printf 'Set ADJOINT_OVERWRITE=1 to replace this exact stage.\n' >&2
      return 2
    fi
    rm -rf -- "${stage_resolved}"
  fi
  mkdir -p "${stage_resolved}"
}

zc_reset_managed_build() {
  local build_dir="$1"
  local marker="${build_dir}/.zc_tapenade_build"
  local has_payload=0 entry
  if [[ -L "${marker}" ]] || [[ ! -f "${marker}" ]] ||
     [[ "$(cat "${marker}")" != 'managed-zc-tapenade-build-v1' ]]; then
    printf 'ERROR: build target is not owned by this toolchain: %s\n' \
      "${build_dir}" >&2
    return 2
  fi
  while IFS= read -r entry; do
    [[ "$(basename "${entry}")" == '.zc_tapenade_build' ]] && continue
    has_payload=1
    break
  done < <(find "${build_dir}" -mindepth 1 -maxdepth 1 -print)
  if [[ "${has_payload}" -eq 0 ]]; then
    return 0
  fi
  if [[ "${ADJOINT_OVERWRITE:-0}" != '1' ]]; then
    printf 'ERROR: build target is nonempty: %s\n' "${build_dir}" >&2
    printf 'Set ADJOINT_OVERWRITE=1 to replace this managed build.\n' >&2
    return 2
  fi
  find "${build_dir}" -mindepth 1 -maxdepth 1 \
    ! -name '.zc_tapenade_build' -exec rm -rf -- {} +
  [[ ! -L "${marker}" && -f "${marker}" ]]
}
