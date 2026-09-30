#!/usr/bin/env bash
set -euo pipefail

audit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
toolchain_guard="${audit_dir}/../tapenade_toolchain/scripts/build_guard.sh"
# shellcheck disable=SC1090
source "${toolchain_guard}"
builder_script_state="$(zc_capture_path_state "${BASH_SOURCE[0]}")"
builder_script_hash="$(zc_sha256_file "${BASH_SOURCE[0]}")"
guard_script_state="$(zc_capture_path_state "${toolchain_guard}")"
guard_script_hash="$(zc_sha256_file "${toolchain_guard}")"
run_dir="${1:?usage: build_run_tangent.sh RUN_DIR O0|O2|O3 OUTPUT}"
optimization="${2:?usage: build_run_tangent.sh RUN_DIR O0|O2|O3 OUTPUT}"
output_requested="${3:?usage: build_run_tangent.sh RUN_DIR O0|O2|O3 OUTPUT}"
case "${optimization}" in O0|O2|O3) ;; *)
  printf 'ERROR: optimization must be O0, O2, or O3\n' >&2
  exit 2
esac
run_dir="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${run_dir}")"
if [[ -L "${output_requested}" ]]; then
  printf 'ERROR: tangent output may not be a symbolic link: %s\n' \
    "${output_requested}" >&2
  exit 2
fi
output="$(zc_assert_lexical_strict_child_no_symlinks \
  "${audit_dir}/build" "${output_requested}" 'tangent output')"
output="$(zc_resolve_path "${output}")"
case "${output}" in
  "${audit_dir}/build/"*) ;;
  *)
    printf 'ERROR: output must be a strict child of %s/build\n' \
      "${audit_dir}" >&2
    exit 2
    ;;
esac
compiled="${run_dir}/coupled_compiled"
snapshot="${compiled}/build_input_snapshot"
generated="${snapshot}/generated/tangent"
prepared="${snapshot}/prepared"
kernel_source="${snapshot}/kernel_source"
driver="${snapshot}/audit/tangent_path_driver.F"
one_step_driver="${snapshot}/audit/tangent_driver.F"
build_manifest="${compiled}/build_manifest.txt"
input_manifest="${compiled}/build_input_manifest.sha256"
output_manifest="${output}.build_manifest.txt"
source_manifest="${output}.build_inputs.sha256"
one_step_output="${output}_one_step"

case "${output}" in
  "${run_dir}"|"${run_dir}/"*)
    printf 'ERROR: tangent output overlaps the immutable coupled build\n' >&2
    exit 2
    ;;
esac

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

for required in "${build_manifest}" "${input_manifest}" "${driver}" \
  "${one_step_driver}" \
  "${generated}/zc_kernel_api_d.f" "${generated}/zc_kernel_nino3_d.f" \
  "${prepared}/zc_kernel_state.inc"; do
  test -s "${required}" || {
    printf 'ERROR: tangent-build input is missing: %s\n' "${required}" >&2
    exit 2
  }
done

zc_verify_sha256_manifest_exact "${snapshot}" "${input_manifest}"
expected_input_manifest="$(awk -F= '$1=="build_input_manifest_sha256" {print $2}' "${build_manifest}")"
[[ "$(sha256_file "${input_manifest}")" == "${expected_input_manifest}" ]] || {
  printf 'ERROR: coupled build does not bind the input manifest\n' >&2
  exit 2
}
coupled_manifest_state="$(zc_capture_path_state "${build_manifest}")"
coupled_input_manifest_state="$(zc_capture_path_state "${input_manifest}")"
snapshot_state="$(zc_capture_path_state "${snapshot}")"

fc="$(awk -F= '$1=="compiler_path" {sub($1 "=", ""); print}' "${build_manifest}")"
expected_fc_hash="$(awk -F= '$1=="compiler_sha256" {print $2}' "${build_manifest}")"
test -x "${fc}" || {
  printf 'ERROR: recorded Fortran compiler is unavailable: %s\n' "${fc}" >&2
  exit 2
}
if [[ "${expected_fc_hash}" != unavailable ]] &&
   [[ "$(sha256_file "${fc}")" != "${expected_fc_hash}" ]]; then
  printf 'ERROR: recorded Fortran compiler hash changed: %s\n' "${fc}" >&2
  exit 2
fi
fc_state="$(zc_capture_path_state "${fc}")"

outputs=("${output}" "${one_step_output}" "${output_manifest}" \
  "${source_manifest}")
output_states=()
for member in "${outputs[@]}"; do
  if [[ -L "${member}" ]]; then
    printf 'ERROR: tangent output-set member may not be a symbolic link: %s\n' \
      "${member}" >&2
    exit 2
  fi
  if [[ -e "${member}" && ! -f "${member}" ]]; then
    printf 'ERROR: tangent output-set member is not a regular file: %s\n' \
      "${member}" >&2
    exit 2
  fi
  if [[ -e "${member}" && "${ADJOINT_OVERWRITE:-0}" != 1 ]]; then
    printf 'ERROR: tangent output-set member already exists: %s\n' \
      "${member}" >&2
    printf 'Set ADJOINT_OVERWRITE=1 to replace this exact output set.\n' >&2
    exit 2
  fi
  output_states+=("$(zc_capture_path_state "${member}")")
done
mkdir -p "$(dirname "${output}")"
stage="$(zc_create_stage_workspace "${audit_dir}/build" tangent-build)"
stage_identity="$(zc_path_identity "${stage}")"
objects="${stage}/objects"
compile_cwd="${stage}/controlled_cwd"
mkdir "${objects}" "${compile_cwd}"
publication_backup=""
publication_backup_identity=""
publication_committed=0
staged_outputs=()

cleanup_publication() {
  local status="$?"
  local index member staged_member current_state staged_identity restored=1
  trap - EXIT HUP INT TERM
  if [[ "${publication_committed}" -eq 0 ]]; then
    # Derive completed hard-link operations from inode identity rather than
    # bookkeeping that a signal could interrupt between mutation and append.
    for index in "${!outputs[@]}"; do
      member="${outputs[${index}]}"
      staged_member="${staged_outputs[${index}]:-}"
      if [[ -e "${member}" && -n "${staged_member}" && \
            "${member}" -ef "${staged_member}" ]]; then
        staged_identity="$(zc_path_identity "${staged_member}")"
        zc_unlink_if_identity "${member}" "${staged_identity}" || {
          restored=0
          status=2
        }
      fi
    done
    for index in "${!outputs[@]}"; do
      member="${outputs[${index}]}"
      if [[ -n "${publication_backup}" && ! -e "${member}" && \
            -e "${publication_backup}/original-${index}" ]]; then
        ln "${publication_backup}/original-${index}" "${member}" || {
          restored=0
          status=2
        }
      elif [[ -n "${publication_backup}" && \
              -e "${publication_backup}/original-${index}" ]]; then
        current_state="$(zc_capture_path_state "${member}" 2>/dev/null || true)"
        if [[ "${current_state}" != "${output_states[${index}]}" ]]; then
          printf 'ERROR: refusing to replace concurrent tangent output: %s\n' \
            "${member}" >&2
          restored=0
          status=2
        fi
      else
        current_state="$(zc_capture_path_state "${member}" 2>/dev/null || true)"
        if [[ "${current_state}" != "${output_states[${index}]}" ]]; then
          printf 'ERROR: tangent output changed during failed publication: %s\n' \
            "${member}" >&2
          restored=0
          status=2
        fi
      fi
    done
  fi
  if [[ -n "${publication_backup}" && -d "${publication_backup}" && \
        "$(zc_path_identity "${publication_backup}")" == \
          "${publication_backup_identity}" && \
        -f "${publication_backup}/.zc_tangent_backup" && \
        ! -L "${publication_backup}/.zc_tangent_backup" && \
        "$(cat "${publication_backup}/.zc_tangent_backup")" == \
          managed-zc-tangent-backup-v1 && "${restored}" -eq 1 ]]; then
    rm -rf -- "${publication_backup}"
  fi
  zc_remove_stage_workspace "${audit_dir}/build" "${stage}" \
    "${stage_identity}" || status=2
  exit "${status}"
}
trap cleanup_publication EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

flags=(-std=legacy "-${optimization}" -ffixed-line-length-none)
includes=(-I"${generated}" -I"${prepared}" -I"${kernel_source}")
link_flags=()
if [[ "$(uname -s)" == Darwin ]]; then
  macos_target="${MACOSX_DEPLOYMENT_TARGET:-$(sw_vers -productVersion | awk -F. '{print $1 ".0"}')}"
  sdk_path="$(xcrun --show-sdk-path)"
  flags+=("-mmacosx-version-min=${macos_target}")
  link_flags=(-Wl,-syslibroot,"${sdk_path}")
fi

inputs=()
link_objects=()
for source in "${generated}"/*_d.f; do
  inputs+=("${source}")
  name="$(basename "${source}" .f)"
  (cd "${compile_cwd}" && \
    "${fc}" "${flags[@]}" "${includes[@]}" -c "${source}" \
      -o "${objects}/${name}.o")
  link_objects+=("${objects}/${name}.o")
done
inputs+=("${driver}" "${one_step_driver}")
(cd "${compile_cwd}" && \
  "${fc}" "${flags[@]}" "${includes[@]}" -c "${driver}" \
    -o "${objects}/tangent_path_driver.o")
(cd "${compile_cwd}" && \
  "${fc}" "${flags[@]}" "${includes[@]}" -c "${one_step_driver}" \
    -o "${objects}/tangent_driver.o")

passive=(
  close_files UGETIO constc setup openfl MDNRIS setup2 USPKD nrdhist
  GGUBFS GGNQF MERFI initdat UERTST
)
for name in "${passive[@]}"; do
  source="${kernel_source}/${name}.F"
  if [[ ! -f "${source}" ]]; then
    source="${kernel_source}/ZC_lib_routines/${name}.F"
  fi
  test -s "${source}"
  inputs+=("${source}")
  (cd "${compile_cwd}" && \
    "${fc}" "${flags[@]}" "${includes[@]}" -c "${source}" \
      -o "${objects}/passive_${name}.o")
  link_objects+=("${objects}/passive_${name}.o")
done

while IFS= read -r include_file; do
  inputs+=("${include_file}")
done < <(find "${generated}" "${prepared}" "${kernel_source}" \
  -maxdepth 1 -type f \( -name '*.inc' -o -name '*.common' \) -print | \
  LC_ALL=C sort)

staged_path="${stage}/path_executable"
staged_one_step="${stage}/one_step_executable"
staged_source_manifest="${stage}/build_inputs.sha256"
staged_output_manifest="${stage}/build_manifest.txt"
(cd "${compile_cwd}" && \
  "${fc}" "${flags[@]}" "${link_flags[@]}" \
    "${objects}/tangent_path_driver.o" "${link_objects[@]}" \
    -o "${staged_path}")
(cd "${compile_cwd}" && \
  "${fc}" "${flags[@]}" "${link_flags[@]}" \
    "${objects}/tangent_driver.o" "${link_objects[@]}" \
    -o "${staged_one_step}")

# Nothing in the immutable coupled snapshot, its external binding manifests,
# the compiler, or this builder may change between preflight and completed
# compilation. This catches transient/concurrent mutation before publication.
zc_verify_path_state "${snapshot}" "${snapshot_state}"
zc_verify_sha256_manifest_exact "${snapshot}" "${input_manifest}"
zc_verify_path_state "${build_manifest}" "${coupled_manifest_state}"
zc_verify_path_state "${input_manifest}" "${coupled_input_manifest_state}"
zc_verify_path_state "${BASH_SOURCE[0]}" "${builder_script_state}"
zc_verify_path_state "${toolchain_guard}" "${guard_script_state}"
zc_verify_path_state "${fc}" "${fc_state}"
if [[ "${expected_fc_hash}" != unavailable ]]; then
  [[ "$(sha256_file "${fc}")" == "${expected_fc_hash}" ]] || {
    printf 'ERROR: recorded compiler changed during tangent build\n' >&2
    exit 2
  }
fi
{
  for input in "${inputs[@]}"; do
    printf '%s  %s\n' "$(sha256_file "${input}")" \
      "$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "${input}" "${snapshot}")"
  done
} | LC_ALL=C sort -u > "${staged_source_manifest}"
{
  printf 'schema=zc-tangent-audit-build-v1\n'
  printf 'builder_script_sha256=%s\n' "${builder_script_hash}"
  printf 'build_guard_sha256=%s\n' "${guard_script_hash}"
  printf 'optimization=%s\n' "${optimization}"
  printf 'compiler_path=%s\n' "${fc}"
  printf 'compiler_sha256=%s\n' "$(sha256_file "${fc}")"
  printf 'compiler_version=%s\n' "$("${fc}" --version | head -n 1)"
  printf 'flags=%s\n' "${flags[*]}"
  printf 'link_flags=%s\n' "${link_flags[*]}"
  printf 'controlled_compile_cwd=%s\n' '<UNIQUE_EMPTY_STAGING>/controlled_cwd'
  printf 'coupled_build_manifest_sha256=%s\n' \
    "$(sha256_file "${build_manifest}")"
  printf 'coupled_build_input_manifest_sha256=%s\n' \
    "$(sha256_file "${input_manifest}")"
  printf 'tangent_build_inputs_sha256=%s\n' \
    "$(sha256_file "${staged_source_manifest}")"
  printf 'path_executable_sha256=%s\n' "$(sha256_file "${staged_path}")"
  printf 'one_step_executable_sha256=%s\n' \
    "$(sha256_file "${staged_one_step}")"
  # Backward-compatible alias for older audit tooling.
  printf 'executable_sha256=%s\n' "$(sha256_file "${staged_path}")"
} > "${staged_output_manifest}"

# Publish only after both executables and both provenance sidecars exist.
# Revalidate immediately before commit. Hard links give every target an
# atomic no-clobber operation; the EXIT/signal handler rolls back a partial
# four-file set. Overwrite first moves old members into a unique authenticated
# sibling so they remain recoverable until all four new links exist.
for member in "${outputs[@]}"; do
  if [[ -L "${member}" || ( -e "${member}" && ! -f "${member}" ) ]]; then
    printf 'ERROR: tangent output-set member changed type before commit: %s\n' \
      "${member}" >&2
    exit 2
  fi
  if [[ -e "${member}" && "${ADJOINT_OVERWRITE:-0}" != 1 ]]; then
    printf 'ERROR: tangent output-set member appeared before commit: %s\n' \
      "${member}" >&2
    exit 2
  fi
done
for index in "${!outputs[@]}"; do
  zc_verify_path_state "${outputs[${index}]}" "${output_states[${index}]}"
done
zc_verify_path_state "${snapshot}" "${snapshot_state}"
zc_verify_sha256_manifest_exact "${snapshot}" "${input_manifest}"
zc_verify_path_state "${build_manifest}" "${coupled_manifest_state}"
zc_verify_path_state "${input_manifest}" "${coupled_input_manifest_state}"
zc_verify_path_state "${BASH_SOURCE[0]}" "${builder_script_state}"
zc_verify_path_state "${toolchain_guard}" "${guard_script_state}"
zc_verify_path_state "${fc}" "${fc_state}"
staged_outputs=("${staged_path}" "${staged_one_step}" \
  "${staged_output_manifest}" "${staged_source_manifest}")
publication_backup="$(mktemp -d \
  "${audit_dir}/build/.tangent-backup.XXXXXX")"
printf '%s\n' managed-zc-tangent-backup-v1 > \
  "${publication_backup}/.zc_tangent_backup"
publication_backup_identity="$(zc_path_identity "${publication_backup}")"
for index in "${!outputs[@]}"; do
  if [[ -e "${outputs[${index}]}" ]]; then
    zc_verify_path_state "${outputs[${index}]}" \
      "${output_states[${index}]}"
    ln "${outputs[${index}]}" \
      "${publication_backup}/original-${index}"
    zc_verify_path_state "${outputs[${index}]}" \
      "${output_states[${index}]}"
    [[ "${outputs[${index}]}" -ef \
       "${publication_backup}/original-${index}" ]]
    zc_unlink_if_identity "${outputs[${index}]}" \
      "$(zc_path_identity "${publication_backup}/original-${index}")"
  fi
  zc_test_failpoint "tangent_after_backup_${index}"
done
for index in "${!outputs[@]}"; do
  [[ ! -e "${outputs[${index}]}" && ! -L "${outputs[${index}]}" ]]
  ln "${staged_outputs[${index}]}" "${outputs[${index}]}"
  zc_test_failpoint "tangent_after_publish_${index}"
done
for index in "${!outputs[@]}"; do
  [[ "${outputs[${index}]}" -ef "${staged_outputs[${index}]}" ]]
done
[[ -f "${publication_backup}/.zc_tangent_backup" ]] &&
  [[ ! -L "${publication_backup}/.zc_tangent_backup" ]] &&
  [[ "$(zc_path_identity "${publication_backup}")" == \
    "${publication_backup_identity}" ]] &&
  [[ "$(cat "${publication_backup}/.zc_tangent_backup")" == \
    managed-zc-tangent-backup-v1 ]]
zc_test_failpoint tangent_before_commit
for index in "${!outputs[@]}"; do
  [[ "${outputs[${index}]}" -ef "${staged_outputs[${index}]}" ]]
done
[[ "$(zc_path_identity "${publication_backup}")" == \
  "${publication_backup_identity}" ]]
publication_committed=1
rm -rf -- "${publication_backup}"
publication_backup=""
trap - EXIT HUP INT TERM
zc_remove_stage_workspace "${audit_dir}/build" "${stage}" \
  "${stage_identity}"
stage=""

printf 'Tangent path executable: %s\n' "${output}"
printf 'SHA-256: %s\n' "$(sha256_file "${output}")"
printf 'Tangent one-step executable: %s\n' "${one_step_output}"
printf 'SHA-256: %s\n' "$(sha256_file "${one_step_output}")"
printf 'Build manifest: %s\n' "${output_manifest}"
