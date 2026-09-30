#!/usr/bin/env bash
set -euo pipefail

toolchain_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${toolchain_dir}/VERSION.env"
kernel_source="${1:?usage: run_coupled_toolchain_linux.sh KERNEL_SOURCE TAPENADE_HOME [BUILD_DIR]}"
requested_tapenade_home="${2:?usage: run_coupled_toolchain_linux.sh KERNEL_SOURCE TAPENADE_HOME [BUILD_DIR]}"
tapenade_home="${requested_tapenade_home}"
requested_build="${3:-${toolchain_dir}/build/coupled_release}"
# shellcheck disable=SC1091
source "${toolchain_dir}/scripts/build_guard.sh"
if [[ -L "${requested_tapenade_home}" ]]; then
  printf 'ERROR: Tapenade home may not be a symbolic link: %s\n' \
    "${requested_tapenade_home}" >&2
  exit 2
fi
kernel_source="$(zc_resolve_path "${kernel_source}")"
tapenade_home="$(zc_resolve_path "${tapenade_home}")"
build_dir="$(zc_claim_build_dir "${toolchain_dir}" "${requested_build}" \
  "${kernel_source}" "${tapenade_home}")"
[[ "$(dirname "${build_dir}")" == "${toolchain_dir}/build" ]] || {
  printf 'ERROR: full coupled build must be an immediate child of %s/build\n' \
    "${toolchain_dir}" >&2
  exit 2
}

# Complete read-only preflight. The wrapper also builds transactionally below,
# so a patch, generation, compilation, or linking failure preserves the prior
# managed build.
for command_name in python3 patch cpp "${FC:-gfortran}" "${CC:-cc}" \
  "${AR:-ar}" java; do
  command -v "${command_name}" >/dev/null || {
    printf 'ERROR: full-toolchain command is unavailable: %s\n' \
      "${command_name}" >&2
    exit 2
  }
done
java_version="$(java -version 2>&1 | head -n 1)"
if ! grep -Eq 'version "17([.]|")' <<<"${java_version}"; then
  printf 'ERROR: Java 17 is required; observed: %s\n' "${java_version}" >&2
  exit 2
fi

kernel_inputs=(
  zc_kernel_api.F ssta.F ztmfc1.F cforce.F mloop.F akcalc.F bndary.F
  uhcalc.F uhinit.F tridag.F zc_kernel_state.inc zeq.common
  modified_means.common openfl.F setup.F constc.F nrdhist.F initdat.F
  setup2.F close_files.F
)
for relative_name in "${kernel_inputs[@]}"; do
  test -s "${kernel_source}/${relative_name}" || {
    printf 'ERROR: full-toolchain input is missing: %s\n' \
      "${kernel_source}/${relative_name}" >&2
    exit 2
  }
done
for relative_name in FFT2C.F GGNQF.F GGUBFS.F MDNRIS.F MERFI.F \
  UERTST.F UGETIO.F USPKD.F; do
  test -s "${kernel_source}/ZC_lib_routines/${relative_name}" || {
    printf 'ERROR: full-toolchain library input is missing: %s\n' \
      "${kernel_source}/ZC_lib_routines/${relative_name}" >&2
    exit 2
  }
done
for relative_name in bin/tapenade bin/linux/fortranParser \
  ADFirstAidKit/adStack.c ADFirstAidKit/adStack.h \
  ADFirstAidKit/adComplex.h; do
  test -s "${tapenade_home}/${relative_name}" || {
    printf 'ERROR: full-toolchain Tapenade input is missing: %s\n' \
      "${tapenade_home}/${relative_name}" >&2
    exit 2
  }
done
test -x "${tapenade_home}/bin/tapenade"
test -x "${tapenade_home}/bin/linux/fortranParser"
zc_verify_tapenade_install "${tapenade_home}" "${toolchain_dir}" \
  "${TAPENADE_VERSION}" "${TAPENADE_REVISION}" \
  "${TAPENADE_ARCHIVE_SHA256}"

for driver in coupled/zc_kernel_nino3.F \
  coupled/zc_adjoint_path_driver.F coupled/zc_nino3_adjoint_driver.F \
  ../full_tangent_audit/tangent_driver.F \
  ../full_tangent_audit/tangent_path_driver.F; do
  test -s "${toolchain_dir}/${driver}" || {
    printf 'ERROR: toolchain driver is missing: %s\n' "${driver}" >&2
    exit 2
  }
done
for patch_file in "${toolchain_dir}/patches/"*.patch; do
  test -s "${patch_file}" || {
    printf 'ERROR: toolchain patch is missing: %s\n' "${patch_file}" >&2
    exit 2
  }
done
for stage_script in prepare_coupled_sources.sh generate_coupled_nino3.sh \
  instrument_generated_reverse.sh compile_coupled_adjoint.sh; do
  test -x "${toolchain_dir}/scripts/${stage_script}" || {
    printf 'ERROR: toolchain stage is not executable: %s\n' \
      "${stage_script}" >&2
    exit 2
  }
done

# Refuse replacement before doing the potentially costly fresh build, but do
# not delete anything yet.
has_payload=0
while IFS= read -r entry; do
  [[ "$(basename "${entry}")" == '.zc_tapenade_build' ]] && continue
  has_payload=1
  break
done < <(find "${build_dir}" -mindepth 1 -maxdepth 1 -print)
if [[ "${has_payload}" -eq 1 && "${ADJOINT_OVERWRITE:-0}" != 1 ]]; then
  printf 'ERROR: build target is nonempty: %s\n' "${build_dir}" >&2
  printf 'Set ADJOINT_OVERWRITE=1 to replace this managed build.\n' >&2
  exit 2
fi
initial_build_state="$(zc_capture_path_state "${build_dir}")"
wrapper_script_state="$(zc_capture_path_state "${BASH_SOURCE[0]}")"
wrapper_guard_state="$(zc_capture_path_state \
  "${toolchain_dir}/scripts/build_guard.sh")"

wrapper_workspace="$(zc_create_stage_workspace \
  "${toolchain_dir}/build" coupled-wrapper)"
wrapper_workspace_identity="$(zc_path_identity "${wrapper_workspace}")"
staging_dir="${wrapper_workspace}/payload"
mkdir "${staging_dir}"
printf '%s\n' managed-zc-tapenade-build-v1 > \
  "${staging_dir}/.zc_tapenade_build"
cleanup_wrapper_workspace() {
  local status="$?"
  trap - EXIT HUP INT TERM
  zc_remove_stage_workspace "${toolchain_dir}/build" \
    "${wrapper_workspace}" "${wrapper_workspace_identity}" || status=2
  exit "${status}"
}
trap cleanup_wrapper_workspace EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"${toolchain_dir}/scripts/prepare_coupled_sources.sh" \
  "${kernel_source}" "${staging_dir}"
"${toolchain_dir}/scripts/generate_coupled_nino3.sh" \
  "${tapenade_home}" "${staging_dir}"
"${toolchain_dir}/scripts/instrument_generated_reverse.sh" \
  "${staging_dir}/coupled_generated/reverse"
"${toolchain_dir}/scripts/compile_coupled_adjoint.sh" \
  "${tapenade_home}" "${kernel_source}" "${staging_dir}"

# Reauthenticate the exact target state captured before the long build, then
# publish through the shared signal-aware transaction. The transaction records
# intent before either rename and refuses to displace a concurrent replacement.
zc_verify_path_state "${BASH_SOURCE[0]}" "${wrapper_script_state}"
zc_verify_path_state "${toolchain_dir}/scripts/build_guard.sh" \
  "${wrapper_guard_state}"
zc_verify_tapenade_install "${tapenade_home}" "${toolchain_dir}" \
  "${TAPENADE_VERSION}" "${TAPENADE_REVISION}" \
  "${TAPENADE_ARCHIVE_SHA256}"
zc_test_failpoint wrapper_before_publication
zc_publish_stage_directory "${toolchain_dir}/build" "${build_dir}" \
  "${staging_dir}" "${initial_build_state}" managed
staging_dir=""
trap - EXIT HUP INT TERM
zc_remove_stage_workspace "${toolchain_dir}/build" \
  "${wrapper_workspace}" "${wrapper_workspace_identity}"
wrapper_workspace=""

printf 'Full coupled Tapenade toolchain completed in %s\n' "${build_dir}"
