#!/usr/bin/env bash
set -euo pipefail

toolchain_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${toolchain_dir}/VERSION.env"
requested_tapenade_home="${1:?usage: compile_coupled_adjoint.sh TAPENADE_HOME KERNEL_SOURCE [BUILD_DIR]}"
tapenade_home="${requested_tapenade_home}"
kernel_source="${2:?usage: compile_coupled_adjoint.sh TAPENADE_HOME KERNEL_SOURCE [BUILD_DIR]}"
build_dir="${3:-${toolchain_dir}/build/coupled_release}"
prepared_dir="${build_dir}/coupled_prepared"
generated_dir="${build_dir}/coupled_generated"
compile_target="${build_dir}/coupled_compiled"

# shellcheck disable=SC1091
source "${toolchain_dir}/scripts/build_guard.sh"
if [[ -L "${requested_tapenade_home}" ]]; then
  printf 'ERROR: Tapenade home may not be a symbolic link: %s\n' \
    "${requested_tapenade_home}" >&2
  exit 2
fi
tapenade_home="$(zc_resolve_path "${tapenade_home}")"
kernel_source="$(zc_resolve_path "${kernel_source}")"
build_dir="$(zc_claim_build_dir "${toolchain_dir}" "${build_dir}" \
  "${tapenade_home}" "${kernel_source}")"
prepared_dir="${build_dir}/coupled_prepared"
generated_dir="${build_dir}/coupled_generated"
compile_target="${build_dir}/coupled_compiled"

fc="${FC:-gfortran}"
cc="${CC:-cc}"
archiver="${AR:-ar}"
build_mode="${ADJOINT_BUILD_MODE:-debug}"

primal_names=(
  zc_kernel_api ssta ztmfc1 cforce mloop akcalc bndary uhcalc uhinit
  tridag
)
support_names=(openfl setup constc nrdhist initdat setup2)
tangent_audit_support_names=(close_files)
zc_library_names=(GGNQF GGUBFS MDNRIS MERFI UERTST UGETIO USPKD)
kernel_include_names=(zeq.common modified_means.common)
compile_script_names=(
  build_guard.sh prepare_coupled_sources.sh generate_coupled_nino3.sh
  instrument_generated_reverse.sh generate_coupled_on_cluster.sh
  install_tapenade_linux.sh compile_coupled_adjoint.sh
  run_coupled_toolchain_linux.sh run_coupled_case.py
)
normalization_patch_names=(
  kernel_api_ad_ready.patch kernel_api_scalar_helpers.patch
  ssta_ad_io_free.patch mloop_io_free.patch mloop_diagnostics_io_free.patch
  cforce_no_hidden_save.patch cforce_ad_ready.patch
  ztmfc1_feedback_loop.patch ztmfc1_ad_constants.patch
  remove_blank_local_saves.patch stress_selective_save.patch
  terminal_push_initialization.patch fft2c_no_equivalence.patch
)
instrumentation_patch_names=(
  generated_reverse_branch_tape.patch generated_reverse_defined_scratch.patch
)

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

resolved_command() {
  local command_name="$1"
  local command_path
  command_path="$(command -v "${command_name}")" || {
    printf 'ERROR: required command is unavailable: %s\n' \
      "${command_name}" >&2
    exit 2
  }
  zc_resolve_path "${command_path}"
}

command_hash_or_unavailable() {
  local command_path="$1"
  if [[ -f "${command_path}" ]]; then
    sha256_file "${command_path}"
  else
    printf unavailable
  fi
}

case "${build_mode}" in
  debug)
    fflags=(-std=legacy -O0 -g -fcheck=all -fbacktrace
      -fallow-argument-mismatch -ffixed-line-length-none)
    ;;
  release)
    fflags=(-std=legacy -O2 -fbacktrace -fallow-argument-mismatch
      -ffixed-line-length-none)
    ;;
  *)
    printf 'ERROR: ADJOINT_BUILD_MODE must be debug or release\n' >&2
    exit 2
    ;;
esac
# The untouched ZC Makefile uses -O3.  Retaining that optimization level for
# the normalized primal and immutable setup routines is required for bitwise
# replay of its single-precision trajectory.  Derivative code uses the mode
# selected above.
primal_fflags=(-std=legacy -O3 -fbacktrace -fallow-argument-mismatch
  -ffixed-line-length-none)

# Homebrew GCC may retain an obsolete configured sysroot after a macOS/Xcode
# update.  Supplying the active SDK explicitly keeps the build reproducible on
# macOS and is a no-op on Linux.
platform_flags=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  sdk_path="$(xcrun --sdk macosx --show-sdk-path)"
  sdk_major="$(xcrun --sdk macosx --show-sdk-version | cut -d. -f1)"
  platform_flags=(-isysroot "${sdk_path}" -mmacosx-version-min="${sdk_major}.0")
fi

fc_path="$(resolved_command "${fc}")"
cc_path="$(resolved_command "${cc}")"
ar_path="$(resolved_command "${archiver}")"
fc_version="$("${fc_path}" --version 2>&1 | head -n 1)"
cc_version="$("${cc_path}" --version 2>&1 | head -n 1)"
ar_version="$("${ar_path}" --version 2>&1 | head -n 1 || true)"
fc_hash="$(command_hash_or_unavailable "${fc_path}")"
cc_hash="$(command_hash_or_unavailable "${cc_path}")"
ar_hash="$(command_hash_or_unavailable "${ar_path}")"
if [[ -z "${ar_version}" || "${ar_version}" == usage:* ]]; then
  ar_version=unavailable
fi
cflags=(-O0 -g)

zc_verify_tapenade_install "${tapenade_home}" "${toolchain_dir}" \
  "${TAPENADE_VERSION}" "${TAPENADE_REVISION}" \
  "${TAPENADE_ARCHIVE_SHA256}"

for required in \
  "${prepared_dir}/zc_kernel_api.f" \
  "${generated_dir}/reverse/zc_kernel_api_b.f" \
  "${generated_dir}/tangent/zc_kernel_api_d.f" \
  "${tapenade_home}/ADFirstAidKit/adStack.c" \
  "${tapenade_home}/ADFirstAidKit/adStack.h" \
  "${tapenade_home}/ADFirstAidKit/adComplex.h" \
  "${kernel_source}/openfl.F"; do
  test -s "${required}" || {
    printf 'ERROR: required file missing: %s\n' "${required}" >&2
    exit 2
  }
done
for name in "${primal_names[@]}"; do
  test -s "${prepared_dir}/${name}.f" || {
    printf 'ERROR: required prepared source is missing: %s.f\n' "${name}" >&2
    exit 2
  }
done
test -s "${prepared_dir}/ZC_lib_routines/fft2c.f"
for name in "${support_names[@]}"; do
  test -s "${kernel_source}/${name}.F" || {
    printf 'ERROR: required passive source is missing: %s.F\n' "${name}" >&2
    exit 2
  }
done
for name in "${tangent_audit_support_names[@]}"; do
  test -s "${kernel_source}/${name}.F" || {
    printf 'ERROR: required tangent-audit support is missing: %s.F\n' \
      "${name}" >&2
    exit 2
  }
done
for name in "${zc_library_names[@]}"; do
  test -s "${kernel_source}/ZC_lib_routines/${name}.F" || {
    printf 'ERROR: required ZC library source is missing: %s.F\n' "${name}" >&2
    exit 2
  }
done
for name in "${kernel_include_names[@]}"; do
  test -s "${kernel_source}/${name}" || {
    printf 'ERROR: required passive include is missing: %s\n' "${name}" >&2
    exit 2
  }
done
for required in \
  "${toolchain_dir}/coupled/zc_adjoint_path_driver.F" \
  "${toolchain_dir}/coupled/zc_nino3_adjoint_driver.F" \
  "${toolchain_dir}/coupled/zc_kernel_nino3.F" \
  "${toolchain_dir}/../full_tangent_audit/tangent_driver.F" \
  "${toolchain_dir}/../full_tangent_audit/tangent_path_driver.F" \
  "${prepared_dir}/prepared_source_manifest.sha256" \
  "${prepared_dir}/prepared_manifest_binding.txt" \
  "${prepared_dir}/preparation_provenance.txt" \
  "${prepared_dir}/upstream_preprocessed_manifest.sha256" \
  "${prepared_dir}/kernel_source_input_manifest.sha256" \
  "${generated_dir}/generated_source_manifest.pre_instrument.sha256" \
  "${generated_dir}/generated_source_manifest.sha256" \
  "${generated_dir}/generation_provenance.txt" \
  "${generated_dir}/generation_manifest_binding.txt" \
  "${generated_dir}/instrumentation_provenance.txt" \
  "${generated_dir}/instrumentation_manifest_binding.txt"; do
  test -s "${required}" || {
    printf 'ERROR: required build/provenance input is missing: %s\n' \
      "${required}" >&2
    exit 2
  }
done
for script_name in "${compile_script_names[@]}"; do
  test -s "${toolchain_dir}/scripts/${script_name}" || {
    printf 'ERROR: required toolchain script is missing: %s\n' \
      "${script_name}" >&2
    exit 2
  }
done
for patch_file in "${toolchain_dir}/patches/"*.patch; do
  test -s "${patch_file}" || {
    printf 'ERROR: required toolchain patch is missing: %s\n' \
      "${patch_file}" >&2
    exit 2
  }
done
if ! compgen -G "${generated_dir}/reverse/*.f" >/dev/null ||
   ! compgen -G "${generated_dir}/tangent/*.f" >/dev/null; then
  printf 'ERROR: generated derivative source set is incomplete\n' >&2
  exit 2
fi

# Verify both content manifests and every cross-stage binding before a
# requested overwrite is allowed to replace a previously good compile stage.
prepared_manifest="${prepared_dir}/prepared_source_manifest.sha256"
prepared_binding="${prepared_dir}/prepared_manifest_binding.txt"
preparation_provenance="${prepared_dir}/preparation_provenance.txt"
upstream_manifest="${prepared_dir}/upstream_preprocessed_manifest.sha256"
kernel_source_manifest="${prepared_dir}/kernel_source_input_manifest.sha256"
pre_manifest="${generated_dir}/generated_source_manifest.pre_instrument.sha256"
final_manifest="${generated_dir}/generated_source_manifest.sha256"
generation_provenance="${generated_dir}/generation_provenance.txt"
generation_binding="${generated_dir}/generation_manifest_binding.txt"
instrumentation_provenance="${generated_dir}/instrumentation_provenance.txt"
instrumentation_binding="${generated_dir}/instrumentation_manifest_binding.txt"

zc_verify_sha256_manifest_exact "${prepared_dir}" "${prepared_manifest}" \
  prepared_source_manifest.sha256 prepared_manifest_binding.txt \
  upstream_preprocessed_manifest.sha256 kernel_source_input_manifest.sha256
zc_verify_binding_value "${prepared_binding}" binding_schema \
  zc-prepared-manifest-binding-v1
zc_verify_binding_file_hash "${prepared_binding}" \
  prepared_source_manifest_sha256 "${prepared_manifest}"
zc_verify_binding_file_hash "${prepared_binding}" \
  preparation_provenance_sha256 "${preparation_provenance}"
zc_verify_binding_file_hash "${prepared_binding}" \
  upstream_preprocessed_manifest_sha256 "${upstream_manifest}"
zc_verify_binding_file_hash "${prepared_binding}" \
  kernel_source_input_manifest_sha256 "${kernel_source_manifest}"
zc_verify_binding_value "${preparation_provenance}" \
  kernel_source_input_manifest_sha256 \
  "$(sha256_file "${kernel_source_manifest}")"
zc_verify_sha256_manifest "${kernel_source}" "${kernel_source_manifest}" \
  >/dev/null

zc_verify_sha256_manifest_exact "${generated_dir}" "${final_manifest}" \
  generated_source_manifest.pre_instrument.sha256 \
  generated_source_manifest.sha256 generation_provenance.txt \
  generation_manifest_binding.txt instrumentation_provenance.txt \
  instrumentation_manifest_binding.txt
zc_verify_binding_value "${generation_binding}" binding_schema \
  zc-generation-manifest-binding-v1
zc_verify_binding_file_hash "${generation_binding}" \
  generated_pre_manifest_sha256 "${pre_manifest}"
zc_verify_binding_file_hash "${generation_binding}" \
  generation_provenance_sha256 "${generation_provenance}"
zc_verify_binding_file_hash "${generation_binding}" \
  prepared_source_manifest_sha256 "${prepared_manifest}"
zc_verify_binding_file_hash "${generation_binding}" \
  prepared_manifest_binding_sha256 "${prepared_binding}"
zc_verify_binding_value "${generation_binding}" \
  tapenade_install_receipt_sha256 \
  "$(sha256_file "${tapenade_home}/.zc_tapenade_install_receipt.txt")"
zc_verify_binding_value "${generation_binding}" \
  tapenade_tree_manifest_sha256 \
  "$(sha256_file "${tapenade_home}/.zc_tapenade_tree_manifest.sha256")"

zc_verify_binding_value "${instrumentation_binding}" binding_schema \
  zc-instrumentation-manifest-binding-v1
zc_verify_binding_file_hash "${instrumentation_binding}" \
  generated_source_manifest_sha256 "${final_manifest}"
zc_verify_binding_file_hash "${instrumentation_binding}" \
  instrumentation_provenance_sha256 "${instrumentation_provenance}"
zc_verify_binding_file_hash "${instrumentation_binding}" \
  generation_manifest_binding_sha256 "${generation_binding}"
zc_verify_binding_file_hash "${instrumentation_binding}" \
  generated_pre_manifest_sha256 "${pre_manifest}"
zc_verify_binding_value "${instrumentation_provenance}" \
  pre_instrument_manifest_sha256 "$(sha256_file "${pre_manifest}")"
zc_verify_binding_value "${instrumentation_provenance}" \
  generation_provenance_sha256 "$(sha256_file "${generation_provenance}")"
zc_verify_binding_value "${instrumentation_provenance}" \
  generation_manifest_binding_sha256 "$(sha256_file "${generation_binding}")"
zc_verify_binding_value "${instrumentation_provenance}" \
  prepared_source_manifest_sha256 "$(sha256_file "${prepared_manifest}")"
zc_verify_binding_value "${instrumentation_provenance}" \
  prepared_manifest_binding_sha256 "$(sha256_file "${prepared_binding}")"
zc_verify_binding_value "${generation_provenance}" \
  prepared_source_manifest_sha256 "$(sha256_file "${prepared_manifest}")"
zc_verify_binding_value "${generation_provenance}" \
  prepared_manifest_binding_sha256 "$(sha256_file "${prepared_binding}")"
zc_verify_binding_value "${generation_provenance}" \
  tapenade_install_receipt_sha256 \
  "$(sha256_file "${tapenade_home}/.zc_tapenade_install_receipt.txt")"
zc_verify_binding_value "${generation_provenance}" \
  tapenade_tree_manifest_sha256 \
  "$(sha256_file "${tapenade_home}/.zc_tapenade_tree_manifest.sha256")"
zc_verify_binding_file_hash "${preparation_provenance}" script_sha256 \
  "${toolchain_dir}/scripts/prepare_coupled_sources.sh"
zc_verify_binding_file_hash "${preparation_provenance}" build_guard_sha256 \
  "${toolchain_dir}/scripts/build_guard.sh"
zc_verify_binding_file_hash "${preparation_provenance}" version_env_sha256 \
  "${toolchain_dir}/VERSION.env"
zc_verify_binding_file_hash "${preparation_provenance}" \
  scalar_head_source_sha256 \
  "${toolchain_dir}/coupled/zc_kernel_nino3.F"
for patch_name in "${normalization_patch_names[@]}"; do
  zc_verify_binding_file_hash "${preparation_provenance}" \
    "patch_${patch_name}_sha256" "${toolchain_dir}/patches/${patch_name}"
done
zc_verify_binding_file_hash "${generation_provenance}" script_sha256 \
  "${toolchain_dir}/scripts/generate_coupled_nino3.sh"
zc_verify_binding_file_hash "${generation_provenance}" build_guard_sha256 \
  "${toolchain_dir}/scripts/build_guard.sh"
zc_verify_binding_file_hash "${generation_provenance}" version_env_sha256 \
  "${toolchain_dir}/VERSION.env"
zc_verify_binding_file_hash "${generation_provenance}" \
  installer_script_sha256 \
  "${toolchain_dir}/scripts/install_tapenade_linux.sh"
zc_verify_binding_file_hash "${generation_provenance}" \
  cluster_helper_script_sha256 \
  "${toolchain_dir}/scripts/generate_coupled_on_cluster.sh"
zc_verify_binding_file_hash "${instrumentation_provenance}" script_sha256 \
  "${toolchain_dir}/scripts/instrument_generated_reverse.sh"
zc_verify_binding_file_hash "${instrumentation_provenance}" \
  build_guard_sha256 "${toolchain_dir}/scripts/build_guard.sh"
for patch_name in "${instrumentation_patch_names[@]}"; do
  zc_verify_binding_file_hash "${instrumentation_provenance}" \
    "patch_${patch_name}_sha256" "${toolchain_dir}/patches/${patch_name}"
done
[[ "$(sha256_file "${toolchain_dir}/coupled/zc_kernel_nino3.F")" == \
  "$(sha256_file "${prepared_dir}/zc_kernel_nino3.f")" ]] || {
  printf 'ERROR: prepared scalar head differs from the current toolchain head\n' \
    >&2
  exit 2
}

# Refuse to compile an uninstrumented reverse.  These assertions cover both
# branch-tape publication and deterministic definitions for Tapenade's split
# FWD scratch values.
grep -q 'tape(3) = kernel_atm_reset' \
  "${generated_dir}/reverse/zc_kernel_api_b.f"
grep -q 'h_minus_eng = 0.0' "${generated_dir}/reverse/ssta_b.f"
grep -q 'beta_factor(i, j) = 0.0' "${generated_dir}/reverse/ztmfc1_b.f"
grep -q 'c2 = 0.0D0' "${generated_dir}/reverse/fft2c_b.f"

# Only now, after all tools and inputs have passed read-only preflight, may we
# create an isolated candidate.  The existing compiled stage is not touched
# until the candidate and all of its inputs have been reverified.
zc_assert_stage_replacement_allowed "${compile_target}" 0
compile_target_state="$(zc_capture_path_state "${compile_target}")"
compile_script_state="$(zc_capture_path_state "${BASH_SOURCE[0]}")"
build_guard_state="$(zc_capture_path_state \
  "${toolchain_dir}/scripts/build_guard.sh")"
fc_state="$(zc_capture_path_state "${fc_path}")"
cc_state="$(zc_capture_path_state "${cc_path}")"
ar_state="$(zc_capture_path_state "${ar_path}")"

compile_workspace="$(zc_create_stage_workspace "${build_dir}" compile)"
compile_workspace_identity="$(zc_path_identity "${compile_workspace}")"
compile_dir="${compile_workspace}/payload"
primal_objects="${compile_dir}/primal"
support_objects="${compile_dir}/support"
reverse_objects="${compile_dir}/reverse"
tangent_objects="${compile_dir}/tangent"
input_snapshot="${compile_dir}/build_input_snapshot"
build_input_manifest="${compile_dir}/build_input_manifest.sha256"
patch_manifest="${compile_dir}/toolchain_patch_manifest.sha256"
compile_recipe="${compile_dir}/compile_recipe.txt"
controlled_cwd="${compile_dir}/controlled_cwd"
mkdir -p "${primal_objects}" "${support_objects}" \
  "${reverse_objects}" "${tangent_objects}" "${input_snapshot}" \
  "${controlled_cwd}"

cleanup_compile_workspace() {
  local status="$?"
  trap - EXIT HUP INT TERM
  zc_remove_stage_workspace "${build_dir}" "${compile_workspace}" \
    "${compile_workspace_identity}" || status=2
  exit "${status}"
}
trap cleanup_compile_workspace EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

{
  printf 'recipe_schema=zc-coupled-compile-v1\n'
  printf 'build_mode=%s\n' "${build_mode}"
  printf 'fc_path=%s\n' "${fc_path}"
  printf 'fc_sha256=%s\n' "${fc_hash}"
  printf 'fc_version=%s\n' "${fc_version}"
  printf 'cc_path=%s\n' "${cc_path}"
  printf 'cc_sha256=%s\n' "${cc_hash}"
  printf 'cc_version=%s\n' "${cc_version}"
  printf 'ar_path=%s\n' "${ar_path}"
  printf 'ar_sha256=%s\n' "${ar_hash}"
  printf 'ar_version=%s\n' "${ar_version}"
  printf 'primal_flags=%s\n' "${primal_fflags[*]}"
  printf 'derivative_flags=%s\n' "${fflags[*]}"
  printf 'platform_flags=%s\n' "${platform_flags[*]}"
  printf 'c_flags=%s\n' "${cflags[*]}"
  printf '%s\n' \
    'c_compile=<CC> <C_FLAGS> <PLATFORM_FLAGS> -c <ADSTACK.c> -I <ADSTACK_INCLUDE> -o <ADSTACK.o>'
  printf '%s\n' \
    'controlled_compile_cwd=<COMPILE_STAGE>/controlled_cwd (initially empty)'
  printf 'primal_units=%s fft2c\n' "${primal_names[*]}"
  printf 'support_units=%s\n' "${support_names[*]}"
  printf 'tangent_audit_support_units=%s\n' \
    "${tangent_audit_support_names[*]}"
  printf 'zc_library_units=%s\n' "${zc_library_names[*]}"
  printf '%s\n' \
    'path_link=<FC> <DERIVATIVE_FLAGS> <PLATFORM_FLAGS> -o zc_adjoint_make_path <PATH_DRIVER.o> <PRIMAL_OBJECTS> <SUPPORT_OBJECTS>'
  printf '%s\n' \
    'reverse_link=<FC> <DERIVATIVE_FLAGS> <PLATFORM_FLAGS> -o zc_nino3_adjoint <REVERSE_OBJECTS> <PRIMAL_SSTA.o> <SUPPORT_OBJECTS>'
  printf '%s\n' \
    'tangent_archive=<AR> rcs libzc_tangent.a <TANGENT_OBJECTS>'
} > "${compile_recipe}"

snapshot_input() {
  local source_file="$1"
  local logical_name="$2"
  local target_file="${input_snapshot}/${logical_name}"
  if [[ -L "${source_file}" || ! -f "${source_file}" || \
        ! -s "${source_file}" ]]; then
    printf 'ERROR: build input is missing or empty: %s\n' "${source_file}" >&2
    exit 2
  fi
  mkdir -p "$(dirname "${target_file}")"
  cp -p "${source_file}" "${target_file}"
}

snapshot_input "${toolchain_dir}/coupled/zc_adjoint_path_driver.F" \
  toolchain/coupled/zc_adjoint_path_driver.F
snapshot_input "${toolchain_dir}/coupled/zc_nino3_adjoint_driver.F" \
  toolchain/coupled/zc_nino3_adjoint_driver.F
snapshot_input "${toolchain_dir}/coupled/zc_kernel_nino3.F" \
  toolchain/coupled/zc_kernel_nino3.F
snapshot_input "${toolchain_dir}/../full_tangent_audit/tangent_path_driver.F" \
  audit/tangent_path_driver.F
snapshot_input "${toolchain_dir}/../full_tangent_audit/tangent_driver.F" \
  audit/tangent_driver.F
snapshot_input "${toolchain_dir}/VERSION.env" toolchain/VERSION.env
for script_name in "${compile_script_names[@]}"; do
  snapshot_input "${toolchain_dir}/scripts/${script_name}" \
    "toolchain/scripts/${script_name}"
done
for patch_file in "${toolchain_dir}/patches/"*.patch; do
  snapshot_input "${patch_file}" \
    "toolchain/patches/$(basename "${patch_file}")"
done

# Build the patch inventory from the immutable candidate copy, never from a
# separate pass over the live patch directory. This ensures the manifest and
# every patch later bound into the compile snapshot describe identical bytes.
mkdir -p "${input_snapshot}/manifests"
(
  cd "${input_snapshot}/toolchain/patches"
  find . -type f -print | sed 's#^\./##' | LC_ALL=C sort |
    while IFS= read -r relative_name; do
      printf '%s  %s\n' "$(sha256_file "${relative_name}")" \
        "${relative_name}"
    done
) > "${input_snapshot}/manifests/toolchain_patch_manifest.sha256"
zc_verify_sha256_manifest_exact \
  "${input_snapshot}/toolchain/patches" \
  "${input_snapshot}/manifests/toolchain_patch_manifest.sha256"

while IFS= read -r source_file; do
  relative_name="${source_file#"${prepared_dir}/"}"
  snapshot_input "${source_file}" "prepared/${relative_name}"
done < <(
  find "${prepared_dir}" -type f \
    \( -name '*.f' -o -name '*.inc' -o -name '*.common' \) -print |
    LC_ALL=C sort
)
for provenance_name in upstream_preprocessed_manifest.sha256 \
  kernel_source_input_manifest.sha256 \
  prepared_source_manifest.sha256 normalization_hashes.txt \
  preparation_provenance.txt prepared_manifest_binding.txt; do
  snapshot_input "${prepared_dir}/${provenance_name}" \
    "prepared/${provenance_name}"
done

while IFS= read -r source_file; do
  relative_name="${source_file#"${generated_dir}/"}"
  snapshot_input "${source_file}" "generated/${relative_name}"
done < <(
  find "${generated_dir}" -type f \
    \( -name '*.f' -o -name '*.common' -o -name '*.msg' \) -print |
    LC_ALL=C sort
)
for provenance_name in tapenade_version.txt \
  generated_source_manifest.pre_instrument.sha256 \
  generated_source_manifest.sha256 generation_provenance.txt \
  generation_manifest_binding.txt instrumentation_provenance.txt \
  instrumentation_manifest_binding.txt; do
  snapshot_input "${generated_dir}/${provenance_name}" \
    "generated/${provenance_name}"
done

for name in "${support_names[@]}"; do
  snapshot_input "${kernel_source}/${name}.F" "kernel_source/${name}.F"
done
for name in "${tangent_audit_support_names[@]}"; do
  snapshot_input "${kernel_source}/${name}.F" "kernel_source/${name}.F"
done
for name in "${zc_library_names[@]}"; do
  snapshot_input "${kernel_source}/ZC_lib_routines/${name}.F" \
    "kernel_source/ZC_lib_routines/${name}.F"
done
for name in "${kernel_include_names[@]}"; do
  snapshot_input "${kernel_source}/${name}" "kernel_source/${name}"
done
# Preserve the complete kernel identity boundary used at preparation time,
# including the raw active sources. Only the normalized prepared copies are
# compiled; this origin snapshot proves they and the passive link inputs came
# from one kernel build.
while read -r _digest relative_name; do
  test -n "${relative_name}" || continue
  snapshot_input "${kernel_source}/${relative_name}" \
    "kernel_origin/${relative_name}"
done < "${kernel_source_manifest}"
snapshot_input "${tapenade_home}/ADFirstAidKit/adStack.c" \
  tapenade/ADFirstAidKit/adStack.c
snapshot_input "${tapenade_home}/ADFirstAidKit/adStack.h" \
  tapenade/ADFirstAidKit/adStack.h
snapshot_input "${tapenade_home}/ADFirstAidKit/adComplex.h" \
  tapenade/ADFirstAidKit/adComplex.h
snapshot_input "${tapenade_home}/.zc_tapenade_tree_manifest.sha256" \
  tapenade/.zc_tapenade_tree_manifest.sha256
snapshot_input "${tapenade_home}/.zc_tapenade_install_receipt.txt" \
  tapenade/.zc_tapenade_install_receipt.txt
snapshot_input "${compile_recipe}" manifests/compile_recipe.txt

kernel_parent="$(cd "${kernel_source}/.." && pwd)"
kernel_parent_report="${kernel_parent}/build_report.json"
if [[ -s "${kernel_parent_report}" ]]; then
  snapshot_input "${kernel_parent_report}" kernel_parent/build_report.json
fi
for provenance_name in source_manifest.sha256 source_manifest.json; do
  if [[ -s "${kernel_parent}/${provenance_name}" ]]; then
    snapshot_input "${kernel_parent}/${provenance_name}" \
      "kernel_parent/${provenance_name}"
  fi
done

(
  cd "${input_snapshot}"
  find . -type f -print | sed 's#^\./##' | LC_ALL=C sort |
    while IFS= read -r relative_name; do
      printf '%s  %s\n' "$(sha256_file "${relative_name}")" "${relative_name}"
    done
) > "${build_input_manifest}"

write_tree_inventory() {
  local inventory_root="$1"
  local inventory_file="$2"
  (
    cd "${inventory_root}"
    find . -type f -print | sed 's#^\./##' | LC_ALL=C sort |
      while IFS= read -r relative_name; do
        printf '%s  %s\n' "$(sha256_file "${relative_name}")" \
          "${relative_name}"
      done
  ) > "${inventory_file}"
}

prepared_inventory="${compile_dir}/prepared_compile_inputs.sha256"
tangent_inventory="${compile_dir}/tangent_compile_inputs.sha256"
reverse_inventory="${compile_dir}/reverse_compile_inputs.sha256"
write_tree_inventory "${input_snapshot}/prepared" "${prepared_inventory}"
write_tree_inventory "${input_snapshot}/generated/tangent" \
  "${tangent_inventory}"
write_tree_inventory "${input_snapshot}/generated/reverse" \
  "${reverse_inventory}"
zc_verify_sha256_manifest_exact "${input_snapshot}/prepared" \
  "${prepared_inventory}"
zc_verify_sha256_manifest_exact "${input_snapshot}/generated/tangent" \
  "${tangent_inventory}"
zc_verify_sha256_manifest_exact "${input_snapshot}/generated/reverse" \
  "${reverse_inventory}"

# The compiler consumes only this staged snapshot.  Remove write permission so
# ordinary build tools cannot accidentally rewrite a source or provenance
# input.  Exact state and content are checked again after every tool returns.
find "${input_snapshot}" -type f -exec chmod a-w {} +
find "${input_snapshot}" -type d -exec chmod a-w {} +
build_input_manifest_state="$(zc_capture_path_state \
  "${build_input_manifest}")"
prepared_inventory_state="$(zc_capture_path_state "${prepared_inventory}")"
tangent_inventory_state="$(zc_capture_path_state "${tangent_inventory}")"
reverse_inventory_state="$(zc_capture_path_state "${reverse_inventory}")"
snapshot_state="$(zc_capture_path_state "${input_snapshot}")"

# Every source/provenance value reported after compilation is read from the
# immutable snapshot below. The live preparation/generation trees are never
# consulted again, so concurrent edits cannot make the summary describe a
# different input set from the one actually compiled.
snapshot_prepared_manifest="${input_snapshot}/prepared/prepared_source_manifest.sha256"
snapshot_prepared_binding="${input_snapshot}/prepared/prepared_manifest_binding.txt"
snapshot_generated_manifest="${input_snapshot}/generated/generated_source_manifest.sha256"
snapshot_generated_pre_manifest="${input_snapshot}/generated/generated_source_manifest.pre_instrument.sha256"
snapshot_generation_provenance="${input_snapshot}/generated/generation_provenance.txt"
snapshot_generation_binding="${input_snapshot}/generated/generation_manifest_binding.txt"
snapshot_instrumentation_provenance="${input_snapshot}/generated/instrumentation_provenance.txt"
snapshot_instrumentation_binding="${input_snapshot}/generated/instrumentation_manifest_binding.txt"
snapshot_tapenade_version="${input_snapshot}/generated/tapenade_version.txt"
snapshot_tapenade_receipt="${input_snapshot}/tapenade/.zc_tapenade_install_receipt.txt"
snapshot_tapenade_tree_manifest="${input_snapshot}/tapenade/.zc_tapenade_tree_manifest.sha256"
snapshot_patch_manifest="${input_snapshot}/manifests/toolchain_patch_manifest.sha256"
snapshot_compile_recipe="${input_snapshot}/manifests/compile_recipe.txt"
snapshot_kernel_parent_report="${input_snapshot}/kernel_parent/build_report.json"
snapshot_version_env="${input_snapshot}/toolchain/VERSION.env"
snapshot_kernel_origin="${input_snapshot}/kernel_origin"
snapshot_kernel_source_manifest="${input_snapshot}/prepared/kernel_source_input_manifest.sha256"

verify_tapenade_snapshot_member() {
  local relative_name="$1"
  local copied_file="${input_snapshot}/tapenade/${relative_name}"
  local expected count
  expected="$(awk -v wanted="${relative_name}" \
    '$2 == wanted {print $1}' "${snapshot_tapenade_tree_manifest}")"
  count="$(awk -v wanted="${relative_name}" \
    '$2 == wanted {n += 1} END {print n + 0}' \
    "${snapshot_tapenade_tree_manifest}")"
  [[ "${count}" == 1 && -n "${expected}" ]] || {
    printf 'ERROR: Tapenade tree manifest does not uniquely bind %s\n' \
      "${relative_name}" >&2
    exit 2
  }
  [[ ! -L "${copied_file}" && -f "${copied_file}" && \
     "$(sha256_file "${copied_file}")" == "${expected}" ]] || {
    printf 'ERROR: copied Tapenade runtime input is not receipt-bound: %s\n' \
      "${relative_name}" >&2
    exit 2
  }
}

zc_verify_binding_value "${snapshot_tapenade_receipt}" receipt_schema \
  zc-tapenade-install-receipt-v1
zc_verify_binding_file_hash "${snapshot_tapenade_receipt}" \
  tree_manifest_sha256 "${snapshot_tapenade_tree_manifest}"
zc_verify_binding_file_hash "${snapshot_tapenade_receipt}" \
  installer_script_sha256 \
  "${input_snapshot}/toolchain/scripts/install_tapenade_linux.sh"
zc_verify_binding_file_hash "${snapshot_tapenade_receipt}" \
  build_guard_sha256 \
  "${input_snapshot}/toolchain/scripts/build_guard.sh"
zc_verify_binding_file_hash "${snapshot_tapenade_receipt}" \
  version_env_sha256 "${snapshot_version_env}"
verify_tapenade_snapshot_member ADFirstAidKit/adStack.c
verify_tapenade_snapshot_member ADFirstAidKit/adStack.h
verify_tapenade_snapshot_member ADFirstAidKit/adComplex.h

zc_verify_sha256_manifest_exact "${input_snapshot}/prepared" \
  "${snapshot_prepared_manifest}" prepared_source_manifest.sha256 \
  prepared_manifest_binding.txt upstream_preprocessed_manifest.sha256 \
  kernel_source_input_manifest.sha256
zc_verify_binding_file_hash "${snapshot_prepared_binding}" \
  prepared_source_manifest_sha256 "${snapshot_prepared_manifest}"
zc_verify_binding_file_hash "${snapshot_prepared_binding}" \
  kernel_source_input_manifest_sha256 "${snapshot_kernel_source_manifest}"
zc_verify_sha256_manifest "${snapshot_kernel_origin}" \
  "${snapshot_kernel_source_manifest}" >/dev/null
for name in "${support_names[@]}" "${tangent_audit_support_names[@]}"; do
  [[ "$(sha256_file "${input_snapshot}/kernel_source/${name}.F")" == \
    "$(sha256_file "${snapshot_kernel_origin}/${name}.F")" ]] || {
    printf 'ERROR: passive source changed while creating build snapshot: %s.F\n' \
      "${name}" >&2
    exit 2
  }
done
for name in "${zc_library_names[@]}"; do
  [[ "$(sha256_file \
    "${input_snapshot}/kernel_source/ZC_lib_routines/${name}.F")" == \
    "$(sha256_file \
      "${snapshot_kernel_origin}/ZC_lib_routines/${name}.F")" ]] || {
    printf 'ERROR: passive library changed while creating snapshot: %s.F\n' \
      "${name}" >&2
    exit 2
  }
done
for name in "${kernel_include_names[@]}"; do
  [[ "$(sha256_file "${input_snapshot}/kernel_source/${name}")" == \
    "$(sha256_file "${snapshot_kernel_origin}/${name}")" ]] || {
    printf 'ERROR: passive include changed while creating snapshot: %s\n' \
      "${name}" >&2
    exit 2
  }
done
zc_verify_sha256_manifest_exact "${input_snapshot}/generated" \
  "${snapshot_generated_manifest}" \
  generated_source_manifest.pre_instrument.sha256 \
  generated_source_manifest.sha256 generation_provenance.txt \
  generation_manifest_binding.txt instrumentation_provenance.txt \
  instrumentation_manifest_binding.txt
zc_verify_binding_file_hash "${snapshot_generation_binding}" \
  prepared_source_manifest_sha256 "${snapshot_prepared_manifest}"
zc_verify_binding_file_hash "${snapshot_generation_binding}" \
  generated_pre_manifest_sha256 "${snapshot_generated_pre_manifest}"
zc_verify_binding_file_hash "${snapshot_generation_binding}" \
  generation_provenance_sha256 "${snapshot_generation_provenance}"
zc_verify_binding_value "${snapshot_generation_provenance}" \
  prepared_source_manifest_sha256 \
  "$(sha256_file "${snapshot_prepared_manifest}")"
zc_verify_binding_value "${snapshot_generation_provenance}" \
  tapenade_install_receipt_sha256 \
  "$(sha256_file "${snapshot_tapenade_receipt}")"
zc_verify_binding_value "${snapshot_generation_provenance}" \
  tapenade_tree_manifest_sha256 \
  "$(sha256_file "${snapshot_tapenade_tree_manifest}")"
zc_verify_binding_value "${snapshot_generation_binding}" \
  tapenade_install_receipt_sha256 \
  "$(sha256_file "${snapshot_tapenade_receipt}")"
zc_verify_binding_value "${snapshot_generation_binding}" \
  tapenade_tree_manifest_sha256 \
  "$(sha256_file "${snapshot_tapenade_tree_manifest}")"
zc_verify_binding_file_hash "${snapshot_instrumentation_binding}" \
  generated_source_manifest_sha256 "${snapshot_generated_manifest}"
zc_verify_binding_file_hash "${snapshot_instrumentation_binding}" \
  generation_manifest_binding_sha256 "${snapshot_generation_binding}"
zc_verify_binding_file_hash "${snapshot_instrumentation_binding}" \
  instrumentation_provenance_sha256 \
  "${snapshot_instrumentation_provenance}"
zc_verify_binding_file_hash "${snapshot_instrumentation_binding}" \
  generated_pre_manifest_sha256 "${snapshot_generated_pre_manifest}"
zc_verify_binding_value "${snapshot_instrumentation_provenance}" \
  pre_instrument_manifest_sha256 \
  "$(sha256_file "${snapshot_generated_pre_manifest}")"
zc_verify_binding_value "${snapshot_instrumentation_provenance}" \
  generation_manifest_binding_sha256 \
  "$(sha256_file "${snapshot_generation_binding}")"
zc_verify_binding_file_hash \
  "${input_snapshot}/prepared/preparation_provenance.txt" script_sha256 \
  "${input_snapshot}/toolchain/scripts/prepare_coupled_sources.sh"
zc_verify_binding_file_hash "${snapshot_generation_provenance}" \
  script_sha256 \
  "${input_snapshot}/toolchain/scripts/generate_coupled_nino3.sh"
zc_verify_binding_file_hash "${snapshot_instrumentation_provenance}" \
  script_sha256 \
  "${input_snapshot}/toolchain/scripts/instrument_generated_reverse.sh"
[[ "$(sha256_file \
  "${input_snapshot}/toolchain/coupled/zc_kernel_nino3.F")" == \
  "$(sha256_file "${input_snapshot}/prepared/zc_kernel_nino3.f")" ]] || {
  printf 'ERROR: scalar head changed while creating build snapshot\n' >&2
  exit 2
}
for patch_name in "${normalization_patch_names[@]}"; do
  zc_verify_binding_file_hash \
    "${input_snapshot}/prepared/preparation_provenance.txt" \
    "patch_${patch_name}_sha256" \
    "${input_snapshot}/toolchain/patches/${patch_name}"
done
for patch_name in "${instrumentation_patch_names[@]}"; do
  zc_verify_binding_file_hash "${snapshot_instrumentation_provenance}" \
    "patch_${patch_name}_sha256" \
    "${input_snapshot}/toolchain/patches/${patch_name}"
done

verify_compile_inputs_unchanged() {
  zc_verify_path_state "${input_snapshot}" "${snapshot_state}"
  zc_verify_sha256_manifest_exact "${input_snapshot}" \
    "${build_input_manifest}"
  zc_verify_path_state "${build_input_manifest}" \
    "${build_input_manifest_state}"
  zc_verify_sha256_manifest_exact "${input_snapshot}/prepared" \
    "${prepared_inventory}"
  zc_verify_sha256_manifest_exact "${input_snapshot}/generated/tangent" \
    "${tangent_inventory}"
  zc_verify_sha256_manifest_exact "${input_snapshot}/generated/reverse" \
    "${reverse_inventory}"
  zc_verify_path_state "${prepared_inventory}" \
    "${prepared_inventory_state}"
  zc_verify_path_state "${tangent_inventory}" \
    "${tangent_inventory_state}"
  zc_verify_path_state "${reverse_inventory}" \
    "${reverse_inventory_state}"
  zc_verify_sha256_manifest_exact \
    "${input_snapshot}/toolchain/patches" \
    "${snapshot_patch_manifest}"
  zc_verify_path_state "${BASH_SOURCE[0]}" "${compile_script_state}"
  zc_verify_path_state "${toolchain_dir}/scripts/build_guard.sh" \
    "${build_guard_state}"
  zc_verify_path_state "${fc_path}" "${fc_state}"
  zc_verify_path_state "${cc_path}" "${cc_state}"
  zc_verify_path_state "${ar_path}" "${ar_state}"
}

verify_compile_inputs_unchanged

# Compile only from the immutable snapshot whose hashes appear in the build
# manifest.  This closes the gap where an original source could change after
# provenance capture but before compilation or linking.
compile_prepared_dir="${input_snapshot}/prepared"
compile_generated_dir="${input_snapshot}/generated"
compile_kernel_source="${input_snapshot}/kernel_source"
compile_tapenade_home="${input_snapshot}/tapenade"
compile_toolchain="${input_snapshot}/toolchain"

compile_fortran() {
  local source_file="$1"
  local output_file="$2"
  shift 2
  (cd "${controlled_cwd}" && \
    "${fc_path}" "${fflags[@]}" "${platform_flags[@]}" "$@" \
      -c "${source_file}" -o "${output_file}")
}

compile_primal_fortran() {
  local source_file="$1"
  local output_file="$2"
  shift 2
  (cd "${controlled_cwd}" && \
    "${fc_path}" "${primal_fflags[@]}" "${platform_flags[@]}" "$@" \
      -c "${source_file}" -o "${output_file}")
}

for name in "${primal_names[@]}"; do
  compile_primal_fortran "${compile_prepared_dir}/${name}.f" \
    "${primal_objects}/${name}.o" -I "${compile_prepared_dir}"
done
compile_primal_fortran "${compile_prepared_dir}/ZC_lib_routines/fft2c.f" \
  "${primal_objects}/fft2c.o" -I "${compile_prepared_dir}"

for name in "${support_names[@]}"; do
  compile_primal_fortran "${compile_kernel_source}/${name}.F" \
    "${support_objects}/${name}.o" -I "${compile_kernel_source}"
done
for name in "${zc_library_names[@]}"; do
  compile_primal_fortran \
    "${compile_kernel_source}/ZC_lib_routines/${name}.F" \
    "${support_objects}/${name}.o" -I "${compile_kernel_source}"
done
(cd "${controlled_cwd}" && \
  "${cc_path}" "${cflags[@]}" "${platform_flags[@]}" -c \
    "${compile_tapenade_home}/ADFirstAidKit/adStack.c" \
    -I "${compile_tapenade_home}/ADFirstAidKit" \
    -o "${support_objects}/adStack.o")

compile_fortran "${compile_toolchain}/coupled/zc_adjoint_path_driver.F" \
  "${primal_objects}/zc_adjoint_path_driver.o" \
  -I "${compile_prepared_dir}"
(cd "${controlled_cwd}" && \
  "${fc_path}" "${fflags[@]}" "${platform_flags[@]}" \
    -o "${compile_dir}/zc_adjoint_make_path" \
    "${primal_objects}/zc_adjoint_path_driver.o" \
    "${primal_objects}/zc_kernel_api.o" "${primal_objects}/ssta.o" \
    "${primal_objects}/ztmfc1.o" "${primal_objects}/cforce.o" \
    "${primal_objects}/mloop.o" "${primal_objects}/akcalc.o" \
    "${primal_objects}/bndary.o" "${primal_objects}/uhcalc.o" \
    "${primal_objects}/uhinit.o" "${primal_objects}/tridag.o" \
    "${primal_objects}/fft2c.o" "${support_objects}/"*.o)

for source_file in "${compile_generated_dir}/reverse/"*.f; do
  name="$(basename "${source_file}" .f)"
  compile_fortran "${source_file}" "${reverse_objects}/${name}.o" \
    -I "${compile_generated_dir}/reverse" -I "${compile_prepared_dir}"
done
compile_fortran "${compile_toolchain}/coupled/zc_nino3_adjoint_driver.F" \
  "${reverse_objects}/zc_nino3_adjoint_driver.o" \
  -I "${compile_generated_dir}/reverse" -I "${compile_prepared_dir}"
(cd "${controlled_cwd}" && \
  "${fc_path}" "${fflags[@]}" "${platform_flags[@]}" \
    -o "${compile_dir}/zc_nino3_adjoint" \
    "${reverse_objects}/"*.o "${primal_objects}/ssta.o" \
    "${support_objects}/"*.o)

for source_file in "${compile_generated_dir}/tangent/"*.f; do
  name="$(basename "${source_file}" .f)"
  compile_fortran "${source_file}" "${tangent_objects}/${name}.o" \
    -I "${compile_generated_dir}/tangent" -I "${compile_prepared_dir}"
done
(cd "${controlled_cwd}" && \
  "${ar_path}" rcs "${compile_dir}/libzc_tangent.a" \
    "${tangent_objects}/"*.o)

zc_test_failpoint compile_after_tool_execution
verify_compile_inputs_unchanged
cp "${snapshot_patch_manifest}" "${patch_manifest}"

{
  printf 'build_mode=%s\n' "${build_mode}"
  printf 'primal_flags=%s\n' "${primal_fflags[*]}"
  printf 'derivative_flags=%s\n' "${fflags[*]}"
  printf 'platform_flags=%s\n' "${platform_flags[*]}"
  printf 'compiler=%s\n' "${fc_version}"
  printf 'compiler_path=%s\n' "${fc_path}"
  printf 'compiler_sha256=%s\n' "${fc_hash}"
  printf 'c_compiler=%s\n' "${cc_version}"
  printf 'c_compiler_path=%s\n' "${cc_path}"
  printf 'c_compiler_sha256=%s\n' "${cc_hash}"
  printf 'c_flags=%s\n' "${cflags[*]}"
  printf '%s\n' \
    'controlled_compile_cwd=<COMPILE_STAGE>/controlled_cwd (initially empty)'
  printf 'archiver=%s\n' "${ar_version}"
  printf 'archiver_path=%s\n' "${ar_path}"
  printf 'archiver_sha256=%s\n' "${ar_hash}"
  printf 'tapenade_version=%s\n' \
    "$(head -n 1 "${snapshot_tapenade_version}")"
  printf 'tapenade_revision=%s\n' \
    "$(awk -F= '$1=="TAPENADE_REVISION" {print $2}' \
      "${snapshot_version_env}")"
  printf 'tapenade_archive_sha256=%s\n' \
    "$(awk -F= '$1=="TAPENADE_ARCHIVE_SHA256" {print $2}' \
      "${snapshot_version_env}")"
  printf 'tapenade_install_receipt_sha256=%s\n' \
    "$(sha256_file "${snapshot_tapenade_receipt}")"
  printf 'tapenade_tree_manifest_sha256=%s\n' \
    "$(sha256_file "${snapshot_tapenade_tree_manifest}")"
  printf 'prepared_source_manifest_sha256=%s\n' \
    "$(sha256_file "${snapshot_prepared_manifest}")"
  printf 'generated_source_manifest_sha256=%s\n' \
    "$(sha256_file "${snapshot_generated_manifest}")"
  printf 'prepared_manifest_binding_sha256=%s\n' \
    "$(sha256_file "${snapshot_prepared_binding}")"
  printf 'generation_manifest_binding_sha256=%s\n' \
    "$(sha256_file "${snapshot_generation_binding}")"
  printf 'instrumentation_manifest_binding_sha256=%s\n' \
    "$(sha256_file "${snapshot_instrumentation_binding}")"
  printf 'build_input_manifest_sha256=%s\n' \
    "$(sha256_file "${build_input_manifest}")"
  printf 'prepared_compile_inputs_sha256=%s\n' \
    "$(sha256_file "${prepared_inventory}")"
  printf '%s\n' \
    'prepared_compile_inputs_root=build_input_snapshot/prepared'
  printf 'tangent_compile_inputs_sha256=%s\n' \
    "$(sha256_file "${tangent_inventory}")"
  printf '%s\n' \
    'tangent_compile_inputs_root=build_input_snapshot/generated/tangent'
  printf 'reverse_compile_inputs_sha256=%s\n' \
    "$(sha256_file "${reverse_inventory}")"
  printf '%s\n' \
    'reverse_compile_inputs_root=build_input_snapshot/generated/reverse'
  printf 'toolchain_patch_manifest_sha256=%s\n' \
    "$(sha256_file "${snapshot_patch_manifest}")"
  printf 'compile_recipe_sha256=%s\n' \
    "$(sha256_file "${snapshot_compile_recipe}")"
  if [[ -s "${snapshot_kernel_parent_report}" ]]; then
    printf 'kernel_parent_build_report_sha256=%s\n' \
      "$(sha256_file "${snapshot_kernel_parent_report}")"
  else
    printf 'kernel_parent_build_report_sha256=unavailable\n'
  fi
  printf 'path_executable_sha256=%s\n' \
    "$(sha256_file "${compile_dir}/zc_adjoint_make_path")"
  printf 'adjoint_executable_sha256=%s\n' \
    "$(sha256_file "${compile_dir}/zc_nino3_adjoint")"
  printf 'tangent_archive_sha256=%s\n' \
    "$(sha256_file "${compile_dir}/libzc_tangent.a")"
} > "${compile_dir}/build_manifest.txt"

for required_output in \
  "${compile_dir}/zc_adjoint_make_path" \
  "${compile_dir}/zc_nino3_adjoint" \
  "${compile_dir}/libzc_tangent.a" \
  "${compile_dir}/build_manifest.txt" \
  "${build_input_manifest}" \
  "${patch_manifest}" \
  "${prepared_inventory}" "${tangent_inventory}" \
  "${reverse_inventory}"; do
  [[ ! -L "${required_output}" && -s "${required_output}" ]] || {
    printf 'ERROR: compiled output is missing, empty, or symlinked: %s\n' \
      "${required_output}" >&2
    exit 2
  }
done
[[ "$(sha256_file "${patch_manifest}")" == \
   "$(sha256_file "${snapshot_patch_manifest}")" ]] || {
  printf 'ERROR: published patch manifest differs from compiled snapshot\n' >&2
  exit 2
}

# Reverify at the last possible point, then atomically replace only the exact
# target state observed before compilation.  A concurrent replacement is left
# untouched and causes publication to fail.
verify_compile_inputs_unchanged
zc_verify_path_state "${compile_target}" "${compile_target_state}"
zc_test_failpoint compile_before_publication
zc_publish_stage_directory "${build_dir}" "${compile_target}" \
  "${compile_dir}" "${compile_target_state}" standard
compile_dir=""
trap - EXIT HUP INT TERM
zc_remove_stage_workspace "${build_dir}" "${compile_workspace}" \
  "${compile_workspace_identity}"
compile_workspace=""

printf 'Compiled coupled path/adjoint executables in %s\n' \
  "${compile_target}"
