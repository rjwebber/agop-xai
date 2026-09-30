#!/usr/bin/env bash
set -euo pipefail

toolchain_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
requested_reverse_dir="${1:?usage: instrument_generated_reverse.sh REVERSE_DIR}"
reverse_dir="${requested_reverse_dir}"
# shellcheck disable=SC1091
source "${toolchain_dir}/scripts/build_guard.sh"
if [[ -L "${requested_reverse_dir}" ]]; then
  printf 'ERROR: reverse directory may not be a symbolic link: %s\n' \
    "${requested_reverse_dir}" >&2
  exit 2
fi
reverse_dir="$(zc_resolve_path "${reverse_dir}")"
generated_dir="$(dirname "${reverse_dir}")"
build_dir="$(dirname "${generated_dir}")"
prepared_dir="${build_dir}/coupled_prepared"
if [[ "$(basename "${reverse_dir}")" != reverse ]] ||
   [[ "$(basename "${generated_dir}")" != coupled_generated ]]; then
  printf 'ERROR: reverse directory must be BUILD/coupled_generated/reverse\n' >&2
  exit 2
fi
claimed_build="$(zc_claim_build_dir "${toolchain_dir}" "${build_dir}")"
[[ "${claimed_build}" == "${build_dir}" ]]
command -v patch >/dev/null || {
  printf 'ERROR: patch is required for reverse instrumentation\n' >&2
  exit 2
}
patch_path="$(zc_resolve_path "$(command -v patch)")"
patch_state="$(zc_capture_path_state "${patch_path}")"

# Authenticate the immutable Tapenade output and its complete upstream chain
# before applying either in-place instrumentation patch.
pre_manifest="${generated_dir}/generated_source_manifest.pre_instrument.sha256"
generation_provenance="${generated_dir}/generation_provenance.txt"
generation_binding="${generated_dir}/generation_manifest_binding.txt"
prepared_manifest="${prepared_dir}/prepared_source_manifest.sha256"
prepared_binding="${prepared_dir}/prepared_manifest_binding.txt"
preparation_provenance="${prepared_dir}/preparation_provenance.txt"
upstream_manifest="${prepared_dir}/upstream_preprocessed_manifest.sha256"
kernel_source_manifest="${prepared_dir}/kernel_source_input_manifest.sha256"
zc_verify_sha256_manifest_exact "${generated_dir}" "${pre_manifest}" \
  generated_source_manifest.pre_instrument.sha256 \
  generation_provenance.txt generation_manifest_binding.txt
zc_verify_binding_value "${generation_binding}" binding_schema \
  zc-generation-manifest-binding-v1
zc_verify_binding_file_hash "${generation_binding}" \
  generated_pre_manifest_sha256 "${pre_manifest}"
zc_verify_binding_file_hash "${generation_binding}" \
  generation_provenance_sha256 "${generation_provenance}"
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
zc_verify_binding_file_hash "${generation_binding}" \
  prepared_source_manifest_sha256 "${prepared_manifest}"
zc_verify_binding_file_hash "${generation_binding}" \
  prepared_manifest_binding_sha256 "${prepared_binding}"
zc_verify_binding_file_hash "${generation_provenance}" \
  prepared_source_manifest_sha256 "${prepared_manifest}"
zc_verify_binding_file_hash "${generation_provenance}" \
  prepared_manifest_binding_sha256 "${prepared_binding}"
zc_verify_binding_value "${generation_binding}" \
  tapenade_install_receipt_sha256 \
  "$(zc_binding_value "${generation_provenance}" \
    tapenade_install_receipt_sha256)"
zc_verify_binding_value "${generation_binding}" \
  tapenade_tree_manifest_sha256 \
  "$(zc_binding_value "${generation_provenance}" \
    tapenade_tree_manifest_sha256)"

for patch_name in generated_reverse_branch_tape.patch \
  generated_reverse_defined_scratch.patch; do
  test -s "${toolchain_dir}/patches/${patch_name}" || {
    printf 'ERROR: required generated-source patch is missing: %s\n' \
      "${patch_name}" >&2
    exit 2
  }
done

for required in zc_kernel_api_b.f ssta_b.f ztmfc1_b.f cforce_b.f \
  fft2c_b.f tridag_b.f uhinit_b.f uhcalc_b.f; do
  test -s "${reverse_dir}/${required}" || {
    printf 'ERROR: generated reverse source is missing: %s\n' \
      "${reverse_dir}/${required}" >&2
    exit 2
  }
done
if grep -q 'tape(3) = kernel_atm_reset' \
   "${reverse_dir}/zc_kernel_api_b.f"; then
  printf 'ERROR: reverse sources are already instrumented: %s\n' \
    "${reverse_dir}" >&2
  exit 2
fi

# Patch an immutable copied generation tree with immutable copied patches.
# The original stage is left untouched until the completed instrumented tree
# and its provenance have passed all checks.
generated_target="${generated_dir}"
prepared_live="${prepared_dir}"
generated_target_state="$(zc_capture_path_state "${generated_target}")"
prepared_live_state="$(zc_capture_path_state "${prepared_live}")"
script_state="$(zc_capture_path_state "${BASH_SOURCE[0]}")"
guard_state="$(zc_capture_path_state \
  "${toolchain_dir}/scripts/build_guard.sh")"
workspace="$(zc_create_stage_workspace "${build_dir}" instrument)"
workspace_identity="$(zc_path_identity "${workspace}")"
generated_dir="${workspace}/payload"
reverse_dir="${generated_dir}/reverse"
input_snapshot="${workspace}/inputs"
prepared_dir="${input_snapshot}/prepared"
generated_input="${input_snapshot}/generated_pre"
snapshot_toolchain="${input_snapshot}/toolchain"
mkdir -p "${generated_dir}" "${prepared_dir}" \
  "${generated_input}" "${snapshot_toolchain}/scripts" \
  "${snapshot_toolchain}/patches"

cleanup_instrument_workspace() {
  local status="$?"
  trap - EXIT HUP INT TERM
  zc_remove_stage_workspace "${build_dir}" "${workspace}" \
    "${workspace_identity}" || status=2
  exit "${status}"
}
trap cleanup_instrument_workspace EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

cp -Rp "${generated_target}/." "${generated_input}/"
cp -Rp "${generated_input}/." "${generated_dir}/"
cp -Rp "${prepared_live}/." "${prepared_dir}/"
for patch_name in generated_reverse_branch_tape.patch \
  generated_reverse_defined_scratch.patch; do
  cp "${toolchain_dir}/patches/${patch_name}" \
    "${snapshot_toolchain}/patches/${patch_name}"
done
cp "${toolchain_dir}/scripts/instrument_generated_reverse.sh" \
  "${snapshot_toolchain}/scripts/instrument_generated_reverse.sh"
cp "${toolchain_dir}/scripts/build_guard.sh" \
  "${snapshot_toolchain}/scripts/build_guard.sh"
zc_verify_path_state "${generated_target}" "${generated_target_state}"
zc_verify_path_state "${prepared_live}" "${prepared_live_state}"

pre_manifest="${generated_dir}/generated_source_manifest.pre_instrument.sha256"
generation_provenance="${generated_dir}/generation_provenance.txt"
generation_binding="${generated_dir}/generation_manifest_binding.txt"
prepared_manifest="${prepared_dir}/prepared_source_manifest.sha256"
prepared_binding="${prepared_dir}/prepared_manifest_binding.txt"
preparation_provenance="${prepared_dir}/preparation_provenance.txt"
upstream_manifest="${prepared_dir}/upstream_preprocessed_manifest.sha256"
kernel_source_manifest="${prepared_dir}/kernel_source_input_manifest.sha256"
zc_verify_sha256_manifest_exact "${generated_dir}" "${pre_manifest}" \
  generated_source_manifest.pre_instrument.sha256 \
  generation_provenance.txt generation_manifest_binding.txt
zc_verify_sha256_manifest_exact "${generated_input}" \
  "${generated_input}/generated_source_manifest.pre_instrument.sha256" \
  generated_source_manifest.pre_instrument.sha256 \
  generation_provenance.txt generation_manifest_binding.txt
zc_verify_sha256_manifest_exact "${prepared_dir}" "${prepared_manifest}" \
  prepared_source_manifest.sha256 prepared_manifest_binding.txt \
  upstream_preprocessed_manifest.sha256 kernel_source_input_manifest.sha256
find "${input_snapshot}" -type f -exec chmod a-w {} +
find "${input_snapshot}" -type d -exec chmod a-w {} +
generated_input_state="$(zc_capture_path_state "${generated_input}")"
prepared_input_state="$(zc_capture_path_state "${prepared_dir}")"
patch_one_state="$(zc_capture_path_state \
  "${snapshot_toolchain}/patches/generated_reverse_branch_tape.patch")"
patch_two_state="$(zc_capture_path_state \
  "${snapshot_toolchain}/patches/generated_reverse_defined_scratch.patch")"
tool_script_state="$(zc_capture_path_state \
  "${snapshot_toolchain}/scripts/instrument_generated_reverse.sh")"
tool_guard_state="$(zc_capture_path_state \
  "${snapshot_toolchain}/scripts/build_guard.sh")"

"${patch_path}" --quiet --fuzz=0 -V none \
  --directory "${reverse_dir}" --strip 0 \
  < "${snapshot_toolchain}/patches/generated_reverse_branch_tape.patch"
"${patch_path}" --quiet --fuzz=0 -V none \
  --directory "${reverse_dir}" --strip 0 \
  < "${snapshot_toolchain}/patches/generated_reverse_defined_scratch.patch"

# BSD and GNU patch can otherwise leave backup copies when a hunk applies with
# offsets.  They are not build inputs and made prior manifests platform-
# dependent.  The generated tree has a strict intentional-artifact contract.
find "${reverse_dir}" -type f \( -name '*.orig' -o -name '*~' \) -delete
if find "${reverse_dir}" -type f -name '*.rej' -print -quit | grep -q .; then
  printf 'ERROR: rejected generated-source patch remains in %s\n' \
    "${reverse_dir}" >&2
  exit 2
fi

grep -q 'kernel_nsst = nsst' "${reverse_dir}/ssta_b.f"
grep -q 'kernel_atm_iter = ad_count' "${reverse_dir}/ztmfc1_b.f"
test "$(grep -c 'kernel_atm_reset = 1' "${reverse_dir}/ztmfc1_b.f")" -eq 2
grep -q 'tape(3) = kernel_atm_reset' "${reverse_dir}/zc_kernel_api_b.f"
grep -q 'h_minus_eng = 0.0' "${reverse_dir}/ssta_b.f"
grep -q 'beta_factor(i, j) = 0.0' "${reverse_dir}/ztmfc1_b.f"
grep -q 'a = 0.0' "${reverse_dir}/ztmfc1_b.f"
grep -q 'at = 0.0' "${reverse_dir}/ztmfc1_b.f"
grep -q 'rx = 0.0' "${reverse_dir}/cforce_b.f"
grep -q 'c2 = 0.0D0' "${reverse_dir}/fft2c_b.f"
grep -q 'dn = 0.0' "${reverse_dir}/tridag_b.f"
grep -q 'd(j) = 0.0' "${reverse_dir}/uhinit_b.f"
grep -q 'r1(j) = 0.0' "${reverse_dir}/uhcalc_b.f"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
{
  printf 'stage=instrument_generated_reverse\n'
  printf 'script_sha256=%s\n' \
    "$(sha256_file \
      "${snapshot_toolchain}/scripts/instrument_generated_reverse.sh")"
  printf 'build_guard_sha256=%s\n' \
    "$(sha256_file "${snapshot_toolchain}/scripts/build_guard.sh")"
  printf 'patch_program=%s\n' "${patch_path}"
  printf 'patch_sha256=%s\n' "$(sha256_file "${patch_path}")"
  printf 'patch_version=%s\n' \
    "$("${patch_path}" --version 2>&1 | head -n 1)"
  printf '%s\n' \
    'patch_flags=--quiet --fuzz=0 -V none --directory <REVERSE> --strip 0'
  printf 'pre_instrument_manifest_sha256=%s\n' \
    "$(sha256_file "${pre_manifest}")"
  printf 'generation_provenance_sha256=%s\n' \
    "$(sha256_file "${generation_provenance}")"
  printf 'generation_manifest_binding_sha256=%s\n' \
    "$(sha256_file "${generation_binding}")"
  printf 'prepared_source_manifest_sha256=%s\n' \
    "$(sha256_file "${prepared_manifest}")"
  printf 'prepared_manifest_binding_sha256=%s\n' \
    "$(sha256_file "${prepared_binding}")"
  for patch_name in generated_reverse_branch_tape.patch \
    generated_reverse_defined_scratch.patch; do
    printf 'patch_%s_sha256=%s\n' "${patch_name}" \
      "$(sha256_file "${snapshot_toolchain}/patches/${patch_name}")"
  done
} > "${generated_dir}/instrumentation_provenance.txt"
manifest_inputs="$(
  cd "${generated_dir}"
  find tangent reverse -type f \
    \( -name '*.f' -o -name '*.common' -o -name '*.msg' \) -print
  printf '%s\n' tapenade_version.txt
)"
{
  while IFS= read -r relative_name; do
    test -n "${relative_name}" || continue
    printf '%s  %s\n' \
      "$(sha256_file "${generated_dir}/${relative_name}")" "${relative_name}"
  done <<<"$(LC_ALL=C sort <<<"${manifest_inputs}")"
} > "${generated_dir}/generated_source_manifest.sha256"
if grep -Eq '\.(orig|rej)( |$)|~( |$)' \
   "${generated_dir}/generated_source_manifest.sha256"; then
  printf 'ERROR: backup artifact entered generated-source manifest\n' >&2
  exit 2
fi
{
  printf 'binding_schema=zc-instrumentation-manifest-binding-v1\n'
  printf 'generated_source_manifest_sha256=%s\n' \
    "$(sha256_file "${generated_dir}/generated_source_manifest.sha256")"
  printf 'instrumentation_provenance_sha256=%s\n' \
    "$(sha256_file "${generated_dir}/instrumentation_provenance.txt")"
  printf 'generation_manifest_binding_sha256=%s\n' \
    "$(sha256_file "${generation_binding}")"
  printf 'generated_pre_manifest_sha256=%s\n' \
    "$(sha256_file "${pre_manifest}")"
} > "${generated_dir}/instrumentation_manifest_binding.txt"
zc_verify_sha256_manifest_exact "${generated_dir}" \
  "${generated_dir}/generated_source_manifest.sha256" \
  generated_source_manifest.pre_instrument.sha256 \
  generated_source_manifest.sha256 generation_provenance.txt \
  generation_manifest_binding.txt instrumentation_provenance.txt \
  instrumentation_manifest_binding.txt
zc_verify_binding_file_hash \
  "${generated_dir}/instrumentation_manifest_binding.txt" \
  generated_source_manifest_sha256 \
  "${generated_dir}/generated_source_manifest.sha256"
zc_verify_binding_file_hash \
  "${generated_dir}/instrumentation_manifest_binding.txt" \
  instrumentation_provenance_sha256 \
  "${generated_dir}/instrumentation_provenance.txt"

# The pre-instrument generation, preparation, patch, and executing-script
# copies remain immutable throughout patching. Also ensure neither live input
# stage changed before the atomic transition replaces the generated stage.
zc_test_failpoint instrument_after_tool_execution
zc_verify_path_state "${generated_input}" "${generated_input_state}"
zc_verify_sha256_manifest_exact "${generated_input}" \
  "${generated_input}/generated_source_manifest.pre_instrument.sha256" \
  generated_source_manifest.pre_instrument.sha256 \
  generation_provenance.txt generation_manifest_binding.txt
zc_verify_path_state "${prepared_dir}" "${prepared_input_state}"
zc_verify_sha256_manifest_exact "${prepared_dir}" "${prepared_manifest}" \
  prepared_source_manifest.sha256 prepared_manifest_binding.txt \
  upstream_preprocessed_manifest.sha256 kernel_source_input_manifest.sha256
zc_verify_path_state \
  "${snapshot_toolchain}/patches/generated_reverse_branch_tape.patch" \
  "${patch_one_state}"
zc_verify_path_state \
  "${snapshot_toolchain}/patches/generated_reverse_defined_scratch.patch" \
  "${patch_two_state}"
zc_verify_path_state \
  "${snapshot_toolchain}/scripts/instrument_generated_reverse.sh" \
  "${tool_script_state}"
zc_verify_path_state "${snapshot_toolchain}/scripts/build_guard.sh" \
  "${tool_guard_state}"
zc_verify_path_state "${BASH_SOURCE[0]}" "${script_state}"
zc_verify_path_state "${toolchain_dir}/scripts/build_guard.sh" \
  "${guard_state}"
zc_verify_path_state "${patch_path}" "${patch_state}"
zc_verify_path_state "${generated_target}" "${generated_target_state}"
zc_verify_path_state "${prepared_live}" "${prepared_live_state}"
zc_publish_stage_directory "${build_dir}" "${generated_target}" \
  "${generated_dir}" "${generated_target_state}" transition

trap - EXIT HUP INT TERM
zc_remove_stage_workspace "${build_dir}" "${workspace}" \
  "${workspace_identity}"
workspace=""
generated_dir="${generated_target}"
reverse_dir="${generated_dir}/reverse"
printf 'Instrumented generated reverse branch-tape outputs in %s\n' \
  "${reverse_dir}"
