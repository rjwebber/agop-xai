#!/usr/bin/env bash
set -euo pipefail

toolchain_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${toolchain_dir}/VERSION.env"
requested_tapenade_home="${1:?usage: generate_coupled_nino3.sh TAPENADE_HOME [BUILD_DIR]}"
tapenade_home="${requested_tapenade_home}"
build_dir="${2:-${toolchain_dir}/build/coupled_release}"
prepared_dir="${build_dir}/coupled_prepared"
generated_dir="${build_dir}/coupled_generated"
tapenade="${tapenade_home}/bin/tapenade"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# shellcheck disable=SC1091
source "${toolchain_dir}/scripts/build_guard.sh"
if [[ -L "${requested_tapenade_home}" ]]; then
  printf 'ERROR: Tapenade home may not be a symbolic link: %s\n' \
    "${requested_tapenade_home}" >&2
  exit 2
fi
tapenade_home="$(zc_resolve_path "${tapenade_home}")"
tapenade="${tapenade_home}/bin/tapenade"
build_dir="$(zc_claim_build_dir "${toolchain_dir}" "${build_dir}" \
  "${tapenade_home}")"
prepared_dir="${build_dir}/coupled_prepared"
generated_dir="${build_dir}/coupled_generated"

for required in "${tapenade}" \
  "${tapenade_home}/bin/linux/fortranParser"; do
  test -x "${required}" || {
    printf 'ERROR: required Tapenade executable is missing: %s\n' \
      "${required}" >&2
    exit 2
  }
done
java_path="$(command -v java)" || {
  printf 'ERROR: Java 17 is required for Tapenade generation\n' >&2
  exit 2
}
java_path="$(zc_resolve_path "${java_path}")"
java_version="$("${java_path}" -version 2>&1 | head -n 1)"
if ! grep -Eq 'version "17([.]|")' <<<"${java_version}"; then
  printf 'ERROR: Java 17 is required; observed: %s\n' "${java_version}" >&2
  exit 2
fi
zc_verify_tapenade_install "${tapenade_home}" "${toolchain_dir}" \
  "${TAPENADE_VERSION}" "${TAPENADE_REVISION}" \
  "${TAPENADE_ARCHIVE_SHA256}"
version_output="$("${tapenade}" -version 2>&1)"
grep -Fq "Tapenade ${TAPENADE_VERSION}" <<<"${version_output}"
grep -Fq "Revision: ${TAPENADE_REVISION}" <<<"${version_output}"

# Validate the complete generation input before replacing an existing stage.
prepared_manifest="${prepared_dir}/prepared_source_manifest.sha256"
prepared_binding="${prepared_dir}/prepared_manifest_binding.txt"
preparation_provenance="${prepared_dir}/preparation_provenance.txt"
upstream_manifest="${prepared_dir}/upstream_preprocessed_manifest.sha256"
kernel_source_manifest="${prepared_dir}/kernel_source_input_manifest.sha256"
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
zc_verify_binding_file_hash "${preparation_provenance}" \
  scalar_head_source_sha256 \
  "${toolchain_dir}/coupled/zc_kernel_nino3.F"
[[ "$(sha256_file "${toolchain_dir}/coupled/zc_kernel_nino3.F")" == \
  "$(sha256_file "${prepared_dir}/zc_kernel_nino3.f")" ]] || {
  printf 'ERROR: prepared scalar head differs from the current toolchain head\n' \
    >&2
  exit 2
}
for provenance_input in VERSION.env scripts/build_guard.sh \
  scripts/install_tapenade_linux.sh \
  scripts/generate_coupled_on_cluster.sh \
  scripts/generate_coupled_nino3.sh; do
  test -s "${toolchain_dir}/${provenance_input}" || {
    printf 'ERROR: generation provenance input is missing: %s\n' \
      "${toolchain_dir}/${provenance_input}" >&2
    exit 2
  }
done
for required in zc_kernel_nino3.f zc_kernel_api.f ssta.f ztmfc1.f \
  cforce.f mloop.f akcalc.f bndary.f uhcalc.f uhinit.f tridag.f \
  ZC_lib_routines/fft2c.f; do
  test -s "${prepared_dir}/${required}" || {
    printf 'ERROR: required prepared input is missing: %s\n' \
      "${prepared_dir}/${required}" >&2
    exit 2
  }
done
zc_assert_stage_replacement_allowed "${generated_dir}" 0
generated_target_state="$(zc_capture_path_state "${generated_dir}")"
script_state="$(zc_capture_path_state "${BASH_SOURCE[0]}")"
guard_state="$(zc_capture_path_state \
  "${toolchain_dir}/scripts/build_guard.sh")"
version_state="$(zc_capture_path_state "${toolchain_dir}/VERSION.env")"
installer_state="$(zc_capture_path_state \
  "${toolchain_dir}/scripts/install_tapenade_linux.sh")"
cluster_helper_state="$(zc_capture_path_state \
  "${toolchain_dir}/scripts/generate_coupled_on_cluster.sh")"
java_state="$(zc_capture_path_state "${java_path}")"

workspace="$(zc_create_stage_workspace "${build_dir}" generate)"
workspace_identity="$(zc_path_identity "${workspace}")"
generated_target="${generated_dir}"
prepared_live="${prepared_dir}"
tapenade_live="${tapenade_home}"
generated_dir="${workspace}/payload"
input_snapshot="${workspace}/inputs"
prepared_dir="${input_snapshot}/prepared"
tapenade_home="${input_snapshot}/tapenade"
snapshot_toolchain="${input_snapshot}/toolchain"
mkdir -p "${generated_dir}/tangent" "${generated_dir}/reverse" \
  "${prepared_dir}" "${tapenade_home}" "${snapshot_toolchain}/scripts"

cleanup_generate_workspace() {
  local status="$?"
  trap - EXIT HUP INT TERM
  zc_remove_stage_workspace "${build_dir}" "${workspace}" \
    "${workspace_identity}" || status=2
  exit "${status}"
}
trap cleanup_generate_workspace EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

cp -Rp "${prepared_live}/." "${prepared_dir}/"
cp -Rp "${tapenade_live}/." "${tapenade_home}/"
for provenance_input in VERSION.env scripts/build_guard.sh \
  scripts/install_tapenade_linux.sh \
  scripts/generate_coupled_on_cluster.sh \
  scripts/generate_coupled_nino3.sh; do
  cp "${toolchain_dir}/${provenance_input}" \
    "${snapshot_toolchain}/${provenance_input}"
done
tapenade="${tapenade_home}/bin/tapenade"
version_output="$("${tapenade}" -version 2>&1)"
grep -Fq "Tapenade ${TAPENADE_VERSION}" <<<"${version_output}"
grep -Fq "Revision: ${TAPENADE_REVISION}" <<<"${version_output}"
prepared_manifest="${prepared_dir}/prepared_source_manifest.sha256"
prepared_binding="${prepared_dir}/prepared_manifest_binding.txt"
preparation_provenance="${prepared_dir}/preparation_provenance.txt"
upstream_manifest="${prepared_dir}/upstream_preprocessed_manifest.sha256"
kernel_source_manifest="${prepared_dir}/kernel_source_input_manifest.sha256"
zc_verify_sha256_manifest_exact "${prepared_dir}" "${prepared_manifest}" \
  prepared_source_manifest.sha256 prepared_manifest_binding.txt \
  upstream_preprocessed_manifest.sha256 kernel_source_input_manifest.sha256
zc_verify_tapenade_install "${tapenade_home}" "${snapshot_toolchain}" \
  "${TAPENADE_VERSION}" "${TAPENADE_REVISION}" \
  "${TAPENADE_ARCHIVE_SHA256}"
find "${input_snapshot}" -type f -exec chmod a-w {} +
find "${input_snapshot}" -type d -exec chmod a-w {} +
prepared_snapshot_state="$(zc_capture_path_state "${prepared_dir}")"
tapenade_snapshot_state="$(zc_capture_path_state "${tapenade_home}")"
toolchain_snapshot_state="$(zc_capture_path_state "${snapshot_toolchain}")"
sources=(
  "${prepared_dir}/zc_kernel_nino3.f"
  "${prepared_dir}/zc_kernel_api.f"
  "${prepared_dir}/ssta.f"
  "${prepared_dir}/ztmfc1.f"
  "${prepared_dir}/cforce.f"
  "${prepared_dir}/mloop.f"
  "${prepared_dir}/akcalc.f"
  "${prepared_dir}/bndary.f"
  "${prepared_dir}/uhcalc.f"
  "${prepared_dir}/uhinit.f"
  "${prepared_dir}/tridag.f"
  "${prepared_dir}/ZC_lib_routines/fft2c.f"
)
head_spec='zc_kernel_nino3(nino3)/(state_r_in)'
nocheckpoint_units='zc_kernel_window zc_kernel_unpack zc_kernel_advance_common zc_kernel_pack zc_unpack_real zc_unpack_real_scalar zc_unpack_real_2d zc_unpack_complex zc_pack_real zc_pack_real_scalar zc_pack_real_2d zc_pack_complex ssta getn zavg zavg_filter zatmc zqgen trid stress cforce mloop uhinit akcalc bndary uhcalc tridag fft2c'

"${tapenade}" -tangent -head "${head_spec}" -I "${prepared_dir}" \
  -O "${generated_dir}/tangent" "${sources[@]}"
"${tapenade}" -reverse -nocheckpoint "${nocheckpoint_units}" \
  -head "${head_spec}" -I "${prepared_dir}" \
  -O "${generated_dir}/reverse" "${sources[@]}"

test -s "${generated_dir}/tangent/zc_kernel_nino3_d.f"
test -s "${generated_dir}/reverse/zc_kernel_nino3_b.f"
printf '%s\n' "${version_output}" > "${generated_dir}/tapenade_version.txt"

{
  printf 'stage=generate_coupled_nino3\n'
  printf 'script_sha256=%s\n' \
    "$(sha256_file \
      "${snapshot_toolchain}/scripts/generate_coupled_nino3.sh")"
  printf 'build_guard_sha256=%s\n' \
    "$(sha256_file "${snapshot_toolchain}/scripts/build_guard.sh")"
  printf 'version_env_sha256=%s\n' \
    "$(sha256_file "${snapshot_toolchain}/VERSION.env")"
  printf 'installer_script_sha256=%s\n' \
    "$(sha256_file \
      "${snapshot_toolchain}/scripts/install_tapenade_linux.sh")"
  printf 'cluster_helper_script_sha256=%s\n' \
    "$(sha256_file \
      "${snapshot_toolchain}/scripts/generate_coupled_on_cluster.sh")"
  printf 'prepared_source_manifest_sha256=%s\n' \
    "$(sha256_file "${prepared_manifest}")"
  printf 'prepared_manifest_binding_sha256=%s\n' \
    "$(sha256_file "${prepared_binding}")"
  printf 'preparation_provenance_sha256=%s\n' \
    "$(sha256_file "${preparation_provenance}")"
  printf '%s\n' 'tapenade_executable=<VERIFIED_INPUT_SNAPSHOT>/bin/tapenade'
  printf 'tapenade_executable_sha256=%s\n' \
    "$(sha256_file "${tapenade}")"
  printf 'fortran_parser_sha256=%s\n' \
    "$(sha256_file "${tapenade_home}/bin/linux/fortranParser")"
  printf 'tapenade_expected_archive_sha256=%s\n' \
    "${TAPENADE_ARCHIVE_SHA256}"
  printf 'tapenade_observed_archive_sha256=%s\n' \
    "$(zc_binding_value \
      "${tapenade_home}/.zc_tapenade_install_receipt.txt" \
      observed_archive_sha256)"
  printf 'tapenade_install_receipt_sha256=%s\n' \
    "$(sha256_file \
      "${tapenade_home}/.zc_tapenade_install_receipt.txt")"
  printf 'tapenade_tree_manifest_sha256=%s\n' \
    "$(sha256_file \
      "${tapenade_home}/.zc_tapenade_tree_manifest.sha256")"
  printf 'tapenade_tree_file_count=%s\n' \
    "$(zc_binding_value \
      "${tapenade_home}/.zc_tapenade_install_receipt.txt" \
      tree_file_count)"
  printf 'tapenade_revision=%s\n' "${TAPENADE_REVISION}"
  printf 'java_path=%s\n' "${java_path}"
  printf 'java_sha256=%s\n' "$(sha256_file "${java_path}")"
  printf 'java_version=%s\n' "${java_version}"
  printf 'tangent_head=%s\n' "${head_spec}"
  printf 'reverse_nocheckpoint_units=%s\n' "${nocheckpoint_units}"
  printf '%s\n' \
    'tangent_recipe=tapenade -tangent -head <HEAD> -I <PREPARED> -O <TANGENT> <SOURCES>'
  printf '%s\n' \
    'reverse_recipe=tapenade -reverse -nocheckpoint <UNITS> -head <HEAD> -I <PREPARED> -O <REVERSE> <SOURCES>'
} > "${generated_dir}/generation_provenance.txt"
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
} > "${generated_dir}/generated_source_manifest.pre_instrument.sha256"
{
  printf 'binding_schema=zc-generation-manifest-binding-v1\n'
  printf 'generated_pre_manifest_sha256=%s\n' \
    "$(sha256_file \
      "${generated_dir}/generated_source_manifest.pre_instrument.sha256")"
  printf 'generation_provenance_sha256=%s\n' \
    "$(sha256_file "${generated_dir}/generation_provenance.txt")"
  printf 'prepared_source_manifest_sha256=%s\n' \
    "$(sha256_file "${prepared_manifest}")"
  printf 'prepared_manifest_binding_sha256=%s\n' \
    "$(sha256_file "${prepared_binding}")"
  printf 'tapenade_install_receipt_sha256=%s\n' \
    "$(sha256_file \
      "${tapenade_home}/.zc_tapenade_install_receipt.txt")"
  printf 'tapenade_tree_manifest_sha256=%s\n' \
    "$(sha256_file \
      "${tapenade_home}/.zc_tapenade_tree_manifest.sha256")"
} > "${generated_dir}/generation_manifest_binding.txt"
zc_verify_sha256_manifest_exact "${generated_dir}" \
  "${generated_dir}/generated_source_manifest.pre_instrument.sha256" \
  generated_source_manifest.pre_instrument.sha256 \
  generation_provenance.txt generation_manifest_binding.txt
zc_verify_binding_file_hash \
  "${generated_dir}/generation_manifest_binding.txt" \
  generated_pre_manifest_sha256 \
  "${generated_dir}/generated_source_manifest.pre_instrument.sha256"
zc_verify_binding_file_hash \
  "${generated_dir}/generation_manifest_binding.txt" \
  generation_provenance_sha256 \
  "${generated_dir}/generation_provenance.txt"

# Tapenade consumed only copied, checksum-bound inputs. Reverify those exact
# copies and the live executing scripts/tool immediately before publication.
zc_test_failpoint generate_after_tool_execution
zc_verify_path_state "${prepared_dir}" "${prepared_snapshot_state}"
zc_verify_sha256_manifest_exact "${prepared_dir}" "${prepared_manifest}" \
  prepared_source_manifest.sha256 prepared_manifest_binding.txt \
  upstream_preprocessed_manifest.sha256 kernel_source_input_manifest.sha256
zc_verify_path_state "${tapenade_home}" "${tapenade_snapshot_state}"
zc_verify_tapenade_install "${tapenade_home}" "${snapshot_toolchain}" \
  "${TAPENADE_VERSION}" "${TAPENADE_REVISION}" \
  "${TAPENADE_ARCHIVE_SHA256}"
zc_verify_path_state "${snapshot_toolchain}" "${toolchain_snapshot_state}"
zc_verify_path_state "${BASH_SOURCE[0]}" "${script_state}"
zc_verify_path_state "${toolchain_dir}/scripts/build_guard.sh" \
  "${guard_state}"
zc_verify_path_state "${toolchain_dir}/VERSION.env" "${version_state}"
zc_verify_path_state \
  "${toolchain_dir}/scripts/install_tapenade_linux.sh" \
  "${installer_state}"
zc_verify_path_state \
  "${toolchain_dir}/scripts/generate_coupled_on_cluster.sh" \
  "${cluster_helper_state}"
zc_verify_path_state "${java_path}" "${java_state}"
zc_publish_stage_directory "${build_dir}" "${generated_target}" \
  "${generated_dir}" "${generated_target_state}" standard

trap - EXIT HUP INT TERM
zc_remove_stage_workspace "${build_dir}" "${workspace}" \
  "${workspace_identity}"
workspace=""
generated_dir="${generated_target}"
printf 'Generated coupled Nino-3 tangent/reverse in %s\n' "${generated_dir}"
