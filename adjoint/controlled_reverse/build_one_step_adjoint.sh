#!/usr/bin/env bash
set -euo pipefail

driver_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd "${driver_dir}/../.." && pwd -P)"
canonical_build="${1:-${repository_root}/adjoint/tapenade_toolchain/build/coupled_run23_final3}"
requested_output="${2:-${driver_dir}/build}"
compiled="${canonical_build}/coupled_compiled"
driver="${driver_dir}/zc_one_step_adjoint_driver.F"
primal_driver="${repository_root}/adjoint/controlled_window/zc_one_step_driver.F"
tangent_driver="${repository_root}/adjoint/controlled_window/zc_one_step_tangent_driver.F"
runtime="${repository_root}/adjoint/variable_window/zc_variable_runtime.F"
object_manifest="${repository_root}/adjoint/variable_window/canonical_final3_objects.sha256"

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

manifest_value() {
  local key="$1"
  local file="$2"
  local count value
  count="$(awk -F= -v wanted="${key}" '$1 == wanted {n += 1} END {print n + 0}' "${file}")"
  value="$(awk -F= -v wanted="${key}" '$1 == wanted {sub(/^[^=]*=/, ""); print}' "${file}")"
  if [[ "${count}" != 1 || -z "${value}" ]]; then
    printf 'ERROR: manifest key is not unique: %s\n' "${key}" >&2
    exit 2
  fi
  printf '%s' "${value}"
}

if [[ -L "${canonical_build}" || ! -d "${compiled}" ]]; then
  printf 'ERROR: canonical final3 build is unavailable: %s\n' \
    "${canonical_build}" >&2
  exit 2
fi
build_manifest="${compiled}/build_manifest.txt"
for source in "${driver}" "${primal_driver}" "${tangent_driver}" \
  "${runtime}" "${object_manifest}" \
  "${build_manifest}"; do
  if [[ -L "${source}" || ! -s "${source}" ]]; then
    printf 'ERROR: required build input is missing or symbolic: %s\n' \
      "${source}" >&2
    exit 2
  fi
done

count=0
while read -r expected relative extra; do
  [[ -n "${expected:-}" ]] || continue
  if [[ -n "${extra:-}" || ! "${expected}" =~ ^[0-9a-f]{64}$ || \
        -z "${relative:-}" || "${relative}" = /* || \
        "${relative}" == *".."* ]]; then
    printf 'ERROR: invalid canonical object-manifest entry\n' >&2
    exit 2
  fi
  candidate="${compiled}/${relative}"
  if [[ -L "${candidate}" || ! -s "${candidate}" || \
        "$(sha256_file "${candidate}")" != "${expected}" ]]; then
    printf 'ERROR: canonical object mismatch: %s\n' "${candidate}" >&2
    exit 2
  fi
  count=$((count + 1))
done < "${object_manifest}"
if [[ "${count}" -ne 45 ]]; then
  printf 'ERROR: expected 45 canonical object entries\n' >&2
  exit 2
fi

expected_flags='-std=legacy -O0 -g -fcheck=all -fbacktrace -fallow-argument-mismatch -ffixed-line-length-none'
if [[ "$(manifest_value derivative_flags "${build_manifest}")" != \
      "${expected_flags}" ]]; then
  printf 'ERROR: canonical derivative flags changed\n' >&2
  exit 2
fi
read -r -a derivative_flags <<< "${expected_flags}"
platform_text="$(manifest_value platform_flags "${build_manifest}")"
platform_flags=()
if [[ -n "${platform_text}" ]]; then
  read -r -a platform_flags <<< "${platform_text}"
fi
requested_fc="${FC:-$(manifest_value compiler_path "${build_manifest}")}"
if [[ "${requested_fc}" == */* ]]; then
  fc_path="${requested_fc}"
else
  fc_path="$(command -v "${requested_fc}" || true)"
fi
if [[ -z "${fc_path}" || -L "${fc_path}" || ! -x "${fc_path}" ]]; then
  printf 'ERROR: canonical compiler unavailable: %s\n' "${requested_fc}" >&2
  exit 2
fi
fc_path="$(cd "$(dirname "${fc_path}")" && pwd -P)/$(basename "${fc_path}")"
compiler_sha256="$(manifest_value compiler_sha256 "${build_manifest}")"
if [[ "$(sha256_file "${fc_path}")" != "${compiler_sha256}" ]]; then
  printf 'ERROR: compiler differs from canonical build\n' >&2
  exit 2
fi

output_parent="$(dirname "${requested_output}")"
output_name="$(basename "${requested_output}")"
mkdir -p "${output_parent}"
output_parent="$(cd "${output_parent}" && pwd -P)"
output_dir="${output_parent}/${output_name}"
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  printf 'ERROR: output already exists: %s\n' "${output_dir}" >&2
  exit 2
fi
staging="$(mktemp -d "${output_parent}/.${output_name}.staging.XXXXXX")"
cleanup() {
  local status="$?"
  trap - EXIT HUP INT TERM
  if [[ -n "${staging:-}" && -d "${staging}" && \
        "${staging}" == "${output_parent}/.${output_name}.staging."* ]]; then
    rm -rf -- "${staging}"
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

include_dir="${compiled}/build_input_snapshot/prepared"
generated_dir="${compiled}/build_input_snapshot/generated"
primal_names=(
  zc_kernel_api ssta ztmfc1 cforce mloop akcalc bndary uhcalc uhinit tridag
)
for name in "${primal_names[@]}"; do
  source="${include_dir}/${name}.f"
  if [[ -L "${source}" || ! -s "${source}" ]]; then
    printf 'ERROR: immutable primal source is unavailable: %s\n' \
      "${source}" >&2
    exit 2
  fi
  "${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
    -I "${include_dir}" -c "${source}" -o "${staging}/${name}.o"
done
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -I "${include_dir}" -c "${include_dir}/ZC_lib_routines/fft2c.f" \
  -o "${staging}/fft2c.o"
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -I "${include_dir}" -c "${primal_driver}" \
  -o "${staging}/zc_one_step_driver.o"
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -I "${include_dir}" -c "${tangent_driver}" \
  -o "${staging}/zc_one_step_tangent_driver.o"
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -I "${include_dir}" -c "${runtime}" \
  -o "${staging}/zc_variable_runtime.o"
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -I "${include_dir}" -c "${driver}" \
  -o "${staging}/zc_one_step_adjoint_driver.o"

reverse_objects=(
  "${compiled}/reverse/akcalc_b.o"
  "${compiled}/reverse/bndary_b.o"
  "${compiled}/reverse/cforce_b.o"
  "${compiled}/reverse/fft2c_b.o"
  "${compiled}/reverse/mloop_b.o"
  "${compiled}/reverse/ssta_b.o"
  "${compiled}/reverse/tridag_b.o"
  "${compiled}/reverse/uhcalc_b.o"
  "${compiled}/reverse/uhinit_b.o"
  "${compiled}/reverse/zc_kernel_api_b.o"
  "${compiled}/reverse/zc_kernel_nino3_b.o"
  "${compiled}/reverse/ztmfc1_b.o"
)
support_objects=("${compiled}/support/"*.o)
primal_objects=(
  "${staging}/zc_kernel_api.o"
  "${staging}/ssta.o"
  "${staging}/ztmfc1.o"
  "${staging}/cforce.o"
  "${staging}/mloop.o"
  "${staging}/akcalc.o"
  "${staging}/bndary.o"
  "${staging}/uhcalc.o"
  "${staging}/uhinit.o"
  "${staging}/tridag.o"
  "${staging}/fft2c.o"
)
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -o "${staging}/zc_one_step_primal" \
  "${staging}/zc_one_step_driver.o" "${primal_objects[@]}" \
  "${support_objects[@]}"
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -o "${staging}/zc_one_step_tangent" \
  "${staging}/zc_one_step_tangent_driver.o" \
  "${compiled}/tangent/"*.o "${support_objects[@]}"
"${fc_path}" "${derivative_flags[@]}" "${platform_flags[@]}" \
  -o "${staging}/zc_one_step_adjoint" \
  "${staging}/zc_one_step_adjoint_driver.o" \
  "${staging}/zc_variable_runtime.o" \
  "${reverse_objects[@]}" "${staging}/ssta.o" \
  "${support_objects[@]}"

{
  printf 'schema=zc-controlled-one-step-consistent-build-v2\n'
  printf 'driver_sha256=%s\n' "$(sha256_file "${driver}")"
  printf 'primal_driver_sha256=%s\n' "$(sha256_file "${primal_driver}")"
  printf 'tangent_driver_sha256=%s\n' "$(sha256_file "${tangent_driver}")"
  printf 'runtime_sha256=%s\n' "$(sha256_file "${runtime}")"
  printf 'canonical_build_manifest_sha256=%s\n' \
    "$(sha256_file "${build_manifest}")"
  printf 'canonical_object_manifest_sha256=%s\n' \
    "$(sha256_file "${object_manifest}")"
  printf 'compiler_path=%s\n' "${fc_path}"
  printf 'compiler_sha256=%s\n' "${compiler_sha256}"
  printf 'derivative_flags=%s\n' "${expected_flags}"
  printf 'platform_flags=%s\n' "${platform_text}"
  printf 'scientific_primal_flags=%s\n' "${expected_flags}"
  printf 'zc_one_step_primal_sha256=%s\n' \
    "$(sha256_file "${staging}/zc_one_step_primal")"
  printf 'zc_one_step_tangent_sha256=%s\n' \
    "$(sha256_file "${staging}/zc_one_step_tangent")"
  printf 'adjoint_executable_sha256=%s\n' \
    "$(sha256_file "${staging}/zc_one_step_adjoint")"
} > "${staging}/build_provenance.txt"

mv "${staging}" "${output_dir}"
staging=''
trap - EXIT HUP INT TERM
printf '%s\n' "${output_dir}/zc_one_step_adjoint"
