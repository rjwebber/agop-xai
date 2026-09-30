#!/usr/bin/env bash
set -u -o pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="${PYTHON_EXECUTABLE:-python}"
output_root="${repository_root}/scratch/zc_pooled_composite_covariance_action/core4-cnn-lead10-seed42-direct-first-adaptive-v1"
log_dir="${output_root}/logs"
mkdir -p "${log_dir}"

overall_status=0
for target_event in extreme_el_nino extreme_la_nina; do
    event_log="${log_dir}/${target_event}.log"
    {
        printf '%s | starting %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${target_event}"
        "${python_executable}" -u \
            "${repository_root}/scripts/run_zc_pooled_composite_covariance_action.py" \
            --target-event "${target_event}" \
            --output-root "${output_root}" \
            --covariance-centered-cache-gib 24
        arm_status=$?
        printf '%s | finished %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${target_event}"
        printf 'exit_status=%d\n' "${arm_status}"
        exit "${arm_status}"
    } 2>&1 | tee -a "${event_log}"
    arm_status=${PIPESTATUS[0]}
    if [[ ${arm_status} -ne 0 ]]; then
        overall_status=${arm_status}
    fi
done

exit "${overall_status}"
