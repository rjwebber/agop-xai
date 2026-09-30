#!/usr/bin/env bash
set -euo pipefail

# Measure the manuscript XAI configuration on one four-core CPU pod. Large
# diagnostic arrays live only in the pod's temporary directory; compact JSON
# reports and a hardware/software log are retained in the repository.

repository_dir="${HOME}/agop-xai"
container_repository_dir="/home/${USER}/agop-xai"
launcher="/opt/launch-sh/bin/launch.sh"
pod_name="${XAI_TIMING_POD_NAME:-agop-xai-cpu4-timing-v1}"
result_dir="outputs/manuscript/benchmarks/xai_cpu4"
export K8S_TIMEOUT_SECONDS="${K8S_TIMEOUT_SECONDS:-172800}"

if [[ ! -x "${launcher}" ]]; then
    echo "UCSD Research Cluster launcher not found: ${launcher}" >&2
    exit 1
fi
if [[ ! -f "${repository_dir}/data/processed/zc-v3/metadata.json" ]]; then
    echo "Processed zc-v3 data are missing below ${repository_dir}." >&2
    exit 1
fi
if [[ ! -f "${repository_dir}/artifacts/zc-v3/models/core4/cnn/lead-10m/years-10000/seed-000042/completed.json" ]]; then
    echo "The core4 CNN lead-10 seed-42 checkpoint is missing." >&2
    exit 1
fi

mkdir -p "${repository_dir}/${result_dir}"

"${launcher}" \
    -B \
    -p low \
    -N "${pod_name}" \
    -c 4 \
    -m 16 \
    bash -lc \
    "set -euo pipefail
     cd '${container_repository_dir}'
     export OMP_NUM_THREADS=4
     export MKL_NUM_THREADS=4
     export OPENBLAS_NUM_THREADS=4
     export NUMEXPR_NUM_THREADS=4
     export OMP_DYNAMIC=FALSE
     scratch_dir=\$(mktemp -d /tmp/agop-xai-cpu4-timing.XXXXXX)
     trap 'rm -rf \"\${scratch_dir}\"' EXIT
     mkdir -p '${result_dir}'
     {
       date
       echo 'CPU allocation: 4 cores; memory allocation: 16 GiB'
       lscpu
       python - <<'PY'
import platform
import numpy
import scipy
import torch

print(f'platform={platform.platform()}')
print(f'python={platform.python_version()}')
print(f'numpy={numpy.__version__}')
print(f'scipy={scipy.__version__}')
print(f'torch={torch.__version__}')
print(f'torch_num_threads={torch.get_num_threads()}')
print(f'torch_num_interop_threads={torch.get_num_interop_threads()}')
PY
       python scripts/benchmark_fresh_agop.py \
         --data-dir data/processed/zc-v3 \
         --artifacts-dir artifacts/zc-v3 \
         --output-dir \"\${scratch_dir}/agop\" \
         --input-profile core4 \
         --architecture cnn \
         --lead-months 10 \
         --seed 42 \
         --device cpu \
         --reference-count 0 \
         --gradient-batch-size 1024 \
         --overwrite
       python scripts/benchmark_fresh_xai_methods.py \
         --data-dir data/processed/zc-v3 \
         --artifacts-dir artifacts/zc-v3 \
         --agop-benchmark-dir \"\${scratch_dir}/agop\" \
         --output-dir \"\${scratch_dir}/methods\" \
         --input-profile core4 \
         --architecture cnn \
         --lead-months 10 \
         --seed 42 \
         --device cpu \
         --neighbor-percent 1 \
         --neighbor-samples 1 \
         --ig-steps 1024 \
         --gradient-shap-samples 1024 \
         --gradient-batch-size 1024 \
         --fused-pair-batch-size 1024 \
         --execution-modes current \
         --overwrite
       cp \"\${scratch_dir}/agop/report.json\" '${result_dir}/agop_report.json'
       cp \"\${scratch_dir}/methods/report.json\" '${result_dir}/method_report.json'
       sha256sum '${result_dir}/agop_report.json' '${result_dir}/method_report.json'
       date
     } 2>&1 | tee '${result_dir}/benchmark.log'"

echo "Submitted ${pod_name}."
echo "Follow it with: source /opt/launch-sh/lib/kubevars.sh"
echo "                kubectl logs -f ${pod_name}"
