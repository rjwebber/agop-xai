#!/usr/bin/env bash
set -euo pipefail

# Evaluate the canonical five CNN seeds for revised manuscript Figure 7.
#
# This runner is intentionally serial.  Each seed uses the same GPU and writes
# a separate completed bundle, so a terminated run can resume at a seed
# boundary.  The per-seed generator hash-validates completed bundles before
# skipping them.  Its ordinary --overwrite repairs a partial seed bundle but
# does not rebuild a valid exact AGOP cache; rebuilding those caches requires
# the separate, intentionally absent --overwrite-agop flag.

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_dir}"

device="${FIGURE7_DEVICE:-cuda}"
case "${device}" in
    auto|cpu|mps|cuda) ;;
    *)
        echo "FIGURE7_DEVICE must be one of auto, cpu, mps, or cuda." >&2
        exit 2
        ;;
esac

data_dir="data/processed/zc-v3"
artifacts_dir="artifacts/zc-v3"
bundle_root="artifacts/zc-v3/manuscript/figure7"
agop_cache_root="scratch/figure7_exact_agop"
figure_output="outputs/manuscript/figures/figure7_zcv3_core4.pdf"
summary_output="outputs/manuscript/figures/figure7_zcv3_core4_scores.csv"
seeds=(42 43 44 45 46)
leads=(1 2 3 4 5 6 7 8 9 10 11 12)

required_paths=(
    "${data_dir}/metadata.json"
    "scripts/generate_fresh_figure7_data.py"
    "scripts/make_fresh_figure7.py"
)
for path in "${required_paths[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required Figure 7 input is missing: ${repository_dir}/${path}" >&2
        exit 1
    fi
done
if [[ ! -d "${artifacts_dir}/figure5" ]]; then
    echo "Figure 5 model caches are missing: ${repository_dir}/${artifacts_dir}/figure5" >&2
    echo "Copy the completed five-seed Figure 5 artifacts before running Figure 7." >&2
    exit 1
fi

mkdir -p \
    "${bundle_root}" \
    "${agop_cache_root}" \
    "$(dirname "${figure_output}")"
export PYTHONUNBUFFERED=1

for seed in "${seeds[@]}"; do
    seed_directory="${bundle_root}/seed-$(printf '%06d' "${seed}")"
    echo "$(date) Starting or validating Figure 7 seed ${seed}."
    python scripts/generate_fresh_figure7_data.py \
        --data-dir "${data_dir}" \
        --artifacts-dir "${artifacts_dir}" \
        --agop-cache-root "${agop_cache_root}" \
        --output-dir "${seed_directory}" \
        --lead-months "${leads[@]}" \
        --seed "${seed}" \
        --device "${device}" \
        --resume-complete \
        --overwrite
    echo "$(date) Figure 7 seed ${seed} is complete and validated."
done

echo "$(date) All five seed bundles are complete; aggregating and rendering Figure 7."
python scripts/make_fresh_figure7.py \
    --bundle-root "${bundle_root}" \
    --output "${figure_output}" \
    --summary-output "${summary_output}" \
    --overwrite

echo "$(date) Five-seed Figure 7 completed: ${repository_dir}/${figure_output}"
