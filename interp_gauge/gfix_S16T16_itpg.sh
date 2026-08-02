#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input_template="${script_dir}/input_S16T16_interpolating.txt"
gauge_dir="${script_dir}/../ensemble/S16T16"
output_dir="${script_dir}/../ensemble/S16T16_itpg"
epsilons=(0.5 0.3 0.1)

cd "${script_dir}"

if [[ ! -x ./GLU ]]; then
    echo "ERROR: expected executable ./GLU in ${script_dir}" >&2
    exit 1
fi

if [[ ! -f "${input_template}" ]]; then
    echo "ERROR: missing input template ${input_template}" >&2
    exit 1
fi

if [[ ! -d "${gauge_dir}" ]]; then
    echo "ERROR: missing gauge directory ${gauge_dir}" >&2
    exit 1
fi

mkdir -p "${output_dir}"

tmp_inputs=()
cleanup() {
    if ((${#tmp_inputs[@]} > 0)); then
        rm -f "${tmp_inputs[@]}"
    fi
}
trap cleanup EXIT

config_numbers=()
for gauge_file in "${gauge_dir}"/wilson_b6.[0-9]*; do
    [[ -e "${gauge_file}" ]] || continue
    n_conf="${gauge_file##*.}"
    [[ "${n_conf}" =~ ^[0-9]+$ ]] || continue
    config_numbers+=("${n_conf}")
done

if ((${#config_numbers[@]} == 0)); then
    echo "ERROR: found no numeric wilson_b6.<n> files in ${gauge_dir}" >&2
    exit 1
fi

mapfile -t config_numbers < <(printf "%s\n" "${config_numbers[@]}" | sort -n)

start_time=$(date +%s)

echo "Start time: $(date)"
echo "Input template: ${input_template}"
echo "Gauge directory: ${gauge_dir}"
echo "Output directory: ${output_dir}"
echo "GF_EPSILON values: ${epsilons[*]}"
echo "Configurations: ${#config_numbers[@]}"

for epsilon in "${epsilons[@]}"; do
    epsilon_tag="${epsilon/./p}"
    tmp_input="$(mktemp "${script_dir}/input_S16T16_interpolating_eps${epsilon_tag}.XXXXXX.txt")"
    tmp_inputs+=("${tmp_input}")

    sed -E "s/^([[:space:]]*GF_EPSILON[[:space:]]*=[[:space:]]*).*/\\1${epsilon}/" \
        "${input_template}" > "${tmp_input}"

    echo " "
    echo "Processing GF_EPSILON = ${epsilon}"

    for n_conf in "${config_numbers[@]}"; do
        echo " "
        echo "Processing configuration ${n_conf} at GF_EPSILON = ${epsilon}"

        gauge_file="${gauge_dir}/wilson_b6.${n_conf}"
        gfixed_file="${output_dir}/wilson_b6.itpg.eps${epsilon_tag}.${n_conf}"

        config_start_time=$(date +%s)
        echo "Start time for config ${n_conf}, epsilon ${epsilon}: $(date)"

        ./GLU -i "${tmp_input}" -c "${gauge_file}" -o "${gfixed_file}"

        config_end_time=$(date +%s)
        config_elapsed_time=$((config_end_time - config_start_time))

        echo " "
        echo "Time for config ${n_conf}, epsilon ${epsilon}: $((config_elapsed_time / 3600)) hours $(((config_elapsed_time % 3600) / 60)) minutes $((config_elapsed_time % 60)) seconds"
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo " "
echo "Total time: ${total_time} seconds"
