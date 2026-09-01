#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
: > logs/noise_sweep.log

for NOISE in 1.0 1.2 1.5; do
    echo "=== NOISE STD = ${NOISE} ===" | tee -a logs/noise_sweep.log

    python3 -u sequence_generation_v2.py \
        --sequences-csv out_sequences/sequences_host_pair.csv \
        --ctgan-model-path ctgan_model_mv.pkl \
        --delta-t-mode lstm \
        --delta-t-noise-std "${NOISE}" \
        --n-synthetic-sequences 2000 \
        --output-dir "out_sequences_v2_noise_${NOISE}" \
        2>&1 | tee -a "logs/noise_sweep_full_${NOISE}.log" \
        | grep --line-buffered -A2 "delta_t_seconds :" \
        | tee -a logs/noise_sweep.log

    echo "" | tee -a logs/noise_sweep.log
done

echo "Terminé. Résumé dans logs/noise_sweep.log"
