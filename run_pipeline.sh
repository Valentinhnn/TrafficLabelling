#!/usr/bin/env bash
set -euo pipefail

# Config: all overridable via env vars.
SEQUENCES_CSV="${SEQUENCES_CSV:-out_sequences/sequences_host_pair.csv}"
CTGAN_MODEL_PATH="${CTGAN_MODEL_PATH:-ctgan_model_mv.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-out_sequences_v2}"
LOG_DIR="${LOG_DIR:-logs}"
EPOCHS="${EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
STOCK_SIZE="${STOCK_SIZE:-50000}"
N_SYNTHETIC_SEQUENCES="${N_SYNTHETIC_SEQUENCES:-2000}"
DELTA_T_CORR_THRESHOLD="${DELTA_T_CORR_THRESHOLD:-0.05}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQ_GEN_SCRIPT="${SCRIPT_DIR}/sequence_generation_v2.py"
FIGURES_SCRIPT="${SCRIPT_DIR}/generate_report_figures.py"

# Fail early with a clear message if a required file is missing.
for f in "$SEQUENCES_CSV" "$CTGAN_MODEL_PATH" "$SEQ_GEN_SCRIPT" "$FIGURES_SCRIPT"; do
    if [ ! -f "$f" ]; then
        echo "[ERREUR] Fichier introuvable : $f"
        echo "[ERREUR] Vérifie le chemin, ou passe-le en variable d'env, ex :"
        echo "  SEQUENCES_CSV=... CTGAN_MODEL_PATH=... nohup bash run_pipeline.sh > logs/orchestration.log 2>&1 &"
        exit 1
    fi
done

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
rm -f "${OUTPUT_DIR}/PIPELINE_DONE" "${OUTPUT_DIR}/PIPELINE_FAILED"

CURRENT_STEP="initialisation"
# On failure: writes a clear marker file naming the step that failed.
on_error() {
    echo "[ÉCHEC] Étape en cours au moment de l'erreur : ${CURRENT_STEP}" | tee "${OUTPUT_DIR}/PIPELINE_FAILED"
    echo "[ÉCHEC] Consulte ${LOG_DIR}/ pour le détail de l'erreur." | tee -a "${OUTPUT_DIR}/PIPELINE_FAILED"
}
trap on_error ERR

echo "=========================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Démarrage du pipeline"
echo "=========================================================="

# Step 1: quick diagnostic run to measure real delta_t autocorrelation.
CURRENT_STEP="diagnostic autocorrélation delta_t"
echo "[$(date '+%H:%M:%S')] ${CURRENT_STEP}..."
python3 "$SEQ_GEN_SCRIPT" \
    --sequences-csv "$SEQUENCES_CSV" \
    --diagnostic-only \
    --output-dir "$OUTPUT_DIR" \
    | tee "${LOG_DIR}/diagnostic.log"

# Extracts the numeric autocorrelation value from the diagnostic log,
# then picks lstm vs iid mode based on the threshold.
CORR_VALUE=$(grep -oP '(?<=host_pair : )-?[0-9]+\.[0-9]+' "${LOG_DIR}/diagnostic.log" | head -1)

if [ -z "$CORR_VALUE" ]; then
    echo "[WARN] Impossible d'extraire l'autocorrélation du log -> fallback sur --delta-t-mode lstm par défaut"
    DELTA_T_MODE="lstm"
else
    ABS_CORR=$(python3 -c "print(abs(${CORR_VALUE}))")
    IS_ABOVE=$(python3 -c "print(1 if ${ABS_CORR} >= ${DELTA_T_CORR_THRESHOLD} else 0)")
    if [ "$IS_ABOVE" = "1" ]; then
        DELTA_T_MODE="lstm"
    else
        DELTA_T_MODE="iid"
    fi
    echo "[INFO] Autocorrélation delta_t = ${CORR_VALUE} (|.|=${ABS_CORR}, seuil=${DELTA_T_CORR_THRESHOLD}) -> delta-t-mode = ${DELTA_T_MODE}"
fi

# Step 2: full generation run (trains the LSTM if mode=lstm, then generates).
CURRENT_STEP="génération des séquences synthétiques (delta-t-mode=${DELTA_T_MODE})"
echo "[$(date '+%H:%M:%S')] ${CURRENT_STEP}..."
python3 "$SEQ_GEN_SCRIPT" \
    --sequences-csv "$SEQUENCES_CSV" \
    --ctgan-model-path "$CTGAN_MODEL_PATH" \
    --epochs "$EPOCHS" --hidden-dim "$HIDDEN_DIM" --stock-size "$STOCK_SIZE" \
    --n-synthetic-sequences "$N_SYNTHETIC_SEQUENCES" \
    --delta-t-mode "$DELTA_T_MODE" \
    --force-retrain \
    --output-dir "$OUTPUT_DIR" \
    | tee "${LOG_DIR}/sequence_generation_v2.log"

# Step 3: report figures + summary draft.
CURRENT_STEP="génération des figures et du résumé (étape 8)"
echo "[$(date '+%H:%M:%S')] ${CURRENT_STEP}..."
python3 "$FIGURES_SCRIPT" \
    --real-sequences-csv "${OUTPUT_DIR}/real_sequences_used.csv" \
    --synth-sequences-csv "${OUTPUT_DIR}/synthetic_sequences.csv" \
    --delta-t-mode "$DELTA_T_MODE" \
    --output-dir "${OUTPUT_DIR}/figures" \
    | tee "${LOG_DIR}/report_figures.log"

trap - ERR
echo "delta_t_mode=${DELTA_T_MODE}" > "${OUTPUT_DIR}/PIPELINE_DONE"
echo "$(date '+%Y-%m-%d %H:%M:%S')" >> "${OUTPUT_DIR}/PIPELINE_DONE"
echo "=========================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline terminé avec succès (mode=${DELTA_T_MODE})"
echo "Résultats : ${OUTPUT_DIR}/synthetic_sequences.csv"
echo "Figures   : ${OUTPUT_DIR}/figures/"
echo "Résumé    : ${OUTPUT_DIR}/figures/step8_summary_draft.md"
echo "=========================================================="
