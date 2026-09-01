#!/usr/bin/env bash
###############################################################################
# run_pipeline.sh
# ================
# Orchestration complète : diagnostic auto -> génération des séquences
# synthétiques -> figures + résumé pour le rapport. Conçu pour tourner en
# arrière-plan via nohup, indépendamment de la session SSH.
#
# CONÇU POUR ÊTRE LANCÉ AINSI (pas juste "bash run_pipeline.sh") :
#
#     nohup bash run_pipeline.sh > logs/orchestration.log 2>&1 &
#     disown
#
# Explication :
#   - nohup  : le process ignore le signal SIGHUP envoyé quand la session SSH
#              se ferme (fermeture du laptop, déconnexion réseau, etc.) -> il
#              continue de tourner sur le SERVEUR, indépendamment de ton
#              laptop (qui n'est de toute façon que le client SSH ici).
#   - &      : lance en arrière-plan, rend la main au shell immédiatement.
#   - disown : détache le process de la table des jobs du shell courant, pour
#              qu'un "exit"/fermeture de la session ne lui envoie rien du tout
#              (nohup seul suffit déjà dans la plupart des cas, disown est une
#              ceinture de sécurité supplémentaire, sans effet de bord).
#
# SUIVI DE L'EXÉCUTION (depuis une nouvelle session SSH si besoin) :
#   tail -f logs/orchestration.log          # log global (ce script)
#   tail -f logs/sequence_generation_v2.log # log de la génération elle-même
#   ls out_sequences_v2/PIPELINE_DONE       # existe seulement si tout a réussi
#   ls out_sequences_v2/PIPELINE_FAILED     # existe si une étape a échoué
#                                           # (contient le nom de l'étape en échec)
#   ps aux | grep sequence_generation_v2    # vérifier que le process tourne encore
#
# DÉCISION AUTOMATIQUE lstm vs iid :
#   Une passe --diagnostic-only (quelques secondes, pas de GPU/CTGAN) calcule
#   l'autocorrélation lag-1 réelle du delta_t. Si |autocorr| >= DELTA_T_CORR_THRESHOLD
#   (0.05 par défaut, cf. la logique déjà discutée avec Claude), le pipeline
#   utilise --delta-t-mode lstm (entraînement complet). Sinon, --delta-t-mode
#   iid (bootstrap direct, pas d'entraînement) -- pas besoin d'être présent
#   pour trancher.
#
# Usage : adapte les variables ci-dessous à ton environnement puis lance avec
# nohup comme indiqué plus haut.
###############################################################################
set -euo pipefail

# ----------------------------------------------------------------------
# CONFIGURATION -- valeurs par défaut, surchageables via variables d'env
# (ex: SEQUENCES_CSV=chemin/vers/fichier.csv nohup bash run_pipeline.sh ...)
# ----------------------------------------------------------------------
SEQUENCES_CSV="${SEQUENCES_CSV:-out_sequences/sequences_host_pair.csv}"
CTGAN_MODEL_PATH="${CTGAN_MODEL_PATH:-ctgan_model_mv.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-out_sequences_v2}"
LOG_DIR="${LOG_DIR:-logs}"
EPOCHS="${EPOCHS:-50}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
STOCK_SIZE="${STOCK_SIZE:-50000}"
N_SYNTHETIC_SEQUENCES="${N_SYNTHETIC_SEQUENCES:-2000}"
DELTA_T_CORR_THRESHOLD="${DELTA_T_CORR_THRESHOLD:-0.05}"   # seuil |autocorrélation| en dessous duquel on bascule en iid

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQ_GEN_SCRIPT="${SCRIPT_DIR}/sequence_generation_v2.py"
FIGURES_SCRIPT="${SCRIPT_DIR}/generate_report_figures.py"

# ----------------------------------------------------------------------
# VÉRIFICATION DES CHEMINS -- échec propre et explicite si un fichier
# attendu n'existe pas, plutôt qu'un traceback Python confus 10 minutes
# après le lancement en arrière-plan.
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# GESTION D'ERREUR : si une étape échoue, on l'écrit dans un fichier
# marqueur clair plutôt que de laisser un traceback perdu dans un log.
# ----------------------------------------------------------------------
CURRENT_STEP="initialisation"
on_error() {
    echo "[ÉCHEC] Étape en cours au moment de l'erreur : ${CURRENT_STEP}" | tee "${OUTPUT_DIR}/PIPELINE_FAILED"
    echo "[ÉCHEC] Consulte ${LOG_DIR}/ pour le détail de l'erreur." | tee -a "${OUTPUT_DIR}/PIPELINE_FAILED"
}
trap on_error ERR

echo "=========================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Démarrage du pipeline"
echo "=========================================================="

# ----------------------------------------------------------------------
# ÉTAPE 1 : diagnostic rapide (autocorrélation réelle du delta_t)
# ----------------------------------------------------------------------
CURRENT_STEP="diagnostic autocorrélation delta_t"
echo "[$(date '+%H:%M:%S')] ${CURRENT_STEP}..."
python3 "$SEQ_GEN_SCRIPT" \
    --sequences-csv "$SEQUENCES_CSV" \
    --diagnostic-only \
    --output-dir "$OUTPUT_DIR" \
    | tee "${LOG_DIR}/diagnostic.log"

# extraction de la valeur numérique de l'autocorrélation depuis le log
# (format : "[DIAG] Autocorrélation lag-1 réelle de log1p(delta_t) ... : X.XXXX (moyenne ...")
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

# ----------------------------------------------------------------------
# ÉTAPE 2 : génération complète (entraînement si lstm + génération)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# ÉTAPE 3 : figures + brouillon de résumé pour le rapport
# ----------------------------------------------------------------------
CURRENT_STEP="génération des figures et du résumé (étape 8)"
echo "[$(date '+%H:%M:%S')] ${CURRENT_STEP}..."
python3 "$FIGURES_SCRIPT" \
    --real-sequences-csv "${OUTPUT_DIR}/real_sequences_used.csv" \
    --synth-sequences-csv "${OUTPUT_DIR}/synthetic_sequences.csv" \
    --delta-t-mode "$DELTA_T_MODE" \
    --output-dir "${OUTPUT_DIR}/figures" \
    | tee "${LOG_DIR}/report_figures.log"

# ----------------------------------------------------------------------
# FIN
# ----------------------------------------------------------------------
trap - ERR   # désarme le trap d'erreur, tout s'est bien passé
echo "delta_t_mode=${DELTA_T_MODE}" > "${OUTPUT_DIR}/PIPELINE_DONE"
echo "$(date '+%Y-%m-%d %H:%M:%S')" >> "${OUTPUT_DIR}/PIPELINE_DONE"
echo "=========================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline terminé avec succès (mode=${DELTA_T_MODE})"
echo "Résultats : ${OUTPUT_DIR}/synthetic_sequences.csv"
echo "Figures   : ${OUTPUT_DIR}/figures/"
echo "Résumé    : ${OUTPUT_DIR}/figures/step8_summary_draft.md"
echo "=========================================================="